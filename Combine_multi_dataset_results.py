import os
import math
import pickle
import pathlib
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import create_save_dir_name, load_dataset, normalize_data, clip_and_normalize_data

MAX_SCENE_COLS = 6
# DATASET_LIST = ['HYDICE', 
#            'ABU_beach_3', 'ABU_airport_4', 'ABU_urban_3', 'ABU_beach_2', 'ABU_urban_1', 
#            'ABU_airport_1', 'ABU_airport_2', 'ABU_airport_3', 'ABU_urban_4', 'ABU_urban_5', 'ABU_urban_2', 
#            'ABU_beach_4', 'ABU_beach_1',
#            'Indiana', 'Salinas', 'WHU-HI', 'cooke_city']
DATASET_LIST = ['Pavia', 'PaviaU', 'Salinas', 'WHU-HI']
MODEL_LIST = ['sample', 'ledoit_wolf', 'mrcd_auto_0.75_identity', 'mrcd_auto_0.75_equicorrelation', 'kmrcd_0.75_rbf']

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
base_filepath_results = os.path.join(base_filepath, 'Results')

argument_parser = argparse.ArgumentParser(
    description='Combine the classical (RX/AMF/ACE) model results and visualizations across multiple '
                'datasets for one fixed scaler/scaling_scope setting (see Process_results.py and '
                'Visualize_classical_results.py, which must have already been run for each dataset).')
argument_parser.add_argument('--datasets', type=str, nargs='+', default=DATASET_LIST, help='Datasets to combine (default: auto-discover every dataset under the Results directory that has a summary for the given scaler/scaling_scope)')
argument_parser.add_argument('--models', type=str, nargs='+', default=MODEL_LIST, help='Background models to include, matched against the "{model_name}_{background_model}" result directory suffix (e.g. sample, ledoit_wolf, mrcd_auto_0.75_identity); pass nothing/None to include every background model found')
argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (must match Process_results.py)')
argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample', 'all'], help='Scaling scope for the Scaler (must match Process_results.py)')
argument_parser.add_argument('--subsample', type=str, default='random', help='Subsampling method (must match Process_results.py/main.py)')
argument_parser.add_argument('--subsample_amount', type=int, default=10000, help='Amount of data points sampled (must match Process_results.py/main.py)')
args = argument_parser.parse_args()

print('Loading yaml config')
with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings_template = yaml.safe_load(f)
metrics_to_calc = experiment_settings_template['metrics']


def discover_datasets(scaler, scaling_scope, subsample, subsample_amount):
    # Each dataset is a subdirectory of Results/ (skip our own "Combined_*" output dirs) that
    # has already been through Process_results.py for this scaler/scaling_scope/subsample, i.e.
    # has a Results_summary.xlsx under Scaler_{scaler}_{scaling_scope}/Subsample_{subsample}_{amount}.
    found = []
    if not os.path.isdir(base_filepath_results):
        return found
    for entry in sorted(os.listdir(base_filepath_results)):
        if entry.startswith("Combined_"):
            continue
        entry_path = os.path.join(base_filepath_results, entry)
        summary_path = os.path.join(entry_path, f"Subsample_{subsample}")
        if subsample != 'none':
            summary_path = summary_path + f"_{subsample_amount}"
        summary_path = os.path.join(summary_path,
                                     f"Scaler_{scaler}_{scaling_scope}", 'Results_summary.xlsx')
        if os.path.isdir(entry_path) and os.path.exists(summary_path):
            found.append(entry)
    return found


if args.datasets is None:
    args.datasets = discover_datasets(args.scaler, args.scaling_scope, args.subsample, args.subsample_amount)
    print(f"Auto-discovered datasets: {args.datasets}")
    if not args.datasets:
        raise SystemExit(f"No datasets found under {base_filepath_results} with a Results_summary.xlsx for "
                          f"Scaler_{args.scaler}_{args.scaling_scope}/Subsample_{args.subsample}_{args.subsample_amount}. "
                          f"Run Process_results.py first.")

combined_save_dir = os.path.join(base_filepath_results, 'Combined',
                                  f"Scaler_{args.scaler}_{args.scaling_scope}",
                                  f"Subsample_{args.subsample}")
if args.subsample != 'none':
    combined_save_dir = combined_save_dir + f"_{args.subsample_amount}"
os.makedirs(combined_save_dir, exist_ok=True)


def select_models(available_models, requested_backgrounds):
    # Result directories are named "{model_name}_{background_model}" (see create_save_dir_name),
    # e.g. "base_rx_sample" or "base_amf_mrcd_auto_0.75_identity". requested_backgrounds selects
    # by that background_model suffix, keeping every detector (rx/amf/ace, ...) that uses it.
    if requested_backgrounds is None:
        return available_models
    selected = [m for m in available_models if any(m.endswith(f"_{bg}") for bg in requested_backgrounds)]
    missing = [bg for bg in requested_backgrounds if not any(m.endswith(f"_{bg}") for m in available_models)]
    if missing:
        print(f"Warning: requested background models not found and will be skipped: {missing}")
    return selected


def parse_mean_std(series):
    means, stds = [], []
    for v in series:
        try:
            mean_str, std_str = str(v).split('±')
            means.append(float(mean_str))
            stds.append(float(std_str))
        except:
            means.append(float(str(v)))
            stds.append(float(0.))
    return np.array(means), np.array(stds)


# --- Gather each dataset's summary results (produced by Process_results.py) ---
def gather_dataset_summaries(datasets, summary_filename):
    dataset_dirs = {}
    metrics_by_dataset = {}
    perc_by_dataset = {}
    for dataset in datasets:
        experiment_settings = dict(experiment_settings_template)
        experiment_settings['dataset'] = dataset
        experiment_settings['scaler'] = args.scaler
        experiment_settings['scaling_scope'] = args.scaling_scope
        experiment_settings['Scaler'] = {'name': args.scaler, 'scaling_scope': args.scaling_scope}
        experiment_settings['subsample'] = args.subsample
        experiment_settings['subsample_amount'] = args.subsample_amount
        experiment_settings['Subsample'] = {'name': args.subsample, 'amount': args.subsample_amount}

        save_dir = create_save_dir_name(base_filepath_results, None, experiment_settings)
        summary_path = os.path.join(save_dir, summary_filename)
        if not os.path.exists(summary_path):
            print(f"Skipping {dataset} for {summary_filename}: not found at {summary_path}.")
            continue

        dataset_dirs[dataset] = save_dir
        metrics_by_dataset[dataset] = pd.read_excel(summary_path, sheet_name='Metrics', index_col=0)
        perc_by_dataset[dataset] = pd.read_excel(summary_path, sheet_name='Percentage correct', index_col=0)
    return dataset_dirs, metrics_by_dataset, perc_by_dataset


def write_combined_summary(metrics_by_dataset, perc_by_dataset, datasets_found, models, out_path):
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        for m in metrics_to_calc:
            rows = {}
            for model in models:
                rows[model] = [metrics_by_dataset[d].loc[model, m] if model in metrics_by_dataset[d].index else None
                                for d in datasets_found]
            pd.DataFrame.from_dict(rows, orient='index', columns=datasets_found).to_excel(writer, sheet_name=m, index=True)
        for dataset in datasets_found:
            perc_by_dataset[dataset].to_excel(writer, sheet_name=f"PC {dataset}"[:31], index=True)
    print(f"Saved combined summary to {out_path}")


def plot_combined_metrics(metrics_by_dataset, datasets_found, models, title_suffix, out_path):
    n_datasets = len(datasets_found)
    x = np.arange(len(models))
    width = 0.8 / max(n_datasets, 1)
    fig, axes = plt.subplots(1, len(metrics_to_calc), figsize=(6 * len(metrics_to_calc), 5), squeeze=False)
    for i, m in enumerate(metrics_to_calc):
        ax = axes[0, i]
        for j, dataset in enumerate(datasets_found):
            df = metrics_by_dataset[dataset]
            present = [model in df.index for model in models]
            means = np.full(len(models), np.nan)
            stds = np.full(len(models), np.nan)
            vals_mean, vals_std = parse_mean_std(df.loc[[model for model in models if model in df.index], m])
            means[present] = vals_mean
            stds[present] = vals_std
            ax.bar(x + j * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3, label=dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title(m)
        ax.set_ylabel(m)
        ax.legend(fontsize=8)
    fig.suptitle(f"Classical model metric comparison across datasets{title_suffix} (Scaler={args.scaler}, "
                 f"scaling_scope={args.scaling_scope}, subsample={args.subsample}_{args.subsample_amount})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


dataset_dirs, metrics_by_dataset, perc_by_dataset = gather_dataset_summaries(args.datasets, 'Results_summary.xlsx')
if not metrics_by_dataset:
    raise SystemExit("No dataset summaries found; nothing to combine.")

datasets_found = list(metrics_by_dataset.keys())
models = sorted(set().union(*(df.index for df in metrics_by_dataset.values())))
models = select_models(models, args.models)

write_combined_summary(metrics_by_dataset, perc_by_dataset, datasets_found, models,
                        os.path.join(combined_save_dir, 'Combined_results_summary.xlsx'))
plot_combined_metrics(metrics_by_dataset, datasets_found, models, "",
                       os.path.join(combined_save_dir, "Combined_metrics_comparison.png"))

# --- Binary results: same combination, scored from Results_summary_binary.xlsx (Scores_binary
# vs. ground truth, see Process_results.py). Datasets without it yet are skipped. ---
_, metrics_by_dataset_binary, perc_by_dataset_binary = gather_dataset_summaries(args.datasets, 'Results_summary_binary.xlsx')
if not metrics_by_dataset_binary:
    print("No binary dataset summaries found; skipping combined binary summary/comparison.")
else:
    datasets_found_binary = list(metrics_by_dataset_binary.keys())
    models_binary = sorted(set().union(*(df.index for df in metrics_by_dataset_binary.values())))
    models_binary = select_models(models_binary, args.models)
    write_combined_summary(metrics_by_dataset_binary, perc_by_dataset_binary, datasets_found_binary, models_binary,
                            os.path.join(combined_save_dir, 'Combined_results_summary_binary.xlsx'))
    plot_combined_metrics(metrics_by_dataset_binary, datasets_found_binary, models_binary, " (binary)",
                           os.path.join(combined_save_dir, "Combined_metrics_comparison_binary.png"))

# --- Scene visualization: one figure per dataset (visual composite, labels, and every model's
# score map), wrapped across multiple rows once there are too many models for one row to read. ---
print("Building per-dataset scene visualizations")
for dataset in datasets_found:
    save_dir = dataset_dirs[dataset]
    models_present = [m for m in metrics_by_dataset[dataset].index if m in models]

    model_scores = {}
    for model in models_present:
        pickle_path = os.path.join(save_dir, model, "Raw_results.pickle")
        if not os.path.exists(pickle_path):
            continue
        with open(pickle_path, "rb") as f:
            x = pickle.load(f)
        model_scores[model] = clip_and_normalize_data(x["Scores"][0])
        del x

    if not model_scores:
        print(f"Skipping {dataset} in scene visualization: no Raw_results.pickle found for any model.")
        continue

    raw_data, _, label_array, label_ids, wavelengths = load_dataset(base_path=base_filepath, dataset_name=dataset)
    bgr_targets = [470, 540, 690]  # approximate blue/green/red wavelengths (nm)
    b_idx, g_idx, r_idx = [np.argmin(np.abs(wavelengths - t)) for t in bgr_targets]
    visual = np.stack([
        normalize_data(raw_data[0, :, :, r_idx]),
        normalize_data(raw_data[0, :, :, g_idx]),
        normalize_data(raw_data[0, :, :, b_idx]),
    ], axis=-1)

    panels = [("Visual (BGR composite)", visual, None), ("Labels", label_array[0], 'nipy_spectral')]
    panels += [(model, score, None) for model, score in model_scores.items()]

    ncols = min(MAX_SCENE_COLS, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for idx, (title, img, cmap) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        if img.shape[-1]==3:
            img=img.squeeze()
        im = ax.imshow(normalize_data(img), cmap=cmap)
        if img.shape[-1]!=3:
            fig.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    fig.suptitle(f"{dataset}: visual composite, labels, and model outputs (Scaler={args.scaler}, "
                 f"scaling_scope={args.scaling_scope}, subsample={args.subsample}_{args.subsample_amount})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(combined_save_dir, f"Combined_scene_visualization_{dataset}.png"))
    plt.close(fig)

print(f"Saved combined visualizations to {combined_save_dir}")

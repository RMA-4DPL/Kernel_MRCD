import os
import pathlib
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import create_save_dir_name

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
base_filepath_results = os.path.join(base_filepath, 'Results')

argument_parser = argparse.ArgumentParser(
    description='Combine the classical (RX/AMF/ACE) model results and visualizations across multiple '
                'datasets for one fixed scaler/scaling_scope setting (see Process_results.py and '
                'Visualize_classical_results.py, which must have already been run for each dataset).')
argument_parser.add_argument('--datasets', type=str, nargs='+', default=None, help='Datasets to combine (default: auto-discover every dataset under the Results directory that has a summary for the given scaler/scaling_scope)')
argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (must match Process_results.py)')
argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample'], help='Scaling scope for the Scaler (must match Process_results.py)')
args = argument_parser.parse_args()

print('Loading yaml config')
with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings_template = yaml.safe_load(f)
metrics_to_calc = experiment_settings_template['metrics']


def discover_datasets(scaler, scaling_scope):
    # Each dataset is a subdirectory of Results/ (skip our own "Combined_*" output dirs) that
    # has already been through Process_results.py for this scaler/scaling_scope, i.e. has a
    # Results_summary.xlsx under Scaler_{scaler}_{scaling_scope}.
    found = []
    if not os.path.isdir(base_filepath_results):
        return found
    for entry in sorted(os.listdir(base_filepath_results)):
        if entry.startswith("Combined_"):
            continue
        entry_path = os.path.join(base_filepath_results, entry)
        summary_path = os.path.join(entry_path, f"Scaler_{scaler}_{scaling_scope}", 'Results_summary.xlsx')
        if os.path.isdir(entry_path) and os.path.exists(summary_path):
            found.append(entry)
    return found


if args.datasets is None:
    args.datasets = discover_datasets(args.scaler, args.scaling_scope)
    print(f"Auto-discovered datasets: {args.datasets}")
    if not args.datasets:
        raise SystemExit(f"No datasets found under {base_filepath_results} with a Results_summary.xlsx for "
                          f"Scaler_{args.scaler}_{args.scaling_scope}. Run Process_results.py first.")

combined_save_dir = os.path.join(base_filepath_results, "Combined_" + "_".join(args.datasets), f"Scaler_{args.scaler}_{args.scaling_scope}")
os.makedirs(combined_save_dir, exist_ok=True)


def parse_mean_std(series):
    means, stds = [], []
    for v in series:
        mean_str, std_str = str(v).split('±')
        means.append(float(mean_str))
        stds.append(float(std_str))
    return np.array(means), np.array(stds)


# --- Gather each dataset's summary results (produced by Process_results.py) ---
dataset_dirs = {}
metrics_by_dataset = {}
perc_by_dataset = {}
for dataset in args.datasets:
    experiment_settings = dict(experiment_settings_template)
    experiment_settings['dataset'] = dataset
    experiment_settings['scaler'] = args.scaler
    experiment_settings['scaling_scope'] = args.scaling_scope
    experiment_settings['Scaler'] = {'name': args.scaler, 'scaling_scope': args.scaling_scope}

    save_dir = create_save_dir_name(base_filepath_results, None, experiment_settings)
    summary_path = os.path.join(save_dir, 'Results_summary.xlsx')
    if not os.path.exists(summary_path):
        print(f"Skipping {dataset}: no summary found at {summary_path}. Run Process_results.py for it first.")
        continue

    dataset_dirs[dataset] = save_dir
    metrics_by_dataset[dataset] = pd.read_excel(summary_path, sheet_name='Metrics', index_col=0)
    perc_by_dataset[dataset] = pd.read_excel(summary_path, sheet_name='Percentage correct', index_col=0)

if not metrics_by_dataset:
    raise SystemExit("No dataset summaries found; nothing to combine.")

datasets_found = list(metrics_by_dataset.keys())
models = sorted(set().union(*(df.index for df in metrics_by_dataset.values())))

# --- Combined excel: one sheet per metric, rows=model, columns=dataset ---
combined_summary_path = os.path.join(combined_save_dir, 'Combined_results_summary.xlsx')
with pd.ExcelWriter(combined_summary_path, engine='xlsxwriter') as writer:
    for m in metrics_to_calc:
        rows = {}
        for model in models:
            rows[model] = [metrics_by_dataset[d].loc[model, m] if model in metrics_by_dataset[d].index else None
                            for d in datasets_found]
        pd.DataFrame.from_dict(rows, orient='index', columns=datasets_found).to_excel(writer, sheet_name=m, index=True)
    for dataset in datasets_found:
        perc_by_dataset[dataset].to_excel(writer, sheet_name=f"PC {dataset}"[:31], index=True)
print(f"Saved combined summary to {combined_summary_path}")

# --- Combined metric comparison: grouped bar chart per metric, grouped by dataset ---
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
fig.suptitle(f"Classical model metric comparison across datasets (Scaler={args.scaler}, scaling_scope={args.scaling_scope})")
fig.tight_layout()
fig.savefig(os.path.join(combined_save_dir, "Combined_metrics_comparison.png"))
plt.close(fig)

# --- Combined scene visualization: stack each dataset's existing scene visualization vertically ---
scene_images = {}
for dataset in datasets_found:
    scene_path = os.path.join(dataset_dirs[dataset], "Classical_scene_visualization.png")
    if os.path.exists(scene_path):
        scene_images[dataset] = plt.imread(scene_path)
    else:
        print(f"Skipping {dataset} in combined scene visualization: {scene_path} not found "
              f"(run Visualize_classical_results.py for it first).")

if scene_images:
    fig, axes = plt.subplots(len(scene_images), 1, figsize=(12, 4 * len(scene_images)), squeeze=False)
    for row, (dataset, img) in enumerate(scene_images.items()):
        ax = axes[row, 0]
        ax.imshow(img)
        ax.set_title(dataset)
        ax.axis('off')
    fig.suptitle("Visual composite, labels, and classical model outputs per dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(combined_save_dir, "Combined_scene_visualization.png"))
    plt.close(fig)
else:
    print("No per-dataset scene visualizations found; skipping combined scene visualization.")

print(f"Saved combined visualizations to {combined_save_dir}")

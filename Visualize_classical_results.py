import os
import pickle
import pathlib
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import create_save_dir_name, load_mudcad_file, normalize_data

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/DECAT/First test"

argument_parser = argparse.ArgumentParser(
    description='Visualize the classical (RX/LRX/CRD) model results produced by Process_results_DL.py for a single configuration')
argument_parser.add_argument('--subsample', type=int, default=None, help='Subsample factor for spatial dimensions (overrides experiment_settings Subsample; must match Process_results_DL.py)')
argument_parser.add_argument('--vis_scale', type=str, default=None, help="Scale factor for the visual RGB bands (overrides experiment_settings Vis_scale; pass 'none' to disable; must match Process_results_DL.py)")
argument_parser.add_argument('--ac_model', type=str, default=None, help='Atmospheric correction model name (overrides experiment_settings AC_model; must match Process_results_DL.py)')
argument_parser.add_argument('--test_split_method', type=str, default='random', choices=['random', 'seasonal'], help='Method used to split the test set (must match Process_results_DL.py)')
argument_parser.add_argument('--test_season', type=str, default='autumn', help='Season used for the test set when test_split_method is seasonal (must match Process_results_DL.py)')
args = argument_parser.parse_args()

print('Loading yaml config')
with open(os.path.join(base_filepath_configs, "model_configs.yaml"), "r") as f:
    model_configs = yaml.safe_load(f)
with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings = yaml.safe_load(f)
metrics_to_calc = experiment_settings['metrics']
# Mirror Process_results_DL.py so the reconstructed save_dir matches where it wrote its output.
experiment_settings['test_split_method'] = args.test_split_method
experiment_settings['test_season'] = args.test_season
experiment_settings.setdefault('loss_function', 'weighted_mse')
if args.subsample is not None:
    experiment_settings['Subsample'] = args.subsample
if args.ac_model is not None:
    experiment_settings['AC_model'] = {'name': args.ac_model}
if args.vis_scale is not None:
    experiment_settings['Vis_scale'] = None if args.vis_scale.lower() in ('none', 'null') else float(args.vis_scale)

# Process_results_DL.py writes its summary/detailed files to the save_dir of whichever model is
# last in {**dl_model_configs, **classical_model_configs}; since that merge always puts the
# classical models last, the last model here (loaded from model_configs.yaml alone) lands on
# the same save_dir without needing DL_model_configs.yaml at all.
last_model = list(model_configs.keys())[-1]
summary_save_dir = create_save_dir_name(base_filepath, last_model, experiment_settings)
summary_save_dir = summary_save_dir.split(last_model)[0]

# Restrict to the models listed in result_processing_config.yaml, if any are given (applied
# after the save_dir above so it doesn't move depending on which subset is visualized).
models_to_evaluate = experiment_settings.get('models')
if models_to_evaluate:
    model_configs = {k: v for k, v in model_configs.items() if k in models_to_evaluate}

summary_path = os.path.join(summary_save_dir, 'DL_Results_summary.xlsx')
detailed_path = os.path.join(summary_save_dir, 'DL_Results_detailed.xlsx')

if not os.path.exists(summary_path):
    raise SystemExit(f"No summary found at {summary_path}. Run Process_results_DL.py first.")

metrics_df = pd.read_excel(summary_path, sheet_name='Metrics', index_col=0)
perc_summary_df = pd.read_excel(summary_path, sheet_name='Percentage correct', index_col=0)
detailed_xls = pd.ExcelFile(detailed_path) if os.path.exists(detailed_path) else None
models = [model for model in model_configs if model in metrics_df.index]

for model in model_configs:
    if model not in models:
        print(f"Skipping {model}: no results found in {os.path.basename(summary_path)}.")


def parse_mean_std(series):
    means, stds = [], []
    for v in series:
        mean_str, std_str = str(v).split('±')
        means.append(float(mean_str))
        stds.append(float(std_str))
    return np.array(means), np.array(stds)


# --- Metric comparison: mean +/- std bar chart per metric ---
fig, axes = plt.subplots(1, len(metrics_to_calc), figsize=(5 * len(metrics_to_calc), 5), squeeze=False)
for i, m in enumerate(metrics_to_calc):
    ax = axes[0, i]
    means, stds = parse_mean_std(metrics_df.loc[models, m])
    ax.bar(models, means, yerr=stds, capsize=4)
    ax.set_title(m)
    ax.set_ylabel(m)
    ax.tick_params(axis='x', rotation=45)
fig.suptitle("Classical model (RX/LRX/CRD) metric comparison")
fig.tight_layout()
fig.savefig(os.path.join(summary_save_dir, "Classical_metrics_comparison.png"))
plt.close(fig)

# --- Percentage correct per category: grouped bar chart across models ---
categories = list(perc_summary_df.columns)
n_models = len(models)
x = np.arange(len(categories))
width = 0.8 / max(n_models, 1)
fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), 6))
for i, model in enumerate(models):
    means, stds = parse_mean_std(perc_summary_df.loc[model])
    ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3, label=model)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_ylabel("Fraction correctly flagged")
ax.set_title("Percentage correct per category")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(summary_save_dir, "Classical_percentage_correct.png"))
plt.close(fig)

# --- Per-sample metric distributions (box plot) from the detailed results file ---
if detailed_xls is not None:
    fig, axes = plt.subplots(1, len(metrics_to_calc), figsize=(5 * len(metrics_to_calc), 5), squeeze=False)
    for i, m in enumerate(metrics_to_calc):
        ax = axes[0, i]
        if m in detailed_xls.sheet_names:
            metric_detail_df = pd.read_excel(detailed_xls, sheet_name=m, index_col=0)
            models_present = [model for model in models if model in metric_detail_df.index]
            data = [metric_detail_df.loc[model].dropna().values for model in models_present]
            ax.boxplot(data, tick_labels=models_present)
        ax.set_title(f"{m} across test samples")
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(summary_save_dir, "Classical_metric_distributions.png"))
    plt.close(fig)

# --- RX-family (RX/LRX/CRD) metric vs. window size ---
# LRX/CRD are swept over window sizes in model_configs.yaml; RX has none and is drawn as a
# flat reference line.
rx_family = {}
for model, config in model_configs.items():
    if model not in models:
        continue
    window = config.get('window')
    outer_window = window[1] if window else None
    rx_family.setdefault(config['model_name'], []).append((model, outer_window))

if rx_family:
    fig, axes = plt.subplots(1, len(metrics_to_calc), figsize=(5 * len(metrics_to_calc), 5), squeeze=False)
    for i, m in enumerate(metrics_to_calc):
        ax = axes[0, i]
        means, stds = parse_mean_std(metrics_df[m])
        model_to_value = dict(zip(metrics_df.index, means))
        model_to_std = dict(zip(metrics_df.index, stds))
        for model_name, entries in rx_family.items():
            windowed = sorted((w, mdl) for mdl, w in entries if w is not None)
            if windowed:
                x_vals = [w for w, _ in windowed]
                y_vals = [model_to_value[mdl] for _, mdl in windowed]
                yerr = [model_to_std[mdl] for _, mdl in windowed]
                ax.errorbar(x_vals, y_vals, yerr=yerr, marker='o', capsize=3, label=model_name)
            for mdl, w in entries:
                if w is None:
                    ax.axhline(model_to_value[mdl], linestyle='--', alpha=0.6, label=f"{mdl} (no window)")
        ax.set_title(m)
        ax.set_xlabel("Outer window size")
        ax.set_ylabel(m)
        ax.legend(fontsize=7)
    fig.suptitle("RX-family (RX/LRX/CRD) metric vs. window size")
    fig.tight_layout()
    fig.savefig(os.path.join(summary_save_dir, "Classical_rx_window_comparison.png"))
    plt.close(fig)
else:
    print("No RX-family (RX/LRX/CRD) results found; skipping window comparison plot.")

# --- Random sample gallery: visual bands, labels, and every classical model's output score map ---
# Classical models score the whole dataset (no train/test split of their own), so samples are
# drawn from the full dataset rather than a held-out test split.
print("Building random sample gallery")
list_of_dirs = []
for root, dirs, files in os.walk(os.path.join(base_filepath, "MUCAD", "dataset"), topdown=False):
    if not dirs:
        list_of_dirs.append(root)

label_array = None
model_scores = {}
for model in models:
    save_dir = create_save_dir_name(base_filepath, model, experiment_settings)
    pickle_path = os.path.join(save_dir, "Raw_results.pickle")
    if not os.path.exists(pickle_path):
        print(f"Skipping {model} in sample gallery: {pickle_path} not found.")
        continue
    with open(pickle_path, "rb") as f:
        x = pickle.load(f)
    if label_array is None:
        label_array = x["Labels"]
    model_scores[model] = x["Scores"]
    del x

if not model_scores:
    print("No classical model results found; skipping random sample gallery.")
else:
    n_gallery = min(5, len(label_array))
    rng = np.random.default_rng(4)
    sample_indices = rng.choice(len(label_array), size=n_gallery, replace=False)

    models_with_scores = [model for model in models if model in model_scores]
    ncols = 2 + len(models_with_scores)
    fig, axes = plt.subplots(n_gallery, ncols, figsize=(4 * ncols, 4 * n_gallery), squeeze=False)
    for row, global_idx in enumerate(sample_indices):
        _, visual_data, _ = load_mudcad_file(list_of_dirs[global_idx], load_vis=True)
        axes[row, 0].imshow(normalize_data(visual_data))
        axes[row, 0].set_title(f"Sample {global_idx} - Visual")
        axes[row, 0].axis('off')

        temp_labels = np.sum(label_array[global_idx].astype(np.int16), axis=-1)
        axes[row, 1].imshow(temp_labels)
        axes[row, 1].set_title("Labels")
        axes[row, 1].axis('off')

        for col, model in enumerate(models_with_scores, start=2):
            ax = axes[row, col]
            ax.imshow(model_scores[model][global_idx])
            ax.set_title(model, fontsize=8)
            ax.axis('off')
    fig.suptitle("Random samples: visual bands, labels, and classical model outputs")
    fig.tight_layout()
    fig.savefig(os.path.join(summary_save_dir, "Classical_sample_gallery.png"))
    plt.close(fig)

print(f"Saved visualizations to {summary_save_dir}")

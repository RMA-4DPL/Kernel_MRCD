import os
import pickle
import pathlib
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import create_save_dir_name, load_dataset, normalize_data

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
base_filepath_results = os.path.join(base_filepath, 'Results')

argument_parser = argparse.ArgumentParser(
    description='Visualize the classical (RX/AMF/ACE) model results produced by Process_results.py')
argument_parser.add_argument('--dataset', type=str, default='Salinas', help='Select which dataset to load (default: Salinas; must match Process_results.py)')
argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (overrides experiment_settings Scaler; must match Process_results.py)')
argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample'], help='Scaling scope for the Scaler (overrides experiment_settings Scaler scaling_scope; must match Process_results.py)')
args = argument_parser.parse_args()

print('Loading yaml config')
with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings = yaml.safe_load(f)
metrics_to_calc = experiment_settings['metrics']
# Mirror Process_results.py so the reconstructed save_dir matches where it wrote its output.
for arg in vars(args):
    experiment_settings[arg] = getattr(args, arg)
if args.scaler is not None:
    experiment_settings.setdefault('Scaler', {})['name'] = args.scaler
    if args.scaling_scope is not None:
        experiment_settings.setdefault('Scaler', {})['scaling_scope'] = args.scaling_scope

# Process_results.py discovers models from the results directory itself (one subdirectory per
# "{model_name}_{background_model}" combination, see LXR_test.py) rather than from
# model_configs.yaml, so the save_dir here doesn't depend on any model name either.
summary_save_dir = create_save_dir_name(base_filepath_results, None, experiment_settings)

summary_path = os.path.join(summary_save_dir, 'Results_summary.xlsx')
detailed_path = os.path.join(summary_save_dir, 'Results_detailed.xlsx')

if not os.path.exists(summary_path):
    raise SystemExit(f"No summary found at {summary_path}. Run Process_results.py first.")

metrics_df = pd.read_excel(summary_path, sheet_name='Metrics', index_col=0)
perc_summary_df = pd.read_excel(summary_path, sheet_name='Percentage correct', index_col=0)
detailed_xls = pd.ExcelFile(detailed_path) if os.path.exists(detailed_path) else None
models = list(metrics_df.index)


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
fig.suptitle("Classical model (RX/AMF/ACE) metric comparison")
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

# --- Scene visualization: BGR composite, labels, and every model's output score map ---
# Salinas is a single scene (see LXR_test.py/load_salinas), so there is one visual composite
# per run rather than a random gallery of samples.
print("Building scene visualization")
raw_data, data_array, label_array, label_ids = load_dataset(base_path=base_filepath, dataset_name=args.dataset)

n_bands = raw_data.shape[-1]
wavelengths = np.linspace(400, 2500, n_bands)
bgr_targets = [495, 555, 760]  # approximate blue/green/red wavelengths (nm)
b_idx, g_idx, r_idx = [np.argmin(np.abs(wavelengths - t)) for t in bgr_targets]
visual = np.stack([
    normalize_data(raw_data[0, :, :, r_idx]),
    normalize_data(raw_data[0, :, :, g_idx]),
    normalize_data(raw_data[0, :, :, b_idx]),
], axis=-1)

model_scores = {}
for model in models:
    pickle_path = os.path.join(summary_save_dir, model, "Raw_results.pickle")
    if not os.path.exists(pickle_path):
        print(f"Skipping {model} in scene visualization: {pickle_path} not found.")
        continue
    with open(pickle_path, "rb") as f:
        x = pickle.load(f)
    model_scores[model] = normalize_data(x["Scores"][0])
    del x

if not model_scores:
    print("No classical model results found; skipping scene visualization.")
else:
    models_with_scores = [model for model in models if model in model_scores]
    ncols = 2 + len(models_with_scores)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    axes = axes[0]

    axes[0].imshow(visual)
    axes[0].set_title("Visual (BGR composite)")
    axes[0].axis('off')

    axes[1].imshow(label_array[0], cmap='nipy_spectral')
    axes[1].set_title("Labels")
    axes[1].axis('off')

    for col, model in enumerate(models_with_scores, start=2):
        axes[col].imshow(model_scores[model])
        axes[col].set_title(model, fontsize=8)
        axes[col].axis('off')
    fig.suptitle(f"{experiment_settings['dataset']}: visual composite, labels, and classical model outputs")
    fig.tight_layout()
    fig.savefig(os.path.join(summary_save_dir, "Classical_scene_visualization.png"))
    plt.close(fig)

    # --- Score histograms: background vs foreground, and per-class, for each model ---
    print("Building score histograms")
    label_map = label_array[0]
    background_mask = label_map == 0
    foreground_mask = ~background_mask
    category_ids = sorted(cid for cid in label_ids if cid != 0)

    fig, axes = plt.subplots(len(models_with_scores), 2, figsize=(12, 4 * len(models_with_scores)), squeeze=False)
    for row, model in enumerate(models_with_scores):
        scores = model_scores[model]

        ax_bg_fg = axes[row, 0]
        ax_bg_fg.hist(scores[background_mask], bins=50, alpha=0.5, density=True, label='Background')
        ax_bg_fg.hist(scores[foreground_mask], bins=50, alpha=0.5, density=True, label='Foreground')
        ax_bg_fg.set_title(f"{model}: background vs foreground")
        ax_bg_fg.set_xlabel("Score")
        ax_bg_fg.set_ylabel("Density")
        ax_bg_fg.legend(fontsize=8)

        ax_classes = axes[row, 1]
        for cat_id in category_ids:
            class_mask = label_map == cat_id
            if np.any(class_mask):
                ax_classes.hist(scores[class_mask], bins=50, alpha=0.4, density=True, label=label_ids[cat_id][0])
        ax_classes.set_title(f"{model}: per-class score distribution")
        ax_classes.set_xlabel("Score")
        ax_classes.set_ylabel("Density")
        ax_classes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(summary_save_dir, "Classical_score_histograms.png"))
    plt.close(fig)

print(f"Saved visualizations to {summary_save_dir}")

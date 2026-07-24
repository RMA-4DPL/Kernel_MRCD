import os
import pickle
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from helper_functions import create_save_dir_name, load_dataset, normalize_data, clip_and_normalize_data

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
base_filepath_results = os.path.join(base_filepath, 'Results')

argument_parser = argparse.ArgumentParser(
    description='Build a 4-panel detection-map figure: RGB composite, ground truth, '
                'RX with sample covariance, and RX with kernel-MRCD background.')
argument_parser.add_argument('--dataset', type=str, default='Salinas')
argument_parser.add_argument('--scaler', type=str, default='Standard')
argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample', 'all'])
argument_parser.add_argument('--subsample', type=str, default='random')
argument_parser.add_argument('--subsample_amount', type=int, default=10000)
argument_parser.add_argument('--kmrcd_config', type=str, default='kmrcd_0.75_rbf',
                              help='Name of the kernel-MRCD background_config subdirectory to use as (d)')
argument_parser.add_argument('--out', type=str, default=None, help='Output PNG path (default: alongside the other result figures)')
args = argument_parser.parse_args()

experiment_settings = {
    'dataset': args.dataset,
    'Scaler': {'name': args.scaler, 'scaling_scope': args.scaling_scope},
    'Subsample': {'name': args.subsample, 'amount': args.subsample_amount},
}
summary_save_dir = create_save_dir_name(base_filepath_results, None, experiment_settings)

sample_cov_dir = os.path.join(summary_save_dir, 'base_rx_sample', 'Raw_results.pickle')
kmrcd_dir = os.path.join(summary_save_dir, f'base_rx_{args.kmrcd_config}', 'Raw_results.pickle')
for p in (sample_cov_dir, kmrcd_dir):
    if not os.path.exists(p):
        raise SystemExit(f"Missing results file: {p}")

print("Loading dataset")
raw_data, data_array, label_array, label_ids, wavelengths = load_dataset(base_path=base_filepath, dataset_name=args.dataset)

rgb_targets = [690, 540, 470]  # approximate red/green/blue wavelengths (nm)
r_idx, g_idx, b_idx = [np.argmin(np.abs(wavelengths - t)) for t in rgb_targets]
rgb_composite = np.stack([
    normalize_data(raw_data[0, :, :, r_idx]),
    normalize_data(raw_data[0, :, :, g_idx]),
    normalize_data(raw_data[0, :, :, b_idx]),
], axis=-1)

label_map = label_array[0]
category_ids = sorted(label_ids.keys())
n_cats = len(category_ids)
# Background (id 0) rendered as black; remaining classes get a fixed, distinguishable
# qualitative palette (tab20) so class identity is comparable across figures.
class_colors = ['black'] + [plt.get_cmap('tab20')(i % 20) for i in range(n_cats - 1)]
label_cmap = ListedColormap(class_colors)
label_norm = BoundaryNorm(np.arange(-0.5, n_cats + 0.5, 1), label_cmap.N)
remapped_labels = np.searchsorted(category_ids, label_map)


def load_scores(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
    scores = x['Scores']
    if scores.ndim == 4:
        scores = np.max(scores, axis=1)
    return clip_and_normalize_data(scores[0])


print("Loading RX (sample covariance) scores")
scores_sample = load_scores(sample_cov_dir)
print("Loading RX (kernel-MRCD background) scores")
scores_kmrcd = load_scores(kmrcd_dir)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(rgb_composite)

axes[1].imshow(remapped_labels, cmap=label_cmap, norm=label_norm)

im_c = axes[2].imshow(scores_sample, cmap='viridis', vmin=0, vmax=1)
# fig.colorbar(im_c, ax=axes[1, 0], fraction=0.046, pad=0.04)

im_d = axes[3].imshow(scores_kmrcd, cmap='viridis', vmin=0, vmax=1)
# fig.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

panel_labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, label in zip(axes.ravel(), panel_labels):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label, fontsize=15)

#fig.suptitle(f"{args.dataset}: detection maps")
fig.tight_layout()

out_path = args.out or os.path.join(summary_save_dir, f"Detection_maps_{args.kmrcd_config}.png")
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved figure to {out_path}")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy
import pathlib
from helper_functions import load_salinas
import argparse
import scipy.io


argument_parser = argparse.ArgumentParser(description='Data exploration for KMRCD')
argument_parser.add_argument('--dataset', type=str, default='Salinas', help='Select which dataset to load (default:Salinas).')
args = argument_parser.parse_args()

np.random.seed(4)
base_filepath = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
dataset_filepath = os.path.join(base_filepath, 'Dataset', args.dataset)
figure_filepath = os.path.join(base_filepath, 'Figures')

salinas_data, salinas_corrected_data, salinas_labels, labels_ids = load_salinas(dataset_filepath=dataset_filepath)

wavelengths = np.linspace(400, 2500, 224)
wavelengths = np.delete(wavelengths, 223)
wavelengths = np.delete(wavelengths, np.s_[153:167])
wavelengths = np.delete(wavelengths, np.s_[107:112])

BGR = [495, 555, 760]
BGR_indices = [np.argmin(np.abs(wavelengths - target)) for target in BGR]
BGR_wavelengths = wavelengths[BGR_indices]

raw_wavelengths = np.linspace(400, 2500, 224)
above_r_wavelengths = np.linspace(BGR_wavelengths[2], wavelengths[-1], num=6)[1:]


def band_indices(targets, wl, n_bands):
    idx = [np.argmin(np.abs(wl - target)) for target in targets]
    return [min(i, n_bands - 1) for i in idx]


def normalize(band):
    band = band.astype(np.float32)
    return (band - band.min()) / (band.max() - band.min() + 1e-8)


def bgr_composite(data, idx):
    b, g, r = idx
    return np.stack([normalize(data[:, :, r]), normalize(data[:, :, g]), normalize(data[:, :, b])], axis=-1)


raw_bgr_idx = band_indices(BGR, raw_wavelengths, salinas_data.shape[2])
raw_above_idx = band_indices(above_r_wavelengths, raw_wavelengths, salinas_data.shape[2])

fig, axes = plt.subplots(6, 2, figsize=(8, 20))

axes[0, 0].imshow(bgr_composite(salinas_data, raw_bgr_idx))
axes[0, 0].set_title('Raw BGR composite')
axes[0, 1].imshow(salinas_labels, cmap='nipy_spectral')
axes[0, 1].set_title('Labels')

for row in range(1, 6):
    raw_band = raw_above_idx[row - 1]
    axes[row, 0].imshow(normalize(salinas_data[:, :, raw_band]), cmap='gray')
    axes[row, 0].set_title(f'Raw {raw_wavelengths[raw_band]:.0f}nm')
    axes[row, 1].axis('off')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
os.makedirs(figure_filepath, exist_ok=True)
fig.savefig(os.path.join(figure_filepath, 'bgr_and_bands.png'), dpi=150)
plt.show()

spectral_wavelengths = wavelengths[:salinas_corrected_data.shape[2]]
class_ids = [c for c in sorted(labels_ids.keys()) if c != 0]

fig2, ax2 = plt.subplots(figsize=(12, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(class_ids)))
for color, class_id in zip(colors, class_ids):
    spectra = salinas_corrected_data[salinas_labels == class_id]
    ax2.plot(spectral_wavelengths, spectra.mean(axis=0), label=labels_ids[class_id][0], color=color)

ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Reflectance')
ax2.set_title('Average spectrum per class (Salinas corrected)')
ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
plt.tight_layout()
fig2.savefig(os.path.join(figure_filepath, 'average_spectrum_per_class.png'), dpi=150)
plt.show()

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
from helper_functions import load_mudcad_file

np.random.seed(4)
# base_filepath = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/DECAT/First test"
# Load gsd and labels from yaml files
print('Loading yaml config')
with open(os.path.join(base_filepath, "MUCAD", "dataset", "gsd.yaml"), "r") as f:
    gsd = yaml.safe_load(f)
with open(os.path.join(base_filepath, "MUCAD", "dataset", "labels_rgb.yaml"), "r") as f:
    labels_rgb = yaml.safe_load(f)

loading = "all"
max_amount_to_show = 5

# Create a list of data files to read
print('Start reading files')
list_of_dirs = []
for root, dirs, files in os.walk(os.path.join(base_filepath, "MUCAD", "dataset"), topdown=False):
   if not dirs:
       list_of_dirs.append(root)

nr_of_captures = len(list_of_dirs)
# assert nr_of_captures == 853 #The dataset consists of 853 captures

if loading == "all":
    list_of_dirs_to_load = list_of_dirs
    indices = np.arange(max_amount_to_show)
elif loading == "random":
    indices = np.random.randint(nr_of_captures, size=max_amount_to_show)
    list_of_dirs_to_load = [list_of_dirs[ind] for ind in indices]
elif type(loading) == list:
    list_of_dirs_to_load = list_of_dirs[loading]
    indices = np.arange(max_amount_to_show)

# Setup dataframes
print('Setting up arrays')
feature_order = ["vis", "blue", "green", "red", "eir", "nir", "lwir"]
feature_length = [3, 1, 1, 1, 1, 1, 1]
col_names = ["Season", "Date", "Location", "Area", "Flight_number", "Region", "Capture_number"]

metadata_array = np.empty((len(list_of_dirs_to_load), len(col_names)), dtype=object)
data_array = np.empty((len(list_of_dirs_to_load), 512, 512, len(feature_order)+2))
label_array = np.empty((len(list_of_dirs_to_load), 512, 512, 3), dtype=np.uint8)
label_count_array = np.zeros((len(list_of_dirs_to_load), len(labels_rgb['categories'])), dtype=np.uint32)

print('Loading data')
def _load_one(args):
    d, directory = args
    metadata_values, data, labels = load_mudcad_file(directory)
    if not os.path.exists(os.path.join(directory, 'Data.npz')):
        np.savez(os.path.join(directory, 'Data.npz'), data=data, labels=labels)
    return d, metadata_values, data, labels

with ThreadPoolExecutor(max_workers=min(32, len(list_of_dirs_to_load))) as executor:
    results = executor.map(_load_one, enumerate(list_of_dirs_to_load))
    for d, metadata_values, data, labels in tqdm(results, total=len(list_of_dirs_to_load)):
        metadata_array[d] = metadata_values
        data_array[d] = data
        label_array[d] = labels

        temp_labels = label_array[d].reshape(-1, label_array[d].shape[-1])
        for u, unique in enumerate(labels_rgb['categories'].values()):
            count = len(np.where(temp_labels == unique)[0])
            index = list(labels_rgb['categories'].values()).index(unique)
            label_count_array[d, index] = count

print('Creating metadata dataframe')

label_count = pd.DataFrame(label_count_array, columns=list(labels_rgb['categories'].keys()))
metadata = pd.DataFrame(metadata_array, columns=col_names)
print(metadata)
print(label_count)

for d, directory in enumerate(list_of_dirs_to_load):
    label_count.iloc[[d]].to_csv(os.path.join(directory, "label_counts.csv"), index=False)

for r,row in enumerate(data_array[indices]):
    fig = plt.figure(figsize=(20,10))
    row_index = 0
    for f, feature in enumerate(feature_order):
        plt.subplot(2,4,f+1)
        if feature_length[f]==1:
            plt.imshow(row[:,:,row_index:row_index+feature_length[f]], cmap='gray')
        else:
            plt.imshow(row[:,:,row_index:row_index+feature_length[f]])
        plt.title(feature)
        row_index += feature_length[f]
    plt.subplot(2,4,len(feature_order)+1)
    plt.imshow(label_array[r])
    os.makedirs(os.path.join(base_filepath, "Figures", f'Sample_{r}'), exist_ok=True)
    plt.savefig(os.path.join(base_filepath, "Figures", f'Sample_{r}', "bands.png"))

for r, row in enumerate(label_count_array[indices]):
    labels_of_interest = row != 0
    labels_of_interest = label_count.columns.values[labels_of_interest]
    colors_of_interest = [labels_rgb['categories'][label] for label in labels_of_interest]
    spectra_of_interest = np.empty((len(labels_of_interest), len(feature_order)+2))
    for l,label in enumerate(labels_of_interest):
        color_of_interest = np.array(labels_rgb['categories'][label])
        indices_of_interest = np.where(label_array[r].reshape(-1,3)==color_of_interest)[0]
        data_of_interest = copy.deepcopy(data_array[r]).reshape(-1,data_array[r].shape[-1])
        data_of_interest = data_of_interest[indices_of_interest]
        spectra_of_interest[l] = np.mean(data_of_interest,axis=0)

    
    fig = plt.figure(figsize=(10,5))
    for s,spectrum in enumerate(spectra_of_interest):
        plt.plot(spectrum, label=labels_of_interest[s], color=tuple(np.array(colors_of_interest[s])/256))
    plt.legend()
    plt.savefig(os.path.join(base_filepath, "Figures", f'Sample_{r}', 'Average_class_spectra_before_correction.png'))

    average_spectrum = np.mean(data_array[r].reshape(-1, data_array[r].shape[-1]),axis=0)
    plt.figure(figsize=(10,5))
    plt.plot(average_spectrum, label='Average scene spectrum')
    plt.legend()
    plt.savefig(os.path.join(base_filepath, "Figures", f'Sample_{r}', 'Average_scene_spectrum.png'))

    fig = plt.figure(figsize=(10,5))
    for s,spectrum in enumerate(spectra_of_interest):
        plt.plot(spectrum/average_spectrum, label=labels_of_interest[s], color=tuple(np.array(colors_of_interest[s])/256))
    plt.legend()
    plt.savefig(os.path.join(base_filepath, "Figures", f'Sample_{r}', 'Average_class_spectra_after_correction.png'))

    
pass

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score
import os
import cv2

def normalize_data(X):
    return (X - np.min(X))/(np.max(X)- np.min(X))

def standardize_data(X):
    return (X - np.mean(X))/np.std(X)

def calc_metric(y_true, y_pred, metric="AUC"):
    metric_dict = {"AUC": roc_auc_score,
                   'IOU': jaccard_score,
                   "AP": average_precision_score,
                   "FP": false_positive}

    if metric in metric_dict:
        return metric_dict[metric](y_true, y_pred)
    else:
        print(f"{metric} is not currently supported.")
    
def false_positive(y_true, y_score):
    temp_scores = np.sign(y_score - np.percentile(y_score, 99.5))
    temp_scores[temp_scores<0] = 0
    temp_labels = np.abs(y_true-1).reshape(-1,1)
    masked_labels = temp_labels * temp_scores
    return np.sum(masked_labels)/len(masked_labels)

def load_mudcad_file(directory, load_vis=False):
    feature_order = ["vis", "blue", "green", "red", "eir", "nir", "lwir"]
    feature_length = [3, 1, 1, 1, 1, 1, 1]
    if load_vis:
        feature_order = feature_order[0]
        feature_length = feature_length[0]

    metadata = directory.split('MUCAD')[1]
    metadata = metadata.split(os.sep)[2:]

    if os.path.exists(os.path.join(directory, 'Data.npz')):
        data = np.load(os.path.join(directory, 'Data.npz'))
        data_array = data['data']
        if load_vis:
            data_array = data_array[:, :, 0:3]
        labels = data['labels']
    else:
        data_array_index = 0
        data_array = np.zeros((512, 512, sum(feature_length)))
        for f,feature in enumerate(feature_order):
            file_to_load = os.path.join(directory, feature + '.png')
            data = cv2.imread(file_to_load, cv2.IMREAD_UNCHANGED)
            bands = data.shape[-1]
            if bands != 3:
                data = data[:,:,np.newaxis]
                bands = 1
            assert bands == feature_length[f]
            data_array[:, :, data_array_index:data_array_index + bands] = data/256
            data_array_index += bands

        file_to_load = os.path.join(directory, 'label.png')
        labels = cv2.imread(file_to_load, cv2.IMREAD_UNCHANGED)

    return (metadata, data_array, labels)

def create_save_dir_name(base_path, model_name, experiment_settings):
    base = os.path.join(base_path, 'Results')
    if 'AC_model' in experiment_settings:
        base= os.path.join(base, f"AC_{experiment_settings['AC_model']['name']}")
    if 'Scaler' in experiment_settings:
        base = os.path.join(base, f"Scaler_{experiment_settings['Scaler']['name']}")
        if 'scaling_scope' in experiment_settings['Scaler']:
            base = base + f"_{experiment_settings['Scaler']['scaling_scope']}"
    if 'Subsample' in experiment_settings:
        base = os.path.join(base, f"Subsample_{experiment_settings['Subsample']}")
    if 'Vis_scale' in experiment_settings:
        base = os.path.join(base, f"Vis_{experiment_settings['Vis_scale']}")
    base = os.path.join(base, f"{model_name}")
    if 'auto' in model_name:
        if 'test_split_method' in experiment_settings:
            base = os.path.join(base, f"Test_split_{experiment_settings['test_split_method']}")
            if 'test_season' in experiment_settings and experiment_settings['test_split_method'].lower() == 'seasonal':
                base = os.path.join(base, f"Test_season_{experiment_settings['test_season']}")
        if 'loss_function' in experiment_settings:
            base = os.path.join(base, f"{experiment_settings['loss_function']}")

    return base

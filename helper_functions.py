import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score
import os
import cv2
import scipy.io

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

def load_salinas(dataset_filepath):
    salinas_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'Salinas.mat'))['salinas']
    salinas_corrected_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'Salinas_corrected.mat'))['salinas_corrected']
    salinas_labels = scipy.io.loadmat(os.path.join(dataset_filepath, 'Salinas_gt.mat'))['salinas_gt']
    labels_ids = {0: ("Background", 56975),
        1: ("Broccoli_green_weeds_1", 2009),
        2: ("Broccoli_green_weeds_2", 3726),
        3: ("Fallow", 1976),
        4: ("Fallow_rough_plow", 1394),
        5: ("Fallow_smooth", 2678),
        6: ("Stubble", 3959),
        7: ("Celery", 3579),
        8: ("Grapes_untrained", 11271),
        9: ("Soil_vinyard_develop", 6203),
        10: ("Corn_senesced_green_weeds", 3278),
        11: ("Lettuce_romaine_4wk", 1068),
        12: ("Lettuce_romaine_5wk", 1927),
        13: ("Lettuce_romaine_6wk", 916),
        14: ("Lettuce_romaine_7wk", 1070),
        15: ("Vinyard_untrained", 7268),
        16: ("Vinyard_vertical_trellis", 1807),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(salinas_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]
    
    return salinas_data, salinas_corrected_data, salinas_data, labels_ids
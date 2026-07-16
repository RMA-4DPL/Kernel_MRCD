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
    salinas_data = salinas_data.astype(np.float32)/256
    salinas_corrected_data = salinas_corrected_data.astype(np.float32)/256
    
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
    
    return salinas_data[None, ...], salinas_corrected_data[None, ...], salinas_labels[None, ...], labels_ids

def load_salinas_A(dataset_filepath):
    salinas_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'SalinasA.mat'))['salinasA']
    salinas_corrected_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'SalinasA_corrected.mat'))['salinasA_corrected']
    salinas_labels = scipy.io.loadmat(os.path.join(dataset_filepath, 'SalinasA_gt.mat'))['salinasA_gt']
    salinas_data = salinas_data.astype(np.float32)/256
    salinas_corrected_data = salinas_corrected_data.astype(np.float32)/256

    labels_ids = {0: ("Background", 1790),
    1: ("Broccoli_green_weeds_1", 391),
    10: ("Corn_senesced_green_weeds", 1343),
    11: ("Lettuce_romaine_4wk", 616),
    12: ("Lettuce_romaine_5wk", 1525),
    13: ("Lettuce_romaine_6wk", 674),
    14: ("Lettuce_romaine_7wk", 799),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(salinas_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]
    
    return salinas_data[None, ...], salinas_corrected_data[None, ...], salinas_labels[None, ...], labels_ids

def load_hydice(dataset_filepath):
    hydice_mat = scipy.io.loadmat(os.path.join(dataset_filepath, 'HYDICE-urban.mat'))
    hydice_data = hydice_mat['data']
    hydice_labels = hydice_mat['map']

    labels_ids = {0: ("Background", 7979),
        1: ("Anomaly", 21),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(hydice_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]

    return hydice_data[None, ...], hydice_data[None, ...], hydice_labels[None, ...], labels_ids

def create_save_dir_name(base, model_name, experiment_settings):
    base = os.path.join(base, experiment_settings['dataset'])
    if 'AC_model' in experiment_settings:
        base= os.path.join(base, f"AC_{experiment_settings['AC_model']['name']}")
    if 'Scaler' in experiment_settings:
        base = os.path.join(base, f"Scaler_{experiment_settings['Scaler']['name']}")
        if 'scaling_scope' in experiment_settings['Scaler']:
            base = base + f"_{experiment_settings['Scaler']['scaling_scope']}"
    if model_name is not None:
        base = os.path.join(base, f"{model_name}_{experiment_settings['background_model']}")

    return base

def load_dataset(base_path, dataset_name):
    datasets = {'Salinas': load_salinas,
                'Salinas_A': load_salinas_A,
                'HYDICE': load_hydice}
    if dataset_name in datasets:
        dataset_filepath = os.path.join(base_path, 'Dataset', dataset_name)
        raw_data, data, labels, labels_ids = datasets[dataset_name](dataset_filepath)
        return raw_data, data, labels, labels_ids
    else:
        print(f"Dataset {dataset_name} is not available.")

def calc_cov(X):
    X_t = X.reshape(-1, X.shape[-1])
    X_t = (X_t - np.mean(X_t, axis=0))
    cov = X_t.T @ X_t
    cov = cov / (X_t.shape[0] -1)
    return cov


def qn_scale_overwrite(a, c=2.219144465985076, axis=0):
    """Qn robust scale estimator (Rousseeuw & Croux), matching
    statsmodels.robust.scale.qn_scale without the statsmodels dependency.

    Qn(a) = c * {|a[i] - a[j]| : i < j}_(k), k = C([n/2] + 1, 2)
    """
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 0:
        raise ValueError("a should have at least one dimension")
    if a.size == 0:
        return np.nan
    if a.ndim == 1:
        return _qn_1d(a, c)
    indices = ~np._core.numeric.greater_equal.outer(np.arange(a.shape[0], dtype=np.int32),
                                np.arange(0, a.shape[0], dtype=np.int32))
    return np.apply_along_axis(_qn_1d, axis, a, c, indices=indices)


def _qn_1d(x, c, indices=None):
    n = x.shape[0]
    x = x[:, None]
    
    if n == 1:
        return 0.0
    #indices_row, indices_col = np.triu_indices(n, k=1)
    if indices is None:
        pairwise = np.abs(x - x.T)[~np._core.numeric.greater_equal.outer(np.arange(n, dtype=np.int32),
                                    np.arange(0, n, dtype=np.int32))]
    else:
        pairwise = np.abs(x - x.T)[indices]
    h = n // 2 + 1
    k = h * (h - 1) // 2
    pairwise.partition(k-1)
    return c * pairwise[k-1]
    # return c * np.partition(pairwise, k - 1)[k - 1]

def get_int_dtype(n):
    log2_n = np.log2(n)
    if log2_n<16:
        target_dtype=np.uint16
    elif log2_n<32:
        target_dtype=np.uint32
    return target_dtype
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score
import os
import glob
import functools
import cv2
import scipy.io
import spectral
import spectral.io.erdas as erdas

def normalize_data(X):
    return (X - np.min(X))/(np.max(X)- np.min(X))

def clip_and_normalize_data(X, lower_percentile=0.5, upper_percentile=99.5):
    lo, hi = np.percentile(X, [lower_percentile, upper_percentile])
    return normalize_data(np.clip(X, lo, hi))

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

def load_pavia(dataset_filepath):
    pavia_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'Pavia.mat'))['pavia']
    pavia_labels = scipy.io.loadmat(os.path.join(dataset_filepath, 'Pavia_gt.mat'))['pavia_gt']

    labels_ids = {0: ("Background", 635488),
        1: ("Water", 65971),
        2: ("Trees", 7598),
        3: ("Asphalt", 3090),
        4: ("Self-Blocking Bricks", 2685),
        5: ("Bitumen", 6584),
        6: ("Tiles", 9248),
        7: ("Shadows", 7287),
        8: ("Meadows", 42826),
        9: ("Bare Soil", 2863),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(pavia_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]

    return pavia_data[None, ...], pavia_data[None, ...], pavia_labels[None, ...], labels_ids

def load_pavia_u(dataset_filepath):
    paviaU_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'PaviaU.mat'))['paviaU']
    paviaU_labels = scipy.io.loadmat(os.path.join(dataset_filepath, 'PaviaU_gt.mat'))['paviaU_gt']

    labels_ids = {0: ("Background", 164624),
        1: ("Asphalt", 6631),
        2: ("Meadows", 18649),
        3: ("Gravel", 2099),
        4: ("Trees", 3064),
        5: ("Painted metal sheets", 1345),
        6: ("Bare Soil", 5029),
        7: ("Bitumen", 1330),
        8: ("Self-Blocking Bricks", 3682),
        9: ("Shadows", 947),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(paviaU_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]

    return paviaU_data[None, ...], paviaU_data[None, ...], paviaU_labels[None, ...], labels_ids

def load_sandiego(dataset_filepath):
    sandiego_data = scipy.io.loadmat(os.path.join(dataset_filepath, 'Sandiego.mat'))['Sandiego']
    # No ground-truth file is available for this dataset, so everything is labeled as background.
    sandiego_labels = np.zeros(sandiego_data.shape[:2], dtype=np.uint8)

    labels_ids = {0: ("Background", int(sandiego_labels.size))}

    return sandiego_data[None, ...], sandiego_data[None, ...], sandiego_labels[None, ...], labels_ids

def load_indiana(dataset_filepath):
    indiana_data = erdas.open(os.path.join(dataset_filepath, '92AV3C.lan')).load()
    indiana_labels = erdas.open(os.path.join(dataset_filepath, '92AV3GT.GIS')).load().squeeze(axis=-1).astype(np.uint8)

    labels_ids = {0: ("Background", 10659),
        1: ("Alfalfa", 54),
        2: ("Corn-notill", 1434),
        3: ("Corn-mintill", 834),
        4: ("Corn", 234),
        5: ("Grass-pasture", 497),
        6: ("Grass-trees", 747),
        7: ("Grass-pasture-mowed", 26),
        8: ("Hay-windrowed", 489),
        9: ("Oats", 20),
        10: ("Soybean-notill", 968),
        11: ("Soybean-mintill", 2468),
        12: ("Soybean-clean", 614),
        13: ("Wheat", 212),
        14: ("Woods", 1294),
        15: ("Buildings-Grass-Trees-Drives", 380),
        16: ("Stone-Steel-Towers", 95),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(indiana_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]

    return indiana_data[None, ...], indiana_data[None, ...], indiana_labels[None, ...], labels_ids

def load_whu_hi(dataset_filepath):
    whu_hi_data = np.asarray(spectral.open_image(os.path.join(dataset_filepath, 'WHU-Hi-LongKou.hdr')).load())
    whu_hi_labels = np.asarray(spectral.open_image(os.path.join(dataset_filepath, 'WHU-Hi-LongKou_gt.hdr')).load()).squeeze(axis=-1).astype(np.uint8)

    labels_ids = {0: ("Unclassified", 15458),
        1: ("Corn", 34511),
        2: ("Cotton", 8374),
        3: ("Sesame", 3031),
        4: ("Broad-leaf soybean", 63212),
        5: ("Narrow-leaf soybean", 4151),
        6: ("Rice", 11854),
        7: ("Water", 67056),
        8: ("Roads and houses", 7124),
        9: ("Mixed weed", 5229),
    }

    # Check that label counts match
    unique_labels, counts = np.unique(whu_hi_labels, return_counts=True)
    for l, label in enumerate(unique_labels):
        assert counts[l] == labels_ids[label][1]

    return whu_hi_data[None, ...], whu_hi_data[None, ...], whu_hi_labels[None, ...], labels_ids

def load_abu(dataset_filepath, filename):
    abu_mat = scipy.io.loadmat(os.path.join(dataset_filepath, filename))
    abu_data = abu_mat['data']
    abu_labels = abu_mat['map']

    unique_labels, counts = np.unique(abu_labels, return_counts=True)
    labels_ids = {int(label): (name, int(count))
                  for label, name, count in zip(unique_labels, ["Background", "Anomaly"], counts)}

    return abu_data[None, ...], abu_data[None, ...], abu_labels[None, ...], labels_ids

def _parse_envi_roi_points(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # ROI points are listed as consecutive (id, x, y) blocks, one block per
    # "; ROI npts: N" entry declared earlier in the file, in the same order.
    roi_npts = [int(line.split(':')[1]) for line in lines if line.strip().startswith('; ROI npts:')]
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('; ID'))
    point_lines = [line.split() for line in lines[header_idx + 1:] if line.strip()]

    rois = []
    idx = 0
    for npts in roi_npts:
        block = point_lines[idx:idx + npts]
        rois.append([(int(x), int(y)) for _, x, y in block])
        idx += npts
    return rois

def load_cooke_city(dataset_filepath):
    cooke_city_data = np.asarray(spectral.open_image(os.path.join(dataset_filepath, 'HyMap', 'self_test_refl.hdr')).load())

    H, W = cooke_city_data.shape[:2]
    cooke_city_labels = np.zeros((H, W), dtype=np.uint8)
    roi_filepaths = glob.glob(os.path.join(dataset_filepath, 'ROI', '**', '*.txt'), recursive=True)
    for roi_filepath in roi_filepaths:
        for points in _parse_envi_roi_points(roi_filepath):
            # A single-point ROI marks a sub-pixel target's center; the
            # surrounding 'Sub'/'Guard' ROIs are exclusion bands, not targets.
            if len(points) == 1:
                x, y = points[0]
                cooke_city_labels[y, x] = 1

    labels_ids = {0: ("Background", int(np.sum(cooke_city_labels == 0))),
        1: ("Target", int(np.sum(cooke_city_labels == 1))),
    }

    return cooke_city_data[None, ...], cooke_city_data[None, ...], cooke_city_labels[None, ...], labels_ids

def create_save_dir_name(base, model_name, experiment_settings):
    base = os.path.join(base, experiment_settings['dataset'])
    if 'AC_model' in experiment_settings:
        base= os.path.join(base, f"AC_{experiment_settings['AC_model']['name']}")
    if 'Subsample' in experiment_settings:
        base = os.path.join(base, f"Subsample_{experiment_settings['Subsample']['name']}")
        if experiment_settings['Subsample']['name'] != 'none':
            base = base + f"_{experiment_settings['Subsample']['amount']}"
    if 'Scaler' in experiment_settings:
        base = os.path.join(base, f"Scaler_{experiment_settings['Scaler']['name']}")
        if 'scaling_scope' in experiment_settings['Scaler']:
            base = base + f"_{experiment_settings['Scaler']['scaling_scope']}"
    if model_name is not None:
        base = os.path.join(base, f"{model_name}_{experiment_settings['background_model']}")

    return base

ABU_SCENE_COUNTS = {'airport': 4, 'beach': 4, 'urban': 5}

def load_dataset(base_path, dataset_name):
    datasets = {'Salinas': load_salinas,
                'Salinas_A': load_salinas_A,
                'HYDICE': load_hydice,
                'Pavia': load_pavia,
                'PaviaU': load_pavia_u,
                'SanDiego': load_sandiego,
                'Indiana': load_indiana,
                'WHU-HI': load_whu_hi,
                'cooke_city': load_cooke_city}
    for category, count in ABU_SCENE_COUNTS.items():
        for i in range(1, count + 1):
            datasets[f"ABU_{category}_{i}"] = functools.partial(load_abu, filename=f"abu-{category}-{i}.mat")
    # Some datasets share a directory rather than having their own.
    dataset_dirs = {'PaviaU': 'Pavia', 'cooke_city': os.path.join('cooke_city', 'self_test')}
    dataset_dirs.update({f"ABU_{category}_{i}": 'Airport-Beach-Urban'
                         for category, count in ABU_SCENE_COUNTS.items() for i in range(1, count + 1)})
    if dataset_name in datasets:
        dataset_filepath = os.path.join(base_path, 'Dataset', dataset_dirs.get(dataset_name, dataset_name))
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
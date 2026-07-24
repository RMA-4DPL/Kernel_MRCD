import numpy as np
import pandas as pd
import os
import yaml
from tqdm import tqdm
from helper_functions import calc_metric, create_save_dir_name
import pickle
import pathlib
import argparse

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
np.random.seed(4)

#MODEL_LIST = ['sample', 'ledoit_wolf', 'mrcd_auto_0.75_identity', 'kmrcd_0.75_rbf']
MODEL_LIST = ['mrcd_auto_0.75_identity', 'kmrcd_0.75_rbf']

argument_parser = argparse.ArgumentParser(description='Summarize classical AD model results (see LXR_test.py)')
argument_parser.add_argument('--dataset', type=str, default='Salinas', help='Select which dataset to load (default:Salinas).')
argument_parser.add_argument('--recalculate', action='store_true', help='Recalculate metrics even if already present in Results_summary.xlsx')
argument_parser.set_defaults(recalculate=False)
argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (overrides experiment_settings Scaler)')
argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample', 'all'], help='Scaling scope for the Scaler (overrides experiment_settings Scaler scaling_scope)')
argument_parser.add_argument('--subsample', type=str, default='random', help='Subsampling method (must match main.py)')
argument_parser.add_argument('--subsample_amount', type=int, default=10000, help='Amount of data points sampled (must match main.py)')
argument_parser.add_argument('--models', type=str, nargs='+', default=MODEL_LIST, help='Background models to include, matched against the "{model_name}_{background_model}" result directory suffix (e.g. sample, ledoit_wolf, mrcd_auto_0.75_identity); default (None) includes every background model found')
args = argument_parser.parse_args()

base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
base_filepath_results = os.path.join(base_filepath, 'Results')

with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings = yaml.safe_load(f)
metrics_to_calc = experiment_settings['metrics']
# For per-target models (AMF/ACE) each metric is also computed per label (see compute_sample_metrics);
# these companion columns hold the std across labels, per row, alongside the base mean-over-labels metric.
metrics_to_calc_all = metrics_to_calc + [f"{m}_label_std" for m in metrics_to_calc]
for arg in vars(args):
    experiment_settings[arg] = getattr(args, arg)
if args.scaler is not None:
    experiment_settings.setdefault('Scaler', {})['name'] = args.scaler
    if args.scaling_scope is not None:
        experiment_settings.setdefault('Scaler', {})['scaling_scope'] = args.scaling_scope
if args.subsample is not None:
    experiment_settings.setdefault('Subsample', {})['name'] = args.subsample
    experiment_settings.setdefault('Subsample', {})['amount'] = args.subsample_amount


def _fill_nan_scores(score_row):
    """Temporary fix for NaN values in scores, replace with mean of surrounding values."""
    nan_indices = np.where(np.isnan(score_row))[0]
    for index in nan_indices:
        score_row[index] = np.nanmean(score_row[index-5:index+5])
    return score_row


def compute_sample_metrics(scores, label_array, anomaly_labels, category_ids, metrics_to_calc, desc, per_label_ids=None):
    """Score every sample against the true anomaly_labels (anomaly vs background, from label_array).
    Used for both the default 'Scores' map and the 'Scores_binary' map -- in both cases the
    comparison is score-vs-ground-truth-label, never score-vs-score.

    For per-target models (AMF/ACE), `scores` is (L, num_labels, H, W) -- one score map per target
    label, see main.py's Scores_per_label_ids -- and each label's own map is scored against that
    label vs. everything else (label_row == label_id), and against its own top-0.5% pixels for the
    percentage-correct table. The per-row metric is then the mean across labels, with the returned
    label-std dict holding the std across labels (all NaN when scores is a single combined map,
    e.g. RX or Scores_binary)."""
    per_label = scores.ndim == 4
    L = scores.shape[0]
    metric_per_sample = {m: np.zeros((L,), dtype=np.float32) for m in metrics_to_calc}
    metric_per_sample_label_std = {m: np.full((L,), np.nan, dtype=np.float32) for m in metrics_to_calc}
    perc_correct_per_sample = np.full((L, len(category_ids)), np.nan, dtype=np.float32)

    for r in tqdm(range(L), total=L, desc=desc):
        label_row = label_array[r].reshape(-1)
        anomaly_row = anomaly_labels[r].reshape(-1)

        if per_label:
            label_scores = {label_id: _fill_nan_scores(scores[r, l].reshape(-1).astype(np.float32))
                             for l, label_id in enumerate(per_label_ids) if label_id in category_ids}

            for m in metrics_to_calc:
                per_label_values = [calc_metric((label_row == label_id).astype(anomaly_row.dtype), score_row, metric=m)
                                     for label_id, score_row in label_scores.items()]
                metric_per_sample[m][r] = np.mean(per_label_values)
                metric_per_sample_label_std[m][r] = np.std(per_label_values)

            for c, cat_id in enumerate(category_ids):
                score_row = label_scores[cat_id]
                threshold = np.percentile(score_row, 99.5)
                predicted_anomaly = score_row >= threshold
                is_category = label_row == cat_id
                count_true = np.count_nonzero(is_category)
                count_pred = np.count_nonzero(is_category & predicted_anomaly)
                perc_correct_per_sample[r, c] = count_pred/count_true if count_true > 0 else np.nan
        else:
            score_row = _fill_nan_scores(scores[r].reshape(-1).astype(np.float32))

            for m in metrics_to_calc:
                metric_per_sample[m][r] = calc_metric(anomaly_row, score_row, metric=m)

            threshold = np.percentile(score_row, 99.5)
            predicted_anomaly = score_row >= threshold
            for c, cat_id in enumerate(category_ids):
                is_category = label_row == cat_id
                count_true = np.count_nonzero(is_category)
                count_pred = np.count_nonzero(is_category & predicted_anomaly)
                perc_correct_per_sample[r, c] = count_pred/count_true if count_true > 0 else np.nan

    return metric_per_sample, perc_correct_per_sample, metric_per_sample_label_std


def compute_background_class_distribution(background_indices, label_array, experiment_configs, uses_kernel, category_ids, desc):
    """For each row, recover which ground-truth class the pixels selected as background
    (the MCD/MRCD/KMRCD h-subset) belong to, and tally the class distribution.

    Only the subsampled pixel *vectors* are stored in Raw_results.pickle (see main.py's
    "Subsamples"/"Background_pixels"), not their original spatial indices, so the random
    subsample selection (np.random.seed(4)-based, see subsampler.py) is deterministically
    replayed here to map Background_indices back onto label_array.
    """
    L, H, W = label_array.shape
    subsample_cfg = experiment_configs.get('Subsample', {}) or {}
    subsample_name = subsample_cfg.get('name', 'none')
    subsample_amount = subsample_cfg.get('amount')
    counts_per_sample = np.zeros((L, len(category_ids)), dtype=np.float32)

    for r in tqdm(range(L), total=L, desc=desc):
        row_H, row_W = label_array[r].shape
        # Mirrors main.py's row[::2,::2] downsampling, applied before kernel-background
        # selection on large rows when no explicit subsampling is configured.
        if uses_kernel and subsample_name == 'none' and row_H * row_W > 80000:
            label_row = label_array[r][::2, ::2].reshape(-1)
        else:
            label_row = label_array[r].reshape(-1)

        if subsample_name == 'random':
            np.random.seed(4)
            n_pixels = label_row.shape[0]
            sampled_positions = np.random.choice(np.arange(n_pixels), size=min(n_pixels, subsample_amount), replace=False)
            candidate_labels = label_row[sampled_positions]
        else:
            candidate_labels = label_row

        selected_labels = candidate_labels[background_indices[r]]
        for c, cid in enumerate(category_ids):
            counts_per_sample[r, c] = np.count_nonzero(selected_labels == cid)

    totals = counts_per_sample.sum(axis=1, keepdims=True)
    percentages_per_sample = np.divide(counts_per_sample, totals, out=np.zeros_like(counts_per_sample), where=totals > 0)
    return percentages_per_sample


def summarize(metric_dict, perc_correct_dict):
    metric_keys = list(metric_dict.keys())
    metrics_summary = {}
    perc_summary = {}
    for model in metric_dict[metric_keys[0]]:
        metrics = np.zeros((len(metric_keys),), dtype=object)
        for i, m in enumerate(metric_keys):
            temp = metric_dict[m][model]
            if len(temp)==1:
                metrics[i] = f"{np.nanmean(temp):.3f}"
            else:
                metrics[i] = f"{np.nanmean(temp):.3f} ± {np.nanstd(temp):.3f}"
        metrics_summary[model] = metrics

        temp = perc_correct_dict[model]
        temp_mean = np.nanmean(temp, axis=0)
        temp_std = np.nanstd(temp, axis=0)
        if len(temp)==1:
            perc_summary[model] = [f"{temp_mean[i]:.3f}" for i in range(len(temp_mean))]
        else:
            perc_summary[model] = [f"{temp_mean[i]:.3f} ± {temp_std[i]:.3f}" for i in range(len(temp_mean))]
    return metrics_summary, perc_summary


metric_dict = {}
perc_correct_dict = {}
metric_dict_binary = {}
perc_correct_dict_binary = {}
bg_dist_dict = {}
category_names = None
all_category_names = None
for m in metrics_to_calc_all:
    metric_dict[m] = {}
for m in metrics_to_calc:
    metric_dict_binary[m] = {}

# Models are discovered from the results directory itself (one subdirectory per
# "{model_name}_{background_model}" combination, see LXR_test.py's create_save_dir_name
# call) rather than from model_configs.yaml, so that any model/background_model combination
# that has actually been run shows up here.
results_dir = create_save_dir_name(base_filepath_results, None, experiment_settings)
summary_save_dir = results_dir
# LXR_test.py has no train/test split (see run_experiments.mk), so there is a single
# summary/detailed file per dataset/scaler combination.
# The "_binary" files score the Scores_binary map (single foreground-vs-background target,
# produced by main.py) against the same true anomaly_labels used for the default "Scores" map
# (max over per-category targets). Older results (e.g. from LXR_test.py) have no Scores_binary
# and are simply skipped for the binary file.
summary_filename = "Results_summary.xlsx"
detailed_filename = "Results_detailed.xlsx"
summary_path = os.path.join(summary_save_dir, summary_filename)
detailed_path = os.path.join(summary_save_dir, detailed_filename)
summary_filename_binary = "Results_summary_binary.xlsx"
detailed_filename_binary = "Results_detailed_binary.xlsx"
summary_path_binary = os.path.join(summary_save_dir, summary_filename_binary)
detailed_path_binary = os.path.join(summary_save_dir, detailed_filename_binary)

existing_metrics_df = None
existing_perc_summary_df = None
existing_detailed_metric = {}
existing_detailed_perc = {}

if not args.recalculate and os.path.exists(summary_path):
    try:
        existing_metrics_df = pd.read_excel(summary_path, sheet_name='Metrics', index_col=0)
        existing_perc_summary_df = pd.read_excel(summary_path, sheet_name='Percentage correct', index_col=0)
        if os.path.exists(detailed_path):
            detailed_xls = pd.ExcelFile(detailed_path)
            for m in metrics_to_calc_all:
                if m in detailed_xls.sheet_names:
                    existing_detailed_metric[m] = pd.read_excel(detailed_xls, sheet_name=m, index_col=0)
            for sheet_name in detailed_xls.sheet_names:
                if sheet_name.startswith("PC "):
                    existing_detailed_perc[sheet_name[len("PC "):]] = pd.read_excel(detailed_xls, sheet_name=sheet_name, index_col=0)
    except Exception as e:
        print(f"Could not read existing summary/detailed results, will recompute everything: {e}")
        existing_metrics_df = None
        existing_perc_summary_df = None

existing_metrics_df_binary = None
existing_perc_summary_df_binary = None
existing_detailed_metric_binary = {}
existing_detailed_perc_binary = {}

if not args.recalculate and os.path.exists(summary_path_binary):
    try:
        existing_metrics_df_binary = pd.read_excel(summary_path_binary, sheet_name='Metrics', index_col=0)
        existing_perc_summary_df_binary = pd.read_excel(summary_path_binary, sheet_name='Percentage correct', index_col=0)
        if os.path.exists(detailed_path_binary):
            detailed_xls_binary = pd.ExcelFile(detailed_path_binary)
            for m in metrics_to_calc:
                if m in detailed_xls_binary.sheet_names:
                    existing_detailed_metric_binary[m] = pd.read_excel(detailed_xls_binary, sheet_name=m, index_col=0)
            for sheet_name in detailed_xls_binary.sheet_names:
                if sheet_name.startswith("PC "):
                    existing_detailed_perc_binary[sheet_name[len("PC "):]] = pd.read_excel(detailed_xls_binary, sheet_name=sheet_name, index_col=0)
    except Exception as e:
        print(f"Could not read existing binary summary/detailed results, will recompute everything: {e}")
        existing_metrics_df_binary = None
        existing_perc_summary_df_binary = None

def has_cached_result(model_name):
    if existing_metrics_df is None or existing_perc_summary_df is None:
        return False
    if model_name not in existing_metrics_df.index or model_name not in existing_perc_summary_df.index:
        return False
    if not set(metrics_to_calc_all).issubset(existing_metrics_df.columns):
        return False
    if existing_metrics_df.loc[model_name, metrics_to_calc_all].isna().any():
        return False
    if model_name not in existing_detailed_perc:
        return False
    return all(model_name in existing_detailed_metric.get(m, pd.DataFrame()).index for m in metrics_to_calc_all)

def has_cached_result_binary(model_name):
    if existing_metrics_df_binary is None or existing_perc_summary_df_binary is None:
        return False
    if model_name not in existing_metrics_df_binary.index or model_name not in existing_perc_summary_df_binary.index:
        return False
    if not set(metrics_to_calc).issubset(existing_metrics_df_binary.columns):
        return False
    if existing_metrics_df_binary.loc[model_name, metrics_to_calc].isna().any():
        return False
    if model_name not in existing_detailed_perc_binary:
        return False
    return all(model_name in existing_detailed_metric_binary.get(m, pd.DataFrame()).index for m in metrics_to_calc)

def select_models(available_models, requested_backgrounds):
    # Result directories are named "{model_name}_{background_model}" (see create_save_dir_name),
    # e.g. "base_rx_sample" or "base_amf_mrcd_auto_0.75_identity". requested_backgrounds selects
    # by that background_model suffix, keeping every detector (rx/amf/ace, ...) that uses it.
    if requested_backgrounds is None:
        return available_models
    selected = [m for m in available_models if any(m.endswith(f"_{bg}") for bg in requested_backgrounds)]
    missing = [bg for bg in requested_backgrounds if not any(m.endswith(f"_{bg}") for m in available_models)]
    if missing:
        print(f"Warning: requested background models not found and will be skipped: {missing}")
    return selected


model_dirs = []
if os.path.isdir(results_dir):
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "Raw_results.pickle")):
            model_dirs.append(entry)
model_dirs = select_models(model_dirs, args.models)
print(f"Found {len(model_dirs)} model result director{'y' if len(model_dirs) == 1 else 'ies'} under {results_dir}")

models_found = 0
models_found_binary = 0
for model in model_dirs:
    cached_main = has_cached_result(model)
    if cached_main:
        print(f"Using cached results for model {model} (already present in {summary_filename})")
        for m in metrics_to_calc_all:
            metric_dict[m][model] = existing_detailed_metric[m].loc[model].values
        perc_correct_dict[model] = existing_detailed_perc[model].values
        if category_names is None:
            category_names = list(existing_detailed_perc[model].columns)
        models_found += 1

    cached_binary = has_cached_result_binary(model)
    if cached_binary:
        print(f"Using cached binary results for model {model} (already present in {summary_filename_binary})")
        for m in metrics_to_calc:
            metric_dict_binary[m][model] = existing_detailed_metric_binary[m].loc[model].values
        perc_correct_dict_binary[model] = existing_detailed_perc_binary[model].values
        models_found_binary += 1

    if cached_main and cached_binary:
        continue

    try:
        save_dir = os.path.join(results_dir, model)
        pickle_path = os.path.join(save_dir, "Raw_results.pickle")
        print('Loading results for model: ', model)
        print(f'Loading data from {save_dir}')
        with open(pickle_path, "rb") as f:
            x = pickle.load(f)
        label_array = x["Labels"]  # (L, H, W) categorical class ids, 0 = background -- ground truth for both scores below
        scores = x["Scores"]  # (L, H, W), or (L, num_labels, H, W) for per-target models (AMF/ACE)
        scores_per_label_ids = x.get("Scores_per_label_ids")  # label id per channel of the 4D Scores array, only present for AMF/ACE
        scores_binary = x.get("Scores_binary")  # (L, H, W), only present for main.py results
        label_ids = x["Label_ids"]
        background_indices = x.get("Background_indices")  # per-row indices of the selected background (h-subset) pixels
        experiment_configs = x.get("Experiment_configs") or {}
        background_configs_info = x.get("Background_configs") or {}
        background_config_used = next(iter(background_configs_info.values()), {}) or {}
        uses_kernel = 'kernel' in background_config_used
        del x

        category_ids = sorted(cid for cid in label_ids if cid != 0)
        if args.dataset == 'WHU-HI':
            category_ids = sorted(cid for cid in label_ids if cid != 7 and cid !=0) # Set water and "unclassified" as the "normal class" for WHU-HI dataset
        if category_names is None:
            category_names = [label_ids[cid][0] for cid in category_ids]
        all_category_ids = sorted(label_ids.keys())
        if all_category_names is None:
            all_category_names = [label_ids[cid][0] for cid in all_category_ids]

        # Anomaly ground truth: any non-background class counts as an anomaly. Both "scores"
        # and "scores_binary" are evaluated against this same label-derived ground truth.
        anomaly_labels = (label_array != 0).astype(np.int16)
        if args.dataset == 'WHU-HI':
            anomaly_labels = (label_array != 7).astype(np.int16) # Set water as the "normal class" for WHU-HI dataset

        if not cached_main:
            metric_per_sample, perc_correct_per_sample, metric_per_sample_label_std = compute_sample_metrics(
                scores, label_array, anomaly_labels, category_ids, metrics_to_calc, model, per_label_ids=scores_per_label_ids)
            for m in metrics_to_calc:
                metric_dict[m][model] = metric_per_sample[m]
                metric_dict[f"{m}_label_std"][model] = metric_per_sample_label_std[m]
            perc_correct_dict[model] = perc_correct_per_sample
            models_found += 1

            if background_indices is not None:
                bg_dist_dict[model] = compute_background_class_distribution(
                    background_indices, label_array, experiment_configs, uses_kernel, all_category_ids, f"{model} (background)")
            else:
                print(f"No background indices found for {model}; skipping background class distribution.")

        if not cached_binary:
            if scores_binary is None:
                print(f"No binary scores found for model {model}; copying regular scores instead.")
                if cached_main:
                    metric_per_sample, perc_correct_per_sample, _ = compute_sample_metrics(
                        scores, label_array, anomaly_labels, category_ids, metrics_to_calc, model, per_label_ids=scores_per_label_ids)
                for m in metrics_to_calc:
                    metric_dict_binary[m][model] = metric_per_sample[m]
                perc_correct_dict_binary[model] = perc_correct_per_sample
                models_found_binary += 1
            else:
                metric_per_sample_binary, perc_correct_per_sample_binary, _ = compute_sample_metrics(
                    scores_binary, label_array, anomaly_labels, category_ids, metrics_to_calc, f"{model} (binary)")
                for m in metrics_to_calc:
                    metric_dict_binary[m][model] = metric_per_sample_binary[m]
                perc_correct_dict_binary[model] = perc_correct_per_sample_binary
                models_found_binary += 1
    except Exception as e:
        print(f"Error processing model {model}: {e}")


# Generate result summary
print("Summarizing results")

if models_found == 0:
    print('No results found. Skipping save step.')
else:
    metrics_summary, perc_summary = summarize(metric_dict, perc_correct_dict)
    save_dir = summary_save_dir
    print('Saving results to', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Remove previouses xlsx files so that new files are always "clean".
    for path in (detailed_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
        metric_df = pd.DataFrame.from_dict(metrics_summary, orient='index', columns=metrics_to_calc_all)
        metric_df.to_excel(writer, sheet_name='Metrics', index=True)
        perc_correct_df = pd.DataFrame.from_dict(perc_summary, orient='index', columns=category_names)
        perc_correct_df.to_excel(writer, sheet_name='Percentage correct', index=True)
        if bg_dist_dict:
            bg_dist_summary = {}
            for bg_model, values in bg_dist_dict.items():
                mean = np.nanmean(values, axis=0)
                std = np.nanstd(values, axis=0)
                if len(values) == 1:
                    bg_dist_summary[bg_model] = [f"{mean[i]:.3f}" for i in range(len(mean))]
                else:
                    bg_dist_summary[bg_model] = [f"{mean[i]:.3f} ± {std[i]:.3f}" for i in range(len(mean))]
            bg_dist_df = pd.DataFrame.from_dict(bg_dist_summary, orient='index', columns=all_category_names)
            bg_dist_df.to_excel(writer, sheet_name='Background class distribution', index=True)
    print(f"Saved results to {summary_path}")

if models_found_binary == 0:
    print('No binary results found. Skipping binary save step.')
else:
    metrics_summary_binary, perc_summary_binary = summarize(metric_dict_binary, perc_correct_dict_binary)
    save_dir = summary_save_dir
    print('Saving binary results to', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(summary_path_binary):
        os.remove(summary_path_binary)

    with pd.ExcelWriter(summary_path_binary, engine='xlsxwriter') as writer:
        metric_df = pd.DataFrame.from_dict(metrics_summary_binary, orient='index', columns=metrics_to_calc)
        metric_df.to_excel(writer, sheet_name='Metrics', index=True)
        perc_correct_df = pd.DataFrame.from_dict(perc_summary_binary, orient='index', columns=category_names)
        perc_correct_df.to_excel(writer, sheet_name='Percentage correct', index=True)
    print(f"Saved binary results to {summary_path_binary}")

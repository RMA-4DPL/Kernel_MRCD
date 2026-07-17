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

argument_parser = argparse.ArgumentParser(description='Summarize classical AD model results (see LXR_test.py)')
argument_parser.add_argument('--dataset', type=str, default='HYDICE', help='Select which dataset to load (default:Salinas).')
argument_parser.add_argument('--recalculate', action='store_true', help='Recalculate metrics even if already present in Results_summary.xlsx')
argument_parser.set_defaults(recalculate=False)
argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (overrides experiment_settings Scaler)')
argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample'], help='Scaling scope for the Scaler (overrides experiment_settings Scaler scaling_scope)')
argument_parser.add_argument('--subsample', type=str, default='none', help='Subsampling method (must match main.py)')
argument_parser.add_argument('--subsample_amount', type=int, default=1000, help='Amount of data points sampled (must match main.py)')
args = argument_parser.parse_args()

local_filepath = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
base_filepath_results = os.path.join(base_filepath, 'Results')

with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings = yaml.safe_load(f)
metrics_to_calc = experiment_settings['metrics']
for arg in vars(args):
    experiment_settings[arg] = getattr(args, arg)
if args.scaler is not None:
    experiment_settings.setdefault('Scaler', {})['name'] = args.scaler
    if args.scaling_scope is not None:
        experiment_settings.setdefault('Scaler', {})['scaling_scope'] = args.scaling_scope
if args.subsample is not None:
    experiment_settings.setdefault('Subsample', {})['name'] = args.subsample
    experiment_settings.setdefault('Subsample', {})['amount'] = args.subsample_amount


metric_dict = {}
perc_correct_dict = {}
category_names = None
for m in metrics_to_calc:
    metric_dict[m] = {}

# Models are discovered from the results directory itself (one subdirectory per
# "{model_name}_{background_model}" combination, see LXR_test.py's create_save_dir_name
# call) rather than from model_configs.yaml, so that any model/background_model combination
# that has actually been run shows up here.
results_dir = create_save_dir_name(base_filepath_results, None, experiment_settings)
summary_save_dir = results_dir
# LXR_test.py has no train/test split (see run_experiments.mk), so there is a single
# summary/detailed file per dataset/scaler combination.
summary_filename = "Results_summary.xlsx"
detailed_filename = "Results_detailed.xlsx"
summary_path = os.path.join(summary_save_dir, summary_filename)
detailed_path = os.path.join(summary_save_dir, detailed_filename)

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
            for m in metrics_to_calc:
                if m in detailed_xls.sheet_names:
                    existing_detailed_metric[m] = pd.read_excel(detailed_xls, sheet_name=m, index_col=0)
            for sheet_name in detailed_xls.sheet_names:
                if sheet_name.startswith("PC "):
                    existing_detailed_perc[sheet_name[len("PC "):]] = pd.read_excel(detailed_xls, sheet_name=sheet_name, index_col=0)
    except Exception as e:
        print(f"Could not read existing summary/detailed results, will recompute everything: {e}")
        existing_metrics_df = None
        existing_perc_summary_df = None

def has_cached_result(model_name):
    if existing_metrics_df is None or existing_perc_summary_df is None:
        return False
    if model_name not in existing_metrics_df.index or model_name not in existing_perc_summary_df.index:
        return False
    if not set(metrics_to_calc).issubset(existing_metrics_df.columns):
        return False
    if existing_metrics_df.loc[model_name, metrics_to_calc].isna().any():
        return False
    if model_name not in existing_detailed_perc:
        return False
    return all(model_name in existing_detailed_metric.get(m, pd.DataFrame()).index for m in metrics_to_calc)

model_dirs = []
if os.path.isdir(results_dir):
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "Raw_results.pickle")):
            model_dirs.append(entry)
print(f"Found {len(model_dirs)} model result director{'y' if len(model_dirs) == 1 else 'ies'} under {results_dir}")

models_found = 0
for model in model_dirs:
    if has_cached_result(model):
        print(f"Using cached results for model {model} (already present in {summary_filename})")
        for m in metrics_to_calc:
            metric_dict[m][model] = existing_detailed_metric[m].loc[model].values
        perc_correct_dict[model] = existing_detailed_perc[model].values
        if category_names is None:
            category_names = list(existing_detailed_perc[model].columns)
        models_found += 1
        continue
    try:
        save_dir = os.path.join(results_dir, model)
        pickle_path = os.path.join(save_dir, "Raw_results.pickle")
        print('Loading results for model: ', model)
        print(f'Loading data from {save_dir}')
        with open(pickle_path, "rb") as f:
            x = pickle.load(f)
        label_array = x["Labels"]  # (L, H, W) categorical class ids, 0 = background
        scores = x["Scores"]  # (L, H, W)
        label_ids = x["Label_ids"]
        del x

        L, H, W = scores.shape
        category_ids = sorted(cid for cid in label_ids if cid != 0)
        if category_names is None:
            category_names = [label_ids[cid][0] for cid in category_ids]

        # Anomaly ground truth: any non-background class counts as an anomaly.
        binary_labels = (label_array != 0).astype(np.int16)

        metric_per_sample = {m: np.zeros((L,), dtype=np.float32) for m in metrics_to_calc}
        perc_correct_per_sample = np.full((L, len(category_ids)), np.nan, dtype=np.float32)

        for r in tqdm(range(L), total=L, desc=model):
            score_row = scores[r].reshape(-1).astype(np.float32)
            label_row = label_array[r].reshape(-1)
            binary_row = binary_labels[r].reshape(-1)

            nan_indices = np.where(np.isnan(score_row))[0]
            if len(nan_indices) > 0:
                for index in nan_indices:
                    score_row[index] = np.nanmean(score_row[index-5:index+5])  # Temporary fix for NaN values in scores, replace with mean of surrounding values

            for m in metrics_to_calc:
                metric_per_sample[m][r] = calc_metric(binary_row, score_row, metric=m)

            threshold = np.percentile(score_row, 99.5)
            predicted_anomaly = score_row >= threshold
            for c, cat_id in enumerate(category_ids):
                is_category = label_row == cat_id
                count_true = np.count_nonzero(is_category)
                count_pred = np.count_nonzero(is_category & predicted_anomaly)
                perc_correct_per_sample[r, c] = count_pred/count_true if count_true > 0 else np.nan

        for m in metrics_to_calc:
            metric_dict[m][model] = metric_per_sample[m]

        perc_correct_dict[model] = perc_correct_per_sample
        models_found += 1
    except Exception as e:
        print(f"Error processing model {model}: {e}")
        pass


# Generate result summary
print("Summarizing results")
metrics_summary = {}
perc_summary = {}

for model in metric_dict[metrics_to_calc[0]]:
    metrics = np.zeros((len(metrics_to_calc),), dtype=object)
    for i,m in enumerate(metric_dict):
        temp = metric_dict[m][model]
        metrics[i] = f"{np.nanmean(temp):.3f} ± {np.nanstd(temp):.3f}"
    metrics_summary[model] = metrics

    temp = perc_correct_dict[model]
    temp_mean = np.nanmean(temp, axis=0)
    temp_std = np.nanstd(temp, axis=0)
    perc_summary[model] = [f"{temp_mean[i]:.3f} ± {temp_std[i]:.3f}" for i in range(len(temp_mean))]

if models_found == 0:
    print('No results found. Skipping save step.')
else:
    save_dir = summary_save_dir
    print('Saving results to', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    n_samples = len(next(iter(metric_dict[metrics_to_calc[0]].values())))
    col_names_metric = [f'Sample {r}' for r in range(n_samples)]

    # Remove previouses xlsx files so that new files are always "clean".
    for path in (detailed_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    # with pd.ExcelWriter(os.path.join(save_dir, detailed_filename), engine='xlsxwriter') as writer:
    #     for m in metrics_to_calc:
    #         metric_df = pd.DataFrame.from_dict(metric_dict[m], orient='index', columns=col_names_metric)
    #         metric_df.to_excel(writer, sheet_name=m, index=True)
    #     for key in perc_correct_dict:
    #         perc_correct_df = pd.DataFrame(perc_correct_dict[key], index=col_names_metric, columns=category_names)
    #         perc_correct_df.to_excel(writer, sheet_name=f"PC {key}", index=True)

    with pd.ExcelWriter(os.path.join(save_dir, summary_filename), engine='xlsxwriter') as writer:
        metric_df = pd.DataFrame.from_dict(metrics_summary, orient='index', columns = metrics_to_calc)
        metric_df.to_excel(writer, sheet_name='Metrics', index=True)
        perc_correct_df = pd.DataFrame.from_dict(perc_summary, orient='index', columns=category_names)
        perc_correct_df.to_excel(writer, sheet_name='Percentage correct', index=True)

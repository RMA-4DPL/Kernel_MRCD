import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
from tqdm import tqdm
from helper_functions import normalize_data, calc_metric, load_mudcad_file, create_save_dir_name
import pickle
import pathlib
import argparse

base_filepath_configs = pathlib.Path(__file__).parent.resolve()
base_filepath = "/mnt/userdata/MaMe/SSDdata/DECAT/First test"
np.random.seed(4)

argument_parser = argparse.ArgumentParser(description='Summarize DL model results')
argument_parser.add_argument('--recalculate', action='store_true', help='Recalculate metrics even if already present in DL_Results_summary.xlsx')
argument_parser.set_defaults(recalculate=False)
argument_parser.add_argument('--subsample', type=str, default=None, help='Subsample factor for spatial dimensions (overrides experiment_settings Subsample; must match the value used at train time)')
argument_parser.add_argument('--vis_scale', type=str, default=None, help="Scale factor for the visual RGB bands (overrides experiment_settings Vis_scale; pass 'none' to disable; must match the value used at train time)")
argument_parser.add_argument('--ac_model', type=str, default='IARR', help='Atmospheric correction model name (overrides experiment_settings AC_model; must match the value used at train time)')
argument_parser.add_argument('--test_split_method', type=str, default='random', choices=['random', 'seasonal'], help='Method used to split the test set (must match the value used at train time)')
argument_parser.add_argument('--test_season', type=str, default='autumn', help='Season used for the test set when test_split_method is seasonal (must match the value used at train time)')
argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (overrides experiment_settings Scaler)')
argument_parser.add_argument('--scaling_scope', type=str, default='global', choices=['global', 'per_sample'], help='Scaling scope for the Scaler (overrides experiment_settings Scaler scaling_scope)')
args = argument_parser.parse_args()

# Load gsd and labels from yaml files
print('Loading yaml config')
with open(os.path.join(base_filepath, "MUCAD", "dataset", "gsd.yaml"), "r") as f:
    gsd = yaml.safe_load(f)
with open(os.path.join(base_filepath, "MUCAD", "dataset", "labels_rgb.yaml"), "r") as f:
    labels_rgb = yaml.safe_load(f)
with open(os.path.join(base_filepath_configs, "DL_model_configs.yaml"), "r") as f:
    dl_model_configs = yaml.safe_load(f)
with open(os.path.join(base_filepath_configs, "model_configs.yaml"), "r") as f:
    classical_model_configs = yaml.safe_load(f)
# autoAD models (fold cross-validated, held-out test split) and rx/crd models (LRX/CRD/RX,
# no training, scored over the whole dataset) are processed with different logic below;
# merge both configs here so a single run produces one combined summary.
model_configs = {**dl_model_configs, **classical_model_configs}
with open(os.path.join(base_filepath_configs, "result_processing_config.yaml"), "r") as f:
    experiment_settings = yaml.safe_load(f)
metrics_to_calc = experiment_settings['metrics']
# DL_train.py merges its argparse defaults into experiment_settings before computing save_dir,
# so these have to be mirrored here to reconstruct the same path (must match run_experiments_DL.mk).
# test_split_method/test_season always take the CLI value (default or overridden) since DL_train.py
# always merges its own argparse defaults for these in the same way.
experiment_settings['test_split_method'] = args.test_split_method
experiment_settings['test_season'] = args.test_season
experiment_settings.setdefault('loss_function', 'weighted_mse')
for arg in vars(args):
    experiment_settings[arg] = getattr(args, arg)
if args.subsample is not None:
    experiment_settings['Subsample'] = None if args.subsample.lower() in ('none', 'null') else int(args.subsample)
if args.scaler is not None:
    experiment_settings.setdefault('Scaler', {})['name'] = args.scaler
if args.scaling_scope is not None:
    experiment_settings.setdefault('Scaler', {})['scaling_scope'] = args.scaling_scope
if args.ac_model is not None:
    experiment_settings['AC_model'] = {'name': args.ac_model}
if args.vis_scale is not None:
    experiment_settings['Vis_scale'] = None if args.vis_scale.lower() in ('none', 'null') else float(args.vis_scale)
loading = 'all'
max_amount_to_show = 10

# Create a list of data files to read
print('Start reading files')
list_of_dirs = []
for root, dirs, files in os.walk(os.path.join(base_filepath, "MUCAD", "dataset"), topdown=False):
   if not dirs:
       list_of_dirs.append(root)

nr_of_captures = len(list_of_dirs)

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

metric_dict = {}
perc_correct_dict = {}
category_values = list(labels_rgb['categories'].values())
test_indices = None
for m in metrics_to_calc:
    metric_dict[m] = {}

# The summary/detailed files are written to the save_dir of whichever model is last in
# model_configs (see save_dir computation below the main loop); mirror that here so we can
# check for already-computed results before running the (expensive) metric computation.
last_model = list(model_configs.keys())[-1]
summary_save_dir = create_save_dir_name(base_filepath, last_model, experiment_settings)
summary_save_dir = summary_save_dir.split(last_model)[0]
# Random and seasonal test splits evaluate different samples and are not comparable, so keep
# their results in separate summary/detailed files rather than overwriting one another.
summary_filename = f"DL_Results_summary_{experiment_settings['test_split_method']}.xlsx"
detailed_filename = f"DL_Results_detailed_{experiment_settings['test_split_method']}.xlsx"
summary_path = os.path.join(summary_save_dir, summary_filename)
detailed_path = os.path.join(summary_save_dir, detailed_filename)

# Restrict to the models listed in result_processing_config.yaml, if any are given (applied
# after the save_dir above so the shared summary/detailed path doesn't move depending on which
# subset of models this particular run evaluates).
# models_to_evaluate = experiment_settings.get('models')
# if models_to_evaluate:
#     model_configs = {k: v for k, v in model_configs.items() if k in models_to_evaluate}

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
            for model in model_configs:
                sheet_name = f"PC {model}"
                if sheet_name in detailed_xls.sheet_names:
                    existing_detailed_perc[model] = pd.read_excel(detailed_xls, sheet_name=sheet_name, index_col=0)
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

def is_autoad_model(model_name):
    return 'auto' in model_name.lower()

# rx/crd models score the whole dataset (no train/test split of their own), so to compare
# them against the autoAD models we evaluate everyone on the same held-out test set. Fetch
# that test set once from the first autoAD model that already has results.
for model in model_configs:
    if not is_autoad_model(model):
        continue
    save_dir = create_save_dir_name(base_filepath, model, experiment_settings)
    pickle_path = os.path.join(save_dir, "Raw_results.pickle")
    if not os.path.exists(pickle_path):
        continue
    with open(pickle_path, "rb") as f:
        x = pickle.load(f)
    test_indices = x["Indices"]["Test"]
    del x
    break

models_skipped = 0
for model, config in model_configs.items():
    if has_cached_result(model):
        print(f"Using cached results for model {model} (already present in {summary_filename})")
        for m in metrics_to_calc:
            metric_dict[m][model] = existing_detailed_metric[m].loc[model].values
        perc_correct_dict[model] = existing_detailed_perc[model].values
        continue
    try:
        save_dir = create_save_dir_name(base_filepath, model, experiment_settings)
        if os.path.exists(os.path.join(save_dir,"Raw_results.pickle")):
            print('Loading results for model: ', model)
            print(f'Loading data from {save_dir}')
            with open(os.path.join(save_dir,"Raw_results.pickle"), "rb") as f:
                x = pickle.load(f)
            label_array = x["Labels"]
            if is_autoad_model(model):
                # Scores_test is already restricted to this model's own held-out test split
                # (shape k_folds, n_test, H, W).
                scores_dict = x["Scores_test"]
                test_indices = x["Indices"]["Test"]
                n_folds = len(x["Scores_test"])
            else:
                # rx/crd models score the entire dataset (no folds, no train/test split of their
                # own); restrict to the same test_indices used for the autoAD models so all
                # models are compared on identical samples, and treat it as a single "fold".
                scores_dict = x["Scores"][test_indices][None, ...]
                n_folds = 1
            label_array = label_array[test_indices]
            del x

            if is_autoad_model(model):
                # Plot the training/validation loss history saved by DL_train.py for each fold
                fold_histories = {}
                plt.figure(figsize=(15, 15))
                for fold in range(n_folds):
                    history_path = os.path.join(save_dir, f"Fold_{fold}", "history.csv")
                    if not os.path.exists(history_path):
                        continue
                    history_df = pd.read_csv(history_path)
                    fold_histories[fold] = history_df
                    plt.subplot(2, 2, fold+1)
                    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
                    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"{model} - Fold {fold} training history")
                    plt.legend()
                plt.savefig(os.path.join(save_dir, "history.png"))
                plt.close()

            n_test = len(test_indices)
            n_categories = len(labels_rgb['categories'])

            # Compute metrics per fold (from that fold's own scores) then average the metric values
            # across folds, rather than averaging the scores first and computing a single metric.
            metric_per_fold = {m: np.zeros((n_folds, n_test), dtype=np.float32) for m in metrics_to_calc}
            perc_correct_per_fold = np.full((n_folds, n_test, n_categories), np.nan, dtype=np.float32)

            temp_labels_reshaped = label_array.reshape((len(label_array),-1,label_array.shape[-1])).astype(np.int16)
            temp_labels = np.sum(temp_labels_reshaped,axis=-1)[..., None]
            temp_labels[temp_labels>0] = 1

            for fold in range(n_folds):
                fold_scores = scores_dict[fold].reshape((n_test,-1,1))
                for r, row in tqdm(enumerate(fold_scores), total=n_test, desc=f"Fold {fold}"):
                    nan_indices = np.where(np.isnan(row))[0]
                    if len(nan_indices) > 0:
                        for index in nan_indices:
                            row[index] = np.nanmean(row[index-5:index+5]) # Temporary fix for NaN values in scores, replace with mean of surrounding values
                    for m in metrics_to_calc:
                        metric_per_fold[m][fold, r] = calc_metric(temp_labels[r], fold_scores[r], metric=m)

                    temp_score = np.sign(row - np.percentile(row, 99.5))
                    temp_score[temp_score<0] = 0
                    masked_labels = temp_labels_reshaped[r] * temp_score
                    for u, unique in enumerate(category_values):
                        count_true = np.count_nonzero(temp_labels_reshaped[r] == unique)
                        count_pred = np.count_nonzero(masked_labels == unique)
                        perc_correct_per_fold[fold, r, u] = count_pred/count_true if count_true > 0 else np.nan

            for m in metrics_to_calc:
                metric_dict[m][model] = np.mean(metric_per_fold[m], axis=0)

            perc_correct_dict[model] = np.nanmean(perc_correct_per_fold, axis=0)
        else:
            print(f"Results for model {model} not found. Skipping.")
            models_skipped +=1
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

if models_skipped >= len(model_configs):
    print('No results found. Skipping save step.')
else:
    save_dir = summary_save_dir
    print('Saving results to', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if test_indices is not None:
        col_names_metric = [f'Sample {r}' for r in test_indices]
    else:
        # No model needed fresh computation this run; reuse the sample columns from the cached detailed file.
        col_names_metric = list(existing_detailed_metric[metrics_to_calc[0]].columns)
    # with pd.option_context('display.float_format', '{:,.3f}'.format):
    #     for m in metrics_to_calc:
    #         metric_df = pd.DataFrame.from_dict(metric_dict[m], orient='index', columns=col_names_metric)
    #         print(metric_df)
    #     for key in perc_correct_dict:
    #         perc_correct_df = pd.DataFrame(perc_correct_dict[key], columns=labels_rgb['categories'].keys())
    #         print(perc_correct_df)

    # Remove previouses xlsx files so that new files are always "clean".
    for path in (detailed_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    with pd.ExcelWriter(os.path.join(save_dir, detailed_filename), engine='xlsxwriter') as writer:
        for m in metrics_to_calc:
            metric_df = pd.DataFrame.from_dict(metric_dict[m], orient='index', columns=col_names_metric)
            metric_df.to_excel(writer, sheet_name=m, index=True)
        for key in perc_correct_dict:
            perc_correct_df = pd.DataFrame(perc_correct_dict[key], index=col_names_metric, columns=labels_rgb['categories'].keys())
            perc_correct_df.to_excel(writer, sheet_name=f"PC {key}", index=True)

    with pd.ExcelWriter(os.path.join(save_dir, summary_filename), engine='xlsxwriter') as writer:
        metric_df = pd.DataFrame.from_dict(metrics_summary, orient='index', columns = metrics_to_calc)
        metric_df.to_excel(writer, sheet_name='Metrics', index=True)
        perc_correct_df = pd.DataFrame.from_dict(perc_summary, orient='index', columns=labels_rgb['categories'].keys())
        perc_correct_df.to_excel(writer, sheet_name='Percentage correct', index=True)
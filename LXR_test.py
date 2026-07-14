import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy
from AD_models_GPU import create_AD_model
from Preproc import PreprocPipeline, subsample_data
from helper_functions import load_mudcad_file, create_save_dir_name
import pickle
from multiprocessing import cpu_count
import pathlib
import torch
import random
import time
import argparse 

if __name__ == "__main__":
    base_filepath_configs = pathlib.Path(__file__).parent.resolve()
    base_filepath = "/mnt/userdata/MaMe/SSDdata/DECAT/First test"

    random.seed(4)
    np.random.seed(4)
    seeds = []
    for i in range(3):
        seeds.append(random.randint(10, 10 ** 6))
    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed(seeds[1])
    torch.cuda.manual_seed_all(seeds[2])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    argument_parser = argparse.ArgumentParser(description='Process results of AD models')
    argument_parser.add_argument('--model', type=str, default='base_rx', required=False, help='Name of the model to process')   
    argument_parser.add_argument('--retrain', action='store_true', help='Retrain the model if specified')
    argument_parser.set_defaults(retrain=False)
    argument_parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use (default: 0)')
    argument_parser.add_argument('--subsample', type=str, default=None, help='Subsample factor for spatial dimensions (overrides experiment_settings Subsample)')
    argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (overrides experiment_settings Scaler)')
    argument_parser.add_argument('--scaling_scope', type=str, default='global', choices=['global', 'per_sample'], help='Scaling scope for the Scaler (overrides experiment_settings Scaler scaling_scope)')
    argument_parser.add_argument('--ac_model', type=str, default='IARR', help='Atmospheric correction model name (overrides experiment_settings AC_model)')
    argument_parser.add_argument('--vis_scale', type=str, default=None, help="Scale factor for the visual RGB bands (overrides experiment_settings Vis_scale; pass 'none' to disable)")
    args = argument_parser.parse_args()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load gsd and labels from yaml files
    print('Loading yaml config')
    with open(os.path.join(base_filepath, "MUCAD", "dataset", "gsd.yaml"), "r") as f:
        gsd = yaml.safe_load(f)
    with open(os.path.join(base_filepath, "MUCAD", "dataset", "labels_rgb.yaml"), "r") as f:
        labels_rgb = yaml.safe_load(f)
    with open(os.path.join(base_filepath_configs, "model_configs.yaml"), "r") as f:
        model_configs = yaml.safe_load(f)
    model_configs = {args.model: model_configs[args.model]}
    experiment_settings = {}
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
    max_amount_to_show = 5
    retrain = args.retrain

    model = list(model_configs.keys())[0]
    config = list(model_configs.values())[0]

    save_dir = create_save_dir_name(base_filepath, model, experiment_settings)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir,"Raw_results.pickle")) or retrain:
        # Create a list of data files to read
        print('Start reading files')
        list_of_dirs = []
        for root, dirs, files in os.walk(os.path.join(base_filepath, "MUCAD", "dataset"), topdown=False):
            if not dirs:
                list_of_dirs.append(root)

        # assert len(list_of_dirs) == 853
        nr_of_captures = len(list_of_dirs)

        if loading == "all":
            list_of_dirs_to_load = list_of_dirs
        elif loading == "random":
            indices = np.random.randint(nr_of_captures, size=max_amount_to_show)
            list_of_dirs_to_load = [list_of_dirs[ind] for ind in indices]
        elif type(loading) == list:
            list_of_dirs_to_load = list_of_dirs[loading]
        max_amount_of_threads = int(np.minimum(cpu_count(), len(list_of_dirs_to_load)))

        # Setup dataframes
        print('Setting up arrays')
        feature_order = ["vis", "blue", "green", "red", "eir", "nir", "lwir"]
        feature_length = [3, 1, 1, 1, 1, 1, 1]
        col_names = ["Season", "Date", "Location", "Area", "Flight_number", "Region", "Capture_number"]

        metadata_array = np.zeros((len(list_of_dirs_to_load), len(col_names)), dtype=object)
        data_array = np.zeros((len(list_of_dirs_to_load), 512, 512, len(feature_order)+2), np.float32)
        label_array = np.zeros((len(list_of_dirs_to_load), 512, 512, 3), dtype=np.uint8)
        label_count_array = np.zeros((len(list_of_dirs_to_load), len(labels_rgb['categories'])), dtype=np.uint32)

        print('Loading data')
        def _load_one(args):
            d, directory = args
            metadata_values, data, labels = load_mudcad_file(directory)
            label_counts = pd.read_csv(os.path.join(directory, "label_counts.csv")).values[0]
            return d, metadata_values, data, labels, label_counts

        with ThreadPoolExecutor(max_workers=max_amount_of_threads) as executor:
            results = executor.map(_load_one, enumerate(list_of_dirs_to_load))
            for d, metadata_values, data, labels, label_counts in tqdm(results, total=len(list_of_dirs_to_load)):
                metadata_array[d] = metadata_values
                data_array[d] = data
                label_array[d] = labels
                label_count_array[d] = label_counts

        # print('Creating metadata dataframe')

        # label_count = pd.DataFrame(label_count_array, columns=list(labels_rgb['categories'].keys()))
        # metadata = pd.DataFrame(metadata_array, columns=col_names)

        # print(metadata)
        # print(label_count)

        print('Preprocessing data')
        label_array = subsample_data(label_array, experiment_settings.get('Subsample'))
        data_array = subsample_data(data_array, experiment_settings.get('Subsample'))
        preproc_pipeline = PreprocPipeline(experiment_settings)
        preproc_pipeline.fit(data_array)
        data_array = preproc_pipeline.transform(data_array)
        L, H, W, B = data_array.shape

        print("Performing calculations")
        logging_dict = {}
        print(f"Running {model} model")
        AD_model = create_AD_model(config['model_name'])
        if hasattr(AD_model, 'load_config'):
            print(f"Loading {model} config")
            AD_model.load_config(config)

        scores = np.zeros((L, H, W), dtype=np.float32)
        times = np.zeros((L,), dtype=np.float32)
        run_gpu=False
        if hasattr(AD_model, 'run_gpu'):
            run_gpu = True
            AD_model.set_device(device)
        for r, row in tqdm(enumerate(data_array), total=len(data_array)):
            start_time = time.time()
            if run_gpu: 
                try: # Try to run the model on GPU, if it fails, fall back to CPU execution
                    scores[r] = AD_model.run_gpu(row)
                except RuntimeError as e:
                    print(f"Error running model {model} on GPU for row {r}: {e}")
                    print("Falling back to CPU execution.")
                    scores[r] = AD_model(row)
                    run_gpu = False
            else:
                scores[r] = AD_model(data_array[r])
            times[r] = time.time() - start_time
        logging_dict["Runtime"] = times
        logging_dict["GPU execution"] = run_gpu

        print(f'Saving to {save_dir}')
        with open(os.path.join(save_dir,"Raw_results.pickle"), "wb") as f:
            save_dict = {"Metadata": metadata_array,
                        "Labels": label_array,
                        "Label_count": label_count_array,
                        "Scores": scores,
                        "Model_configs": model_configs,
                        "Experiment_configs": experiment_settings,
                        "Logging": logging_dict}
            pickle.dump(save_dict, f)
        with open(os.path.join(save_dir, "experiment_configs.yaml"), "w") as f:
            yaml.safe_dump(experiment_settings, f)
    else:
        print(f"{model} has already been processed, skipping.")
    pass
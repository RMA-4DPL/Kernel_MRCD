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
from Preproc import PreprocPipeline
from helper_functions import load_dataset, create_save_dir_name
from Background_selection import select_background_model
from subsampler import get_subsampler
from kernels import get_kernel
import pickle
from multiprocessing import cpu_count
import pathlib
import torch
import random
import time
import argparse 

if __name__ == "__main__":

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
    argument_parser.add_argument('--dataset', type=str, default='Salinas_A', help='Select which dataset to load (default:Salinas).')
    argument_parser.add_argument('--model', type=str, default='base_amf', required=False, help='Name of the model to process')   
    argument_parser.add_argument('--retrain', action='store_true', help='Retrain the model if specified')
    argument_parser.set_defaults(retrain=False)
    argument_parser.add_argument('--scaler', type=str, default='Standard', help='Scaler name (overrides experiment_settings Scaler)')
    argument_parser.add_argument('--scaling_scope', type=str, default='per_sample', choices=['global', 'per_sample'], help='Scaling scope for the Scaler (overrides experiment_settings Scaler scaling_scope)')
    argument_parser.add_argument('--background_model', type=str, default='MCD', help='Model to select background sample for statistics (default: Sample).')
    argument_parser.add_argument('--background_config', type=str, default='kmrcd_0.75_rbf', help='Name of the entry in background_configs.yaml to load parameters from (overrides --background_model with its model_name).')
    argument_parser.add_argument('--gpu', type=int, default=3, help='GPU device number to use (default: 0)')
    argument_parser.add_argument('--subsample', type=str, default='random', help='method to use for data subsampling.')
    argument_parser.add_argument('--subsample_amount', type=int, default=100, help='amount of data point to sample')
    args = argument_parser.parse_args()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    local_filepath = pathlib.Path(__file__).parent.resolve()
    base_filepath = "/mnt/userdata/MaMe/SSDdata/Kernel_MRCD"
    base_filepath_results = os.path.join(base_filepath, 'Results')
    dataset_filepath = os.path.join(base_filepath, 'Dataset', args.dataset)
    figure_filepath = os.path.join(base_filepath, 'Figures')

    with open(os.path.join(local_filepath, "model_configs.yaml"), "r") as f:
        model_configs = yaml.safe_load(f)
    model_configs = {args.model: model_configs[args.model]}
    with open(os.path.join(local_filepath, "background_configs.yaml"), "r") as f:
        background_configs = yaml.safe_load(f)
    experiment_settings = {}
    for arg in vars(args):
        experiment_settings[arg] = getattr(args, arg)
    if args.scaler is not None:
        experiment_settings.setdefault('Scaler', {})['name'] = args.scaler
        if args.scaling_scope is not None:
            experiment_settings.setdefault('Scaler', {})['scaling_scope'] = args.scaling_scope
    if args.subsample is not None:
        experiment_settings.setdefault('Subsample', {})['name'] = args.subsample
        experiment_settings.setdefault('Subsample', {})['amount'] = args.subsample_amount
        
    retrain = args.retrain

    model = list(model_configs.keys())[0]
    config = list(model_configs.values())[0]


    background_config = None
    background_model_name = experiment_settings['background_model']
    if args.background_config is not None:
        background_config = background_configs[args.background_config]
        background_model_name = background_config['model_name']
        experiment_settings['background_model'] = args.background_config

    
    save_dir = create_save_dir_name(base_filepath_results, model, experiment_settings)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir,"Raw_results.pickle")) or retrain:


        print('Loading data')
        data_array_raw, data_array, label_array, labels_ids = load_dataset(base_path=base_filepath, dataset_name=args.dataset)

        print('Preprocessing data')
        preproc_pipeline = PreprocPipeline(experiment_settings)
        preproc_pipeline.fit(data_array)
        data_array = preproc_pipeline.transform(data_array)
        L, H, W, B = data_array.shape

        print("Performing calculations")
        logging_dict = {}
        
        background_model = select_background_model(background_model_name)
        if background_config is not None and hasattr(background_model, 'load_config'):
            print(f"Loading {args.background_config} background config")
            background_model.load_config(background_config)

        print(f"Running {model} model")
        AD_model = create_AD_model(config['model_name'])
        if 'kernel' in background_config:
            AD_model.kernel=get_kernel(background_config['kernel'])
        if hasattr(AD_model, 'load_config'):
            print(f"Loading {model} config")
            AD_model.load_config(config)

        scores = np.zeros((L, H, W), dtype=np.float32)
        scores_binary = np.zeros((L, H, W), dtype=np.float32)
        times = np.zeros((L,), dtype=np.float32)
        run_gpu=False
        if hasattr(AD_model, 'run_gpu'):
            AD_model.gpu=True
            AD_model.set_device(device)
        for r, row in tqdm(enumerate(data_array), total=len(data_array)):
            start_time = time.time()
            print('Getting background statistics.')
            if 'kernel' in background_config:
                bg_data = row
                if experiment_settings['Subsample']['name'] != 'none':
                    sampler = get_subsampler(experiment_settings['Subsample']['name'])
                    bg_data = sampler(bg_data, experiment_settings['Subsample']['amount'])
                indices = background_model(bg_data)
                bg = bg_data.reshape((-1, bg_data.shape[-1]))[indices]
            else:
                bg_data = row
                if experiment_settings['Subsample']['name'] != 'none':
                    sampler = get_subsampler(experiment_settings['Subsample']['name'])
                    bg_data = sampler(bg_data, experiment_settings['Subsample']['amount'])
                mean_N, cov = background_model(bg_data)
                bg = bg_data.reshape((-1, bg_data.shape[-1]))
                AD_model.set_mean_N(mean_N)
                AD_model.set_cov(cov)
            print('Calculating anomaly scores.')
            row_computed = False
            while not row_computed:
                try:
                    if model=='base_rx': # Try to run the model on GPU, if it fails, fall back to CPU execution
                        scores[r] = AD_model(row, bg)
                        scores_binary[r] = scores[r]
                    else:
                        temp_scores = np.zeros((len(labels_ids)-1, H, W))
                        for i, id in enumerate(labels_ids):
                            if id != 0:
                                target = np.mean(row[np.where(label_array[r]==id)], axis=0)
                                temp_scores[i-1] = AD_model(row, bg, target)
                        scores[r] = np.max(temp_scores, axis=0)
                        target = np.mean(row[np.where(label_array[r]!=0)], axis=0)
                        scores_binary[r] = AD_model(row, bg, target)
                    row_computed = True
                except RuntimeError as e:
                    print(f"Error running model {model} on GPU for row {r}: {e}")
                    print("Falling back to CPU execution.")
                    AD_model.gpu=False
            # Reset stats for new data
            AD_model.set_mean_N(None)
            AD_model.set_cov(None)
                
            times[r] = time.time() - start_time
        logging_dict["Runtime"] = times
        logging_dict["GPU execution"] = run_gpu

        print(f'Saving to {save_dir}')
        with open(os.path.join(save_dir,"Raw_results.pickle"), "wb") as f:
            save_dict = {"Labels": label_array,
                        "Label_ids": labels_ids,
                        "Scores": scores,
                        "Scores_binary": scores_binary,
                        "Model_configs": model_configs,
                        "Background_configs": {args.background_config: background_config} if background_config is not None else None,
                        "Experiment_configs": experiment_settings,
                        "Logging": logging_dict}
            pickle.dump(save_dict, f)
        with open(os.path.join(save_dir, "experiment_configs.yaml"), "w") as f:
            yaml.safe_dump(experiment_settings, f)
    else:
        print(f"{model} has already been processed, skipping.")
    pass
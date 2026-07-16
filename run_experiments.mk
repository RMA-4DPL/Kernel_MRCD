# Makefile for training different models
# Variables
PYTHON=python
RETRAIN=#--retrain
GPU=--gpu=3

# All models defined in model_configs.yaml
MODELS = base_rx \
         base_amf \
		 base_ace

# Sweep axes:
#   subsample     -> none (yaml default, no subsampling) or 2
#   vis_scale     -> none (disabled) or 3 (yaml default scale factor)
#   ac_model      -> atmospheric correction model (only IARR is implemented in Preproc.py)
#   scaling_scope -> global or per_sample
# (LXR_test.py has no train/test split, so no test_split_method axis here)
SCALERS = none Standard
SCALING_SCOPES = per_sample 
BACKGROUND_CONFIGS = sample ledoit_wolf shrinkage_0.1 diagonal_0.1 mcd_0.75 mrcd_auto_0.75_identity mrcd_auto_0.75_equicorrelation kmrcd_0.75_rbf
DATASETS = Salinas_A HYDICE #Salinas

# Default target
all: train

# Train target
# For each sweep setting (subsample x vis_scale x ac_model x scaling_scope), runs every
# model in MODELS.
train:
	@for background_config in $(BACKGROUND_CONFIGS); do\
		for scaler in $(SCALERS); do \
			for scope in $(SCALING_SCOPES); do \
				for dataset in $(DATASETS); do \
					for model in $(MODELS); do\
						echo "=== model=$$model dataset=$$dataset scaler=$$scaler scaling_scope=$$scope --background_config=$$background_config ==="; \
						$(PYTHON) main.py --model $$model $(GPU) $(RETRAIN) \
							--scaling_scope=$$scope --scaler=$$scaler --dataset=$$dataset --background_config=$$background_config; \
					done; \
				done; \
			done; \
		done; \
	done

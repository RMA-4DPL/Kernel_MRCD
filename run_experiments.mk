# Makefile for training different models
# Variables
PYTHON=python
RETRAIN=--retrain
GPU=--gpu=3

# All models defined in model_configs.yaml
MODELS = base_rx \
         base_amf \
		 base_ace

# Sweep axes:
#   subsample     -> none (no subsampling) or random (see subsampler.py), paired with subsample_amount
#   vis_scale     -> none (disabled) or 3 (yaml default scale factor)
#   ac_model      -> atmospheric correction model (only IARR is implemented in Preproc.py)
#   scaling_scope -> global or per_sample
# (LXR_test.py has no train/test split, so no test_split_method axis here)
SCALERS = Standard #none
SCALING_SCOPES = per_sample
BACKGROUND_CONFIGS = sample ledoit_wolf shrinkage_0.1 diagonal_0.1 mcd_0.5 mcd_0.75 mrcd_auto_0.5_identity mrcd_auto_0.75_identity mrcd_auto_0.75_equicorrelation kmrcd_0.75_rbf
DATASETS = HYDICE Salinas_A \
           ABU_beach_3 ABU_airport_4 ABU_urban_3 ABU_beach_2 ABU_urban_1 \
           ABU_airport_1 ABU_airport_2 ABU_airport_3 ABU_urban_4 ABU_urban_5 ABU_urban_2 \
           ABU_beach_4 ABU_beach_1 \
           Indiana PaviaU Salinas cooke_city SanDiego WHU-HI Pavia
SUBSAMPLES = none
SUBSAMPLE_AMOUNTS = 100 #400 1000

# Default target
all: train

# Train target
# For each sweep setting (subsample x subsample_amount x vis_scale x ac_model x scaling_scope), runs every
# model in MODELS.
train:
	@for dataset in $(DATASETS); do \
		for background_config in $(BACKGROUND_CONFIGS); do\
			for scaler in $(SCALERS); do \
				for subsample in $(SUBSAMPLES); do \
					for amount in $(SUBSAMPLE_AMOUNTS); do \
						for scope in $(SCALING_SCOPES); do \
							for model in $(MODELS); do\
								echo "=== model=$$model dataset=$$dataset scaler=$$scaler scaling_scope=$$scope subsample=$$subsample subsample_amount=$$amount --background_config=$$background_config ==="; \
								$(PYTHON) main.py --model $$model $(GPU) $(RETRAIN) \
									--scaling_scope=$$scope --scaler=$$scaler --dataset=$$dataset --background_config=$$background_config \
									--subsample=$$subsample --subsample_amount=$$amount; \
							done; \
						done; \
					done; \
				done; \
			done; \
		done; \
	done

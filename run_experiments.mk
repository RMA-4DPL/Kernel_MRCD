# Makefile for training different models
# Variables
PYTHON=python
RETRAIN=--retrain
GPU=--gpu=3

# All models defined in model_configs.yaml
MODELS = base_rx \
         lrx_5_15 \
         lrx_11_31 \
         lrx_21_61 \
         lrx_31_91 \
         lrx_41_121 \
         crd_5_15 \
         crd_11_21 \
         crd_21_31 \
         crd_31_41 \
         crd_41_51

# Sweep axes:
#   subsample     -> none (yaml default, no subsampling) or 2
#   vis_scale     -> none (disabled) or 3 (yaml default scale factor)
#   ac_model      -> atmospheric correction model (only IARR is implemented in Preproc.py)
#   scaling_scope -> global or per_sample
# (LXR_test.py has no train/test split, so no test_split_method axis here)
SUBSAMPLES = 2 none
VIS_SCALES = 3 none
AC_MODELS = IARR none
SCALING_SCOPES = global 

# Default target
all: train

# Train target
# For each sweep setting (subsample x vis_scale x ac_model x scaling_scope), runs every
# model in MODELS.
train:
	@for subsample in $(SUBSAMPLES); do \
		for vis in $(VIS_SCALES); do \
			for ac in $(AC_MODELS); do \
				for scope in $(SCALING_SCOPES); do \
					for model in $(MODELS); do \
						echo "=== model=$$model subsample=$$subsample vis=$$vis ac_model=$$ac scaling_scope=$$scope ==="; \
						$(PYTHON) LXR_test.py --model $$model $(GPU) $(RETRAIN) \
							--scaling_scope=$$scope --vis_scale=$$vis --ac_model=$$ac --subsample=$$subsample; \
					done; \
				done; \
			done; \
		done; \
	done

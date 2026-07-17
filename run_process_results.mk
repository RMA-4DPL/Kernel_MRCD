# Makefile for summarizing results across all datasets/settings
# Variables
PYTHON=python
RECALCULATE=#--recalculate

# Sweep axes must match the ones used in run_experiments.mk (Scaler name + scaling_scope +
# subsample/subsample_amount determine the results directory via create_save_dir_name;
# background_config/model are not needed here since Process_results.py discovers every
# "{model}_{background_model}" directory already present under each
# dataset/scaler/scope/subsample combination).
SCALERS = Standard
SCALING_SCOPES = per_sample
DATASETS = Salinas_A HYDICE Salinas
SUBSAMPLES = random #none
SUBSAMPLE_AMOUNTS = 100 400 1000

# Default target
all: process

# Process target
# For each sweep setting (dataset x scaler x scaling_scope x subsample x subsample_amount),
# summarizes every model result directory found on disk into Results_summary.xlsx / Results_detailed.xlsx.
process:
	@for scaler in $(SCALERS); do \
		for subsample in $(SUBSAMPLES); do \
			for amount in $(SUBSAMPLE_AMOUNTS); do \
				for dataset in $(DATASETS); do \
					for scope in $(SCALING_SCOPES); do \
						echo "=== dataset=$$dataset scaler=$$scaler scaling_scope=$$scope subsample=$$subsample subsample_amount=$$amount ==="; \
						$(PYTHON) Process_results.py --dataset=$$dataset --scaler=$$scaler --scaling_scope=$$scope --subsample=$$subsample --subsample_amount=$$amount $(RECALCULATE); \
						$(PYTHON) Visualize_classical_results.py --dataset=$$dataset --scaler=$$scaler --scaling_scope=$$scope --subsample=$$subsample --subsample_amount=$$amount; \
					done; \
				done; \
				$(PYTHON) Combine_multi_dataset_results.py --scaler=$$scaler --subsample=$$subsample --subsample_amount=$$amount; \
			done; \
		done; \
	done
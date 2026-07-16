# Makefile for summarizing results across all datasets/settings
# Variables
PYTHON=python
RECALCULATE=#--recalculate

# Sweep axes must match the ones used in run_experiments.mk (Scaler name + scaling_scope
# determine the results directory via create_save_dir_name; background_config/model are not
# needed here since Process_results.py discovers every "{model}_{background_model}" directory
# already present under each dataset/scaler/scope combination).
SCALERS = none Standard
SCALING_SCOPES = per_sample
DATASETS = Salinas_A HYDICE Salinas

# Default target
all: process

# Process target
# For each sweep setting (dataset x scaler x scaling_scope), summarizes every model result
# directory found on disk into Results_summary.xlsx / Results_detailed.xlsx.
process:
	@for dataset in $(DATASETS); do \
		for scaler in $(SCALERS); do \
			for scope in $(SCALING_SCOPES); do \
				echo "=== dataset=$$dataset scaler=$$scaler scaling_scope=$$scope ==="; \
				$(PYTHON) Process_results.py --dataset=$$dataset --scaler=$$scaler --scaling_scope=$$scope $(RECALCULATE); \
				$(PYTHON) Visualize_classical_results.py --dataset=$$dataste --scaler=$$scaler --scaling_scope==$$scope
			done; \
		done; \
	done

	$(PYTHON) Combine_multi_dataset_results.py --scaler=none
	$(PYTHON) Combine_multi_dataset_results.py --scaler=Standard
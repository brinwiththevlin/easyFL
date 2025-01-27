#!/bin/bash

set -e

# Function to run simulations with consistent parameters
run_simulations() {
    local bad_nodes=$1
    local dataset=$2
    local label_tampering=$3
    local weight_tampering=$4
    local NEW_PATH="simulation_bn${bad_nodes}_ds${dataset}_lt${label_tampering}_wt${weight_tampering}"

    # Simulation configurations
    local client_configs=(
        "25 10"
        "50 10"
        "100 10"
    )

    # Selection methods
    local selection_methods=("kl-kmeans" "random")

    # Iterate through client configurations and selection methods
    for config in "${client_configs[@]}"; do
        read -r clients per_round <<< "$config"

        for selection in "${selection_methods[@]}"; do
            python3 src/config.py \
                --clients "$clients" \
                --per_round "$per_round" \
                --selection "$selection" \
                --under_rep 3 \
                --res_path "$NEW_PATH" \
                --dataset "$dataset" \
                --label_tampering "$label_tampering" \
                --weight_tampering "$weight_tampering"

            # IID simulation
            python3 src/simulation.py \
                --iterations 1000 \
                --iid \
                --clients "$clients" \
                --per_round "$per_round" \
                --selection "$selection" \
                --under_rep 3 \
                --res_path "$NEW_PATH" \
                --bad_nodes "$bad_nodes" \
                --dataset "$dataset" \
                --label_tampering "$label_tampering" \
                --weight_tampering "$weight_tampering"

            # Non-IID simulation
            python3 src/simulation.py \
                --iterations 1000 \
                --clients "$clients" \
                --per_round "$per_round" \
                --selection "$selection" \
                --under_rep 3 \
                --res_path "$NEW_PATH" \
                --bad_nodes "$bad_nodes" \
                --dataset "$dataset" \
                --label_tampering "$label_tampering" \
                --weight_tampering "$weight_tampering"
        done
    done

    # Aggregate results
    python3 src/aggregate_figure.py \
        --results_dir_name "src/results/$NEW_PATH" \
        --bad_nodes "$bad_nodes" \
        --dataset "$dataset" \
        --label_tampering "$label_tampering" \
        --weight_tampering "$weight_tampering"
}


# Main script execution
# Parameter arrays
BAD_NODES=(1)
DATASETS=("MNIST" "cifar10")
LABEL_TAMPERING=("none" "zero" "reverse" "random")
WEIGHT_TAMPERING=("none" "large_neg" "reverse" "random")

# Nested loops to run all parameter combinations
for bad_nodes in "${BAD_NODES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for label_tampering in "${LABEL_TAMPERING[@]}"; do
            for weight_tampering in "${WEIGHT_TAMPERING[@]}"; do
                run_simulations "$bad_nodes" "$dataset" "$label_tampering" "$weight_tampering"
            done
        done
    done
done
#!/bin/bash

set -e

# Function to check if a simulation is complete
#!/bin/bash

check_simulation_complete() {
    local base_path=$1    # e.g. simulation_bn1_dsMNIST_ltzero_wtnone
    local dataset=$2      # e.g. MNIST
    local clients=$3      # e.g. 25
    local per_round=$4    # e.g. 10
    local selection=$5    # e.g. kl-kmeans
    local is_iid=$6      # true/false
    
    # Debug output
    echo "Checking path: src/results/$base_path"
    
    # First check if base directory exists
    if [ ! -d "src/results/$base_path" ]; then
        echo "Base directory not found"
        return 1
    fi
    
    # Build the pattern without timestamp
    local dir_pattern
    if [ "$is_iid" = true ]; then
        dir_pattern="${dataset}_iid_${clients}_${per_round}_${selection}"
    else
        dir_pattern="${dataset}_noniid_3_${clients}_${per_round}_${selection}"
    fi
    
    echo "Looking for pattern: $dir_pattern"
    
    # Find matching directories
    found_dirs=$(find "src/results/$base_path" -type d -name "${dir_pattern}*")
    
    if [ -z "$found_dirs" ]; then
        echo "No matching directories found"
        return 1
    fi
    
    # For each matching directory
    while IFS= read -r dir; do
        echo "Checking directory: $dir"
        # Check for both results.csv and at least one PNG file
        if [ -f "$dir/results.csv" ] && ls "$dir"/*.png >/dev/null 2>&1; then
            echo "Found complete results in: $dir"
            return 0
        fi
    done <<< "$found_dirs"
    
    echo "No complete results found"
    return 1
}

# Test the function
# check_simulation_complete "simulation_bn1_dsMNIST_ltzero_wtnone" "MNIST" "25" "10" "kl-kmeans" true

# Modified run_simulations function that checks completion
run_simulations() {
    local bad_nodes=$1
    local dataset=$2
    local label_tampering=$3
    local weight_tampering=$4
    local NEW_PATH="simulation_bn${bad_nodes}_ds${dataset}_lt${label_tampering}_wt${weight_tampering}"

    echo "Checking simulation group: $NEW_PATH"

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
            # Check IID simulation
            if check_simulation_complete "$NEW_PATH" "$dataset" "$clients" "$per_round" "$selection" true; then
                echo "Skipping completed IID simulation: $clients clients, $selection selection"
            else
                echo "Running IID simulation: $clients clients, $selection selection"
                
                python3 src/config.py \
                    --clients "$clients" \
                    --per_round "$per_round" \
                    --selection "$selection" \
                    --under_rep 3 \
                    --iid\
                    --res_path "$NEW_PATH" \
                    --dataset "$dataset" \
                    --label_tampering "$label_tampering" \
                    --weight_tampering "$weight_tampering"

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
            fi

            # Check non-IID simulation
            if check_simulation_complete "$NEW_PATH" "$dataset" "$clients" "$per_round" "$selection" false; then
                echo "Skipping completed non-IID simulation: $clients clients, $selection selection"
            else
                echo "Running non-IID simulation: $clients clients, $selection selection"
                
                python3 src/config.py \
                    --clients "$clients" \
                    --per_round "$per_round" \
                    --selection "$selection" \
                    --under_rep 3 \
                    --res_path "$NEW_PATH" \
                    --dataset "$dataset" \
                    --label_tampering "$label_tampering" \
                    --weight_tampering "$weight_tampering"

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
            fi
        done
    done

    # Check if all simulations for this configuration are complete before aggregating
    local all_complete=true
    for config in "${client_configs[@]}"; do
        read -r clients per_round <<< "$config"
        for selection in "${selection_methods[@]}"; do
            if ! check_simulation_complete "$NEW_PATH" "$dataset" "$clients" "$per_round" "$selection" true || \
               ! check_simulation_complete "$NEW_PATH" "$dataset" "$clients" "$per_round" "$selection" false; then
                all_complete=false
                break 2
            fi
        done
    done

    if [ "$all_complete" = true ]; then
        echo "Aggregating results for $NEW_PATH"
        python3 src/aggregate_figure.py \
            --results_dir_name "src/results/$NEW_PATH" \
            --bad_nodes "$bad_nodes" \
            --dataset "$dataset" \
            --label_tampering "$label_tampering" \
            --weight_tampering "$weight_tampering"
    else
        echo "Skipping aggregation for incomplete simulation group: $NEW_PATH"
    fi
}

# Main script execution
# Parameter arrays
BAD_NODES=(1)
DATASETS=("MNIST" "cifar10")
LABEL_TAMPERING=("zero" "reverse" "random")
WEIGHT_TAMPERING=("none")

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
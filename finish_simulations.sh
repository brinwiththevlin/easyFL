#!/bin/bash

set -e

# Function to check if a simulation is complete
check_simulation_complete() {
    local base_path=$1
    local dataset=$2
    local clients=$3
    local per_round=$4
    local selection=$5
    local is_iid=$6

    echo "Checking path: src/results/$base_path"

    if [ ! -d "src/results/$base_path" ]; then
        echo "Base directory not found"
        return 1
    fi

    local dir_pattern
    if [ "$is_iid" = true ]; then
        dir_pattern="${dataset}_iid_${clients}_${per_round}_${selection}"
    else
        dir_pattern="${dataset}_noniid_3_${clients}_${per_round}_${selection}"
    fi

    echo "Looking for pattern: $dir_pattern"

    found_dirs=$(find "src/results/$base_path" -type d -name "${dir_pattern}*")

    if [ -z "$found_dirs" ]; then
        echo "No matching directories found"
        return 1
    fi

    while IFS= read -r dir; do
        echo "Checking directory: $dir"
        if [ -f "$dir/results.csv" ] && ls "$dir"/*.png >/dev/null 2>&1; then
            echo "Found complete results in: $dir"
            return 0
        fi
    done <<< "$found_dirs"

    echo "No complete results found"
    return 1
}

# Function to run a single simulation group
run_simulation_group() {
    local dataset=$1
    local label_tampering=$2
    local weight_tampering=$3

    local client_configs=("25 10" "50 10" "100 10")
    # local client_configs=("100 10")
    local selection_methods=("kl-kmeans" "random")

    for config in "${client_configs[@]}"; do
        read -r clients per_round <<< "$config"
        
        # Compute bad_nodes as 10% of clients (rounded)
        local bad_nodes=$(( (clients + 9) / 10 ))  # Ensures rounding up

        local base_path="simulation_bn${bad_nodes}_ds${dataset}_lt${label_tampering}_wt${weight_tampering}"

        echo "Checking simulation group: $base_path with ${bad_nodes} bad nodes"

        for selection in "${selection_methods[@]}"; do
            local iid_complete=false
            local noniid_complete=false

            if check_simulation_complete "$base_path" "$dataset" "$clients" "$per_round" "$selection" true; then
                echo "Skipping completed IID simulation: $clients clients, $selection selection"
                iid_complete=true
            else
                echo "Running IID simulation: $clients clients, $selection selection"
                python3 src/config.py \
                    --clients "$clients" --per_round "$per_round" --selection "$selection" \
                    --under_rep 3 --iid --res_path "$base_path" --dataset "$dataset" \
                    --label_tampering "$label_tampering" --weight_tampering "$weight_tampering" &

                PID=$!
                wait $PID

                
                python3 src/simulation.py \
                    --iterations 1000 --iid --clients "$clients" --per_round "$per_round" \
                    --selection "$selection" --under_rep 3 --res_path "$base_path" \
                    --bad_nodes "$bad_nodes" --dataset "$dataset" \
                    --label_tampering "$label_tampering" --weight_tampering "$weight_tampering"
            fi

            if check_simulation_complete "$base_path" "$dataset" "$clients" "$per_round" "$selection" false; then
                echo "Skipping completed non-IID simulation: $clients clients, $selection selection"
                noniid_complete=true
            else
                echo "Running non-IID simulation: $clients clients, $selection selection"
                python3 src/config.py \
                    --clients "$clients" --per_round "$per_round" --selection "$selection" \
                    --under_rep 3 --res_path "$base_path" --dataset "$dataset" \
                    --label_tampering "$label_tampering" --weight_tampering "$weight_tampering" &

                PID=$!
                wait $PID
                python3 src/simulation.py \
                    --iterations 1000 --clients "$clients" --per_round "$per_round" \
                    --selection "$selection" --under_rep 3 --res_path "$base_path" \
                    --bad_nodes "$bad_nodes" --dataset "$dataset" \
                    --label_tampering "$label_tampering" --weight_tampering "$weight_tampering"
            fi
        done
    done

    # Check if all simulations for this group are complete before aggregating
    local all_complete=true
    for config in "${client_configs[@]}"; do
        read -r clients per_round <<< "$config"
        local bad_nodes=$(( (clients + 9) / 10 ))  # Recompute bad_nodes

        for selection in "${selection_methods[@]}"; do
            if ! check_simulation_complete "$base_path" "$dataset" "$clients" "$per_round" "$selection" true || \
               ! check_simulation_complete "$base_path" "$dataset" "$clients" "$per_round" "$selection" false; then
                all_complete=false
                break 2
            fi
        done
    done

    if [ "$all_complete" = true ]; then
        echo "Aggregating results for $base_path"
        python3 src/aggregate_figure.py \
            --results_dir_name "src/results/$base_path" \
            --bad_nodes "$bad_nodes" \
            --dataset "$dataset" \
            --label_tampering "$label_tampering" \
            --weight_tampering "$weight_tampering"
    else
        echo "Skipping aggregation for incomplete simulation group: $base_path"
    fi
}

# Main script execution
DATASETS=("MNIST")
LABEL_TAMPERING=("zero" "reverse" "random" "none")
WEIGHT_TAMPERING=("none" "large_neg" "reverse" "random")

for dataset in "${DATASETS[@]}"; do
    for label_tampering in "${LABEL_TAMPERING[@]}"; do
        run_simulation_group "$dataset" "$label_tampering" "none"
    done
    for weight_tampering in "${WEIGHT_TAMPERING[@]}"; do
        run_simulation_group "$dataset" none "$weight_tampering"
    done
done

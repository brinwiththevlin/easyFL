#!/bin/bash

# WARNING: only run this script if on a GPU server or if you have sufficient GPU power. may take a long time
set -e

# Find all src/results/simulation* paths and set a new path for the next simulation
COUNT=$(find src/results/test*  -maxdepth 0 -type d | wc -w)
NEW_COUNT=$((COUNT + 1))
NEW_PATH="test${NEW_COUNT}"

# Run simulations with interleaved --iid and non--iid configurations
# 25 clients, 10 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 25 --per_round 10 --selection kmeans --under_rep 3 --res_path $NEW_PATH  --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 25 --per_round 10 --selection kmeans --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 25 --per_round 10 --selection random --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 25 --per_round 10 --selection random --under_rep 3 --res_path $NEW_PATH --bad_nodes 1

# 50 clients, 25 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 50 --per_round 10 --selection kmeans --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 50 --per_round 10 --selection kmeans --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 50 --per_round 10 --selection random --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 50 --per_round 10 --selection random --under_rep 3 --res_path $NEW_PATH --bad_nodes 1

# 100 clients, 50 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 100 --per_round 10 --selection kmeans --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 100 --per_round 10 --selection kmeans --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 100 --per_round 10 --selection random --under_rep 3 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 100 --per_round 10 --selection random --under_rep 3 --res_path $NEW_PATH --bad_nodes 1

# Aggregate results
python3 src/aggregate_figure.py --results_dir_name src/results/$NEW_PATH

COUNT=$(find src/results/simulation*  -maxdepth 0 -type d | wc -w)
NEW_COUNT=$((COUNT + 1))
NEW_PATH="simulation${NEW_COUNT}"

# Run simulations with interleaved --iid and non--iid configurations
# 25 clients, 10 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 25 --per_round 10 --selection kmeans --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 25 --per_round 10 --selection kmeans --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 25 --per_round 10 --selection random --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 25 --per_round 10 --selection random --under_rep 5 --res_path $NEW_PATH --bad_nodes 1

# 50 clients, 25 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 50 --per_round 10 --selection kmeans --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 50 --per_round 10 --selection kmeans --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 50 --per_round 10 --selection random --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 50 --per_round 10 --selection random --under_rep 5 --res_path $NEW_PATH --bad_nodes 1

# 100 clients, 50 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 100 --per_round 10 --selection kmeans --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 100 --per_round 10 --selection kmeans --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 100 --per_round 10 --selection random --under_rep 5 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 100 --per_round 10 --selection random --under_rep 5 --res_path $NEW_PATH --bad_nodes 1

# Aggregate results
python3 src/aggregate_figure.py --results_dir_name src/results/$NEW_PATH

COUNT=$(find src/results/simulation*  -maxdepth 0 -type d | wc -w)
NEW_COUNT=$((COUNT + 1))
NEW_PATH="simulation${NEW_COUNT}"

# Run simulations with interleaved --iid and non--iid configurations
# 25 clients, 10 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 25 --per_round 10 --selection kmeans --under_rep  7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 25 --per_round 10 --selection kmeans --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 25 --per_round 10 --selection random --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 25 --per_round 10 --selection random --under_rep 7 --res_path $NEW_PATH --bad_nodes 1

# 50 clients, 25 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 50 --per_round 10 --selection kmeans --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 50 --per_round 10 --selection kmeans --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 50 --per_round 10 --selection random --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 50 --per_round 10 --selection random --under_rep 7 --res_path $NEW_PATH --bad_nodes 1

# 100 clients, 50 per round, 500 iterations
python3 src/simulation.py --iterations 500 --iid --clients 100 --per_round 10 --selection kmeans --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 100 --per_round 10 --selection kmeans --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --iid --clients 100 --per_round 10 --selection random --under_rep 7 --res_path $NEW_PATH --bad_nodes 1
python3 src/simulation.py --iterations 500 --clients 100 --per_round 10 --selection random --under_rep 7 --res_path $NEW_PATH --bad_nodes 1

# Aggregate results
python3 src/aggregate_figure.py --results_dir_name src/results/$NEW_PATH
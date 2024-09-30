#!/bin/bash

# WARNING: only run this script if on a GPU server or if you have sufficient gpu power. may take a long time
set -e

#find all src/results/simulation* paths and set a new path for next simulation
COUNT=$(find src/results/simulation* -type d -maxdepth 0| wc -w)
NEW_COUNT=$((COUNT + 1))
NEW_PATH="simulation${NEW_COUNT}"


# Run the simulations
# 10 clients, 5 per round, 1000 iterations
python3 src/simulation.py --iterations 1000 --iid --clients 10 --per_round 5 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 10 --per_round 5 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 10 --per_round 5 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 10 --per_round 5 --selection random --res_path $NEW_PATH

python3 src/simulation.py --iterations 1000 --clients 10 --per_round 5 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 10 --per_round 5 --selection pearson  --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 10 --per_round 5 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 10 --per_round 5 --selection random --res_path $NEW_PATH

# 50 clients, 25 per round, 1000 iterations
python3 src/simulation.py --iterations 1000 --iid --clients 50 --per_round 25 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 50 --per_round 25 --selection pearson  --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 50 --per_round 25 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 50 --per_round 25 --selection random --res_path $NEW_PATH

python3 src/simulation.py --iterations 1000 --clients 50 --per_round 25 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 50 --per_round 25 --selection pearson  --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 50 --per_round 25 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 50 --per_round 25 --selection random --res_path $NEW_PATH
# 100 clients, 50 per round, 1000 iterations

python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 25 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 25 --selection pearson  --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 25 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 25 --selection random --res_path $NEW_PATH

python3 src/simulation.py --iterations 1000 --clients 100 --per_round 50 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 100 --per_round 50 --selection pearson  --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 100 --per_round 50 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 100 --per_round 50 --selection random --res_path $NEW_PATH

python3 src/aggregate_figure.py --results_dir_name src/results/$NEW_PATH

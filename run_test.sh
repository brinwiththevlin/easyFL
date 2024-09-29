#!/bin/bash

# WARNING: only run this script if on a GPU server or if you have sufficient gpu power. may take a long time
set -e

COUNT=$(find src/results/test* -maxdepth 0 -type d | wc -w)
NEW_COUNT=$((COUNT + 1))
NEW_PATH="test${NEW_COUNT}"

# Run the simulations
# 10 clients, 5 per round, 200 iterations
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection cosine  --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection random --res_path $NEW_PATH

python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection random --res_path $NEW_PATH

# 50 clients, 25 per round, 200 iterations
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection random --res_path $NEW_PATH

python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection random --res_path $NEW_PATH
# 100 clients, 50 per round, 200 iterations

python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 50 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 50 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 50 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 50 --selection random --res_path $NEW_PATH

python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection pearson --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection kernel --res_path $NEW_PATH
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection random --res_path $NEW_PATH

python3 src/statistic/figures.py --path src/results/$NEW_PATH

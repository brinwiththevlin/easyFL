#!/bin/bash

# WARNING: only run this script if on a GPU server or if you have sufficient gpu power. may take a long time
set -e

# Run the simulations
# 10 clients, 5 per round, 200 iterations
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection graph --similarity cosine  --res_path test
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection graph --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection graph --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection random --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection random --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --iid --clients 10 --per_round 5 --selection random --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection graph --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection graph --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection graph --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection random --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection random --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --clients 10 --per_round 5 --selection random --similarity kernel --res_path test

# 50 clients, 25 per round, 200 iterations
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection graph --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection graph --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection graph --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection random --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection random --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --iid --clients 50 --per_round 25 --selection random --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection graph --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection graph --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection graph --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection random --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection random --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --clients 50 --per_round 25 --selection random --similarity kernel --res_path test
# 100 clients, 50 per round, 200 iterations

python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 25 --selection graph --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 25 --selection graph --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 25 --selection graph --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 25 --selection random --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 25 --selection random --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --iid --clients 100 --per_round 25 --selection random --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection graph --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection graph --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection graph --similarity kernel --res_path test

python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection random --similarity cosine --res_path test
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection random --similarity pearson --res_path test --tolerance 0.7
python3 src/simulation.py --iterations 200 --clients 100 --per_round 50 --selection random --similarity kernel --res_path test

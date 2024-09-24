#!/bin/bash

# WARNING: only run this script if on a GPU server or if you have sufficient gpu power. may take a long time
set -e

# Run the simulations
# 10 clients, 5 per round, 2000 iterations
python3 src/simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --selection graph --similarity cosine
python3 src/simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --selection graph --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --selection graph --similarity kernel

python3 src/simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --selection random --similarity cosine
python3 src/simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --selection random --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --selection random --similarity kernel

python3 src/simulation.py --iterations 2000 --clients 10 --per_round 5 --selection graph --similarity cosine
python3 src/simulation.py --iterations 2000 --clients 10 --per_round 5 --selection graph --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --clients 10 --per_round 5 --selection graph --similarity kernel

python3 src/simulation.py --iterations 2000 --clients 10 --per_round 5 --selection random --similarity cosine
python3 src/simulation.py --iterations 2000 --clients 10 --per_round 5 --selection random --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --clients 10 --per_round 5 --selection random --similarity kernel

# 50 clients, 25 per round, 2000 iterations
python3 src/simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --selection graph --similarity cosine
python3 src/simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --selection graph --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --selection graph --similarity kernel

python3 src/simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --selection random --similarity cosine
python3 src/simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --selection random --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --selection random --similarity kernel

python3 src/simulation.py --iterations 2000 --clients 50 --per_round 25 --selection graph --similarity cosine
python3 src/simulation.py --iterations 2000 --clients 50 --per_round 25 --selection graph --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --clients 50 --per_round 25 --selection graph --similarity kernel

python3 src/simulation.py --iterations 2000 --clients 50 --per_round 25 --selection random --similarity cosine
python3 src/simulation.py --iterations 2000 --clients 50 --per_round 25 --selection random --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --clients 50 --per_round 25 --selection random --similarity kernel
# 100 clients, 50 per round, 2000 iterations

python3 src/simulation.py --iterations 2000 --iid --clients 100 --per_round 25 --selection graph --similarity cosine
python3 src/simulation.py --iterations 2000 --iid --clients 100 --per_round 25 --selection graph --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --iid --clients 100 --per_round 25 --selection graph --similarity kernel

python3 src/simulation.py --iterations 2000 --iid --clients 100 --per_round 25 --selection random --similarity cosine
python3 src/simulation.py --iterations 2000 --iid --clients 100 --per_round 25 --selection random --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --iid --clients 100 --per_round 25 --selection random --similarity kernel

python3 src/simulation.py --iterations 2000 --clients 100 --per_round 50 --selection graph --similarity cosine
python3 src/simulation.py --iterations 2000 --clients 100 --per_round 50 --selection graph --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --clients 100 --per_round 50 --selection graph --similarity kernel

python3 src/simulation.py --iterations 2000 --clients 100 --per_round 50 --selection random --similarity cosine
python3 src/simulation.py --iterations 2000 --clients 100 --per_round 50 --selection random --similarity pearson --tolerance 0.7
python3 src/simulation.py --iterations 2000 --clients 100 --per_round 50 --selection random --similarity kernel


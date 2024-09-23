#!/bin/bash

# WARNING: only run this script if on a GPU server or if you have sufficient gpu power. may take a long time
set -e

python3 simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --similarity cosine --selection graph

python3 simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --similarity cosine --selection graph

python3 simulation.py --iterations 2000 --iid --clients 100 --per_round 50 --similarity cosine --selection graph

python3 simulation.py --iterations 2000 --clients 10 --per_round 5 --similarity cosine --selection graph

python3 simulation.py --iterations 2000 --clients 50 --per_round 25 --similarity cosine --selection graph

python3 simulation.py --iterations 2000 --clients 100 --per_round 50 --similarity cosine --selection graph

python3 simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --similarity euclid --selection graph

python3 simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --similarity euclid --selection graph

python3 simulation.py --iterations 2000 --iid --clients 100 --per_round 50 --similarity euclid --selection graph

python3 simulation.py --iterations 2000 --clients 10 --per_round 5 --similarity euclid --selection graph

python3 simulation.py --iterations 2000 --clients 50 --per_round 25 --similarity euclid --selection graph

python3 simulation.py --iterations 2000 --clients 100 --per_round 50 --similarity euclid --selection graph

python3 simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --similarity kernel --selection graph

python3 simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --similarity kernel --selection graph

python3 simulation.py --iterations 2000 --iid --clients 100 --per_round 50 --similarity kernel --selection graph

python3 simulation.py --iterations 2000 --clients 10 --per_round 5 --similarity kernel --selection graph

python3 simulation.py --iterations 2000 --clients 50 --per_round 25 --similarity kernel --selection graph

python3 simulation.py --iterations 2000 --clients 100 --per_round 50 --similarity kernel --selection graph

python3 simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --similarity cosine --selection random

python3 simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --similarity cosine --selection random

python3 simulation.py --iterations 2000 --iid --clients 100 --per_round 50 --similarity cosine --selection random

python3 simulation.py --iterations 2000 --clients 10 --per_round 5 --similarity cosine --selection random

python3 simulation.py --iterations 2000 --clients 50 --per_round 25 --similarity cosine --selection random

python3 simulation.py --iterations 2000 --clients 100 --per_round 50 --similarity cosine --selection random

python3 simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --similarity euclid --selection random

python3 simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --similarity euclid --selection random

python3 simulation.py --iterations 2000 --iid --clients 100 --per_round 50 --similarity euclid --selection random

python3 simulation.py --iterations 2000 --clients 10 --per_round 5 --similarity euclid --selection random

python3 simulation.py --iterations 2000 --clients 50 --per_round 25 --similarity euclid --selection random

python3 simulation.py --iterations 2000 --clients 100 --per_round 50 --similarity euclid --selection random

python3 simulation.py --iterations 2000 --iid --clients 10 --per_round 5 --similarity kernel --selection random

python3 simulation.py --iterations 2000 --iid --clients 50 --per_round 25 --similarity kernel --selection random

python3 simulation.py --iterations 2000 --iid --clients 100 --per_round 50 --similarity kernel --selection random

python3 simulation.py --iterations 2000 --clients 10 --per_round 5 --similarity kernel --selection random

python3 simulation.py --iterations 2000 --clients 50 --per_round 25 --similarity kernel --selection random

python3 simulation.py --iterations 2000 --clients 100 --per_round 50 --similarity kernel --selection random

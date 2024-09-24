#!/bin/bash

set -e

python3 simulation.py --iterations 200 --iid --clients 10 --per_round 5 --similarity cosine --selection graph --res_path test

python3 simulation.py --iterations 200 --iid --clients 10 --per_round 5 --similarity euclid --selection graph --res_path test

python3 simulation.py --iterations 200 --iid --clients 10 --per_round 5 --similarity kernel --selection graph --res_path test

python3 simulation.py --iterations 200 --iid --clients 10 --per_round 5 --similarity cosine --selection random --res_path test

python3 simulation.py --iterations 200 --iid --clients 10 --per_round 5 --similarity euclid --selection random --res_path test

python3 simulation.py --iterations 200 --iid --clients 10 --per_round 5 --similarity kernel --selection random --res_path test

python3 simulation.py --iterations 200 --clients 10 --per_round 5 --similarity cosine --selection graph --res_path test

python3 simulation.py --iterations 200 --clients 10 --per_round 5 --similarity euclid --selection graph --res_path test

python3 simulation.py --iterations 200 --clients 10 --per_round 5 --similarity kernel --selection graph --res_path test

python3 simulation.py --iterations 200 --clients 10 --per_round 5 --similarity cosine --selection random --res_path test

python3 simulation.py --iterations 200 --clients 10 --per_round 5 --similarity euclid --selection random --res_path test

python3 simulation.py --iterations 200 --clients 10 --per_round 5 --similarity kernel --selection random --res_path test

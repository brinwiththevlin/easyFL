#!/bin/bash

set -e

python3 simulation.py --iid --clients 10 --per_round 5 --simularity cosine

python3 simulation.py --iid --clients 50 --per_round 25 --simularity cosine

python3 simulation.py --iid --clients 100 --per_round 50 --simularity cosine

python3 simulation.py --clients 10 --per_round 5 --simularity cosine

python3 simulation.py --clients 50 --per_round 25 --simularity cosine

python3 simulation.py --clients 100 --per_round 50 --simularity cosine

python3 simulation.py --iid --clients 10 --per_round 5 --simularity euclid

python3 simulation.py --iid --clients 50 --per_round 25 --simularity euclid

python3 simulation.py --iid --clients 100 --per_round 50 --simularity euclid

python3 simulation.py --clients 10 --per_round 5 --simularity euclid

python3 simulation.py --clients 50 --per_round 25 --simularity euclid

python3 simulation.py --clients 100 --per_round 50 --simularity euclid

python3 simulation.py --iid --clients 10 --per_round 5 --simularity kernel

python3 simulation.py --iid --clients 50 --per_round 25 --simularity kernel

python3 simulation.py --iid --clients 100 --per_round 50 --simularity kernel

python3 simulation.py --clients 10 --per_round 5 --simularity kernel

python3 simulation.py --clients 50 --per_round 25 --simularity kernel

python3 simulation.py --clients 100 --per_round 50 --simularity kernel

NEW_PATH="simulation8"
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 10 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 10 --selection random --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 100 --per_round 10 --selection random --res_path $NEW_PATH

# Aggregate results
python3 src/aggregate_figure.py --results_dir_name src/results/$NEW_PATH

NEW_PATH="simulation9"
python3 src/simulation.py --iterations 1000 --iid --clients 50 --per_round 10 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 50 --per_round 10 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 50 --per_round 10 --selection random --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 50 --per_round 10 --selection random --res_path $NEW_PATH

# 100 clients, 50 per round, 1000 iterations
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 10 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 100 --per_round 10 --selection cosine --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --iid --clients 100 --per_round 10 --selection random --res_path $NEW_PATH
python3 src/simulation.py --iterations 1000 --clients 100 --per_round 10 --selection random --res_path $NEW_PATH

# Aggregate results
python3 src/aggregate_figure.py --results_dir_name src/results/$NEW_PATH
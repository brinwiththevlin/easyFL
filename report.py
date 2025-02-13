#!/usr/bin/env python3
import re

def parse_log(log_path):
    scenarios = []
    current_scenario = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Arguments received:"):
                # If we have an active scenario, save it before starting a new one.
                if current_scenario is not None:
                    scenarios.append(current_scenario)
                # Start new scenario with parsed arguments.
                args_line = line[len("Arguments received:"):].strip()
                args = {}
                # Split by space and extract key=value pairs.
                for part in args_line.split():
                    if '=' in part:
                        key, value = part.split("=", 1)
                        args[key] = value
                current_scenario = {"args": args, "logs": []}
            else:
                if current_scenario is not None:
                    current_scenario["logs"].append(line)
    if current_scenario is not None:
        scenarios.append(current_scenario)
    return scenarios

def analyze_scenario(scenario):
    kl_filtered = 0
    kmeans_filtered = 0
    lof_filtered = 0
    passed = None
    pattern = re.compile(r"\*\*\*\{(\d+)\} nodes passed the filter\*\*\*")
    
    for log_line in scenario["logs"]:
        if "filtered out by KL divergence" in log_line:
            kl_filtered += 1
        if "filtered out by KMeans" in log_line:
            kmeans_filtered += 1
        if "filtered out by LOF" in log_line:
            lof_filtered += 1
        m = pattern.search(log_line)
        if m:
            passed = int(m.group(1))
    return kl_filtered, kmeans_filtered, passed

def generate_report(scenarios):
    report_lines = []
    for idx, scenario in enumerate(scenarios, start=1):
        args = scenario["args"]
        kl_filtered, kmeans_filtered, passed = analyze_scenario(scenario)
        report_lines.append(f"Scenario {idx}:")
        report_lines.append("  Arguments:")
        for key, value in args.items():
            report_lines.append(f"    {key}: {value}")
        report_lines.append("  Filtering results:")
        report_lines.append(f"    KL divergence filtered: {kl_filtered}")
        report_lines.append(f"    KMeans filtered: {kmeans_filtered}")
        if passed is not None:
            report_lines.append(f"    Nodes passed filter: {passed}")
        else:
            report_lines.append("    Nodes passed filter: Not reported")
        report_lines.append("")
    return "\n".join(report_lines)

if __name__ == "__main__":
    log_file = "bad_filter.log"  # Adjust path if necessary.
    scenarios = parse_log(log_file)
    report = generate_report(scenarios)
    with open("report.txt", "w") as f:
        f.write(report)
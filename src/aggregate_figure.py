import click
import os
import pandas as pd
import plotly.express as px
from typing import Optional
from config import load_config

# config = load_config()

# @click.command()
# @click.option("--results_dir_name", default=None, help="Path to results directory")
# @click.option("--bad_nodes", default=1, help="Number of malicious nodes")
# @click.option("--dataset", type=click.Choice(['MNIST', 'cifar10']))
# @click.option("--label_tampering", type=click.Choice(["none", "zero", "reverse", "random"]), help="Style of label tampering")
# @click.option("--weight_tampering", type=click.Choice(["none", "large_neg", "reverse", "random"]), help="Style of weight tampering")
# def generate_multitrace_figures(
#     results_dir_name: Optional[str] = None,
#     bad_nodes: int = 1,
#     dataset: str = "MNIST",
#     label_tampering: str = "none",
#     weight_tampering: str = "none"
# ):
#     pass


# import click
# import os
# import pandas as pd
# import plotly.express as px
# from typing import Optional
# from config import load_config

config = load_config()


@click.command()
@click.option("--results_dir_name", default=None, help="Path to results directory")
@click.option("--bad_nodes", default=1, help="Number of malicious nodes")
@click.option("--dataset", type=click.Choice(["MNIST", "cifar10"]))
@click.option(
    "--label_tampering",
    type=click.Choice(["none", "zero", "reverse", "random"]),
    help="Style of label tampering",
)
@click.option(
    "--weight_tampering",
    type=click.Choice(["none", "large_neg", "reverse", "random"]),
    help="Style of weight tampering",
)
def generate_multitrace_figures(
    results_dir_name: Optional[str] = None,
    bad_nodes: int = 1,
    dataset: str = "MNIST",
    label_tampering: str = "none",
    weight_tampering: str = "none",
):
    # Use config value if no path is provided
    if results_dir_name is None:
        results_dir_name = config.results_file_path

    if not os.path.exists(results_dir_name):
        print(
            f"Error: The specified results directory '{results_dir_name}' does not exist."
        )
        return

    # Extract simulation group (folder name of results_dir_name)
    simulation_group = os.path.basename(results_dir_name)

    # Extract experiment run directories (IID & non-IID)
    run_dirs = [
        os.path.join(results_dir_name, d)
        for d in os.listdir(results_dir_name)
        if os.path.isdir(os.path.join(results_dir_name, d))
    ]

    if not run_dirs:
        print(f"No valid result directories found in {results_dir_name}.")
        return

    # Separate IID and non-IID runs
    iids = {d for d in run_dirs if "noniid" not in d}
    noniids = set(run_dirs) - iids

    # Extract correct client splits for both IID and non-IID
    def extract_client_config(dirname: str):
        parts = dirname.split("_")
        if "noniid" in parts:
            # Non-IID format: MNIST_noniid_3_25_10_kl-kmeans
            return f"{parts[3]}_{parts[4]}"  # Extracts "25_10"
        else:
            # IID format: MNIST_iid_25_10_kl-kmeans
            return f"{parts[2]}_{parts[3]}"  # Extracts "25_10"

    client_splits_iid = {extract_client_config(os.path.basename(d)) for d in iids}
    client_splits_noniid = {extract_client_config(os.path.basename(d)) for d in noniids}

    # Group results by client count
    iids_by_clients = {
        c: [d for d in iids if extract_client_config(os.path.basename(d)) == c]
        for c in client_splits_iid
    }
    noniids_by_clients = {
        c: [d for d in noniids if extract_client_config(os.path.basename(d)) == c]
        for c in client_splits_noniid
    }

    def process_results(dirs, prefix):
        for clients, dirs in dirs.items():
            result_dfs = []
            for d in dirs:
                csv_path = os.path.join(d, "results.csv")
                if not os.path.exists(csv_path):
                    print(f"Warning: Missing results.csv in {d}, skipping.")
                    continue

                df = pd.read_csv(csv_path)

                # Extract selection method dynamically
                selection = os.path.basename(d).split("_")[
                    -1
                ]  # Last part is selection method
                df["selection"] = selection
                result_dfs.append(df)

            if not result_dfs:
                continue  # Skip if no valid data

            results_df = pd.concat(result_dfs)
            metrics = list(set(results_df.columns) - {"num_iter", "selection"})

            for metric in metrics:
                fig = px.line(
                    results_df,
                    x="num_iter",
                    y=metric,
                    color="selection",
                    labels={"num_iter": "Iterations", metric: metric},
                    title=f"{metric} over time for {clients} clients:\n{dataset}-bad={bad_nodes}-l={label_tampering}-w={weight_tampering}",
                )
                fig.update_layout(
                    width=1200, height=800, font=dict(size=18), title_font=dict(size=22)
                )

                # Save figures inside the correct simulation directory
                png_file = os.path.join(
                    results_dir_name, f"{prefix}_{clients}_{metric}_over_time.png"
                )
                fig.write_image(png_file, scale=2, engine="kaleido")
                print(f"Figure saved as {png_file}")

    # Process IID and non-IID results
    process_results(iids_by_clients, "iid")
    process_results(noniids_by_clients, "noniid")


if __name__ == "__main__":
    generate_multitrace_figures()

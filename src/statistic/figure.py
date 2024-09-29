import os
from config import config
import pandas as pd
import plotly.express as px
import click


def generate_figures(results_file_name: str | None = None):
    if results_file_name is None:
        results_file_name = config.fl_results_file_path
    results_df: pd.DataFrame = pd.read_csv(results_file_name)
    metrics: list[str] = list(results_df.columns[1:])
    path_header = results_file_name.removesuffix("results.csv")

    for metric in metrics:
        metric_values = results_df[metric]
        fig = px.line(
            metric_values,
            x=results_df.num_iter,
            y=metric_values.values,
            labels={"x": "iterations", "y": metric},
            title=f"{metric} over time for {config.comments}",
        )
        png_file = f"{path_header}/{metric}_over_time.png"
        fig.write_image(png_file)
        print(f"Figure saved as {png_file}")


@click.command()
@click.option("--results_dir_name", default=None, help="path to results directory")
def generate_multitrace_figures(results_dir_name: str | None = None):
    if results_dir_name is None:
        results_dir_name = config.results_file_path

    run_dirs = [
        d
        for d in os.listdir(results_dir_name)
        if os.path.isdir(os.path.join(results_dir_name, d))
    ]
    iids = set([d for d in run_dirs if "iid" in d])
    noniids = set(run_dirs) - iids

    iids_by_clients = {
        d: [c for c in iids if d in c] for d in ["10_5", "50_25", "100_5"]
    }
    noniids_by_clients = {
        d: [c for c in noniids if d in c] for d in ["10_5", "50_25", "100_5"]
    }

    for clients, dirs in iids_by_clients.items():
        resullt_dfs = [
            pd.read_csv(os.path.join(results_dir_name, d, "results.csv")) for d in dirs
        ]
        # add a column for selection/similarity: (random, cosine, pearson, kernel)
        # ex: MNIST_ModelCNNMnist_iid_10_5_cosine_09-26-16:26/

        results_dfs = [
            df.assign(selection=d.split("_")[-2]) for df, d in zip(resullt_dfs, dirs)
        ]
        results_df = pd.concat(results_dfs)

        metrics = list(set(results_df.columns) - {"num_iter", "selection"})
        for metric in metrics:
            fig = px.line(
                results_df,
                x="num_iter",
                y=metric,
                color="selection",
                labels={"x": "iterations", "y": metric},
                title=f"{metric} over time for {clients} clients",
            )
            png_file = os.path.join(
                results_dir_name, f"{clients}_{metric}_over_time.png"
            )
            fig.write_image(png_file)
            print(f"Figure saved as {png_file}")

    for clients, dirs in noniids_by_clients.items():
        resullt_dfs = [
            pd.read_csv(os.path.join(results_dir_name, d, "results.csv")) for d in dirs
        ]
        # add a column for selection/similarity: (random, cosine, pearson, kernel)
        # ex: MNIST_ModelCNNMnist_iid_10_5_cosine_09-26-16:26/
        results_dfs = [
            df.assign(selection=d.split("_")[-2]) for df, d in zip(resullt_dfs, dirs)
        ]
        results_df = pd.concat(results_dfs)

        metrics = list(set(results_df.columns) - {"num_iter", "selection"})
        for metric in metrics:
            fig = px.line(
                results_df,
                x="num_iter",
                y=metric,
                color="selection",
                labels={"x": "iterations", "y": metric},
                title=f"{metric} over time for {clients} clients",
            )
            png_file = os.path.join(
                results_dir_name, f"{clients}_{metric}_over_time.png"
            )
            fig.write_image(png_file)
            print(f"Figure saved as {png_file}")

if __name__ == "__main__":
    generate_multitrace_figures()

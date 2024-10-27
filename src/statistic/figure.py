from config import config
import pandas as pd
import plotly.express as px


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



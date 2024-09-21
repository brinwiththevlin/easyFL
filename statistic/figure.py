import os
from config import config
import pandas as pd
import plotly.express as px


def generate_figures(results_file_name: str = config.fl_results_file_path):
    results_df: pd.DataFrame = pd.read_csv(results_file_name)
    metrics: list[str] = list(results_df.columns[1:])

    for metric in metrics:
        metric_values = results_df[metric]
        fig = px.line(
            metric_values,
            x=metric_values.index,
            y=metric_values.values,
            labels={"x": "iterations", "y": metric},
            title=f"{metric} over time",
        )
        png_file = f"{metric}_over_time_{results_file_name}.png"
        fig.write_image(png_file)
        print(f"FIgure saved as {png_file}")

        os.system(f"/mnt/c/Windows/System32/cmd.exe /c start {png_file}")
    pass

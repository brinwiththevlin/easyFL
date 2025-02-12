from src.config import load_config, Config

config = load_config()
import pandas as pd
import plotly.express as px


import os
import pandas as pd
import plotly.express as px
from typing import Optional
from config import load_config  

config = load_config()

def generate_figures(results_file_name: Optional[str] = None):
    if results_file_name is None:
        results_file_name = config.fl_results_file_path

    results_df = pd.read_csv(results_file_name)

    if "num_iter" not in results_df.columns:
        print(f"Error: 'num_iter' column missing in {results_file_name}")
        return

    metrics = [col for col in results_df.columns if col != "num_iter"]
    path_header = os.path.dirname(results_file_name)

    for metric in metrics:
        fig = px.line(
            results_df,
            x="num_iter",
            y=metric,
            labels={"num_iter": "Iterations", metric: metric},
            title=f"{metric} over time for {config.comments}",
        )
        fig.update_layout(
            width=1200, height=800, font=dict(size=18), title_font=dict(size=22)
        )

        png_file = os.path.join(path_header, f"{metric}_over_time.png")
        fig.write_image(png_file, scale=2, engine="kaleido")
        print(f"Figure saved as {png_file}")


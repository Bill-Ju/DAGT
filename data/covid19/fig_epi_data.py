import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_epi_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    node = np.array(data["node"])
    time = np.array(data["time"])
    case = np.array(data["case"])
    return node, time, case


def plot_epi_case(node, time, case, region, output_dir):
    n_nodes = len(node)
    if n_nodes == 0 or len(time) == 0:
        raise ValueError(f"Empty data for {region}")

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    delta = -1
    target_labels = 8
    dN = max(1, int(np.ceil(n_nodes / target_labels)))
    dT = max(1, len(time) // 4)
    minor_step = max(1, dT // 4)

    for idx in range(n_nodes):
        max_val = case[:, idx].max()
        scale = n_nodes / 20 / max_val if max_val > 0 else 1.0
        series = case[:, idx] * scale + delta * idx
        ax.plot(series, color="black", lw=plt.rcParams["lines.linewidth"] / 2)
        ax.fill_between(
            np.arange(len(time)),
            series,
            delta * idx,
            color="#2933EF",
            alpha=0.6,
            edgecolor=None,
        )

    ax.set_xticks(np.arange(0, len(time), dT))
    ax.set_xticks(np.arange(0, len(time), minor_step), minor=True)
    ax.set_xticklabels(time[::dT], rotation=0, fontsize=0.7 * plt.rcParams["font.size"])
    ax.set_xlabel("Time")

    ax.set_yticks(delta * np.arange(0, n_nodes, dN))
    ax.set_yticks(delta * np.arange(0, n_nodes), minor=True)
    ax.tick_params(axis="both", which="minor", width=plt.rcParams["axes.linewidth"] / 2)
    ax.set_yticklabels(node[::dN], fontsize=0.7 * plt.rcParams["font.size"])
    ax.set_ylabel("Node")

    ax.tick_params(axis="both", which="both", direction="in")
    # ax.set_title(
    #     f"COVID-19 confirmed cases in {n_nodes} counties of {region}",
    #     fontsize=plt.rcParams["font.size"],
    # )

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{region}_case.svg"
    fig.savefig(out_path)
    plt.close(fig)


def main():
    style_path = Path(__file__).resolve().parent / "plot.txt"
    plt.style.use(style_path)
    data_dir = Path("COVID_Network_Data")
    output_dir = Path("COVID_Network_Data/figs/")

    regions = ["Alaska", "Illinois", "Maine","Minnesota", "Utah", "Washington"]
    for region in regions:
        json_path = data_dir / f"COVIDin{region}.json"
        node, time, case = load_epi_json(json_path)
        plot_epi_case(node, time, case, region, output_dir)


if __name__ == "__main__":
    main()

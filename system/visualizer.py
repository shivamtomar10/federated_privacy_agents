# system/visualizer.py

import matplotlib
matplotlib.use('Agg')  # Important for headless environments

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def plot_federated_metrics(history_df, output_dir="metrics"):
    """Standard performance overview plots"""

    if history_df.empty:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_theme(style="whitegrid")

    # 1️⃣ Learning Curve
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history_df, x="round", y="accuracy", hue="country", marker="o")
    plt.title("Federated Learning Progress: Accuracy per Round")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig(f"{output_dir}/learning_curve.png")
    plt.close()

    # 2️⃣ Privacy vs Utility
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=history_df,
        x="epsilon",
        y="accuracy",
        hue="country",
        size="reward",
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title("Privacy-Utility Trade-off (Accuracy vs Epsilon)")
    plt.savefig(f"{output_dir}/privacy_utility.png")
    plt.close()

    # 3️⃣ Data distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=history_df.drop_duplicates('country'),
        x="country",
        y="rows"
    )
    plt.title("Data Distribution Across Hospital Agents")
    plt.savefig(f"{output_dir}/data_distribution.png")
    plt.close()

    print(f"📊 Standard visualizations saved to '{output_dir}/'")


def plot_research_graphs(history_df, norm_history, output_dir="metrics"):
    """Research-grade plots"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_theme(style="whitegrid")

    # 1️⃣ Convergence
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=history_df,
        x="round",
        y="accuracy",
        hue="country",
        marker="o",
        linewidth=2.5
    )
    plt.title("Federated Convergence (Accuracy over Rounds)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.savefig(f"{output_dir}/research_convergence.png")
    plt.close()

    # 2️⃣ Privacy Tax
    plt.figure(figsize=(10, 6))
    sns.regplot(data=history_df, x="epsilon", y="accuracy", scatter=True)
    plt.title("Privacy-Utility Trade-off (Privacy Tax)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(f"{output_dir}/research_privacy_tax.png")
    plt.close()

    # 3️⃣ Norm Stability
    plt.figure(figsize=(10, 6))
    for country, norms in norm_history.items():
        plt.plot(
            range(1, len(norms) + 1),
            norms,
            label=f"{country}",
            marker='x'
        )

    plt.title("Weight Norm Stability (Byzantine Resilience)")
    plt.xlabel("Round")
    plt.ylabel("Norm")
    plt.legend()
    plt.savefig(f"{output_dir}/research_norms.png")
    plt.close()

    print("📈 Research graphs generated successfully")


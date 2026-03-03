import matplotlib
matplotlib.use('Agg') # Add this before importing plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_federated_metrics(history_df, output_dir="metrics"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_theme(style="whitegrid")

    # 1. Learning Curve: Accuracy per Round per Country
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history_df, x="round", y="accuracy", hue="country", marker="o")
    plt.title("Federated Learning Progress: Accuracy per Round")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig(f"{output_dir}/learning_curve.png")
    plt.close()

    # 2. Privacy vs Utility: Reward based on Epsilon
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=history_df, x="epsilon", y="accuracy", size="reward", hue="country", alpha=0.7)
    plt.title("Privacy-Utility Trade-off (Accuracy vs. Epsilon)")
    plt.savefig(f"{output_dir}/privacy_utility.png")
    plt.close()

    # 3. Data Heterogeneity: Sample counts per country
    plt.figure(figsize=(10, 6))
    sns.barplot(data=history_df.drop_duplicates('country'), x="country", y="rows")
    plt.title("Data Distribution Across Hospital Agents (Samples)")
    plt.savefig(f"{output_dir}/data_distribution.png")
    plt.close()

    print(f"📊 Visualizations saved to the '{output_dir}/' folder.")

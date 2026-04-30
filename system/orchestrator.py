# system/orchestrator.py

import pandas as pd
import numpy as np
import os
from collections import Counter

from system.visualizer import plot_federated_metrics, plot_research_graphs

from agents.hospital_agent import HospitalAgent
from agents.global_agent import GlobalAgent
from agents.attack_detector import AttackDetector


class FederatedOrchestrator:
    """
    Multi-Round Federated Learning Orchestrator
    - Handles feature alignment
    - Tracks metrics across rounds
    - Generates research graphs
    - Improved stability + research tracking
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.attack_detector = AttackDetector()
        self.global_agent = GlobalAgent()
        self.final_reports = []
        self.final_total_rows = 0
        self.global_feature_list = None

    def run(self, rounds=5):
        print("\n🚀 Starting Research-Grade Federated Learning")

        global_weights = None

        history_data = []
        norm_history = {country: [] for country in self.datasets.keys()}
        global_norm_history = []  # ✅ NEW

        for r in range(rounds):
            print("\n" + "=" * 60)
            print(f"🌍 FEDERATED ROUND {r + 1}")
            print("=" * 60)

            country_updates = {}
            total_rows_this_round = 0
            round_reports = []

            # ------------------------------
            # 1️⃣ Feature Alignment (Round 0)
            # ------------------------------
            if r == 0:
                print("\n🔎 Handshake: Collecting global feature space...")
                all_features = set()

                for country, path in self.datasets.items():
                    temp_agent = HospitalAgent(country, path)
                    df = temp_agent.load_data()
                    target = temp_agent.infer_target_column(df)

                    features = temp_agent.get_feature_list(df, target)
                    all_features.update(features)

                self.global_feature_list = sorted(list(all_features))
                print(f"🌐 Global feature dimension: {len(self.global_feature_list)}")

            # ------------------------------
            # 2️⃣ Distributed Training (SHUFFLED)
            # ------------------------------
            items = list(self.datasets.items())
            np.random.shuffle(items)  # ✅ remove ordering bias

            for country, path in items:
                print(f"\n🏥 Hospital Node: {country.upper()}")

                hospital = HospitalAgent(
                    country=country,
                    csv_path=path,
                    global_weights=global_weights,
                    global_feature_list=self.global_feature_list
                )

                result = hospital.process()
                country_updates[country] = result

                # Track weight norm
                w_norm = float(np.linalg.norm(result["weights"]))
                norm_history[country].append(w_norm)

                # Extract best strategy
                best_strategy = max(
                    result["strategy_results"],
                    key=lambda x: x["accuracy"]
                )

                history_data.append({
                    "round": r + 1,
                    "country": country,
                    "accuracy": best_strategy["accuracy"],
                    "epsilon": best_strategy["epsilon"],
                    "rows": result["analysis"]["rows"],
                    "reward": best_strategy["reward"]
                })

                round_reports.append(result)
                total_rows_this_round += result["analysis"]["rows"]

            # ------------------------------
            # 3️⃣ Byzantine Filtering
            # ------------------------------
            raw_weights = [u["weights"] for u in country_updates.values()]
            clean_indices = self.attack_detector.inspect(raw_weights)

            if not clean_indices:
                print("⚠️ Byzantine detector rejected all clients → fallback to all")
                clean_indices = list(range(len(country_updates)))

            filtered_updates = {
                k: v for idx, (k, v) in enumerate(country_updates.items())
                if idx in clean_indices
            }

            current_norms = [
                round(norm_history[c][-1], 4)
                for c in self.datasets.keys()
            ]
            print(f"📊 Weight Norms this round: {current_norms}")

            # ------------------------------
            # 4️⃣ Aggregation
            # ------------------------------
            global_weights, meta = self.global_agent.aggregate(filtered_updates)

            if "error" in meta:
                print("⚠️ Aggregation skipped due to no valid updates")

            elif global_weights is not None:
                # ✅ NORMALIZATION (CRITICAL FIX)
                global_weights = global_weights / (np.linalg.norm(global_weights) + 1e-8)

                g_norm = float(np.linalg.norm(global_weights))
                global_norm_history.append(g_norm)

                print(f"✅ Round {r+1} Aggregation Complete. Global Norm: {g_norm:.4f}")

            else:
                print("⚠️ Aggregation failed — keeping previous weights")

            # Save final round
            if r == rounds - 1:
                self.final_reports = round_reports
                self.final_total_rows = total_rows_this_round

        # ------------------------------
        # 5️⃣ Final Report + Graphs
        # ------------------------------
        self._print_final_report()

        history_df = pd.DataFrame(history_data)

        if not os.path.exists("metrics"):
            os.makedirs("metrics")

        plot_federated_metrics(history_df)
        plot_research_graphs(history_df, norm_history)

        print("\n🎉 All Research Graphs generated in 'metrics/'")

        return {
            "global_weights": global_weights,
            "global_norm_history": global_norm_history  # ✅ extra research output
        }

    def _print_final_report(self):
        print("\n🌍 FINAL GLOBAL AI AGENT REPORT\n")

        global_distribution = Counter()

        for report in self.final_reports:
            ana = report["analysis"]

            print(
                f"📌 {report['country'].upper()} | "
                f"Rows: {ana['rows']} | "
                f"DP ε: {report['strategy']['epsilon']} | "
                f"Risk: {report['strategy']['risk_level']}"
            )

            global_distribution.update(ana["class_counts"])

        print(f"\n🌐 Total Data: {self.final_total_rows} rows")
        print(f"🌐 Global Features: {self.global_feature_list}")
        print(f"🌐 Class Distribution: {dict(global_distribution)}")


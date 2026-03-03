# system/orchestrator.py

import pandas as pd
import numpy as np
from collections import Counter

# Visualizer import
from system.visualizer import plot_federated_metrics

# Core imports
from core.schema_inference import infer_schema
from core.sensitivity_detector import detect_sensitive_columns
from core.privacy import apply_privacy
from core.data_sanitizer import sanitize

# Agent imports
from agents.hospital_agent import HospitalAgent
from agents.global_agent import GlobalAgent
from agents.attack_detector import AttackDetector


class FederatedOrchestrator:
    """
    Multi-Round Federated Learning Orchestrator with feature alignment.
    Tracks metrics across rounds to generate performance visualizations.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.attack_detector = AttackDetector()
        self.global_agent = GlobalAgent()

    def run(self, rounds=3):

        print("\n🚀 Starting Multi-Round Federated Learning")
        global_weights = None
        global_feature_list = None

        # This list will store metrics for every round/country for plotting
        history_data = []

        for r in range(rounds):

            print("\n" + "=" * 60)
            print(f"🌍 FEDERATED ROUND {r + 1}")
            print("=" * 60)

            country_updates = {}
            global_numeric_features = set()
            global_class_counts = Counter()
            total_rows = 0
            full_analysis_reports = []

            # --------------------------------------
            # 1️⃣ Feature Alignment Phase (Round 1 only)
            # --------------------------------------
            if r == 0:
                print("\n🔎 Collecting global feature space...")

                all_features = set()

                for country, path in self.datasets.items():
                    temp_agent = HospitalAgent(country, path)
                    df = temp_agent.load_data()

                    schema = infer_schema(df)
                    sensitivity = detect_sensitive_columns(schema)

                    target = temp_agent.infer_target_column(df)
                    df = apply_privacy(df, sensitivity, country)
                    df = sanitize(df)

                    features = temp_agent.get_feature_list(df, target)
                    all_features.update(features)

                global_feature_list = sorted(list(all_features))
                print("🌐 Global feature dimension:", len(global_feature_list))

            # --------------------------------------
            # 2️⃣ Run Hospital Agents
            # --------------------------------------
            for country, path in self.datasets.items():

                print(f"\n🏥 Processing {country.upper()} dataset")

                hospital = HospitalAgent(
                    country=country,
                    csv_path=path,
                    global_weights=global_weights,
                    global_feature_list=global_feature_list
                )

                # Process local training and privacy preservation
                update = hospital.process()
                country_updates[country] = update

                analysis = update["analysis"]
                strategy = update["strategy"]

                # --- CAPTURE METRICS FOR VISUALIZATION ---
                # We pull the best accuracy achieved in the local strategy search
                best_acc = update["strategy_results"][-1]["accuracy"]
                history_data.append({
                    "round": r + 1,
                    "country": country,
                    "accuracy": best_acc,
                    "epsilon": strategy["epsilon"],
                    "reward": update["strategy_results"][-1]["reward"],
                    "rows": analysis["rows"]
                })

                full_analysis_reports.append({
                    "country": country,
                    "target_column": update["target_column"],
                    "analysis": analysis,
                    "strategy": strategy
                })

                global_numeric_features.update(analysis["numeric_features"])
                global_class_counts.update(analysis.get("class_counts", {}))
                total_rows += analysis["rows"]

            # --------------------------------------
            # 3️⃣ Aggregate Global Model
            # --------------------------------------
            global_weights, _ = self.global_agent.aggregate(country_updates)

            if global_weights is None:
                print("⚠️ Warning: Global weights aggregation failed!")
            else:
                print(f"✅ Aggregation complete. Weight Norm: {np.linalg.norm(global_weights):.2f}")

            # --------------------------------------
            # 4️⃣ Final Report (Summary of last round)
            # --------------------------------------
            if r == rounds - 1:
                print("\n🌍 FINAL GLOBAL AI AGENT REPORT\n")

                for report in full_analysis_reports:
                    analysis = report["analysis"]
                    strategy = report["strategy"]

                    print(f"📌 Country: {report['country']}")
                    print(f"Target Column: {report['target_column']}")
                    print(f"Rows: {analysis['rows']}")
                    print(f"Numeric Features: {analysis['numeric_features']}")
                    print(f"Sensitive Features: {analysis['sensitive_features']}")
                    print(f"Sensitive Ratio: {analysis['sensitive_ratio']:.2f}")
                    print(f"Class Imbalance: {analysis['class_imbalance']:.2f}")
                    print(f"DP ε: {strategy['epsilon']}")
                    print(f"Epochs: {strategy['epochs']}")
                    print(f"Risk Level: {strategy['risk_level']}\n")

                print(f"🌐 Total rows across countries: {total_rows}")
                print(f"🌐 Combined numeric features: {sorted(global_numeric_features)}")
                print(f"🌐 Global class distribution: {dict(global_class_counts)}")

        # --------------------------------------
        # 5️⃣ GENERATE VISUALIZATIONS
        # --------------------------------------
        print("\n📊 Generating visual reports for the reviewer...")
        history_df = pd.DataFrame(history_data)
        plot_federated_metrics(history_df)

        print("\n🎉 Multi-Round Federated Learning Completed Successfully")

        return {
            "global_weights": global_weights
        }

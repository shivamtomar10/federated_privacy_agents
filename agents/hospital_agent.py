# agents/hospital_agent.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from core.schema_inference import infer_schema
from core.sensitivity_detector import detect_sensitive_columns
from core.privacy import apply_privacy
from core.data_sanitizer import sanitize
from core.model import FederatedModel

from agents.agent_memory import summarize_memory


class HospitalAgent:

    def __init__(self, country, csv_path, global_weights=None, global_feature_list=None):
        self.country = country
        self.csv_path = csv_path
        self.global_weights = global_weights
        self.global_feature_list = global_feature_list

        # ✅ GLOBAL ENCODER CACHE (important)
        self.encoders = {}

    def load_data(self):
        df = pd.read_csv(self.csv_path, engine="python", on_bad_lines="skip")
        print(f"Loaded {len(df)} rows from {self.country}")
        return df

    def infer_target_column(self, df):
        keywords = ["diagnosis", "disease", "target", "label", "class"]
        for col in df.columns:
            if any(k in col.lower() for k in keywords):
                return col
        raise ValueError("❌ Cannot infer target column")

    def get_feature_list(self, df, target):
        df_features, _ = self.preprocess(df, [], target)
        return sorted(list(df_features.columns))

    # ===============================
    # PREPROCESSING (FIXED)
    # ===============================
    def _encode_column(self, col, values):
        if col not in self.encoders:
            le = LabelEncoder()
            le.fit(values.astype(str))
            self.encoders[col] = le

        return self.encoders[col].transform(values.astype(str))

    def preprocess(self, df, sensitive_cols, target):
        df = df.copy()

        junk_keywords = ["phone", "id", "address", "name", "email", "mobile", "ssn"]

        cols_to_drop = [
            col for col in df.columns
            if any(k in col.lower() for k in junk_keywords) or col in sensitive_cols
        ]

        df = df.drop(columns=cols_to_drop, errors="ignore")

        y = None
        if target in df.columns:
            y = LabelEncoder().fit_transform(df[target].astype(str))
            df = df.drop(columns=[target], errors="ignore")

        # ✅ CONSISTENT encoding
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = self._encode_column(col, df[col])

        df = df.fillna(0)

        return df, y

    # ===============================
    # ANALYSIS
    # ===============================
    def analyze_dataset(self, df, sensitivity, target_column):
        total_rows = len(df)

        df_clean, _ = self.preprocess(df, sensitivity.keys(), target_column)

        class_counts = df[target_column].value_counts().to_dict()
        class_dist = {k: v / total_rows for k, v in class_counts.items()}

        return {
            "rows": total_rows,
            "sensitive_features": [c for c, r in sensitivity.items() if r == "HIGH"],
            "sensitive_ratio": len(sensitivity) / max(len(df.columns), 1),
            "class_counts": class_counts,
            "class_imbalance": max(class_dist.values()) if class_dist else 0,
            "numeric_features": list(df_clean.columns)
        }

    # ===============================
    # STRATEGY
    # ===============================
    def decide_strategies(self, analysis):
        memory_summary = summarize_memory()

        eps_candidates = [1.0, 5.0, 10.0]
        lr_candidates = [0.05, 0.1]

        bad_eps = memory_summary.get("bad_epsilons", set())
        eps_candidates = [e for e in eps_candidates if e not in bad_eps]

        strategies = []

        for eps in eps_candidates:
            for lr in lr_candidates:
                strategies.append({
                    "epochs": 5 if analysis["rows"] < 5000 else 10,
                    "lr": lr,
                    "epsilon": eps,
                    "risk_level": "HIGH" if analysis["class_imbalance"] > 0.9 else "LOW"
                })

        return strategies

    # ===============================
    # TRAINING (FIXED)
    # ===============================
    def train_with_planning(self, df, strategies, target):
        df_model, y = self.preprocess(df, [], target)

        if y is None:
            raise ValueError("❌ Target missing")

        if self.global_feature_list is not None:
            df_model = df_model.reindex(columns=self.global_feature_list, fill_value=0)

        X = df_model.to_numpy(dtype=np.float64)

        # ✅ GLOBAL SAFE NORMALIZATION
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        X = (X - mean) / std

        num_classes = len(np.unique(y))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best = None
        strategy_results = []

        for strategy in strategies:
            model = FederatedModel(X_train.shape[1], num_classes)

            if self.global_weights is not None:
                g = self.global_weights
                r = min(g.shape[0], model.weights.shape[0])
                c = min(g.shape[1], model.weights.shape[1])
                model.weights[:r, :c] = g[:r, :c]

            initial_weights = model.weights.copy()

            for _ in range(strategy["epochs"]):
                logits = X_train @ model.weights

                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)

                one_hot = np.eye(num_classes)[y_train]

                grad = (X_train.T @ (probs - one_hot)) / len(y_train)

                # ✅ PROPER DP NOISE scaling
                grad_norm = np.linalg.norm(grad)
                noise_scale = grad_norm / (strategy["epsilon"] + 1e-8)

                noise = np.random.laplace(0, noise_scale, size=grad.shape)

                model.weights -= strategy["lr"] * (grad + noise)

            accuracy = np.mean(np.argmax(X_val @ model.weights, axis=1) == y_val)

            update = model.weights - initial_weights

            strategy_results.append({
                "country": self.country,
                "accuracy": accuracy,
                "epsilon": strategy["epsilon"],
                "reward": 1
            })

            print(f"🔍 Strategy {strategy} → Val Accuracy={accuracy:.4f}")

            if best is None or accuracy > best[2]:
                best = (update, strategy, accuracy, 1)

        return (*best, strategy_results)

    # ===============================
    # MAIN
    # ===============================
    def process(self):
        df = self.load_data()
        target = self.infer_target_column(df)

        schema = infer_schema(df)
        sensitivity = detect_sensitive_columns(schema)

        df = apply_privacy(df, sensitivity, self.country)
        df = sanitize(df)

        analysis = self.analyze_dataset(df, sensitivity, target)
        strategies = self.decide_strategies(analysis)

        update, strategy, acc, reward, res = self.train_with_planning(df, strategies, target)

        update = np.clip(update, -10.0, 10.0)

        return {
            "country": self.country,
            "target_column": target,
            "weights": update,
            "analysis": analysis,
            "strategy": strategy,
            "strategy_results": res
        }


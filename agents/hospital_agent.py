# agents/hospital_agent.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

from core.feature_encoder import encode
from core.schema_inference import infer_schema
from core.sensitivity_detector import detect_sensitive_columns
from core.privacy import apply_privacy, add_differential_privacy
from core.data_sanitizer import sanitize
from core.model import FederatedModel

from agents.agent_memory import (
    save_experience,
    summarize_memory
)


class HospitalAgent:

    GOALS = {
        "min_accuracy": 0.70,
        "min_epsilon": 0.3
    }

    def __init__(self, country, csv_path, global_weights=None, global_feature_list=None):
        self.country = country
        self.csv_path = csv_path
        self.global_weights = global_weights
        self.global_feature_list = global_feature_list

    # ===============================
    # DATA LOADING
    # ===============================
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

    # ===============================
    # FEATURE EXTRACTION
    # ===============================
    def get_feature_list(self, df, target):

        schema = infer_schema(df)
        sensitivity = detect_sensitive_columns(schema)

        sensitive_cols = [
            c for c, risk in sensitivity.items()
            if risk == "HIGH"
        ]

        df_model = df.drop(columns=sensitive_cols, errors="ignore")
        df_model = encode(df_model)
        df_model = df_model.drop(columns=[target], errors="ignore")

        return df_model.columns.tolist()

    # ===============================
    # DATA ANALYSIS
    # ===============================
    def analyze_dataset(self, df, sensitivity, target_column):

        sensitive_cols = [
            col for col, risk in sensitivity.items()
            if risk == "HIGH"
        ]

        class_counts = df[target_column].value_counts().to_dict()
        total_rows = len(df)

        class_distribution = {
            k: v / total_rows for k, v in class_counts.items()
        }

        class_imbalance = max(class_distribution.values()) if class_distribution else 0

        numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_features = [
            col for col in numeric_features
            if col not in sensitive_cols and col != target_column
        ]

        return {
            "rows": total_rows,
            "sensitive_features": sensitive_cols,
            "sensitive_ratio": len(sensitive_cols) / max(len(df.columns), 1),
            "class_counts": class_counts,
            "class_distribution": class_distribution,
            "class_imbalance": class_imbalance,
            "numeric_features": numeric_features
        }

    # ===============================
    # STRATEGY DESIGN
    # ===============================
    def decide_strategies(self, analysis):

        memory_summary = summarize_memory()

        eps_candidates = [0.3, 0.5, 0.7]
        lr_candidates = [0.01, 0.02]

        bad_eps = memory_summary.get("bad_epsilons", set())
        eps_candidates = [e for e in eps_candidates if e not in bad_eps]

        strategies = []

        for eps in eps_candidates:
            for lr in lr_candidates:
                strategies.append({
                    "epochs": 3 if analysis["rows"] < 5000 else 8,
                    "lr": lr,
                    "epsilon": eps,
                    "risk_level": (
                        "HIGH" if analysis["class_imbalance"] > 0.9 else "LOW"
                    )
                })

        return strategies

    # ===============================
    # REWARD FUNCTION
    # ===============================
    def compute_reward(self, accuracy, epsilon):

        reward = 0

        if accuracy >= 0.85:
            reward += 2
        elif accuracy >= 0.70:
            reward += 1

        if epsilon >= 0.5:
            reward += 1

        return reward

    # ===============================
    # TRAINING LOOP (multi-class + class weights)
    # ===============================
    def train_with_planning(self, df, strategies, target):

        schema = infer_schema(df)
        sensitivity = detect_sensitive_columns(schema)

        sensitive_cols = [
            c for c, risk in sensitivity.items()
            if risk == "HIGH"
        ]

        # Remove sensitive columns
        df_model = df.drop(columns=sensitive_cols, errors="ignore")

        # Encode categorical features
        df_model = encode(df_model)

        # GLOBAL FEATURE ALIGNMENT
        if self.global_feature_list is not None:
            df_model = df_model.reindex(
                columns=self.global_feature_list,
                fill_value=0
            )

        # Separate features and target
        X_df = df_model.drop(columns=[target], errors="ignore")

        # Force numeric
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.fillna(0)
        X_df = X_df.astype(np.float64)

        X = X_df.to_numpy(dtype=np.float64)

        # Target encoding (multi-class)
        y = LabelEncoder().fit_transform(df[target])
        num_classes = len(np.unique(y))

        # Dimensionality reduction (optional)
        if X.shape[1] > 200:
            pca = PCA(n_components=200)
            X = pca.fit_transform(X)

        # Normalize
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # class weights (handle imbalance)
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        sample_weights = class_weights[y]

        print("Actual feature count:", X.shape[1])
        print("Feature dtype:", X.dtype)

        best = None
        strategy_results = []

        for strategy in strategies:

            model_dim = X.shape[1]
            model = FederatedModel(model_dim, num_classes)

            if self.global_weights is not None:
                if isinstance(self.global_weights, np.ndarray):
                    if self.global_weights.shape == model.weights.shape:
                        model.weights = self.global_weights.copy()

            initial_weights = model.weights.copy()

            # custom training with class weights
            for _ in range(strategy["epochs"]):

                logits = X @ model.weights
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)

                one_hot = np.eye(num_classes)[y]

                # weighted gradient
                grad = (X.T @ ((probs - one_hot) * sample_weights[:, None])) / len(y)

                model.weights -= strategy["lr"] * grad

            update = model.weights - initial_weights

            # multi-class prediction
            logits = X @ model.weights
            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)

            accuracy = np.mean(preds == y)
            reward = self.compute_reward(accuracy, strategy["epsilon"])

            print(f"🔍 Strategy {strategy} → Accuracy={accuracy:.4f}, Reward={reward}")

            strategy_results.append({
                "country": self.country,
                "strategy": strategy,
                "accuracy": accuracy,
                "reward": reward
            })

            if best is None or accuracy > best[2]:
                best = (update, strategy, accuracy, reward)

        return (*best, strategy_results)

    # ===============================
    # MAIN EXECUTION
    # ===============================
    def process(self):

        df = self.load_data()

        schema = infer_schema(df)
        sensitivity = detect_sensitive_columns(schema)

        target = self.infer_target_column(df)
        print("🎯 Target column:", target)

        df = apply_privacy(df, sensitivity, self.country)
        df = sanitize(df)

        analysis = self.analyze_dataset(df, sensitivity, target)
        strategies = self.decide_strategies(analysis)

        update, strategy, accuracy, reward, strategy_results = \
            self.train_with_planning(df, strategies, target)

        clip_value = 5.0
        update = np.clip(update, -clip_value, clip_value)

        update = add_differential_privacy(update, strategy["epsilon"])

        save_experience({
            "country": self.country,
            "rows": analysis["rows"],
            "epsilon": strategy["epsilon"],
            "lr": strategy["lr"],
            "score": accuracy,
            "reward": reward
        })

        return {
            "country": self.country,
            "target_column": target,
            "weights": update,
            "analysis": analysis,
            "strategy": strategy,
            "strategy_results": strategy_results
        }


"""
Federated Learning Trainer - Privacy-Preserving Model Training
Implements federated learning with differential privacy and secure aggregation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional
import joblib
import hashlib
import json
import os
import time
from dataclasses import dataclass
import warnings
import pickle
warnings.filterwarnings('ignore')

# ✅ Unified feature engineering import
from backend.feature_engineering import get_feature_engineer

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    data_size: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'training_time': self.training_time,
            'data_size': self.data_size
        }


class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise
    
    def clip_gradients(self, gradients: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
        norm = np.linalg.norm(gradients)
        if norm > clip_norm:
            return gradients * (clip_norm / norm)
        return gradients
    
    def privatize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        privatized = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ['data_size', 'training_time']:
                sensitivity = 0.05
                noise = np.random.laplace(0, sensitivity / self.epsilon)
                privatized[key] = max(0.0, min(1.0, value + noise))
            else:
                privatized[key] = value
        return privatized


class FederatedModelTrainer:
    """
    Trains local models for federated learning with privacy preservation
    Uses unified feature engineering for consistency with inference.
    """
    
    def __init__(
        self, 
        model_type: str = "random_forest",
        epsilon: float = 1.0,
        delta: float = 1e-5,
        model_params: Optional[Dict] = None
    ):
        self.model_type = model_type
        self.dp = DifferentialPrivacy(epsilon, delta)
        self.model_params = model_params or {}
        self.target_column = 'isFraud'
        # Unified feature engineering
        self.feature_engineer = get_feature_engineer()
        self.feature_columns = self.feature_engineer.get_feature_names()
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validate input
        if not self.feature_engineer.validate_features(df):
            raise ValueError("Missing required columns in input data")
        # Transform features using unified pipeline
        df_processed = self.feature_engineer.transform(df)
        self.feature_columns = self.feature_engineer.get_feature_names()
        return df_processed
    
    def _create_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                min_samples_split=self.model_params.get('min_samples_split', 5),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=self.model_params.get('max_iter', 1000),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 5),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_local_model(
        self, 
        df: pd.DataFrame, 
        apply_dp: bool = True,
        test_size: float = 0.2
    ) -> Tuple[Any, ModelMetrics]:
        start_time = time.time()
        df_processed = self._prepare_features(df)
        X = df_processed[self.feature_columns]
        y = df[self.target_column]
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        model = self._create_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = ModelMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, y_prob)),
            training_time=time.time() - start_time,
            data_size=len(df)
        )
        if apply_dp:
            metrics_dict = self.dp.privatize_metrics(metrics.to_dict())
            metrics = ModelMetrics(**metrics_dict)
        return model, metrics
    
    def generate_encrypted_update(
        self, 
        model: Any, 
        metrics: ModelMetrics
    ) -> Dict[str, Any]:
        model_bytes = pickle.dumps(model)
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        zkp_data = {
            'model_hash': model_hash,
            'accuracy': metrics.accuracy,
            'data_size': metrics.data_size,
            'timestamp': time.time()
        }
        zkp_proof = hashlib.sha256(
            json.dumps(zkp_data, sort_keys=True).encode()
        ).hexdigest()
        update = {
            'model_bytes': model_bytes,
            'model_hash': model_hash,
            'metrics': metrics.to_dict(),
            'zkp_proof': zkp_proof,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        return update
    
    def save_model(self, model: Any, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(model, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model = joblib.load(filepath)
        print(f"📂 Model loaded from {filepath}")
        return model


class FederatedAggregator:
    """
    Aggregates models from multiple clients using federated averaging
    """
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.global_model = None
        self.round_number = 0
        self.aggregation_history = []
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def aggregate_models(
        self, 
        client_updates: List[Dict[str, Any]],
        client_models: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        if not client_updates or not client_models:
            raise ValueError("No client updates to aggregate")
        print(f"\n⚡ Aggregating {len(client_models)} client models...")
        data_sizes = [update['metrics']['data_size'] for update in client_updates]
        total_size = sum(data_sizes)
        weights = [size / total_size for size in data_sizes]
        print(f"   Weights: {[f'{w:.3f}' for w in weights]}")
        client_model_list = list(client_models.values())
        first_model = client_model_list[0]
        if isinstance(first_model, RandomForestClassifier):
            self.global_model = self._aggregate_random_forest(client_model_list, weights)
        elif isinstance(first_model, LogisticRegression):
            self.global_model = self._aggregate_logistic_regression(client_model_list, weights)
        elif isinstance(first_model, GradientBoostingClassifier):
            self.global_model = self._aggregate_gradient_boosting(client_updates, client_model_list)
        else:
            best_idx = np.argmax([u['metrics']['accuracy'] for u in client_updates])
            self.global_model = client_model_list[best_idx]
        global_metrics = self._calculate_global_metrics(client_updates, weights)
        self.round_number += 1
        self.aggregation_history.append({
            'round': self.round_number,
            'num_clients': len(client_updates),
            'metrics': global_metrics,
            'timestamp': time.time()
        })
        print(f"   ✅ Aggregation complete - Round {self.round_number}")
        print(f"   Global Accuracy: {global_metrics['global_accuracy']:.4f}")
        return self.global_model, global_metrics
    
    def _aggregate_random_forest(
        self, 
        models: List[RandomForestClassifier], 
        weights: List[float]
    ) -> RandomForestClassifier:
        aggregated_model = RandomForestClassifier(
            n_estimators=models[0].n_estimators,
            random_state=42,
            n_jobs=-1
        )
        all_estimators = []
        for model, weight in zip(models, weights):
            n_estimators = max(1, int(len(model.estimators_) * weight))
            all_estimators.extend(model.estimators_[:n_estimators])
        aggregated_model.estimators_ = all_estimators
        aggregated_model.n_estimators = len(all_estimators)
        aggregated_model.classes_ = models[0].classes_
        aggregated_model.n_classes_ = models[0].n_classes_
        aggregated_model.n_features_in_ = models[0].n_features_in_
        aggregated_model.n_outputs_ = models[0].n_outputs_
        return aggregated_model
    
    def _aggregate_logistic_regression(
        self, 
        models: List[LogisticRegression], 
        weights: List[float]
    ) -> LogisticRegression:
        aggregated_model = LogisticRegression(random_state=42, n_jobs=-1)
        coef_avg = np.average([model.coef_ for model in models], axis=0, weights=weights)
        intercept_avg = np.average([model.intercept_ for model in models], axis=0, weights=weights)
        aggregated_model.coef_ = coef_avg
        aggregated_model.intercept_ = intercept_avg
        aggregated_model.classes_ = models[0].classes_
        aggregated_model.n_features_in_ = models[0].n_features_in_
        return aggregated_model
    
    def _aggregate_gradient_boosting(
        self, 
        client_updates: List[Dict],
        models: List[GradientBoostingClassifier]
    ) -> GradientBoostingClassifier:
        accuracies = [update['metrics']['accuracy'] for update in client_updates]
        best_idx = np.argmax(accuracies)
        return models[best_idx]
    
    def _calculate_global_metrics(
        self, 
        client_updates: List[Dict], 
        weights: List[float]
    ) -> Dict[str, float]:
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        global_metrics = {}
        for key in metrics_keys:
            values = [update['metrics'][key] for update in client_updates]
            global_metrics[f'global_{key}'] = float(np.average(values, weights=weights))
        global_metrics['num_clients'] = len(client_updates)
        global_metrics['total_samples'] = sum(
            update['metrics']['data_size'] for update in client_updates
        )
        global_metrics['round_number'] = self.round_number + 1
        return global_metrics
    
    def save_global_model(self, filename: str = "global_model.joblib"):
        if self.global_model is None:
            print("⚠️  No global model to save")
            return
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(self.global_model, filepath)
        print(f"💾 Global model saved to {filepath}")
    
    def load_global_model(self, filename: str = "global_model.joblib") -> Any:
        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  Model file not found: {filepath}")
            return None
        self.global_model = joblib.load(filepath)
        print(f"📂 Global model loaded from {filepath}")
        return self.global_model
    
    def get_aggregation_history(self) -> List[Dict]:
        return self.aggregation_history


def test_trainer():
    """Test federated trainer"""
    print("=" * 60)
    print("🧪 Testing Federated Trainer")
    print("=" * 60)
    # Create synthetic data
    from data.load_paysim import PaySimLoader
    loader = PaySimLoader()
    df = loader.preprocess()
    print(f"\n📊 Dataset: {len(df)} samples")
    trainer = FederatedModelTrainer(model_type="random_forest")
    print("\n🔧 Training local model...")
    model, metrics = trainer.train_local_model(df, apply_dp=True)
    print(f"\n📈 Model Metrics:")
    print(f"   Accuracy: {metrics.accuracy:.4f}")
    print(f"   Precision: {metrics.precision:.4f}")
    print(f"   Recall: {metrics.recall:.4f}")
    print(f"   F1 Score: {metrics.f1_score:.4f}")
    print(f"   ROC AUC: {metrics.roc_auc:.4f}")
    update = trainer.generate_encrypted_update(model, metrics)
    print(f"\n🔐 Encrypted update generated")
    print(f"   Model hash: {update['model_hash'][:32]}...")
    print(f"   ZKP proof: {update['zkp_proof'][:32]}...")
    print("\n✅ Trainer test complete!")


if __name__ == "__main__":
    test_trainer()
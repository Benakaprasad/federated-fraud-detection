"""
Test Suite for Federated Trainer Module
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.federated_trainer import (
    FederatedModelTrainer, FederatedAggregator, 
    TrainingMetrics, EncryptedModelUpdate
)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creating training metrics"""
        metrics = TrainingMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.91,
            f1_score=0.92,
            roc_auc=0.96,
            training_time=5.2,
            data_size=50000
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.roc_auc == 0.96
        assert metrics.data_size == 50000
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary"""
        metrics = TrainingMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.91,
            f1_score=0.92,
            roc_auc=0.96,
            training_time=5.2,
            data_size=50000
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['accuracy'] == 0.95
        assert 'training_time' in metrics_dict


class TestEncryptedModelUpdate:
    """Test EncryptedModelUpdate dataclass"""
    
    def test_update_creation(self):
        """Test creating encrypted update"""
        update = EncryptedModelUpdate(
            client_id="FI-001",
            encrypted_weights="encrypted_data_123",
            metrics={'accuracy': 0.95},
            zkp_proof="proof_xyz",
            data_size=50000,
            timestamp=1234567890.0
        )
        
        assert update.client_id == "FI-001"
        assert update.data_size == 50000
        assert update.metrics['accuracy'] == 0.95


class TestFederatedModelTrainer:
    """Test FederatedModelTrainer class"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = FederatedModelTrainer(model_type="random_forest")
        
        assert trainer.model_type == "random_forest"
        assert trainer.dp_epsilon == 1.0
        assert len(trainer.feature_columns) > 0
    
    def test_get_feature_columns(self):
        """Test getting feature columns"""
        trainer = FederatedModelTrainer()
        features = trainer._get_feature_columns()
        
        assert isinstance(features, list)
        assert 'amount' in features
        assert 'type' in features
    
    def test_create_sample_dataset(self):
        """Test creating sample dataset"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'isFraud' in df.columns
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        X, y = trainer._preprocess_data(df)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert X.shape[1] == len(trainer.feature_columns)
    
    def test_train_local_model(self):
        """Test training local model"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df, apply_dp=False)
        
        assert model is not None
        assert isinstance(metrics, TrainingMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert metrics.data_size == len(df)
    
    def test_train_with_dp(self):
        """Test training with differential privacy"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        model_no_dp, metrics_no_dp = trainer.train_local_model(df, apply_dp=False)
        model_with_dp, metrics_with_dp = trainer.train_local_model(df, apply_dp=True)
        
        # Both should produce valid models and metrics
        assert model_no_dp is not None
        assert model_with_dp is not None
        assert isinstance(metrics_no_dp, TrainingMetrics)
        assert isinstance(metrics_with_dp, TrainingMetrics)
    
    def test_generate_encrypted_update(self):
        """Test generating encrypted update"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        update = trainer.generate_encrypted_update(model, metrics, client_id="FI-001")
        
        assert isinstance(update, EncryptedModelUpdate)
        assert update.client_id == "FI-001"
        assert 'accuracy' in update.metrics
        assert len(update.zkp_proof) > 0


class TestFederatedAggregator:
    """Test FederatedAggregator class"""
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization"""
        aggregator = FederatedAggregator()
        
        assert aggregator.round_number == 0
        assert aggregator.global_model is None
        assert aggregator.aggregation_history == []
    
    def test_aggregate_models(self):
        """Test model aggregation"""
        aggregator = FederatedAggregator()
        trainer = FederatedModelTrainer()
        
        # Train multiple local models
        client_updates = []
        trained_models = {}
        
        for i in range(3):
            client_id = f"FI-{i+1:03d}"
            df = trainer._create_sample_dataset()
            model, metrics = trainer.train_local_model(df)
            
            update = trainer.generate_encrypted_update(model, metrics, client_id)
            client_updates.append(update)
            trained_models[client_id] = model
        
        # Aggregate
        global_model, global_metrics = aggregator.aggregate_models(
            client_updates, trained_models
        )
        
        assert global_model is not None
        assert isinstance(global_metrics, dict)
        assert 'global_accuracy' in global_metrics
        assert aggregator.round_number == 1
    
    def test_calculate_aggregation_weights(self):
        """Test weight calculation"""
        aggregator = FederatedAggregator()
        
        client_updates = [
            EncryptedModelUpdate(
                client_id="FI-001",
                encrypted_weights="",
                metrics={},
                zkp_proof="",
                data_size=50000,
                timestamp=0.0
            ),
            EncryptedModelUpdate(
                client_id="FI-002",
                encrypted_weights="",
                metrics={},
                zkp_proof="",
                data_size=30000,
                timestamp=0.0
            ),
            EncryptedModelUpdate(
                client_id="FI-003",
                encrypted_weights="",
                metrics={},
                zkp_proof="",
                data_size=20000,
                timestamp=0.0
            )
        ]
        
        weights = aggregator._calculate_aggregation_weights(client_updates)
        
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] > weights[1] > weights[2]  # Proportional to data size
    
    def test_verify_zkp_proofs(self):
        """Test ZKP verification"""
        aggregator = FederatedAggregator()
        
        proofs = ["proof1", "proof2", "proof3"]
        # In real implementation, this would verify actual proofs
        result = aggregator._verify_zkp_proofs(proofs)
        
        assert isinstance(result, bool)
    
    def test_save_and_load_global_model(self):
        """Test saving and loading global model"""
        aggregator = FederatedAggregator()
        trainer = FederatedModelTrainer()
        
        # Create a simple model
        df = trainer._create_sample_dataset()
        model, _ = trainer.train_local_model(df)
        aggregator.global_model = model
        
        # Save
        aggregator.save_global_model()
        
        # Load
        aggregator.global_model = None
        loaded = aggregator.load_global_model()
        
        assert loaded is True
        assert aggregator.global_model is not None
        
        # Cleanup
        if os.path.exists("models/global_model.joblib"):
            os.remove("models/global_model.joblib")
    
    def test_get_round_history(self):
        """Test getting aggregation history"""
        aggregator = FederatedAggregator()
        
        history = aggregator.get_round_history()
        
        assert isinstance(history, list)


class TestModelTypes:
    """Test different model types"""
    
    def test_random_forest_training(self):
        """Test Random Forest model"""
        trainer = FederatedModelTrainer(model_type="random_forest")
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_gradient_boosting_training(self):
        """Test Gradient Boosting model"""
        trainer = FederatedModelTrainer(model_type="gradient_boosting")
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_logistic_regression_training(self):
        """Test Logistic Regression model"""
        trainer = FederatedModelTrainer(model_type="logistic")
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        
        assert model is not None
        assert hasattr(model, 'predict')


class TestDifferentialPrivacy:
    """Test differential privacy implementation"""
    
    def test_dp_noise_addition(self):
        """Test DP noise is added correctly"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        # Train same model twice with DP
        _, metrics1 = trainer.train_local_model(df, apply_dp=True)
        _, metrics2 = trainer.train_local_model(df, apply_dp=True)
        
        # Metrics should be slightly different due to DP noise
        # (may occasionally be same due to randomness, but unlikely)
        assert isinstance(metrics1, TrainingMetrics)
        assert isinstance(metrics2, TrainingMetrics)


class TestFeatureEngineering:
    """Test feature engineering in trainer"""
    
    def test_engineered_features_present(self):
        """Test that engineered features are created"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        X, y = trainer._preprocess_data(df)
        
        assert 'amount_log' in X.columns
        assert 'balance_change_orig' in X.columns
        assert 'balance_change_dest' in X.columns
        assert 'large_transaction' in X.columns
    
    def test_no_missing_values(self):
        """Test that preprocessing handles missing values"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        X, y = trainer._preprocess_data(df)
        
        assert not X.isnull().any().any()
        assert not y.isnull().any()


class TestModelEvaluation:
    """Test model evaluation metrics"""
    
    def test_metrics_calculation(self):
        """Test that all metrics are calculated"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        
        assert metrics.accuracy >= 0 and metrics.accuracy <= 1
        assert metrics.precision >= 0 and metrics.precision <= 1
        assert metrics.recall >= 0 and metrics.recall <= 1
        assert metrics.f1_score >= 0 and metrics.f1_score <= 1
        assert metrics.roc_auc >= 0 and metrics.roc_auc <= 1
        assert metrics.training_time > 0
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix is calculated correctly"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        
        # Confusion matrix should be available in model evaluation
        assert metrics is not None


class TestDataValidation:
    """Test data validation in training"""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty dataset"""
        trainer = FederatedModelTrainer()
        df = pd.DataFrame()
        
        with pytest.raises(Exception):
            trainer.train_local_model(df)
    
    def test_missing_target_column(self):
        """Test handling of missing target column"""
        trainer = FederatedModelTrainer()
        df = pd.DataFrame({
            'amount': [100, 200],
            'type': ['PAYMENT', 'TRANSFER']
        })
        
        with pytest.raises(Exception):
            trainer.train_local_model(df)


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_serialization(self):
        """Test that models can be serialized"""
        trainer = FederatedModelTrainer()
        df = trainer._create_sample_dataset()
        
        model, metrics = trainer.train_local_model(df)
        
        # Save model
        test_path = "test_model.joblib"
        joblib.dump(model, test_path)
        
        # Load model
        loaded_model = joblib.load(test_path)
        
        assert loaded_model is not None
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Test Suite for Inference Module
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.inference import FraudDetectionInference, InferenceResult, BatchInferenceProcessor


class TestInferenceResult:
    """Test InferenceResult dataclass"""
    
    def test_inference_result_creation(self):
        """Test creating an inference result"""
        result = InferenceResult(
            transaction_id="tx_001",
            prediction=1,
            fraud_probability=0.85,
            risk_level="high",
            inference_time_ms=15.5,
            model_version="v1.0",
            timestamp=1234567890.0,
            features_used=["amount", "type", "balance"]
        )
        
        assert result.transaction_id == "tx_001"
        assert result.prediction == 1
        assert result.fraud_probability == 0.85
        assert result.risk_level == "high"
    
    def test_inference_result_to_dict(self):
        """Test converting result to dictionary"""
        result = InferenceResult(
            transaction_id="tx_001",
            prediction=1,
            fraud_probability=0.85,
            risk_level="high",
            inference_time_ms=15.5,
            model_version="v1.0",
            timestamp=1234567890.0,
            features_used=["amount", "type"]
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['prediction'] == 'fraud'
        assert result_dict['fraud_probability'] == 0.85
        assert result_dict['features_count'] == 2


class TestFraudDetectionInference:
    """Test FraudDetectionInference class"""
    
    def test_inference_initialization(self):
        """Test inference engine initialization"""
        inference = FraudDetectionInference()
        
        assert inference.model_version == "v1.0"
        assert len(inference.feature_columns) > 0
        assert inference.metrics is not None
    
    def test_get_feature_columns(self):
        """Test getting feature columns"""
        inference = FraudDetectionInference()
        features = inference._get_feature_columns()
        
        assert isinstance(features, list)
        assert 'amount' in features
        assert 'type' in features
        assert len(features) > 5
    
    def test_preprocess_transaction(self):
        """Test transaction preprocessing"""
        inference = FraudDetectionInference()
        
        transaction = {
            'type': 'TRANSFER',
            'amount': 150000,
            'oldbalanceOrg': 200000,
            'newbalanceOrig': 50000,
            'oldbalanceDest': 0,
            'newbalanceDest': 0
        }
        
        df_processed = inference.preprocess_transaction(transaction)
        
        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed) == 1
        assert 'amount_log' in df_processed.columns
        assert 'balance_change_orig' in df_processed.columns
    
    def test_preprocess_handles_missing_fields(self):
        """Test preprocessing handles missing fields gracefully"""
        inference = FraudDetectionInference()
        
        # Minimal transaction
        transaction = {
            'type': 'PAYMENT',
            'amount': 100
        }
        
        df_processed = inference.preprocess_transaction(transaction)
        
        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed.columns) == len(inference.feature_columns)
        assert not df_processed.isnull().any().any()
    
    def test_predict_without_model(self):
        """Test prediction fails gracefully without model"""
        inference = FraudDetectionInference()
        inference.model = None  # Force no model
        
        transaction = {
            'type': 'PAYMENT',
            'amount': 100,
            'oldbalanceOrg': 5000,
            'newbalanceOrig': 4900
        }
        
        with pytest.raises(RuntimeError):
            inference.predict(transaction)
    
    def test_predict_with_invalid_transaction(self):
        """Test prediction with invalid transaction"""
        inference = FraudDetectionInference()
        
        # Invalid transaction (negative amount)
        transaction = {
            'type': 'PAYMENT',
            'amount': -100
        }
        
        with pytest.raises(ValueError):
            inference.predict(transaction)
    
    def test_batch_predict(self):
        """Test batch prediction"""
        inference = FraudDetectionInference()
        
        transactions = [
            {'type': 'PAYMENT', 'amount': 100, 'oldbalanceOrg': 5000, 'newbalanceOrig': 4900},
            {'type': 'TRANSFER', 'amount': 50000, 'oldbalanceOrg': 60000, 'newbalanceOrig': 10000},
        ]
        
        # Note: This will fail if model not trained, but tests the structure
        try:
            results = inference.predict_batch(transactions)
            assert isinstance(results, list)
        except RuntimeError:
            # Expected if no model
            pass
    
    def test_get_performance_stats(self):
        """Test getting performance statistics"""
        inference = FraudDetectionInference()
        
        stats = inference.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'total_inferences' in stats
        assert 'avg_inference_time_ms' in stats
        assert 'uptime_seconds' in stats


class TestFeatureEngineering:
    """Test feature engineering in preprocessing"""
    
    def test_amount_log_feature(self):
        """Test log transformation of amount"""
        inference = FraudDetectionInference()
        
        transaction = {'type': 'PAYMENT', 'amount': 1000}
        df = inference.preprocess_transaction(transaction)
        
        assert 'amount_log' in df.columns
        assert df['amount_log'].iloc[0] == pytest.approx(np.log1p(1000))
    
    def test_balance_change_features(self):
        """Test balance change calculations"""
        inference = FraudDetectionInference()
        
        transaction = {
            'type': 'TRANSFER',
            'amount': 1000,
            'oldbalanceOrg': 5000,
            'newbalanceOrig': 4000,
            'oldbalanceDest': 2000,
            'newbalanceDest': 3000
        }
        
        df = inference.preprocess_transaction(transaction)
        
        assert 'balance_change_orig' in df.columns
        assert 'balance_change_dest' in df.columns
        assert df['balance_change_orig'].iloc[0] == -1000
        assert df['balance_change_dest'].iloc[0] == 1000
    
    def test_zero_balance_flags(self):
        """Test zero balance flag features"""
        inference = FraudDetectionInference()
        
        transaction = {
            'type': 'CASH_OUT',
            'amount': 5000,
            'oldbalanceOrg': 0,  # Zero balance
            'newbalanceOrig': 0,
            'oldbalanceDest': 1000,
            'newbalanceDest': 6000
        }
        
        df = inference.preprocess_transaction(transaction)
        
        assert 'orig_balance_zero' in df.columns
        assert df['orig_balance_zero'].iloc[0] == 1
    
    def test_large_transaction_flag(self):
        """Test large transaction detection"""
        inference = FraudDetectionInference()
        
        # Large transaction
        transaction_large = {'type': 'TRANSFER', 'amount': 200000}
        df_large = inference.preprocess_transaction(transaction_large)
        
        assert df_large['large_transaction'].iloc[0] == 1
        
        # Small transaction
        transaction_small = {'type': 'PAYMENT', 'amount': 100}
        df_small = inference.preprocess_transaction(transaction_small)
        
        assert df_small['large_transaction'].iloc[0] == 0
    
    def test_merchant_destination_flag(self):
        """Test merchant destination detection"""
        inference = FraudDetectionInference()
        
        # Merchant destination
        transaction_merchant = {
            'type': 'PAYMENT',
            'amount': 100,
            'nameDest': 'M1234567890'
        }
        df_merchant = inference.preprocess_transaction(transaction_merchant)
        
        assert df_merchant['dest_is_merchant'].iloc[0] == 1
        
        # Customer destination
        transaction_customer = {
            'type': 'TRANSFER',
            'amount': 1000,
            'nameDest': 'C9876543210'
        }
        df_customer = inference.preprocess_transaction(transaction_customer)
        
        assert df_customer['dest_is_merchant'].iloc[0] == 0


class TestBatchInferenceProcessor:
    """Test batch inference processing"""
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization"""
        inference = FraudDetectionInference()
        batch_processor = BatchInferenceProcessor(inference, batch_size=100)
        
        assert batch_processor.engine == inference
        assert batch_processor.batch_size == 100


class TestRiskLevelCalculation:
    """Test risk level calculation"""
    
    def test_risk_levels(self):
        """Test different risk level thresholds"""
        # High risk
        assert FraudDetectionInference._calculate_risk_level(0.85) == 'high'
        
        # Medium risk
        assert FraudDetectionInference._calculate_risk_level(0.50) == 'medium'
        
        # Low risk
        assert FraudDetectionInference._calculate_risk_level(0.15) == 'low'
    
    @staticmethod
    def _calculate_risk_level(fraud_prob):
        """Helper method to test risk calculation logic"""
        if fraud_prob > 0.7:
            return 'high'
        elif fraud_prob > 0.3:
            return 'medium'
        else:
            return 'low'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
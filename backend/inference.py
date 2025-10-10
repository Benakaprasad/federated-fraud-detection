"""
Inference Module - Real-time Fraud Detection
FIXED VERSION - Uses unified feature engineering
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import time
import os
from dataclasses import dataclass

from backend.utils.metrics import ModelMetrics, SystemMetrics
from backend.utils.validators import TransactionValidator, ModelValidator
from backend.utils.logger import get_logger

# ✅ IMPORT UNIFIED FEATURE ENGINEERING
from backend.feature_engineering import get_feature_engineer


@dataclass
class InferenceResult:
    """Result from fraud detection inference"""
    transaction_id: str
    prediction: int  # 0=legitimate, 1=fraud
    fraud_probability: float
    risk_level: str  # 'low', 'medium', 'high'
    inference_time_ms: float
    model_version: str
    timestamp: float
    features_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'transaction_id': self.transaction_id,
            'prediction': 'fraud' if self.prediction == 1 else 'legitimate',
            'fraud_probability': self.fraud_probability,
            'risk_level': self.risk_level,
            'inference_time_ms': self.inference_time_ms,
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            'features_count': len(self.features_used)
        }


class FraudDetectionInference:
    """
    Real-time fraud detection inference engine
    FIXED - Uses unified feature engineering for consistency
    """
    
    def __init__(self, model_path: str = "models/global_model.joblib"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.model_version = "v1.0"
        
        # ✅ USE UNIFIED FEATURE ENGINEERING
        self.feature_engineer = get_feature_engineer()
        self.feature_columns = self.feature_engineer.get_feature_names()
        
        self.logger = get_logger('inference')
        self.metrics = SystemMetrics()
        
        # Load model
        self.load_model()
    
    def preprocess_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """
        ✅ FIXED: Preprocess using unified feature engineering
        
        Args:
            transaction: Raw transaction data
        
        Returns:
            Preprocessed DataFrame with engineered features
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Apply unified feature engineering (same as training!)
        df_processed = self.feature_engineer.transform(df)
        
        return df_processed
    
    def load_model(self) -> bool:
        """Load trained model"""
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model not found at {self.model_path}")
            return False
        
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            # Verify model has expected features
            if hasattr(self.model, "feature_names_in_"):
                model_features = list(self.model.feature_names_in_)
                
                # Check if features match
                if model_features != self.feature_columns:
                    self.logger.warning(
                        f"⚠️  Model features ({len(model_features)}) differ from "
                        f"expected features ({len(self.feature_columns)})"
                    )
                    self.logger.info(f"Model features: {model_features[:5]}...")
                    self.logger.info(f"Expected features: {self.feature_columns[:5]}...")
                else:
                    self.logger.info(
                        f"✅ Model features match expected features "
                        f"({len(self.feature_columns)} features)"
                    )
            
            self.logger.info("✅ Model loaded successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, transaction: Dict[str, Any],
                transaction_id: Optional[str] = None) -> InferenceResult:
        """
        ✅ FIXED: Predict fraud with unified feature engineering
        
        Args:
            transaction: Transaction data
            transaction_id: Optional transaction ID
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        if transaction_id is None:
            transaction_id = f"tx_{int(time.time() * 1000)}"
        
        # Validate transaction
        is_valid, errors = TransactionValidator.validate_transaction(transaction)
        if not is_valid:
            self.logger.warning(f"Invalid transaction {transaction_id}: {errors}")
            raise ValueError(f"Invalid transaction: {errors}")
        
        if self.model is None:
            self.logger.error("Model not loaded")
            raise RuntimeError("Model not loaded. Please load a model first.")
        
        try:
            # ✅ Preprocess using unified feature engineering
            df_processed = self.preprocess_transaction(transaction)
            
            # ✅ Features are now guaranteed to match!
            # No need for manual alignment
            
            # Predict
            prediction = int(self.model.predict(df_processed)[0])
            proba = self.model.predict_proba(df_processed)[0]
            fraud_prob = float(proba[1])
            
            # Determine risk level
            if fraud_prob > 0.7:
                risk_level = "high"
            elif fraud_prob > 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000
            self.metrics.record_operation_time("inference", inference_time / 1000)
            
            # Create result
            result = InferenceResult(
                transaction_id=transaction_id,
                prediction=prediction,
                fraud_probability=fraud_prob,
                risk_level=risk_level,
                inference_time_ms=inference_time,
                model_version=self.model_version,
                timestamp=time.time(),
                features_used=list(df_processed.columns),
            )
            
            self.logger.info(
                f"✅ Transaction {transaction_id}: pred={prediction}, "
                f"fraud_prob={fraud_prob:.4f}, time={inference_time:.2f}ms"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Inference error for {transaction_id}: {e}")
            raise RuntimeError(f"Prediction error: {e}")
    
    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[InferenceResult]:
        """
        Batch prediction for multiple transactions
        
        Args:
            transactions: List of transactions
        
        Returns:
            List of inference results
        """
        self.logger.info(f"Batch prediction for {len(transactions)} transactions")
        
        results = []
        for i, tx in enumerate(transactions):
            tx_id = f"batch_tx_{i}_{int(time.time() * 1000)}"
            try:
                result = self.predict(tx, transaction_id=tx_id)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict transaction {tx_id}: {e}")
                # Continue with other transactions
        
        self.logger.info(f"Batch prediction complete: {len(results)}/{len(transactions)} successful")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get inference performance statistics
        
        Returns:
            Performance metrics
        """
        summary = self.metrics.get_summary()
        
        if 'inference' in summary['operations']:
            inference_stats = summary['operations']['inference']
            return {
                'total_inferences': inference_stats['count'],
                'avg_inference_time_ms': inference_stats['avg_time'] * 1000,
                'throughput_per_second': inference_stats['throughput'],
                'uptime_seconds': summary['uptime_seconds']
            }
        
        return {
            'total_inferences': 0,
            'avg_inference_time_ms': 0,
            'throughput_per_second': 0,
            'uptime_seconds': summary['uptime_seconds']
        }


class BatchInferenceProcessor:
    """Process large batches of transactions efficiently"""
    
    def __init__(self, inference_engine: FraudDetectionInference,
                 batch_size: int = 1000):
        """
        Initialize batch processor
        
        Args:
            inference_engine: Inference engine instance
            batch_size: Size of processing batches
        """
        self.engine = inference_engine
        self.batch_size = batch_size
        self.logger = get_logger('batch_inference')
    
    def process_file(self, file_path: str) -> Tuple[List[InferenceResult], Dict[str, Any]]:
        """
        Process transactions from a file
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            (results, summary_stats)
        """
        self.logger.info(f"Processing file: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded {len(df):,} transactions")
        
        # Convert to list of dicts
        transactions = df.to_dict('records')
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(transactions), self.batch_size):
            batch = transactions[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(transactions)-1)//self.batch_size + 1}")
            
            results = self.engine.predict_batch(batch)
            all_results.extend(results)
        
        # Calculate summary statistics
        fraud_count = sum(1 for r in all_results if r.prediction == 1)
        avg_inference_time = np.mean([r.inference_time_ms for r in all_results])
        
        summary = {
            'total_processed': len(all_results),
            'fraud_detected': fraud_count,
            'fraud_rate': fraud_count / len(all_results) if all_results else 0,
            'avg_inference_time_ms': avg_inference_time,
            'total_time_seconds': sum(r.inference_time_ms for r in all_results) / 1000
        }
        
        self.logger.info(f"✅ Batch processing complete")
        self.logger.info(f"   Fraud detected: {fraud_count}/{len(all_results)}")
        self.logger.info(f"   Avg inference time: {avg_inference_time:.2f}ms")
        
        return all_results, summary


def test_inference():
    """Test inference engine"""
    print("=" * 60)
    print("🔍 Testing Fraud Detection Inference (FIXED VERSION)")
    print("=" * 60)
    
    # Create inference engine
    inference = FraudDetectionInference()
    
    # Test single transaction
    print("\n1. Single Transaction Inference:")
    
    test_transaction = {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 250000,
        'oldbalanceOrg': 300000,
        'newbalanceOrig': 50000,
        'oldbalanceDest': 0,
        'newbalanceDest': 0,
        'nameDest': 'C1234567890'
    }
    
    try:
        result = inference.predict(test_transaction)
        print(f"   Transaction ID: {result.transaction_id}")
        print(f"   Prediction: {result.prediction} ({'fraud' if result.prediction == 1 else 'legitimate'})")
        print(f"   Fraud Probability: {result.fraud_probability:.4f}")
        print(f"   Risk Level: {result.risk_level}")
        print(f"   Inference Time: {result.inference_time_ms:.2f}ms")
        print(f"   Features Used: {len(result.features_used)}")
    except Exception as e:
        print(f"   ⚠️  Inference failed (model might not be trained yet): {e}")
    
    # Test batch inference
    print("\n2. Batch Inference:")
    
    batch_transactions = [
        {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 100,
            'oldbalanceOrg': 5000,
            'newbalanceOrig': 4900,
            'oldbalanceDest': 1000,
            'newbalanceDest': 1100,
            'nameDest': 'M123456'
        },
        {
            'step': 2,
            'type': 'CASH_OUT',
            'amount': 50000,
            'oldbalanceOrg': 60000,
            'newbalanceOrig': 10000,
            'oldbalanceDest': 0,
            'newbalanceDest': 0,
            'nameDest': 'C789012'
        },
        {
            'step': 3,
            'type': 'TRANSFER',
            'amount': 150000,
            'oldbalanceOrg': 200000,
            'newbalanceOrig': 50000,
            'oldbalanceDest': 0,
            'newbalanceDest': 0,
            'nameDest': 'C345678'
        }
    ]
    
    try:
        results = inference.predict_batch(batch_transactions)
        print(f"   Processed {len(results)} transactions")
        fraud_count = sum(1 for r in results if r.prediction == 1)
        print(f"   Fraud detected: {fraud_count}")
        print(f"   Avg inference time: {np.mean([r.inference_time_ms for r in results]):.2f}ms")
    except Exception as e:
        print(f"   ⚠️  Batch inference failed: {e}")
    
    # Performance stats
    print("\n3. Performance Statistics:")
    stats = inference.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Inference test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_inference()
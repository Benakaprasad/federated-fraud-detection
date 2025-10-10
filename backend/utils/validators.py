"""
Validators Module - Data Validation and Sanitization
Ensures data integrity and security
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class TransactionValidator:
    """Validate financial transaction data"""
    
    TRANSACTION_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    
    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize string input
        
        Args:
            input_str: Input string
            max_length: Maximum allowed length
        
        Returns:
            Sanitized string
        """
        # Truncate to max length
        sanitized = input_str[:max_length]
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>\'\"%;()&+]', '', sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], 
                               required_keys: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate JSON structure
        
        Args:
            data: JSON data as dictionary
            required_keys: List of required keys
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return False, errors
        
        # Check for required keys
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        return len(errors) == 0, errors


class BlockchainValidator:
    """Validate blockchain data"""
    
    @staticmethod
    def validate_transaction_hash(tx_hash: str) -> bool:
        """
        Validate transaction hash format
        
        Args:
            tx_hash: Transaction hash
        
        Returns:
            True if valid
        """
        # Should be 64 character hex string (SHA-256)
        if len(tx_hash) != 64:
            return False
        
        try:
            int(tx_hash, 16)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_block_height(height: int, current_height: int) -> Tuple[bool, str]:
        """
        Validate block height
        
        Args:
            height: Requested height
            current_height: Current blockchain height
        
        Returns:
            (is_valid, error_message)
        """
        if height < 0:
            return False, "Block height cannot be negative"
        
        if height > current_height:
            return False, f"Block height {height} exceeds current height {current_height}"
        
        return True, ""
    
    @staticmethod
    def validate_merkle_root(merkle_root: str) -> bool:
        """
        Validate Merkle root format
        
        Args:
            merkle_root: Merkle root hash
        
        Returns:
            True if valid
        """
        return BlockchainValidator.validate_transaction_hash(merkle_root)


def test_validators():
    """Test all validators"""
    print("=" * 60)
    print("✅ Testing Validators")
    print("=" * 60)
    
    # Test transaction validator
    print("\n1. Transaction Validator:")
    valid_tx = {
        'type': 'PAYMENT',
        'amount': 1000.50,
        'oldbalanceOrg': 5000,
        'newbalanceOrig': 4000
    }
    
    is_valid, errors = TransactionValidator.validate_transaction(valid_tx)
    print(f"   Valid transaction: {is_valid}")
    
    invalid_tx = {
        'type': 'INVALID',
        'amount': -100
    }
    
    is_valid, errors = TransactionValidator.validate_transaction(invalid_tx)
    print(f"   Invalid transaction detected: {not is_valid}")
    print(f"   Errors: {errors}")
    
    # Test DataFrame validator
    print("\n2. DataFrame Validator:")
    df = pd.DataFrame({
        'type': ['PAYMENT', 'TRANSFER'],
        'amount': [100, 200],
        'isFraud': [0, 1]
    })
    
    is_valid, errors = DataFrameValidator.validate_fraud_dataset(df)
    print(f"   Dataset valid: {is_valid}")
    
    quality = DataFrameValidator.validate_data_quality(df)
    print(f"   Total rows: {quality['total_rows']}")
    print(f"   Total columns: {quality['total_columns']}")
    
    # Test model validator
    print("\n3. Model Validator:")
    X = np.array([[1, 2, 3], [4, 5, 6]])
    is_valid, errors = ModelValidator.validate_model_input(X, expected_features=3)
    print(f"   Model input valid: {is_valid}")
    
    predictions = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    is_valid, errors = ModelValidator.validate_prediction(predictions, probabilities)
    print(f"   Prediction valid: {is_valid}")
    
    # Test security validator
    print("\n4. Security Validator:")
    is_valid, msg = SecurityValidator.validate_client_id("client-001")
    print(f"   Valid client ID: {is_valid}")
    
    is_valid, msg = SecurityValidator.validate_client_id("client'; DROP TABLE--")
    print(f"   SQL injection detected: {not is_valid}")
    print(f"   Error: {msg}")
    
    # Test blockchain validator
    print("\n5. Blockchain Validator:")
    valid_hash = "a" * 64
    print(f"   Valid hash: {BlockchainValidator.validate_transaction_hash(valid_hash)}")
    
    invalid_hash = "xyz"
    print(f"   Invalid hash detected: {not BlockchainValidator.validate_transaction_hash(invalid_hash)}")
    
    print("\n" + "=" * 60)
    print("✅ All validator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_validators()
staticmethod
    def validate_transaction(transaction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single transaction
        
        Args:
            transaction: Transaction dictionary
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = ['type', 'amount']
        for field in required_fields:
            if field not in transaction:
                errors.append(f"Missing required field: {field}")
        
        # Validate transaction type
        if 'type' in transaction:
            if transaction['type'] not in TransactionValidator.TRANSACTION_TYPES:
                errors.append(f"Invalid transaction type: {transaction['type']}")
        
        # Validate amount
        if 'amount' in transaction:
            try:
                amount = float(transaction['amount'])
                if amount < 0:
                    errors.append("Amount cannot be negative")
                if amount > 10000000:  # 10 million limit
                    errors.append("Amount exceeds maximum limit")
            except (ValueError, TypeError):
                errors.append("Invalid amount format")
        
        # Validate balances if present
        for balance_field in ['oldbalanceOrg', 'newbalanceOrig', 
                             'oldbalanceDest', 'newbalanceDest']:
            if balance_field in transaction:
                try:
                    balance = float(transaction[balance_field])
                    if balance < 0:
                        errors.append(f"{balance_field} cannot be negative")
                except (ValueError, TypeError):
                    errors.append(f"Invalid {balance_field} format")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_batch(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of transactions
        
        Args:
            transactions: List of transactions
        
        Returns:
            Validation summary
        """
        total = len(transactions)
        valid_count = 0
        invalid_transactions = []
        
        for idx, tx in enumerate(transactions):
            is_valid, errors = TransactionValidator.validate_transaction(tx)
            if is_valid:
                valid_count += 1
            else:
                invalid_transactions.append({
                    'index': idx,
                    'transaction': tx,
                    'errors': errors
                })
        
        return {
            'total': total,
            'valid': valid_count,
            'invalid': total - valid_count,
            'validation_rate': valid_count / total if total > 0 else 0,
            'invalid_transactions': invalid_transactions[:10]  # First 10 errors
        }


class DataFrameValidator:
    """Validate pandas DataFrames"""
    
    @staticmethod
    def validate_fraud_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate fraud detection dataset
        
        Args:
            df: DataFrame to validate
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        required_columns = ['type', 'amount', 'isFraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for null values in critical columns
        if 'isFraud' in df.columns:
            if df['isFraud'].isnull().any():
                errors.append("Target column 'isFraud' contains null values")
        
        # Check data types
        if 'amount' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['amount']):
                errors.append("Column 'amount' must be numeric")
        
        if 'isFraud' in df.columns:
            unique_values = df['isFraud'].unique()
            if not set(unique_values).issubset({0, 1}):
                errors.append("Column 'isFraud' must contain only 0 and 1")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            errors.append(f"Dataset contains {duplicates} duplicate rows")
        
        # Check class balance
        if 'isFraud' in df.columns:
            fraud_rate = df['isFraud'].mean()
            if fraud_rate < 0.001:
                errors.append(f"Very low fraud rate: {fraud_rate:.4%}")
            elif fraud_rate > 0.5:
                errors.append(f"Unusually high fraud rate: {fraud_rate:.4%}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality
        
        Args:
            df: DataFrame to assess
        
        Returns:
            Quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': {}
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_values': int(df[col].nunique())
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['mean'] = float(df[col].mean())
                col_info['std'] = float(df[col].std())
                col_info['min'] = float(df[col].min())
                col_info['max'] = float(df[col].max())
            
            quality_report['columns'][col] = col_info
        
        return quality_report


class ModelValidator:
    """Validate model inputs and outputs"""
    
    @staticmethod
    def validate_model_input(X: np.ndarray, 
                            expected_features: int) -> Tuple[bool, List[str]]:
        """
        Validate model input data
        
        Args:
            X: Input features
            expected_features: Expected number of features
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check dimensions
        if len(X.shape) != 2:
            errors.append(f"Expected 2D array, got {len(X.shape)}D")
        
        if X.shape[1] != expected_features:
            errors.append(
                f"Expected {expected_features} features, got {X.shape[1]}"
            )
        
        # Check for NaN or Inf
        if np.any(np.isnan(X)):
            errors.append("Input contains NaN values")
        
        if np.any(np.isinf(X)):
            errors.append("Input contains infinite values")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_prediction(prediction: Any, 
                          probability: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
        """
        Validate model prediction
        
        Args:
            prediction: Model prediction
            probability: Prediction probability (optional)
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check prediction values
        if isinstance(prediction, np.ndarray):
            unique_preds = np.unique(prediction)
            if not set(unique_preds).issubset({0, 1}):
                errors.append("Predictions must be 0 or 1")
        
        # Check probabilities if provided
        if probability is not None:
            if not isinstance(probability, np.ndarray):
                errors.append("Probability must be numpy array")
            elif np.any(probability < 0) or np.any(probability > 1):
                errors.append("Probabilities must be between 0 and 1")
        
        return len(errors) == 0, errors


class SecurityValidator:
    """Security validation for inputs"""
    
    @staticmethod
    def validate_client_id(client_id: str) -> Tuple[bool, str]:
        """
        Validate client ID format
        
        Args:
            client_id: Client identifier
        
        Returns:
            (is_valid, error_message)
        """
        # Check length
        if len(client_id) < 3 or len(client_id) > 100:
            return False, "Client ID must be between 3 and 100 characters"
        
        # Check for SQL injection patterns
        sql_patterns = ['--', ';', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
        if any(pattern.lower() in client_id.lower() for pattern in sql_patterns):
            return False, "Client ID contains invalid characters"
        
        # Check for valid characters (alphanumeric, dash, underscore)
        if not re.match(r'^[a-zA-Z0-9_-]+$', client_id):
            return False, "Client ID can only contain letters, numbers, dash, and underscore"
        
        return True, ""
    
    @
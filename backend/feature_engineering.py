"""
Unified Feature Engineering for Fraud Detection
Ensures consistency between training and inference
"""

import numpy as np
import pandas as pd
from typing import List


class FraudFeatureEngineer:
    """
    Centralized feature engineering for fraud detection
    Used by both training and inference pipelines
    """
    
    def __init__(self):
        """Initialize feature engineer with column definitions"""
        # Base columns from raw data
        self.base_columns = [
            'step', 'type', 'amount', 
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ]
        
        # Transaction types for one-hot encoding
        self.transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        
        # Final feature set (in order)
        self.feature_columns = None  # Will be set after first transform
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to dataframe
        
        Args:
            df: Raw transaction dataframe
            
        Returns:
            Processed dataframe with engineered features
        """
        df_processed = df.copy()
        
        # === 1. Log Transform Amount ===
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        
        # === 2. Balance Changes ===
        df_processed['balance_change_orig'] = (
            df_processed['newbalanceOrig'] - df_processed['oldbalanceOrg']
        )
        df_processed['balance_change_dest'] = (
            df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
        )
        
        # === 3. Zero Balance Flags ===
        df_processed['orig_balance_zero'] = (df_processed['oldbalanceOrg'] == 0).astype(int)
        df_processed['dest_balance_zero'] = (df_processed['oldbalanceDest'] == 0).astype(int)
        
        # === 4. Large Transaction Flag ===
        # Threshold at 100k (can be adjusted)
        df_processed['large_transaction'] = (df_processed['amount'] > 100000).astype(int)
        
        # === 5. Merchant Destination ===
        if 'nameDest' in df_processed.columns:
            df_processed['dest_is_merchant'] = (
                df_processed['nameDest'].str.startswith('M').fillna(False).astype(int)
            )
        else:
            df_processed['dest_is_merchant'] = 0
        
        # === 6. One-Hot Encode Transaction Type ===
        if 'type' in df_processed.columns:
            type_dummies = pd.get_dummies(df_processed['type'], prefix='type')
            
            # Ensure all transaction types exist
            for tx_type in self.transaction_types:
                col_name = f'type_{tx_type}'
                if col_name not in type_dummies.columns:
                    type_dummies[col_name] = 0
            
            # Merge with main dataframe
            df_processed = pd.concat([df_processed, type_dummies], axis=1)
        else:
            # Add dummy columns if type is missing
            for tx_type in self.transaction_types:
                df_processed[f'type_{tx_type}'] = 0
        
        # === 7. Define Final Feature Set ===
        if self.feature_columns is None:
            self.feature_columns = [
                # Base features
                'step', 'amount', 'amount_log',
                'oldbalanceOrg', 'newbalanceOrig', 'balance_change_orig',
                'oldbalanceDest', 'newbalanceDest', 'balance_change_dest',
                # Flags
                'orig_balance_zero', 'dest_balance_zero',
                'large_transaction', 'dest_is_merchant',
                # Transaction types (one-hot encoded)
                'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
                'type_PAYMENT', 'type_TRANSFER'
            ]
        
        # === 8. Handle Missing Values ===
        df_processed = df_processed.fillna(0)
        df_processed = df_processed.replace([np.inf, -np.inf], 0)
        
        # === 9. Ensure All Features Exist ===
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # === 10. Select Only Required Features ===
        df_processed = df_processed[self.feature_columns]
        
        return df_processed
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        if self.feature_columns is None:
            # Initialize with default features
            self.feature_columns = [
                'step', 'amount', 'amount_log',
                'oldbalanceOrg', 'newbalanceOrig', 'balance_change_orig',
                'oldbalanceDest', 'newbalanceDest', 'balance_change_dest',
                'orig_balance_zero', 'dest_balance_zero',
                'large_transaction', 'dest_is_merchant',
                'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
                'type_PAYMENT', 'type_TRANSFER'
            ]
        
        return self.feature_columns
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Check if dataframe has all required base columns
        
        Args:
            df: Input dataframe
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest']
        
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️  Missing required columns: {missing}")
            return False
        
        return True


# Singleton instance for shared use
_feature_engineer = FraudFeatureEngineer()


def get_feature_engineer() -> FraudFeatureEngineer:
    """Get the global feature engineer instance"""
    return _feature_engineer


def test_feature_engineering():
    """Test feature engineering pipeline"""
    print("=" * 60)
    print("🧪 Testing Feature Engineering")
    print("=" * 60)
    
    # Create sample transaction
    sample_data = {
        'step': [1, 2, 3],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
        'amount': [9839.64, 181.00, 181.00],
        'oldbalanceOrg': [170136.0, 181.0, 181.0],
        'newbalanceOrig': [160296.36, 0.0, 0.0],
        'oldbalanceDest': [0.0, 0.0, 21182.0],
        'newbalanceDest': [0.0, 0.0, 0.0],
        'nameDest': ['M1234', 'C5678', 'C9012']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\n📊 Original Data:")
    print(df.head())
    
    # Apply feature engineering
    engineer = get_feature_engineer()
    df_processed = engineer.transform(df)
    
    print("\n🔧 Engineered Features:")
    print(f"   Total features: {len(df_processed.columns)}")
    print(f"   Feature names: {df_processed.columns.tolist()}")
    
    print("\n📈 Sample Processed Data:")
    print(df_processed.head())
    
    # Validate
    is_valid = engineer.validate_features(df)
    print(f"\n✅ Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    print("\n" + "=" * 60)
    print("✅ Feature engineering test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_engineering()


# ============================================================
# UPDATED TRAINER INTEGRATION
# ============================================================

"""
Apply this patch to federated_trainer.py:

1. Add import at top:
   from backend.feature_engineering import get_feature_engineer

2. Replace _prepare_features method:
   
   def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
       '''Prepare features using unified feature engineering'''
       engineer = get_feature_engineer()
       
       # Validate input
       if not engineer.validate_features(df):
           raise ValueError("Missing required columns in input data")
       
       # Transform features
       df_processed = engineer.transform(df)
       
       # Update feature columns
       self.feature_columns = engineer.get_feature_names()
       
       return df_processed

3. Remove old feature columns definition from __init__:
   # DELETE these lines:
   self.feature_columns = [...]
   self.type_columns = [...]
"""


# ============================================================
# UPDATED INFERENCE INTEGRATION
# ============================================================

"""
Apply this patch to inference.py:

1. Add import at top:
   from backend.feature_engineering import get_feature_engineer

2. Update __init__ method:
   
   def __init__(self, model_path: str = "models/global_model.joblib"):
       self.model_path = model_path
       self.model = None
       self.model_version = "v1.0"
       
       # Use unified feature engineering
       self.feature_engineer = get_feature_engineer()
       self.feature_columns = self.feature_engineer.get_feature_names()
       
       self.logger = get_logger('inference')
       self.metrics = SystemMetrics()
       self.load_model()

3. Replace _get_feature_columns method:
   # DELETE this entire method - no longer needed

4. Replace preprocess_transaction method:
   
   def preprocess_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
       '''Preprocess using unified feature engineering'''
       # Convert to DataFrame
       df = pd.DataFrame([transaction])
       
       # Apply unified feature engineering
       df_processed = self.feature_engineer.transform(df)
       
       return df_processed

5. Update predict method - remove feature alignment code:
   
   def predict(self, transaction: Dict[str, Any],
               transaction_id: Optional[str] = None) -> InferenceResult:
       # ... (validation code stays same)
       
       try:
           # Preprocess using unified pipeline
           df_processed = self.preprocess_transaction(transaction)
           
           # Predict (features already aligned!)
           prediction = int(self.model.predict(df_processed)[0])
           proba = self.model.predict_proba(df_processed)[0]
           fraud_prob = float(proba[1])
           
           # ... (rest stays same)
"""
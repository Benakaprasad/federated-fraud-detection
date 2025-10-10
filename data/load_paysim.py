"""
PaySim Data Loader - Load and preprocess financial transaction data
Handles data loading, preprocessing, and client data distribution
UPDATED: Works with real Kaggle PaySim dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import sys


class PaySimLoader:
    """Load and preprocess PaySim synthetic financial transactions dataset"""
    
    def __init__(self, data_path: str = "data/paysim.csv"):
        self.data_path = data_path
        self.df = None
        self.preprocessed = False
    
    def load_data(self) -> pd.DataFrame:
        """
        Load PaySim dataset
        If file doesn't exist, generate synthetic data
        """
        if os.path.exists(self.data_path):
            print(f"📂 Loading data from {self.data_path}...")
            try:
                self.df = pd.read_csv(self.data_path)
                
                # Handle different column name formats from Kaggle
                # Some versions use lowercase, some use camelCase
                column_mapping = {
                    'isfraud': 'isFraud',
                    'isflaggedfraud': 'isFlaggedFraud',
                    'nameorg': 'nameOrig',
                    'namedest': 'nameDest',
                    'oldbalanceorg': 'oldbalanceOrg',
                    'newbalanceorg': 'newbalanceOrig',
                    'oldbalancedest': 'oldbalanceDest',
                    'newbalancedest': 'newbalanceDest'
                }
                
                # Rename columns if needed (case-insensitive)
                self.df.columns = self.df.columns.str.strip()  # Remove whitespace
                
                for old_col, new_col in column_mapping.items():
                    for df_col in self.df.columns:
                        if df_col.lower() == old_col.lower() and df_col != new_col:
                            self.df.rename(columns={df_col: new_col}, inplace=True)
                
                print(f"   Loaded {len(self.df):,} transactions")
                print(f"   Columns: {list(self.df.columns)}")
                
                # Verify essential columns exist
                required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                               'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
                               'newbalanceDest', 'isFraud']
                
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    print(f"   ⚠️  Missing columns: {missing_cols}")
                    print(f"   Available columns: {list(self.df.columns)}")
                    print(f"   Generating synthetic data instead...")
                    self.df = self._generate_synthetic_fallback()
                else:
                    print(f"   ✅ All required columns found")
                    
            except Exception as e:
                print(f"   ❌ Error loading CSV: {e}")
                print(f"   Generating synthetic data instead...")
                self.df = self._generate_synthetic_fallback()
        else:
            print(f"⚠️  Data file not found at: {self.data_path}")
            print(f"   Generating synthetic data...")
            self.df = self._generate_synthetic_fallback()
        
        return self.df
    
    def _generate_synthetic_fallback(self, n_samples: int = 100000) -> pd.DataFrame:
        """
        Generate synthetic transaction data similar to PaySim
        
        Args:
            n_samples: Number of transactions to generate
        
        Returns:
            Synthetic DataFrame
        """
        np.random.seed(42)
        
        print(f"🔧 Generating {n_samples:,} synthetic transactions...")
        
        # Transaction types
        types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        type_probs = [0.30, 0.25, 0.25, 0.10, 0.10]
        
        # Generate data
        data = {
            'step': np.random.randint(1, 744, n_samples),  # 744 hours in a month
            'type': np.random.choice(types, n_samples, p=type_probs),
            'amount': np.random.lognormal(5, 2, n_samples),  # Log-normal for realistic amounts
            'nameOrig': [f'C{i:010d}' for i in np.random.randint(0, n_samples // 10, n_samples)],
            'oldbalanceOrg': np.random.lognormal(8, 2, n_samples),
            'newbalanceOrig': np.zeros(n_samples),
            'nameDest': [f'C{i:010d}' if np.random.random() > 0.4 else f'M{i:010d}' 
                        for i in np.random.randint(0, n_samples // 10, n_samples)],
            'oldbalanceDest': np.random.lognormal(8, 2, n_samples),
            'newbalanceDest': np.zeros(n_samples),
            'isFraud': np.zeros(n_samples, dtype=int),
            'isFlaggedFraud': np.zeros(n_samples, dtype=int)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate new balances
        df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
        df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
        
        # Make negative balances zero (not allowed)
        df.loc[df['newbalanceOrig'] < 0, 'newbalanceOrig'] = 0
        
        # Generate fraud cases (3-5% of transactions)
        fraud_rate = 0.04
        n_fraud = int(n_samples * fraud_rate)
        
        # Fraud patterns:
        # 1. Large transfers with balance anomalies
        # 2. Cash out with zero ending balance
        
        fraud_indices = np.random.choice(
            df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].index,
            size=min(n_fraud, len(df[df['type'].isin(['TRANSFER', 'CASH_OUT'])])),
            replace=False
        )
        
        df.loc[fraud_indices, 'isFraud'] = 1
        
        # Fraud pattern: Zero out destination balance
        df.loc[fraud_indices, 'oldbalanceDest'] = 0
        df.loc[fraud_indices, 'newbalanceDest'] = 0
        
        # High value frauds
        df.loc[fraud_indices, 'amount'] = np.random.lognormal(12, 1, len(fraud_indices))
        
        # Flag large frauds
        large_fraud_mask = (df['isFraud'] == 1) & (df['amount'] > 200000)
        df.loc[large_fraud_mask, 'isFlaggedFraud'] = 1
        
        print(f"   ✅ Generated {len(df):,} transactions")
        print(f"   Fraud rate: {df['isFraud'].mean()*100:.2f}%")
        
        return df
    
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the dataset with feature engineering
        
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            self.load_data()
        
        if self.preprocessed:
            return self.df
        
        print("🔧 Preprocessing data...")
        
        # Create copy to avoid modifying original
        df = self.df.copy()
        
        # One-hot encode transaction type
        if 'type' in df.columns:
            type_dummies = pd.get_dummies(df['type'], prefix='type')
            df = pd.concat([df, type_dummies], axis=1)
        
        # Feature engineering
        # 1. Log transform of amount (handle zeros)
        df['amount_log'] = np.log1p(df['amount'])
        
        # 2. Balance changes
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # 3. Zero balance flags
        df['orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
        df['dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
        
        # 4. Large transaction flag
        threshold = df['amount'].quantile(0.95)
        df['large_transaction'] = (df['amount'] > threshold).astype(int)
        
        # 5. Merchant destination flag
        if 'nameDest' in df.columns:
            df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        
        # Handle missing values
        df.fillna(0, inplace=True)
        
        # Remove infinite values
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        self.df = df
        self.preprocessed = True
        
        print(f"   ✅ Preprocessing complete")
        print(f"   Features: {len(df.columns)}")
        print(f"   Samples: {len(df):,}")
        print(f"   Fraud transactions: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
        
        return df
    
    def split_by_client(self, num_clients: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Split data into client datasets (simulating different banks)
        
        Args:
            num_clients: Number of clients to create
        
        Returns:
            Dictionary mapping client_id to DataFrame
        """
        if self.df is None or not self.preprocessed:
            self.preprocess()
        
        print(f"\n📊 Splitting data into {num_clients} client datasets...")
        
        # Shuffle data
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into roughly equal parts
        splits = np.array_split(df_shuffled, num_clients)
        
        client_datasets = {}
        
        for i, split_df in enumerate(splits):
            client_id = f"FI-{i+1:03d}"
            client_name = [
                "Bank_Alpha", "Bank_Beta", "Fintech_Gamma", 
                "Credit_Union_Delta", "Digital_Bank_Epsilon"
            ][i % 5]
            
            client_datasets[client_id] = split_df.copy()
            
            fraud_rate = split_df['isFraud'].mean()
            
            print(f"   {client_id} ({client_name}):")
            print(f"      Samples: {len(split_df):,}")
            print(f"      Fraud rate: {fraud_rate*100:.2f}%")
        
        print(f"   ✅ Data split complete")
        
        return client_datasets
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_transactions': len(self.df),
            'fraud_transactions': int(self.df['isFraud'].sum()),
            'fraud_rate': float(self.df['isFraud'].mean()),
            'transaction_types': self.df['type'].value_counts().to_dict() if 'type' in self.df.columns else {},
            'amount_stats': {
                'mean': float(self.df['amount'].mean()),
                'median': float(self.df['amount'].median()),
                'std': float(self.df['amount'].std()),
                'min': float(self.df['amount'].min()),
                'max': float(self.df['amount'].max())
            }
        }
        
        return stats
    
    def get_sample_batch(self, n: int = 100) -> pd.DataFrame:
        """
        Get a sample batch for testing
        
        Args:
            n: Number of samples
        
        Returns:
            Sample DataFrame
        """
        if self.df is None or not self.preprocessed:
            self.preprocess()
        
        return self.df.sample(n=min(n, len(self.df)), random_state=42)


def test_loader():
    """Test data loader"""
    print("=" * 60)
    print("📊 Testing PaySim Data Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = PaySimLoader()
    
    # Load and preprocess data
    df = loader.preprocess()
    
    print(f"\n📈 Dataset Statistics:")
    stats = loader.get_statistics()
    print(f"   Total Transactions: {stats['total_transactions']:,}")
    print(f"   Fraud Transactions: {stats['fraud_transactions']:,}")
    print(f"   Fraud Rate: {stats['fraud_rate']*100:.2f}%")
    
    print(f"\n💰 Amount Statistics:")
    print(f"   Mean: ${stats['amount_stats']['mean']:,.2f}")
    print(f"   Median: ${stats['amount_stats']['median']:,.2f}")
    print(f"   Max: ${stats['amount_stats']['max']:,.2f}")
    
    # Split into clients
    client_datasets = loader.split_by_client(num_clients=3)
    
    print(f"\n✅ Test complete!")
    
    return loader, client_datasets


if __name__ == "__main__":
    loader, datasets = test_loader()
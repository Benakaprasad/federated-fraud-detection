import pandas as pd
import sys
import os

# Ensure backend is in the path for import
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from feature_engineering import get_feature_engineer

# Load your CSV
df = pd.read_csv(os.path.join('data', 'paysim.csv'))
print(f"✅ Loaded CSV: {len(df)} rows")
print(f"📊 Columns: {list(df.columns)}")

# Apply feature engineering
engineer = get_feature_engineer()
df_processed = engineer.transform(df.head(10))

print(f"\n🔧 After feature engineering:")
print(f"   Total features: {len(df_processed.columns)}")
print(f"   Feature names ({len(df_processed.columns)}):")
for i, col in enumerate(df_processed.columns, 1):
    print(f"      {i}. {col}")

# Save feature list
import joblib
os.makedirs('models', exist_ok=True)
feature_list = df_processed.columns.tolist()
joblib.dump(feature_list, os.path.join('models', 'feature_names.joblib'))
print(f"\n💾 Saved {len(feature_list)} feature names to models/feature_names.joblib")
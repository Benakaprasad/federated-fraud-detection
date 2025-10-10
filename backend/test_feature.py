"""
Feature Engineering Consistency Test
Ensures the feature engineering pipeline produces the expected features and values.
"""

import pandas as pd
import numpy as np
import sys

try:
    from backend.feature_engineering import get_feature_engineer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure backend/feature_engineering.py exists and is importable.")
    sys.exit(1)


class FeatureEngineeringTester:
    """Test suite for unified feature engineering"""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def test(self, name: str, condition: bool, message: str = ""):
        status = "✅ PASS" if condition else "❌ FAIL"
        result = f"{status} - {name}"
        if message:
            result += f": {message}"
        self.results.append(result)
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(result)
        return condition

    def report(self):
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING TEST REPORT")
        print("=" * 70)
        for result in self.results:
            print(result)
        print("\n" + "-" * 70)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
        print("=" * 70)
        return self.failed == 0


def create_sample_data() -> pd.DataFrame:
    """Create sample transaction data"""
    np.random.seed(42)
    n_samples = 10
    data = {
        'step': np.arange(1, n_samples + 1),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'], n_samples),
        'amount': np.random.lognormal(10, 2, n_samples),
        'oldbalanceOrg': np.random.lognormal(11, 2, n_samples),
        'newbalanceOrig': np.random.lognormal(11, 2, n_samples),
        'oldbalanceDest': np.random.lognormal(10, 2, n_samples),
        'newbalanceDest': np.random.lognormal(10, 2, n_samples),
        'nameDest': [f"C{i:010d}" if i % 2 == 0 else f"M{i:010d}" for i in range(n_samples)]
    }
    return pd.DataFrame(data)


def test_feature_engineering_pipeline():
    print("\n" + "=" * 70)
    print("TEST: UNIFIED FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    tester = FeatureEngineeringTester()
    engineer = get_feature_engineer()
    df = create_sample_data()

    # Test transformation
    try:
        df_processed = engineer.transform(df)
        tester.test("Feature transformation runs", True, f"Transformed {len(df)} samples")
    except Exception as e:
        tester.test("Feature transformation runs", False, str(e))
        return tester

    # Test feature count
    expected_feature_count = len(engineer.get_feature_names())
    tester.test(
        "Feature count matches",
        len(df_processed.columns) == expected_feature_count,
        f"Expected {expected_feature_count}, got {len(df_processed.columns)}"
    )

    # Test required features exist
    required_features = [
        'amount_log', 'balance_change_orig', 'balance_change_dest',
        'orig_balance_zero', 'dest_balance_zero', 'large_transaction',
        'dest_is_merchant', 'type_PAYMENT', 'type_TRANSFER'
    ]
    for feature in required_features:
        tester.test(
            f"Feature '{feature}' exists",
            feature in df_processed.columns
        )

    # Test no NaNs or infs
    tester.test(
        "No NaN values",
        not df_processed.isnull().values.any()
    )
    tester.test(
        "No infinite values",
        np.isfinite(df_processed.values).all()
    )

    # Test one-hot encoding for all transaction types
    for tx_type in engineer.transaction_types:
        col = f"type_{tx_type}"
        tester.test(
            f"One-hot column '{col}' exists",
            col in df_processed.columns
        )

    # Test validation
    is_valid = engineer.validate_features(df)
    tester.test("Validation passes on sample data", is_valid)

    # Test that transform is idempotent (running twice gives same result)
    df_processed2 = engineer.transform(df)
    tester.test(
        "Transform is idempotent",
        np.allclose(df_processed.values, df_processed2.values),
        "Second transform matches first"
    )

    # Test that all columns are numeric
    tester.test(
        "All features are numeric",
        all(np.issubdtype(dtype, np.number) for dtype in df_processed.dtypes)
    )

    return tester


if __name__ == "__main__":
    tester = test_feature_engineering_pipeline()
    all_passed = tester.report()
    if all_passed:
        print("\n🎉 All feature engineering tests PASSED!")
    else:
        print("\n❌ Some feature engineering tests FAILED.")
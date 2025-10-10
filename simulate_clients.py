"""
Client Simulator - Simulate multiple financial institutions
Tests federated learning with synthetic clients
"""

import requests
import time
import json
import argparse
from typing import List, Dict, Any
import numpy as np

# Configuration
SERVER_URL = "http://localhost:8000"
NUM_CLIENTS = 3
TRAINING_ROUNDS = 5
DELAY_BETWEEN_ROUNDS = 10  # seconds


class ClientSimulator:
    """Simulates a financial institution client"""
    
    def __init__(self, client_id: str, name: str, data_size: int):
        self.client_id = client_id
        self.name = name
        self.data_size = data_size
        self.registered = False
        self.training_history = []
    
    def register(self) -> bool:
        """Register client with server"""
        try:
            response = requests.post(
                f"{SERVER_URL}/client/register",
                json={
                    "client_id": self.client_id,
                    "name": self.name,
                    "data_size": self.data_size
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {self.name} registered successfully")
                print(f"   Data size: {result['data_size']:,}")
                print(f"   Fraud rate: {result['fraud_rate']:.4f}")
                self.registered = True
                return True
            else:
                print(f"❌ {self.name} registration failed: {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ {self.name} registration error: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get client information from server"""
        try:
            response = requests.get(f"{SERVER_URL}/client/{self.client_id}")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"❌ Error getting {self.name} info: {e}")
            return {}


def simulate_fraud_transactions(num_transactions: int = 5) -> List[Dict[str, Any]]:
    """Generate simulated fraud transactions for testing"""
    transactions = []
    
    for i in range(num_transactions):
        # Generate suspicious transaction
        is_fraud = np.random.random() > 0.5
        
        if is_fraud:
            # Suspicious patterns
            transaction = {
                "type": np.random.choice(["TRANSFER", "CASH_OUT"]),
                "amount": float(np.random.uniform(100000, 500000)),
                "oldbalanceOrg": float(np.random.uniform(200000, 600000)),
                "newbalanceOrig": 0,  # Account emptied
                "oldbalanceDest": 0,
                "newbalanceDest": 0,  # Round-tripping
                "nameDest": f"C{np.random.randint(1000000, 9999999)}"
            }
        else:
            # Normal transaction
            transaction = {
                "type": np.random.choice(["PAYMENT", "DEBIT"]),
                "amount": float(np.random.uniform(10, 5000)),
                "oldbalanceOrg": float(np.random.uniform(5000, 50000)),
                "newbalanceOrig": float(np.random.uniform(4000, 49000)),
                "oldbalanceDest": float(np.random.uniform(1000, 10000)),
                "newbalanceDest": float(np.random.uniform(1100, 11000)),
                "nameDest": f"M{np.random.randint(1000000, 9999999)}"
            }
        
        transactions.append(transaction)
    
    return transactions


def test_fraud_detection(num_tests: int = 5):
    """Test fraud detection with sample transactions"""
    print(f"\n{'='*60}")
    print(f"🔍 Testing Fraud Detection ({num_tests} transactions)")
    print(f"{'='*60}\n")
    
    transactions = simulate_fraud_transactions(num_tests)
    
    results = {
        "total": 0,
        "fraud_detected": 0,
        "legitimate": 0,
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0
    }
    
    for i, tx in enumerate(transactions, 1):
        try:
            response = requests.post(
                f"{SERVER_URL}/detect/fraud",
                json={"transaction_data": tx}
            )
            
            if response.status_code == 200:
                result = response.json()
                results["total"] += 1
                
                if result["prediction"] == "fraud":
                    results["fraud_detected"] += 1
                else:
                    results["legitimate"] += 1
                
                results[f"{result['risk_level']}_risk"] += 1
                
                print(f"Transaction {i}:")
                print(f"  Type: {tx['type']}, Amount: ${tx['amount']:,.2f}")
                print(f"  Prediction: {result['prediction'].upper()}")
                print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
                print(f"  Risk Level: {result['risk_level'].upper()}")
                print()
            else:
                print(f"❌ Transaction {i} failed: {response.text}")
        
        except Exception as e:
            print(f"❌ Error testing transaction {i}: {e}")
        
        time.sleep(0.5)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Fraud Detection Summary")
    print(f"{'='*60}")
    print(f"Total Transactions: {results['total']}")
    print(f"Fraud Detected: {results['fraud_detected']}")
    print(f"Legitimate: {results['legitimate']}")
    print(f"High Risk: {results['high_risk']}")
    print(f"Medium Risk: {results['medium_risk']}")
    print(f"Low Risk: {results['low_risk']}")
    print(f"{'='*60}\n")


def run_federated_training_round(client_ids: List[str]) -> Dict[str, Any]:
    """Execute a federated training round"""
    try:
        response = requests.post(
            f"{SERVER_URL}/federated/train",
            json={"client_ids": client_ids}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Training round failed: {response.text}")
            return {}
    
    except Exception as e:
        print(f"❌ Training round error: {e}")
        return {}


def get_blockchain_stats() -> Dict[str, Any]:
    """Get blockchain statistics"""
    try:
        response = requests.get(f"{SERVER_URL}/blockchain/stats")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"❌ Error getting blockchain stats: {e}")
        return {}


def print_blockchain_summary():
    """Print blockchain summary"""
    stats = get_blockchain_stats()
    
    if stats:
        print(f"\n{'='*60}")
        print("⛓️  Blockchain Summary")
        print(f"{'='*60}")
        print(f"Total Blocks: {stats.get('total_blocks', 0)}")
        print(f"Total Transactions: {stats.get('total_transactions', 0)}")
        print(f"Average Block Time: {stats.get('average_block_time', 0):.2f}s")
        
        if 'transaction_types' in stats:
            print("\nTransaction Types:")
            for tx_type, count in stats['transaction_types'].items():
                print(f"  {tx_type}: {count}")
        
        print(f"{'='*60}\n")


def main():
    """Main simulation function"""
    parser = argparse.ArgumentParser(description="Simulate federated learning clients")
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=TRAINING_ROUNDS, help="Training rounds")
    parser.add_argument("--delay", type=int, default=DELAY_BETWEEN_ROUNDS, help="Delay between rounds (seconds)")
    parser.add_argument("--test-only", action="store_true", help="Only test fraud detection")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("🚀 Federated AI Framework - Client Simulator")
    print(f"{'='*60}\n")
    
    # Check server health
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            print("✅ Server is healthy\n")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure the server is running at {SERVER_URL}")
        return
    
    # If test-only mode, just run fraud detection tests
    if args.test_only:
        test_fraud_detection(num_tests=10)
        print_blockchain_summary()
        return
    
    # Create and register clients
    clients = []
    
    for i in range(args.clients):
        client = ClientSimulator(
            client_id=f"FI-{i+1:03d}",
            name=f"Bank_{chr(65+i)}",  # Bank_A, Bank_B, etc.
            data_size=np.random.randint(40000, 60000)
        )
        
        if client.register():
            clients.append(client)
        
        time.sleep(1)
    
    if not clients:
        print("❌ No clients registered successfully")
        return
    
    print(f"\n✅ {len(clients)} clients registered\n")
    
    # Run federated training rounds
    client_ids = [c.client_id for c in clients]
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"🔄 Federated Training Round {round_num}/{args.rounds}")
        print(f"{'='*60}\n")
        
        result = run_federated_training_round(client_ids)
        
        if result:
            print(f"✅ Round {round_num} completed successfully")
            print(f"   Participating Clients: {result.get('participating_clients', 0)}")
            
            if 'global_metrics' in result:
                metrics = result['global_metrics']
                print(f"   Global Accuracy: {metrics.get('global_accuracy', 0):.4f}")
                print(f"   Global ROC-AUC: {metrics.get('global_roc_auc', 0):.4f}")
        else:
            print(f"❌ Round {round_num} failed")
        
        # Wait before next round (except last round)
        if round_num < args.rounds:
            print(f"\n⏳ Waiting {args.delay} seconds before next round...")
            time.sleep(args.delay)
    
    # Test fraud detection after training
    print(f"\n{'='*60}")
    print("🎯 Testing Trained Model")
    print(f"{'='*60}\n")
    
    test_fraud_detection(num_tests=10)
    
    # Print final blockchain summary
    print_blockchain_summary()
    
    # Get final client info
    print(f"\n{'='*60}")
    print("📊 Final Client Status")
    print(f"{'='*60}\n")
    
    for client in clients:
        info = client.get_info()
        if info:
            print(f"{info['name']}:")
            print(f"  Data Size: {info['data_size']:,}")
            print(f"  Training Rounds: {info.get('training_rounds', 0)}")
            print(f"  Status: {info.get('status', 'unknown')}")
            print()
    
    print("✅ Simulation complete!\n")


if __name__ == "__main__":
    main()
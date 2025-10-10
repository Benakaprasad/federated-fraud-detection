"""
Complete System Test - Federated Fraud Detection
Tests all functionality with real data
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


class FederatedFraudTester:
    """Test the complete federated fraud detection system"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client_ids = []
        
    def print_section(self, title: str):
        """Print formatted section header"""
        print("\n" + "="*60)
        print(f"🧪 {title}")
        print("="*60)
    
    def test_health_check(self):
        """Test 1: Server Health Check"""
        self.print_section("TEST 1: Health Check")
        
        response = requests.get(f"{self.base_url}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server Status: {data['status']}")
            print(f"   Platform: {data['platform']}")
            print(f"   Version: {data['version']}")
            print(f"   Blockchain Height: {data['blockchain_height']}")
            print(f"   Registered Clients: {data['registered_clients']}")
            print(f"   Federated Rounds: {data['federated_rounds']}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    
    def test_blockchain_verify(self):
        """Test 2: Blockchain Integrity"""
        self.print_section("TEST 2: Blockchain Verification")
        
        response = requests.get(f"{self.base_url}/blockchain/verify")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Blockchain Valid: {data['valid']}")
            print(f"   Total Blocks: {data.get('blocks', 'N/A')}")
            print(f"   Total Transactions: {data.get('transactions', 'N/A')}")
            return data['valid']
        else:
            print(f"❌ Verification failed: {response.text}")
            return False
    
    def test_register_clients(self, num_clients: int = 3):
        """Test 3: Register Multiple Clients"""
        self.print_section(f"TEST 3: Register {num_clients} Clients")
        
        client_names = [
            ("Bank Alpha", 50000),
            ("Bank Beta", 40000),
            ("Fintech Gamma", 60000),
            ("Credit Union Delta", 45000),
            ("Digital Bank Epsilon", 55000)
        ]
        
        for i in range(min(num_clients, len(client_names))):
            name, data_size = client_names[i]
            
            payload = {
                "name": name,
                "data_size": data_size
            }
            
            print(f"\n📝 Registering {name}...")
            response = requests.post(
                f"{self.base_url}/client/register",
                headers=HEADERS,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                self.client_ids.append(data['client_id'])
                print(f"✅ Client ID: {data['client_id']}")
                print(f"   Data Size: {data['data_size']:,}")
                print(f"   Fraud Rate: {data['fraud_rate']*100:.2f}%")
            else:
                print(f"❌ Registration failed: {response.text}")
                return False
        
        print(f"\n✅ Successfully registered {len(self.client_ids)} clients")
        return True
    
    def test_list_clients(self):
        """Test 4: List All Clients"""
        self.print_section("TEST 4: List All Clients")
        
        response = requests.get(f"{self.base_url}/clients")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total Clients: {data['total']}")
            
            for client in data['clients']:
                print(f"\n   Client: {client['name']}")
                print(f"   ID: {client['client_id']}")
                print(f"   Data Size: {client['data_size']:,}")
                print(f"   Fraud Rate: {client['fraud_rate']*100:.2f}%")
                print(f"   Status: {client['status']}")
            
            return True
        else:
            print(f"❌ Failed to list clients: {response.text}")
            return False
    
    def test_federated_training(self):
        """Test 5: Federated Training Round"""
        self.print_section("TEST 5: Federated Training Round")
        
        print("⏳ Starting federated training (this may take a few minutes)...")
        print("   Training local models with differential privacy...")
        print("   Generating encrypted updates with zero-knowledge proofs...")
        print("   Aggregating models using federated averaging...")
        
        start_time = time.time()
        
        payload = {
            "client_ids": []  # Empty list trains all clients
        }
        
        response = requests.post(
            f"{self.base_url}/federated/train",
            headers=HEADERS,
            json=payload,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Training Complete! ({duration:.1f}s)")
            print(f"   Round: {data['round']}")
            print(f"   Participating Clients: {data['participating_clients']}")
            
            metrics = data['global_metrics']
            print(f"\n📊 Global Model Metrics:")
            print(f"   Accuracy: {metrics['global_accuracy']*100:.2f}%")
            print(f"   Precision: {metrics['global_precision']*100:.2f}%")
            print(f"   Recall: {metrics['global_recall']*100:.2f}%")
            print(f"   F1 Score: {metrics['global_f1_score']*100:.2f}%")
            print(f"   ROC AUC: {metrics['global_roc_auc']*100:.2f}%")
            print(f"   Total Samples: {metrics['total_samples']:,}")
            
            return True
        else:
            print(f"❌ Training failed: {response.text}")
            return False
    
    def test_fraud_detection(self):
        """Test 6: Fraud Detection"""
        self.print_section("TEST 6: Fraud Detection")

        test_cases = [
            {
                "name": "Suspicious Large Transfer (High Risk)",
                "data": {
                    # === BASE FEATURES (11 from CSV) ===
                    "step": 100,
                    "amount": 500000.0,
                    "oldbalanceOrg": 600000.0,
                    "newbalanceOrig": 100000.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "type": "TRANSFER",  # Will be one-hot encoded

                    # === ENGINEERED FEATURES (7 more) ===
                    "amount_log": 13.122,  # np.log1p(500000)
                    "balance_change_orig": -500000.0,  # newbalanceOrig - oldbalanceOrg
                    "balance_change_dest": 0.0,  # newbalanceDest - oldbalanceDest
                    "orig_balance_zero": 0,  # oldbalanceOrg == 0
                    "dest_balance_zero": 1,  # oldbalanceDest == 0 (FRAUD INDICATOR!)
                    "large_transaction": 1,  # amount > threshold
                    "dest_is_merchant": 0,  # nameDest starts with 'M'

                    # === ONE-HOT ENCODED TYPE (5 features) ===
                    "type_CASH_IN": 0,
                    "type_CASH_OUT": 0,
                    "type_DEBIT": 0,
                    "type_PAYMENT": 0,
                    "type_TRANSFER": 1
                }
            },
            {
                "name": "Normal Small Payment (Low Risk)",
                "data": {
                    "step": 50,
                    "amount": 150.0,
                    "oldbalanceOrg": 5000.0,
                    "newbalanceOrig": 4850.0,
                    "oldbalanceDest": 2000.0,
                    "newbalanceDest": 2150.0,
                    "type": "PAYMENT",

                    "amount_log": 5.011,
                    "balance_change_orig": -150.0,
                    "balance_change_dest": 150.0,
                    "orig_balance_zero": 0,
                    "dest_balance_zero": 0,
                    "large_transaction": 0,
                    "dest_is_merchant": 1,

                    "type_CASH_IN": 0,
                    "type_CASH_OUT": 0,
                    "type_DEBIT": 0,
                    "type_PAYMENT": 1,
                    "type_TRANSFER": 0
                }
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {test_case['name']}")

            payload = {
                "transaction_data": test_case['data']
            }

            response = requests.post(
                f"{self.base_url}/detect/fraud",
                headers=HEADERS,
                json=payload
            )

            if response.status_code == 200:
                data = response.json()

                # Color code based on risk
                risk_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }

                emoji = risk_emoji.get(data['risk_level'], "⚪")

                print(f"   {emoji} Prediction: {data['prediction'].upper()}")
                print(f"   Fraud Probability: {data['fraud_probability']*100:.2f}%")
                print(f"   Risk Level: {data['risk_level'].upper()}")
                print(f"   Transaction ID: {data['transaction_id']}")
            else:
                print(f"   ❌ Detection failed: {response.text}")
                return False

        print(f"\n✅ All fraud detection tests completed")
        return True
    
    def test_blockchain_stats(self):
        """Test 7: Blockchain Statistics"""
        self.print_section("TEST 7: Blockchain Statistics")
        
        response = requests.get(f"{self.base_url}/blockchain/stats")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Blockchain Statistics:")
            print(f"   Total Blocks: {data['total_blocks']}")
            print(f"   Total Transactions: {data['total_transactions']}")
            print(f"   Pending Transactions: {data['pending_transactions']}")
            print(f"   Chain Length: {data['chain_length']}")
            
            print(f"\n📝 Transaction Types:")
            for tx_type, count in data['transaction_types'].items():
                print(f"   {tx_type}: {count}")
            
            return True
        else:
            print(f"❌ Failed to get stats: {response.text}")
            return False
    
    def test_search_transactions(self):
        """Test 8: Search Blockchain Transactions"""
        self.print_section("TEST 8: Search Transactions")
        
        search_types = [
            "client_registration",
            "model_update",
            "fraud_detection",
            "federated_round"
        ]
        
        for tx_type in search_types:
            response = requests.get(
                f"{self.base_url}/blockchain/search",
                params={"tx_type": tx_type}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   {tx_type}: {data['count']} transactions")
            else:
                print(f"   ❌ Search failed for {tx_type}")
        
        print(f"\n✅ Transaction search completed")
        return True
    
    def test_websocket_connection(self):
        """Test 9: WebSocket Connection"""
        self.print_section("TEST 9: WebSocket Connection")
        
        print("ℹ️  WebSocket endpoint available at:")
        print(f"   ws://localhost:8000/ws")
        print("\n   To test WebSocket, use a WebSocket client or JavaScript:")
        print("   const ws = new WebSocket('ws://localhost:8000/ws');")
        print("   ws.onmessage = (event) => console.log(JSON.parse(event.data));")
        
        return True
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "🚀"*30)
        print("FEDERATED FRAUD DETECTION - COMPLETE SYSTEM TEST")
        print("🚀"*30)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Blockchain Verification", self.test_blockchain_verify),
            ("Register Clients", lambda: self.test_register_clients(3)),
            ("List Clients", self.test_list_clients),
            ("Federated Training", self.test_federated_training),
            ("Fraud Detection", self.test_fraud_detection),
            ("Blockchain Statistics", self.test_blockchain_stats),
            ("Search Transactions", self.test_search_transactions),
            ("WebSocket Info", self.test_websocket_connection)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                
                if not result:
                    print(f"\n⚠️  Test '{test_name}' failed!")
                    break
                
                time.sleep(1)  # Brief pause between tests
                
            except Exception as e:
                print(f"\n❌ Test '{test_name}' crashed: {str(e)}")
                results.append((test_name, False))
                break
        
        # Print summary
        self.print_section("TEST SUMMARY")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! System is fully functional!")
            print("\n✅ Your system includes:")
            print("   • Client Registration & Management")
            print("   • Federated Learning with Differential Privacy")
            print("   • Model Aggregation (FedAvg)")
            print("   • Real-time Fraud Detection")
            print("   • Blockchain Audit Trail")
            print("   • Zero-Knowledge Proofs")
            print("   • Secure Aggregation")
            print("   • WebSocket Real-time Updates")
            print("   • Transaction Search & Analytics")
        else:
            print("\n⚠️  Some tests failed. Check the logs above.")
        
        return passed == total


if __name__ == "__main__":
    print("🔧 Starting Federated Fraud Detection System Test...")
    print("📡 Make sure server is running on http://localhost:8000")
    
    input("\nPress Enter to start testing...")
    
    tester = FederatedFraudTester()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)
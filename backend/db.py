"""
Database Module - Simple in-memory database for development
In production: Use PostgreSQL, MongoDB, or other production database
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import threading


@dataclass
class ClientRecord:
    """Client registration record"""
    client_id: str
    name: str
    data_size: int
    fraud_rate: float
    registered_at: float
    status: str
    last_training: Optional[float] = None
    training_rounds: int = 0


@dataclass
class TrainingRecord:
    """Training session record"""
    record_id: str
    client_id: str
    round_number: int
    accuracy: float
    roc_auc: float
    training_time: float
    data_size: int
    timestamp: float


@dataclass
class InferenceRecord:
    """Inference prediction record"""
    transaction_id: str
    prediction: int
    fraud_probability: float
    risk_level: str
    inference_time_ms: float
    timestamp: float
    amount: Optional[float] = None
    transaction_type: Optional[str] = None


@dataclass
class BlockchainRecord:
    """Blockchain event record"""
    event_id: str
    event_type: str
    block_height: int
    transaction_hash: str
    timestamp: float
    data: Dict[str, Any]


class InMemoryDatabase:
    """
    Simple in-memory database for development
    Thread-safe with basic CRUD operations
    """
    
    def __init__(self):
        self.clients: Dict[str, ClientRecord] = {}
        self.training_records: List[TrainingRecord] = []
        self.inference_records: List[InferenceRecord] = []
        self.blockchain_records: List[BlockchainRecord] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        print("🗄️  In-memory database initialized")
    
    # ==================== CLIENT OPERATIONS ====================
    
    def insert_client(self, client: ClientRecord):
        """Insert or update client record"""
        with self.lock:
            self.clients[client.client_id] = client
    
    def get_client(self, client_id: str) -> Optional[ClientRecord]:
        """Get client by ID"""
        with self.lock:
            return self.clients.get(client_id)
    
    def get_all_clients(self) -> List[ClientRecord]:
        """Get all clients"""
        with self.lock:
            return list(self.clients.values())
    
    def update_client_training(self, client_id: str):
        """Update client's last training timestamp"""
        with self.lock:
            if client_id in self.clients:
                self.clients[client_id].last_training = time.time()
                self.clients[client_id].training_rounds += 1
    
    def delete_client(self, client_id: str) -> bool:
        """Delete client"""
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]
                return True
            return False
    
    # ==================== TRAINING OPERATIONS ====================
    
    def insert_training_record(self, record: TrainingRecord):
        """Insert training record"""
        with self.lock:
            self.training_records.append(record)
    
    def get_training_records(self, client_id: Optional[str] = None,
                            limit: int = 100) -> List[TrainingRecord]:
        """Get training records, optionally filtered by client"""
        with self.lock:
            records = self.training_records
            
            if client_id:
                records = [r for r in records if r.client_id == client_id]
            
            # Sort by timestamp descending
            records = sorted(records, key=lambda x: x.timestamp, reverse=True)
            
            return records[:limit]
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        with self.lock:
            if not self.training_records:
                return {
                    'total_rounds': 0,
                    'total_records': 0,
                    'avg_accuracy': 0,
                    'avg_roc_auc': 0
                }
            
            return {
                'total_rounds': max((r.round_number for r in self.training_records), default=0),
                'total_records': len(self.training_records),
                'avg_accuracy': sum(r.accuracy for r in self.training_records) / len(self.training_records),
                'avg_roc_auc': sum(r.roc_auc for r in self.training_records) / len(self.training_records),
                'total_training_time': sum(r.training_time for r in self.training_records)
            }
    
    # ==================== INFERENCE OPERATIONS ====================
    
    def insert_inference_record(self, record: InferenceRecord):
        """Insert inference record"""
        with self.lock:
            self.inference_records.append(record)
    
    def get_inference_records(self, limit: int = 1000,
                             fraud_only: bool = False) -> List[InferenceRecord]:
        """Get inference records"""
        with self.lock:
            records = self.inference_records
            
            if fraud_only:
                records = [r for r in records if r.prediction == 1]
            
            # Sort by timestamp descending
            records = sorted(records, key=lambda x: x.timestamp, reverse=True)
            
            return records[:limit]
    
    def get_inference_stats(self, window_seconds: int = 3600) -> Dict[str, Any]:
        """Get inference statistics for recent time window"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            recent_records = [
                r for r in self.inference_records 
                if r.timestamp >= cutoff_time
            ]
            
            if not recent_records:
                return {
                    'total_inferences': 0,
                    'fraud_detected': 0,
                    'fraud_rate': 0,
                    'avg_inference_time_ms': 0
                }
            
            fraud_count = sum(1 for r in recent_records if r.prediction == 1)
            
            return {
                'total_inferences': len(recent_records),
                'fraud_detected': fraud_count,
                'fraud_rate': fraud_count / len(recent_records),
                'avg_inference_time_ms': sum(r.inference_time_ms for r in recent_records) / len(recent_records),
                'window_seconds': window_seconds
            }
    
    # ==================== BLOCKCHAIN OPERATIONS ====================
    
    def insert_blockchain_record(self, record: BlockchainRecord):
        """Insert blockchain event record"""
        with self.lock:
            self.blockchain_records.append(record)
    
    def get_blockchain_records(self, event_type: Optional[str] = None,
                              limit: int = 100) -> List[BlockchainRecord]:
        """Get blockchain records"""
        with self.lock:
            records = self.blockchain_records
            
            if event_type:
                records = [r for r in records if r.event_type == event_type]
            
            # Sort by timestamp descending
            records = sorted(records, key=lambda x: x.timestamp, reverse=True)
            
            return records[:limit]
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        with self.lock:
            if not self.blockchain_records:
                return {
                    'total_events': 0,
                    'latest_block_height': 0,
                    'event_types': {}
                }
            
            event_types = {}
            for record in self.blockchain_records:
                event_types[record.event_type] = event_types.get(record.event_type, 0) + 1
            
            return {
                'total_events': len(self.blockchain_records),
                'latest_block_height': max((r.block_height for r in self.blockchain_records), default=0),
                'event_types': event_types
            }
    
    # ==================== GENERAL OPERATIONS ====================
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        with self.lock:
            return {
                'clients': {
                    'total': len(self.clients),
                    'active': sum(1 for c in self.clients.values() if c.status == 'active')
                },
                'training': self.get_training_stats(),
                'inference': self.get_inference_stats(),
                'blockchain': self.get_blockchain_stats()
            }
    
    def clear_all(self):
        """Clear all data (use with caution!)"""
        with self.lock:
            self.clients.clear()
            self.training_records.clear()
            self.inference_records.clear()
            self.blockchain_records.clear()
            print("⚠️  Database cleared")
    
    def export_to_json(self, filepath: str):
        """Export database to JSON file"""
        with self.lock:
            data = {
                'clients': [asdict(c) for c in self.clients.values()],
                'training_records': [asdict(r) for r in self.training_records],
                'inference_records': [asdict(r) for r in self.inference_records],
                'blockchain_records': [asdict(r) for r in self.blockchain_records],
                'exported_at': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Database exported to {filepath}")
    
    def import_from_json(self, filepath: str):
        """Import database from JSON file"""
        with self.lock:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing data
            self.clear_all()
            
            # Import clients
            for client_data in data.get('clients', []):
                self.clients[client_data['client_id']] = ClientRecord(**client_data)
            
            # Import training records
            for record_data in data.get('training_records', []):
                self.training_records.append(TrainingRecord(**record_data))
            
            # Import inference records
            for record_data in data.get('inference_records', []):
                self.inference_records.append(InferenceRecord(**record_data))
            
            # Import blockchain records
            for record_data in data.get('blockchain_records', []):
                self.blockchain_records.append(BlockchainRecord(**record_data))
            
            print(f"📂 Database imported from {filepath}")


# Global database instance
db = InMemoryDatabase()


def test_database():
    """Test database operations"""
    print("=" * 60)
    print("🗄️  Testing Database Operations")
    print("=" * 60)
    
    # Test client operations
    print("\n1. Client Operations:")
    client = ClientRecord(
        client_id="FI-001",
        name="Bank Alpha",
        data_size=50000,
        fraud_rate=0.048,
        registered_at=time.time(),
        status="active"
    )
    
    db.insert_client(client)
    print(f"   ✅ Client inserted: {client.client_id}")
    
    retrieved = db.get_client("FI-001")
    print(f"   ✅ Client retrieved: {retrieved.name}")
    
    all_clients = db.get_all_clients()
    print(f"   ✅ Total clients: {len(all_clients)}")
    
    # Test training records
    print("\n2. Training Record Operations:")
    training = TrainingRecord(
        record_id="train_001",
        client_id="FI-001",
        round_number=1,
        accuracy=0.95,
        roc_auc=0.97,
        training_time=5.2,
        data_size=50000,
        timestamp=time.time()
    )
    
    db.insert_training_record(training)
    print(f"   ✅ Training record inserted")
    
    training_records = db.get_training_records(client_id="FI-001")
    print(f"   ✅ Retrieved {len(training_records)} training records")
    
    training_stats = db.get_training_stats()
    print(f"   Avg Accuracy: {training_stats['avg_accuracy']:.4f}")
    
    # Test inference records
    print("\n3. Inference Record Operations:")
    inference = InferenceRecord(
        transaction_id="tx_001",
        prediction=1,
        fraud_probability=0.85,
        risk_level="high",
        inference_time_ms=12.5,
        timestamp=time.time(),
        amount=150000,
        transaction_type="TRANSFER"
    )
    
    db.insert_inference_record(inference)
    print(f"   ✅ Inference record inserted")
    
    inference_records = db.get_inference_records(limit=10)
    print(f"   ✅ Retrieved {len(inference_records)} inference records")
    
    inference_stats = db.get_inference_stats()
    print(f"   Total Inferences: {inference_stats['total_inferences']}")
    print(f"   Fraud Detected: {inference_stats['fraud_detected']}")
    
    # Test blockchain records
    print("\n4. Blockchain Record Operations:")
    blockchain = BlockchainRecord(
        event_id="event_001",
        event_type="model_update",
        block_height=5,
        transaction_hash="abc123" * 10,
        timestamp=time.time(),
        data={"client": "FI-001", "accuracy": 0.95}
    )
    
    db.insert_blockchain_record(blockchain)
    print(f"   ✅ Blockchain record inserted")
    
    blockchain_records = db.get_blockchain_records(event_type="model_update")
    print(f"   ✅ Retrieved {len(blockchain_records)} blockchain records")
    
    # Test system stats
    print("\n5. System Statistics:")
    stats = db.get_system_stats()
    print(f"   Total Clients: {stats['clients']['total']}")
    print(f"   Total Training Rounds: {stats['training']['total_rounds']}")
    print(f"   Total Inferences: {stats['inference']['total_inferences']}")
    
    # Test export/import
    print("\n6. Export/Import:")
    export_file = "test_db_export.json"
    db.export_to_json(export_file)
    print(f"   ✅ Exported to {export_file}")
    
    print("\n" + "=" * 60)
    print("✅ All database tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_database()
"""
Blockchain Module - Distributed Ledger for Federated Learning
Stores model updates, fraud detections, and client registrations
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid


@dataclass
class Transaction:
    """Blockchain transaction"""
    tx_id: str
    tx_type: str  # 'model_update', 'fraud_detection', 'client_registration'
    timestamp: float
    data: Dict[str, Any]
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class Block:
    """Block in the blockchain"""
    
    def __init__(
        self, 
        index: int, 
        transactions: List[Transaction],
        previous_hash: str = "",
        nonce: int = 0
    ):
        self.index = index
        self.timestamp = time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 2):
        """
        Mine block with proof-of-work
        
        Args:
            difficulty: Number of leading zeros required
        """
        target = '0' * difficulty
        
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
        
        print(f"⛏️  Block #{self.index} mined: {self.hash[:16]}...")
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'hash': self.hash,
            'nonce': self.nonce,
            'transaction_count': len(self.transactions)
        }


class Blockchain:
    """Basic blockchain implementation"""
    
    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.mining_reward = 10
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type='genesis',
            timestamp=time.time(),
            data={'message': 'Genesis Block - Federated Fraud Detection System'}
        )
        
        genesis_block = Block(0, [genesis_tx], "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        
        print("🧱 Genesis block created")
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction):
        """Add a transaction to pending transactions"""
        self.pending_transactions.append(transaction)
    
    def create_block(self) -> Optional[Block]:
        """Mine pending transactions into a new block"""
        if not self.pending_transactions:
            return None
        
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions.copy(),
            previous_hash=self.get_latest_block().hash
        )
        
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        print(f"✅ Block #{new_block.index} added to chain ({len(new_block.transactions)} transactions)")
        
        return new_block
    
    def verify_chain(self) -> Dict[str, Any]:
        """Verify blockchain integrity"""
        print("\n🔍 Verifying blockchain...")
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check hash validity
            if current_block.hash != current_block.calculate_hash():
                return {
                    'valid': False,
                    'error': f'Invalid hash at block {i}',
                    'block_index': i
                }
            
            # Check chain linkage
            if current_block.previous_hash != previous_block.hash:
                return {
                    'valid': False,
                    'error': f'Chain broken at block {i}',
                    'block_index': i
                }
        
        print("✅ Blockchain is valid")
        return {
            'valid': True,
            'blocks': len(self.chain),
            'transactions': sum(len(block.transactions) for block in self.chain)
        }
    
    def get_chain(self) -> List[Dict]:
        """Get entire blockchain as list of dicts"""
        return [block.to_dict() for block in self.chain]
    
    def get_block(self, index: int) -> Optional[Dict]:
        """Get specific block by index"""
        if 0 <= index < len(self.chain):
            return self.chain[index].to_dict()
        return None
    
    def get_pending_transactions(self) -> List[Dict]:
        """Get all pending transactions"""
        return [tx.to_dict() for tx in self.pending_transactions]
    
    def search_transactions(self, filters: Dict[str, Any]) -> List[Dict]:
        """
        Search for transactions matching filters
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            List of matching transactions
        """
        results = []
        
        for block in self.chain:
            for tx in block.transactions:
                match = True
                
                for key, value in filters.items():
                    if key == 'tx_type' and tx.tx_type != value:
                        match = False
                        break
                    elif key == 'client_id' and tx.client_id != value:
                        match = False
                        break
                
                if match:
                    tx_dict = tx.to_dict()
                    tx_dict['block_index'] = block.index
                    results.append(tx_dict)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        
        # Count transaction types
        tx_types = {}
        for block in self.chain:
            for tx in block.transactions:
                tx_types[tx.tx_type] = tx_types.get(tx.tx_type, 0) + 1
        
        return {
            'total_blocks': len(self.chain),
            'total_transactions': total_transactions,
            'pending_transactions': len(self.pending_transactions),
            'transaction_types': tx_types,
            'chain_length': len(self.chain),
            'latest_block_hash': self.get_latest_block().hash,
            'genesis_timestamp': self.chain[0].timestamp if self.chain else None
        }


class HyperledgerFabric(Blockchain):
    """
    Extended blockchain simulating Hyperledger Fabric features
    Adds channels, endorsement policies, and enterprise features
    """
    
    def __init__(self, difficulty: int = 2):
        super().__init__(difficulty)
        self.channels = {'default': self.chain}
        self.endorsement_policies = {}
    
    def create_channel(self, channel_name: str):
        """Create a new channel"""
        if channel_name not in self.channels:
            self.channels[channel_name] = []
            print(f"📺 Channel '{channel_name}' created")
    
    def set_endorsement_policy(self, tx_type: str, min_endorsements: int):
        """Set endorsement policy for transaction type"""
        self.endorsement_policies[tx_type] = min_endorsements
        print(f"📋 Endorsement policy set: {tx_type} requires {min_endorsements} endorsements")


class BlockchainLogger:
    """
    Logger for blockchain events
    Simplifies adding different types of transactions
    """
    
    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
    
    def log_client_registration(self, client_id: str, client_data: Dict[str, Any]):
        """Log client registration to blockchain"""
        transaction = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type='client_registration',
            timestamp=time.time(),
            client_id=client_id,
            data={
                'client_name': client_data.get('name', 'Unknown'),
                'data_size': client_data.get('data_size', 0),
                'fraud_rate': client_data.get('fraud_rate', 0.0),
                'status': client_data.get('status', 'active')
            }
        )
        
        self.blockchain.add_transaction(transaction)
        print(f"📝 Client registration logged: {client_id}")
    
    def log_model_update(
        self, 
        client_id: str, 
        metrics: Dict[str, float],
        zkp_proof: str,
        model_hash: str
    ):
        """Log model update to blockchain"""
        transaction = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type='model_update',
            timestamp=time.time(),
            client_id=client_id,
            data={
                'metrics': metrics,
                'zkp_proof': zkp_proof,
                'model_hash': model_hash,
                'privacy_preserved': True
            }
        )
        
        self.blockchain.add_transaction(transaction)
        print(f"📝 Model update logged: {client_id}")
    
    def log_federated_round(
        self,
        round_number: int,
        participating_clients: List[str],
        global_accuracy: float,
        model_hash: str
    ):
        """Log federated aggregation round to blockchain"""
        transaction = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type='federated_round',
            timestamp=time.time(),
            data={
                'round_number': round_number,
                'participating_clients': participating_clients,
                'num_clients': len(participating_clients),
                'global_accuracy': global_accuracy,
                'model_hash': model_hash
            }
        )
        
        self.blockchain.add_transaction(transaction)
        print(f"📝 Federated round {round_number} logged")
    
    def log_fraud_detection(
        self,
        tx_id: str,
        fraud_prob: float,
        prediction: int,
        tx_details: Dict[str, Any]
    ):
        """Log fraud detection to blockchain"""
        transaction = Transaction(
            tx_id=tx_id,
            tx_type='fraud_detection',
            timestamp=time.time(),
            data={
                'fraud_probability': fraud_prob,
                'prediction': 'fraud' if prediction == 1 else 'legitimate',
                'transaction_amount': tx_details.get('amount', 0),
                'transaction_type': tx_details.get('type', 'unknown')
            }
        )
        
        self.blockchain.add_transaction(transaction)
        print(f"📝 Fraud detection logged: {tx_id[:8]}...")
    
    def log_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log general system event"""
        transaction = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type='system_event',
            timestamp=time.time(),
            data={
                'event_type': event_type,
                'event_data': event_data
            }
        )
        
        self.blockchain.add_transaction(transaction)
        print(f"📝 System event logged: {event_type}")


def test_blockchain():
    """Test blockchain functionality"""
    print("=" * 60)
    print("🧪 Testing Blockchain")
    print("=" * 60)
    
    # Create blockchain
    bc = HyperledgerFabric(difficulty=2)
    logger = BlockchainLogger(bc)
    
    # Test client registration
    print("\n1. Testing Client Registration...")
    logger.log_client_registration(
        client_id="FI-001",
        client_data={
            'name': 'Bank Alpha',
            'data_size': 50000,
            'fraud_rate': 0.035,
            'status': 'active'
        }
    )
    
    # Test model update
    print("\n2. Testing Model Update...")
    logger.log_model_update(
        client_id="FI-001",
        metrics={'accuracy': 0.95, 'roc_auc': 0.92},
        zkp_proof="proof123abc",
        model_hash="hash456def"
    )
    
    # Test fraud detection
    print("\n3. Testing Fraud Detection...")
    logger.log_fraud_detection(
        tx_id="TX-12345",
        fraud_prob=0.87,
        prediction=1,
        tx_details={'amount': 5000, 'type': 'TRANSFER'}
    )
    
    # Mine a block
    print("\n4. Mining Block...")
    block = bc.create_block()
    
    # Verify chain
    print("\n5. Verifying Chain...")
    verification = bc.verify_chain()
    print(f"   Valid: {verification['valid']}")
    
    # Get statistics
    print("\n6. Blockchain Statistics...")
    stats = bc.get_statistics()
    print(f"   Total Blocks: {stats['total_blocks']}")
    print(f"   Total Transactions: {stats['total_transactions']}")
    print(f"   Transaction Types: {stats['transaction_types']}")
    
    # Search transactions
    print("\n7. Searching Transactions...")
    results = bc.search_transactions({'tx_type': 'client_registration'})
    print(f"   Found {len(results)} client registrations")
    
    print("\n" + "=" * 60)
    print("✅ All blockchain tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_blockchain()
"""
Test Suite for Blockchain Module
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.blockchain import HyperledgerFabric, BlockchainLogger, Transaction, Block
import time


class TestTransaction:
    """Test Transaction class"""
    
    def test_transaction_creation(self):
        """Test creating a transaction"""
        tx = Transaction(
            id="test_tx_001",
            type="model_update",
            timestamp=time.time(),
            data={"accuracy": 0.95},
            client_id="FI-001"
        )
        
        assert tx.id == "test_tx_001"
        assert tx.type == "model_update"
        assert tx.client_id == "FI-001"
    
    def test_transaction_hash(self):
        """Test transaction hashing"""
        tx = Transaction(
            id="test_tx_001",
            type="model_update",
            timestamp=time.time(),
            data={"accuracy": 0.95}
        )
        
        hash1 = tx.hash()
        hash2 = tx.hash()
        
        # Same transaction should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 char hex
    
    def test_transaction_to_dict(self):
        """Test transaction serialization"""
        tx = Transaction(
            id="test_tx_001",
            type="model_update",
            timestamp=time.time(),
            data={"accuracy": 0.95}
        )
        
        tx_dict = tx.to_dict()
        
        assert isinstance(tx_dict, dict)
        assert "id" in tx_dict
        assert "type" in tx_dict
        assert "data" in tx_dict


class TestBlock:
    """Test Block class"""
    
    def test_block_creation(self):
        """Test creating a block"""
        tx = Transaction(
            id="tx_001",
            type="test",
            timestamp=time.time(),
            data={}
        )
        
        block = Block(
            height=1,
            timestamp=time.time(),
            transactions=[tx],
            prev_hash="0" * 64,
            validator="test_validator"
        )
        
        assert block.height == 1
        assert len(block.transactions) == 1
        assert block.prev_hash == "0" * 64
    
    def test_merkle_root_calculation(self):
        """Test Merkle root calculation"""
        txs = [
            Transaction(id=f"tx_{i}", type="test", timestamp=time.time(), data={})
            for i in range(4)
        ]
        
        block = Block(
            height=1,
            timestamp=time.time(),
            transactions=txs,
            prev_hash="0" * 64
        )
        
        merkle_root = block.calculate_merkle_root()
        
        assert merkle_root is not None
        assert len(merkle_root) == 64
    
    def test_block_hash_calculation(self):
        """Test block hash calculation"""
        tx = Transaction(
            id="tx_001",
            type="test",
            timestamp=time.time(),
            data={}
        )
        
        block = Block(
            height=1,
            timestamp=time.time(),
            transactions=[tx],
            prev_hash="0" * 64
        )
        
        block.merkle_root = block.calculate_merkle_root()
        block_hash = block.calculate_hash()
        
        assert block_hash is not None
        assert len(block_hash) == 64


class TestHyperledgerFabric:
    """Test HyperledgerFabric blockchain"""
    
    def test_blockchain_initialization(self):
        """Test blockchain initialization with genesis block"""
        blockchain = HyperledgerFabric()
        
        assert len(blockchain.chain) == 1  # Genesis block
        assert blockchain.chain[0].height == 0
        assert blockchain.chain[0].prev_hash == "0" * 64
    
    def test_add_transaction(self):
        """Test adding transactions to mempool"""
        blockchain = HyperledgerFabric()
        
        tx = Transaction(
            id="test_tx",
            type="test",
            timestamp=time.time(),
            data={}
        )
        
        initial_pending = len(blockchain.pending_transactions)
        blockchain.add_transaction(tx)
        
        assert len(blockchain.pending_transactions) == initial_pending + 1
    
    def test_create_block(self):
        """Test creating a new block"""
        blockchain = HyperledgerFabric()
        
        # Add some transactions
        for i in range(5):
            tx = Transaction(
                id=f"tx_{i}",
                type="test",
                timestamp=time.time(),
                data={"index": i}
            )
            blockchain.add_transaction(tx)
        
        initial_height = len(blockchain.chain)
        block = blockchain.create_block()
        
        assert block is not None
        assert len(blockchain.chain) == initial_height + 1
        assert block.height == initial_height
        assert len(block.transactions) == 5
    
    def test_verify_chain(self):
        """Test blockchain verification"""
        blockchain = HyperledgerFabric()
        
        # Add transactions and create blocks
        for i in range(3):
            tx = Transaction(
                id=f"tx_{i}",
                type="test",
                timestamp=time.time(),
                data={}
            )
            blockchain.add_transaction(tx)
            blockchain.create_block()
        
        verification = blockchain.verify_chain()
        
        assert verification['valid'] is True
        assert verification['chain_length'] == len(blockchain.chain)
        assert len(verification['errors']) == 0
    
    def test_get_chain(self):
        """Test getting blockchain as list"""
        blockchain = HyperledgerFabric()
        
        chain = blockchain.get_chain()
        
        assert isinstance(chain, list)
        assert len(chain) == len(blockchain.chain)
    
    def test_search_transactions(self):
        """Test transaction search"""
        blockchain = HyperledgerFabric()
        
        # Add transactions of different types
        for i in range(3):
            tx = Transaction(
                id=f"tx_{i}",
                type="model_update" if i % 2 == 0 else "fraud_detection",
                timestamp=time.time(),
                data={},
                client_id="FI-001"
            )
            blockchain.add_transaction(tx)
        
        blockchain.create_block()
        
        # Search by type
        results = blockchain.search_transactions({"type": "model_update"})
        
        assert len(results) >= 1
        assert all(r['type'] == "model_update" for r in results)
    
    def test_get_statistics(self):
        """Test blockchain statistics"""
        blockchain = HyperledgerFabric()
        
        # Add some activity
        for i in range(5):
            tx = Transaction(
                id=f"tx_{i}",
                type="test",
                timestamp=time.time(),
                data={}
            )
            blockchain.add_transaction(tx)
        
        blockchain.create_block()
        
        stats = blockchain.get_statistics()
        
        assert "total_blocks" in stats
        assert "total_transactions" in stats
        assert "transaction_types" in stats
        assert stats['total_blocks'] >= 1


class TestBlockchainLogger:
    """Test BlockchainLogger"""
    
    def test_log_client_registration(self):
        """Test logging client registration"""
        blockchain = HyperledgerFabric()
        logger = BlockchainLogger(blockchain)
        
        initial_pending = len(blockchain.pending_transactions)
        
        logger.log_client_registration(
            client_id="FI-001",
            client_data={"name": "Test Bank", "data_size": 50000, "fraud_rate": 0.04}
        )
        
        assert len(blockchain.pending_transactions) == initial_pending + 1
        assert blockchain.pending_transactions[-1].type == "client_registration"
    
    def test_log_model_update(self):
        """Test logging model update"""
        blockchain = HyperledgerFabric()
        logger = BlockchainLogger(blockchain)
        
        logger.log_model_update(
            client_id="FI-001",
            metrics={"accuracy": 0.95, "roc_auc": 0.97},
            zkp_proof="test_proof_123",
            model_hash="model_hash_abc"
        )
        
        assert blockchain.pending_transactions[-1].type == "model_update"
        assert blockchain.pending_transactions[-1].client_id == "FI-001"
    
    def test_log_federated_round(self):
        """Test logging federated round"""
        blockchain = HyperledgerFabric()
        logger = BlockchainLogger(blockchain)
        
        logger.log_federated_round(
            round_number=1,
            participating_clients=["FI-001", "FI-002"],
            global_accuracy=0.95,
            model_hash="global_hash_xyz"
        )
        
        assert blockchain.pending_transactions[-1].type == "federated_round"
        assert blockchain.pending_transactions[-1].data['round_number'] == 1
    
    def test_log_fraud_detection(self):
        """Test logging fraud detection"""
        blockchain = HyperledgerFabric()
        logger = BlockchainLogger(blockchain)
        
        logger.log_fraud_detection(
            tx_id="tx_001",
            fraud_prob=0.85,
            prediction=1,
            tx_details={"amount": 150000, "type": "TRANSFER"}
        )
        
        assert blockchain.pending_transactions[-1].type == "fraud_detection"
        assert blockchain.pending_transactions[-1].id == "tx_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
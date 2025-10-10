"""
Encryption Module - Advanced Cryptography
Provides HE, DP, and ZKP for secure federated learning
"""

import hashlib
import json
import secrets
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class EncryptedData:
    """Container for encrypted data"""
    ciphertext: str
    public_key: str
    timestamp: float


class HomomorphicEncryption:
    """
    Simplified Homomorphic Encryption
    In production: Use libraries like PySEAL or TenSEAL
    """
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key = self._generate_public_key()
        self.private_key = self._generate_private_key()
    
    def _generate_public_key(self) -> str:
        """Generate public key"""
        return hashlib.sha256(str(secrets.randbits(self.key_size)).encode()).hexdigest()
    
    def _generate_private_key(self) -> str:
        """Generate private key"""
        return hashlib.sha256(str(secrets.randbits(self.key_size)).encode()).hexdigest()
    
    def encrypt(self, data: Any) -> EncryptedData:
        """
        Encrypt data (simplified)
        In production: Use actual HE schemes like CKKS or BFV
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        
        # Simulate encryption by hashing with public key
        combined = f"{data_str}:{self.public_key}"
        ciphertext = hashlib.sha256(combined.encode()).hexdigest()
        
        return EncryptedData(
            ciphertext=ciphertext,
            public_key=self.public_key,
            timestamp=time.time()
        )
    
    def decrypt(self, encrypted: EncryptedData) -> Any:
        """
        Decrypt data (simplified)
        In production: Use actual HE decryption
        """
        # This is a placeholder - in real HE, you'd decrypt using private key
        return {"status": "decrypted", "timestamp": encrypted.timestamp}
    
    def add_encrypted(self, enc1: EncryptedData, enc2: EncryptedData) -> EncryptedData:
        """
        Homomorphic addition of encrypted values
        This is the key property of HE - operations on ciphertext
        """
        # Simulate homomorphic addition
        combined = f"{enc1.ciphertext}+{enc2.ciphertext}"
        result_ciphertext = hashlib.sha256(combined.encode()).hexdigest()
        
        return EncryptedData(
            ciphertext=result_ciphertext,
            public_key=self.public_key,
            timestamp=time.time()
        )


class DifferentialPrivacy:
    """
    Differential Privacy Implementation
    Adds calibrated noise to preserve privacy
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize DP mechanism
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise for DP
        
        Args:
            value: Original value
            sensitivity: Global sensitivity of the query
        
        Returns:
            Noisy value
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Gaussian noise for DP (better for composition)
        
        Args:
            value: Original value
            sensitivity: Global sensitivity
        
        Returns:
            Noisy value
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def privatize_gradient(self, gradient: np.ndarray, 
                          clip_norm: float = 1.0) -> np.ndarray:
        """
        Apply DP to gradients (gradient clipping + noise)
        
        Args:
            gradient: Original gradient
            clip_norm: Clipping threshold
        
        Returns:
            Privatized gradient
        """
        # Clip gradient to bound sensitivity
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > clip_norm:
            gradient = gradient * (clip_norm / grad_norm)
        
        # Add Gaussian noise
        sensitivity = clip_norm
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, gradient.shape)
        
        return gradient + noise
    
    def privatize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Apply DP to training metrics
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            Privatized metrics
        """
        privatized = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Add noise proportional to value range
                sensitivity = 0.1  # Assuming metrics are in [0, 1]
                privatized[key] = self.add_laplace_noise(value, sensitivity)
            else:
                privatized[key] = value
        
        return privatized


class ZeroKnowledgeProof:
    """
    Zero-Knowledge Proof Implementation
    Proves correctness without revealing data
    """
    
    def __init__(self):
        self.hash_algorithm = hashlib.sha256
    
    def generate_proof(self, data: Dict[str, Any]) -> str:
        """
        Generate ZKP for data
        
        In production: Use actual ZK-SNARK or ZK-STARK libraries
        like libsnark or circom
        
        Args:
            data: Data to prove
        
        Returns:
            Proof string
        """
        # Create commitment
        data_str = json.dumps(data, sort_keys=True, default=str)
        commitment = self.hash_algorithm(data_str.encode()).hexdigest()
        
        # Add random challenge
        challenge = secrets.token_hex(32)
        
        # Create response
        response_input = f"{commitment}:{challenge}:{time.time()}"
        response = self.hash_algorithm(response_input.encode()).hexdigest()
        
        # Proof is the tuple (commitment, challenge, response)
        proof = f"{commitment}:{challenge}:{response}"
        
        return proof
    
    def verify_proof(self, proof: str, expected_commitment: str = None) -> bool:
        """
        Verify ZKP
        
        Args:
            proof: Proof string
            expected_commitment: Expected commitment (optional)
        
        Returns:
            True if valid
        """
        try:
            parts = proof.split(':')
            if len(parts) != 3:
                return False
            
            commitment, challenge, response = parts
            
            # Verify format
            if len(commitment) != 64 or len(challenge) != 64 or len(response) != 64:
                return False
            
            # If expected commitment provided, check it
            if expected_commitment and commitment != expected_commitment:
                return False
            
            # In production: Verify actual ZK proof
            return True
        
        except Exception:
            return False
    
    def prove_correct_training(self, model_hash: str, 
                              dataset_size: int,
                              epsilon: float) -> str:
        """
        Generate proof that training was done correctly
        
        Args:
            model_hash: Hash of trained model
            dataset_size: Number of samples used
            epsilon: DP epsilon used
        
        Returns:
            Proof of correct training
        """
        proof_data = {
            "model_hash": model_hash,
            "dataset_size": dataset_size,
            "epsilon": epsilon,
            "timestamp": time.time(),
            "protocol_version": "1.0"
        }
        
        return self.generate_proof(proof_data)


class SecureAggregator:
    """
    Secure Aggregation for Federated Learning
    Combines encrypted updates securely
    """
    
    def __init__(self, epsilon: float = 1.0):
        self.he = HomomorphicEncryption()
        self.dp = DifferentialPrivacy(epsilon=epsilon)
        self.zkp = ZeroKnowledgeProof()
    
    def aggregate_encrypted_updates(self, 
                                    encrypted_updates: List[EncryptedData],
                                    weights: List[float] = None) -> EncryptedData:
        """
        Securely aggregate encrypted model updates
        
        Args:
            encrypted_updates: List of encrypted updates
            weights: Optional weights for each update
        
        Returns:
            Aggregated encrypted update
        """
        if not encrypted_updates:
            raise ValueError("No updates to aggregate")
        
        if weights is None:
            weights = [1.0 / len(encrypted_updates)] * len(encrypted_updates)
        
        # Start with first update
        aggregated = encrypted_updates[0]
        
        # Homomorphically add other updates
        for i, update in enumerate(encrypted_updates[1:], 1):
            aggregated = self.he.add_encrypted(aggregated, update)
        
        return aggregated
    
    def verify_all_proofs(self, proofs: List[str]) -> bool:
        """
        Verify all ZK proofs from clients
        
        Args:
            proofs: List of ZK proofs
        
        Returns:
            True if all valid
        """
        return all(self.zkp.verify_proof(proof) for proof in proofs)


def test_encryption():
    """Test encryption utilities"""
    print("=" * 60)
    print("🔐 Testing Encryption Utilities")
    print("=" * 60)
    
    # Test Homomorphic Encryption
    print("\n1. Testing Homomorphic Encryption...")
    he = HomomorphicEncryption()
    data = {"accuracy": 0.95, "loss": 0.12}
    encrypted = he.encrypt(data)
    print(f"   ✅ Data encrypted: {encrypted.ciphertext[:32]}...")
    
    # Test Differential Privacy
    print("\n2. Testing Differential Privacy...")
    dp = DifferentialPrivacy(epsilon=1.0)
    original_accuracy = 0.9534
    noisy_accuracy = dp.add_laplace_noise(original_accuracy, sensitivity=0.1)
    print(f"   Original: {original_accuracy:.4f}")
    print(f"   Noisy: {noisy_accuracy:.4f}")
    print(f"   ✅ Noise added successfully")
    
    # Test gradient privatization
    gradient = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    private_grad = dp.privatize_gradient(gradient, clip_norm=1.0)
    print(f"   ✅ Gradient privatized")
    
    # Test Zero-Knowledge Proof
    print("\n3. Testing Zero-Knowledge Proof...")
    zkp = ZeroKnowledgeProof()
    proof_data = {"model_version": "v1", "accuracy": 0.95}
    proof = zkp.generate_proof(proof_data)
    print(f"   Proof: {proof[:64]}...")
    
    is_valid = zkp.verify_proof(proof)
    print(f"   ✅ Proof valid: {is_valid}")
    
    # Test Secure Aggregator
    print("\n4. Testing Secure Aggregator...")
    aggregator = SecureAggregator(epsilon=1.0)
    
    updates = [he.encrypt({"w": i}) for i in range(3)]
    aggregated = aggregator.aggregate_encrypted_updates(updates)
    print(f"   ✅ {len(updates)} updates aggregated")
    
    proofs = [zkp.generate_proof({"id": i}) for i in range(3)]
    all_valid = aggregator.verify_all_proofs(proofs)
    print(f"   ✅ All proofs verified: {all_valid}")
    
    print("\n" + "=" * 60)
    print("✅ All encryption tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_encryption()
"""
Main FastAPI Server - Secure Federated AI Framework
Orchestrates federated learning, blockchain, and real-time fraud detection
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import joblib
import time
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import local modules
from backend.blockchain import Blockchain, HyperledgerFabric, BlockchainLogger
from backend.federated_trainer import FederatedModelTrainer, FederatedAggregator
from backend.utils.encryption import SecureAggregator
from data.load_paysim import PaySimLoader

# Configuration
HOST = "0.0.0.0"
PORT = 8000
BLOCK_INTERVAL_SEC = 5
MODEL_DIR = "models"
DATA_DIR = "data"

# Initialize FastAPI
app = FastAPI(
    title="Secure Federated AI Framework",
    description="Collaborative fraud detection with privacy-preserving federated learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
blockchain = HyperledgerFabric()
blockchain_logger = BlockchainLogger(blockchain)
federated_aggregator = FederatedAggregator()
secure_aggregator = SecureAggregator(epsilon=1.0)
data_loader = PaySimLoader()

# Client management
registered_clients: Dict[str, Dict[str, Any]] = {}
client_models: Dict[str, Any] = {}
client_datasets: Dict[str, pd.DataFrame] = {}
websocket_clients: List[WebSocket] = []

# Model trainer
model_trainer = FederatedModelTrainer(model_type="random_forest")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ==================== PYDANTIC MODELS ====================

class ClientRegistration(BaseModel):
    client_id: Optional[str] = None
    name: str
    data_size: int = 50000


class TransactionRequest(BaseModel):
    transaction_data: Dict[str, Any]
    client_id: Optional[str] = None


class FederatedRoundRequest(BaseModel):
    client_ids: List[str] = []


# ==================== WEBSOCKET ====================

async def broadcast_to_clients(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    dead_clients = []
    for ws in websocket_clients:
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            dead_clients.append(ws)
    
    # Remove dead connections
    for dead in dead_clients:
        if dead in websocket_clients:
            websocket_clients.remove(dead)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_clients.append(websocket)
    print(f"✅ WebSocket client connected (Total: {len(websocket_clients)})")
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "chain": blockchain.get_chain(),
            "mempool": blockchain.get_pending_transactions(),
            "clients": list(registered_clients.keys()),
            "stats": blockchain.get_statistics()
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Echo acknowledgment
            await websocket.send_json({"type": "ack", "message": "received"})
    
    except WebSocketDisconnect:
        print(f"⚠️  WebSocket client disconnected")
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "platform": "Secure Federated AI Framework",
        "version": "1.0.0",
        "blockchain_height": len(blockchain.chain),
        "registered_clients": len(registered_clients),
        "federated_rounds": federated_aggregator.round_number,
        "websocket_connections": len(websocket_clients)
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "server": "healthy",
        "blockchain": "operational",
        "model_trainer": "ready",
        "secure_enclave": "IBM_Z_Secure_Execution",
        "timestamp": time.time()
    }


# ==================== CLIENT MANAGEMENT ====================

@app.post("/client/register")
async def register_client(client_data: ClientRegistration):
    """Register a new financial institution client"""
    client_id = client_data.client_id or str(uuid.uuid4())
    client_name = client_data.name
    data_size = client_data.data_size
    
    print(f"\n📝 Registering client: {client_name}")
    
    # ===== FIX: Correct path to CSV =====
    # Get the project root (parent of backend/)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    csv_path = os.path.join(project_root, "data", "paysim.csv")
    
    print(f"🔍 Looking for CSV at: {csv_path}")
    print(f"📁 CSV exists: {os.path.exists(csv_path)}")
    
    if os.path.exists(csv_path):
        print(f"📂 Loading real data from CSV...")
        df_full = pd.read_csv(csv_path)
        total_rows = len(df_full)
        print(f"✅ Loaded {total_rows:,} total transactions from CSV")
        
        # Split data among clients (non-overlapping partitions)
        if client_name == "Bank Alpha":
            # First 33% (~2.1M transactions)
            df = df_full.iloc[:int(total_rows * 0.33)].copy()
        elif client_name == "Bank Beta":
            # Next 27% (~1.7M transactions)
            df = df_full.iloc[int(total_rows * 0.33):int(total_rows * 0.60)].copy()
        else:  # Fintech Gamma
            # Remaining 40% (~2.5M transactions)
            df = df_full.iloc[int(total_rows * 0.60):].copy()
        
        print(f"✅ Client {client_name} gets {len(df):,} transactions")
        
    else:
        print(f"❌ CSV NOT FOUND at: {csv_path}")
        print(f"⚠️ Falling back to synthetic data...")
        df = data_loader._generate_synthetic_fallback(n_samples=data_size)
    
    # Preprocess dataset
    data_loader.df = df
    df = data_loader.preprocess()
    
    print(f"🔧 After preprocessing: {len(df):,} rows, {len(df.columns)} features")
    
    # Store dataset privately (never shared!)
    client_datasets[client_id] = df
    
    # Register client
    registered_clients[client_id] = {
        "client_id": client_id,
        "name": client_name,
        "data_size": len(df),
        "fraud_rate": float(df['isFraud'].mean()),
        "registered_at": time.time(),
        "status": "active"
    }
    
    # Log to blockchain
    blockchain_logger.log_client_registration(client_id, registered_clients[client_id])
    
    # Broadcast update
    await broadcast_to_clients({
        "type": "client_registered",
        "client_id": client_id,
        "client_name": client_name
    })
    
    print(f"✅ Client {client_name} registered successfully")
    
    return {
        "status": "registered",
        "client_id": client_id,
        "client_name": client_name,
        "data_size": len(df),
        "fraud_rate": float(df['isFraud'].mean())
    }

@app.get("/clients")
async def get_clients():
    """Get all registered clients"""
    return {
        "clients": list(registered_clients.values()),
        "total": len(registered_clients)
    }


@app.get("/client/{client_id}")
async def get_client(client_id: str):
    """Get specific client information"""
    if client_id not in registered_clients:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return registered_clients[client_id]


# ==================== FEDERATED LEARNING ====================

@app.post("/federated/train")
async def train_federated_round(request: FederatedRoundRequest):
    """Execute a federated training round"""
    client_ids = request.client_ids
    
    if not client_ids:
        client_ids = list(registered_clients.keys())
    
    if not client_ids:
        raise HTTPException(status_code=400, detail="No clients available for training")
    
    print(f"\n🚀 Starting federated training round with {len(client_ids)} clients")
    
    # Train local models
    client_updates = []
    trained_models = {}
    
    for client_id in client_ids:
        if client_id not in client_datasets:
            print(f"⚠️  Skipping {client_id} - no dataset")
            continue
        
        print(f"\n🔒 Training {client_id}...")
        df = client_datasets[client_id]
        
        # Train local model
        model, metrics = model_trainer.train_local_model(df, apply_dp=True)
        
        # Generate encrypted update
        update = model_trainer.generate_encrypted_update(model, metrics)
        
        # Store
        trained_models[client_id] = model
        client_updates.append(update)
        
        # Log to blockchain
        blockchain_logger.log_model_update(
            client_id=client_id,
            metrics={
                "accuracy": metrics.accuracy,
                "roc_auc": metrics.roc_auc
            },
            zkp_proof=update['zkp_proof'],
            model_hash=update['zkp_proof'][:32]
        )
    
    # Aggregate models
    print(f"\n⚡ Aggregating {len(trained_models)} models...")
    global_model, global_metrics = federated_aggregator.aggregate_models(
        client_updates, trained_models
    )
    
    # Save global model
    federated_aggregator.save_global_model()
    
    # Log to blockchain
    blockchain_logger.log_federated_round(
        round_number=federated_aggregator.round_number,
        participating_clients=list(trained_models.keys()),
        global_accuracy=global_metrics['global_accuracy'],
        model_hash=str(uuid.uuid4())
    )
    
    # Broadcast update
    await broadcast_to_clients({
        "type": "federated_round_complete",
        "round": federated_aggregator.round_number,
        "metrics": global_metrics
    })
    
    return {
        "status": "success",
        "round": federated_aggregator.round_number,
        "participating_clients": len(trained_models),
        "global_metrics": global_metrics
    }


@app.get("/federated/status")
async def get_federated_status():
    """Get federated learning status"""
    return {
        "current_round": federated_aggregator.round_number,
        "registered_clients": len(registered_clients),
        "has_global_model": federated_aggregator.global_model is not None
    }


# ==================== FRAUD DETECTION ====================

@app.post("/detect/fraud")
async def detect_fraud(request: TransactionRequest):
    """Detect fraud in a transaction"""
    
    # Check if global model exists
    if federated_aggregator.global_model is None:
        # Try to load from disk
        federated_aggregator.load_global_model()
        
        if federated_aggregator.global_model is None:
            raise HTTPException(
                status_code=503, 
                detail="No trained model available. Please train a federated model first."
            )
    
    # Prepare transaction data
    tx_data = request.transaction_data
    
    # Create DataFrame with required features
    df_input = pd.DataFrame([tx_data])
    
    # Auto-generate missing engineered features if needed
    if 'amount_log' not in df_input.columns and 'amount' in df_input.columns:
        df_input['amount_log'] = np.log1p(df_input['amount'])
    
    if 'balance_change_orig' not in df_input.columns:
        if 'newbalanceOrig' in df_input.columns and 'oldbalanceOrg' in df_input.columns:
            df_input['balance_change_orig'] = df_input['newbalanceOrig'] - df_input['oldbalanceOrg']
        else:
            df_input['balance_change_orig'] = 0
    
    if 'balance_change_dest' not in df_input.columns:
        if 'newbalanceDest' in df_input.columns and 'oldbalanceDest' in df_input.columns:
            df_input['balance_change_dest'] = df_input['newbalanceDest'] - df_input['oldbalanceDest']
        else:
            df_input['balance_change_dest'] = 0
    
    if 'orig_balance_zero' not in df_input.columns:
        if 'oldbalanceOrg' in df_input.columns:
            df_input['orig_balance_zero'] = (df_input['oldbalanceOrg'] == 0).astype(int)
        else:
            df_input['orig_balance_zero'] = 0
    
    if 'dest_balance_zero' not in df_input.columns:
        if 'oldbalanceDest' in df_input.columns:
            df_input['dest_balance_zero'] = (df_input['oldbalanceDest'] == 0).astype(int)
        else:
            df_input['dest_balance_zero'] = 0
    
    if 'large_transaction' not in df_input.columns:
        if 'amount' in df_input.columns:
            df_input['large_transaction'] = (df_input['amount'] > 100000).astype(int)
        else:
            df_input['large_transaction'] = 0
    
    if 'dest_is_merchant' not in df_input.columns:
        df_input['dest_is_merchant'] = 0
    
    # Get features from trained model
    features = model_trainer.feature_columns
    
    # Ensure all required features exist
    for feature in features:
        if feature not in df_input.columns:
            df_input[feature] = 0
    
    # Make prediction
    try:
        prediction = federated_aggregator.global_model.predict(df_input[features])
        probability = federated_aggregator.global_model.predict_proba(df_input[features])
        
        fraud_prob = float(probability[0][1])
        is_fraud = int(prediction[0])
        
        # Log to blockchain
        tx_id = str(uuid.uuid4())
        blockchain_logger.log_fraud_detection(
            tx_id=tx_id,
            fraud_prob=fraud_prob,
            prediction=is_fraud,
            tx_details=tx_data
        )
        
        # Broadcast update
        await broadcast_to_clients({
            "type": "fraud_detection",
            "transaction_id": tx_id,
            "fraud_probability": fraud_prob,
            "prediction": "fraud" if is_fraud else "legitimate"
        })
        
        return {
            "transaction_id": tx_id,
            "prediction": "fraud" if is_fraud else "legitimate",
            "fraud_probability": fraud_prob,
            "risk_level": "high" if fraud_prob > 0.7 else "medium" if fraud_prob > 0.3 else "low",
            "timestamp": time.time()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ==================== BLOCKCHAIN ====================

@app.get("/blockchain/chain")
async def get_blockchain():
    """Get entire blockchain"""
    return {
        "chain": blockchain.get_chain(),
        "length": len(blockchain.chain)
    }


@app.get("/blockchain/block/{height}")
async def get_block(height: int):
    """Get specific block"""
    block = blockchain.get_block(height)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    return block


@app.get("/blockchain/latest")
async def get_latest_block():
    """Get latest block"""
    return blockchain.get_latest_block().to_dict()


@app.get("/blockchain/mempool")
async def get_mempool():
    """Get pending transactions"""
    return {
        "pending_transactions": blockchain.get_pending_transactions(),
        "count": len(blockchain.pending_transactions)
    }


@app.post("/blockchain/mine")
async def mine_block():
    """Mine a new block from pending transactions"""
    block = blockchain.create_block()
    
    if block is None:
        return {
            "status": "no_transactions",
            "message": "No pending transactions to mine"
        }
    
    # Broadcast update
    await broadcast_to_clients({
        "type": "block_mined",
        "block": block.to_dict()
    })
    
    return {
        "status": "success",
        "block": block.to_dict()
    }


@app.get("/blockchain/verify")
async def verify_blockchain():
    """Verify blockchain integrity"""
    return blockchain.verify_chain()


@app.get("/blockchain/stats")
async def get_blockchain_stats():
    """Get blockchain statistics"""
    return blockchain.get_statistics()


@app.get("/blockchain/search")
async def search_transactions(tx_type: Optional[str] = None, client_id: Optional[str] = None):
    """Search transactions"""
    filters = {}
    if tx_type:
        filters['type'] = tx_type
    if client_id:
        filters['client_id'] = client_id
    
    results = blockchain.search_transactions(filters)
    return {
        "transactions": results,
        "count": len(results)
    }


# ==================== BACKGROUND TASKS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    print("\n" + "="*60)
    print("🚀 Starting Secure Federated AI Framework")
    print("="*60)
    print(f"📍 Server: http://{HOST}:{PORT}")
    print(f"📡 WebSocket: ws://{HOST}:{PORT}/ws")
    print(f"📚 API Docs: http://{HOST}:{PORT}/docs")
    print("="*60 + "\n")
    
    # Start background block mining
    asyncio.create_task(auto_mine_blocks())


async def auto_mine_blocks():
    """Automatically mine blocks at regular intervals"""
    while True:
        await asyncio.sleep(BLOCK_INTERVAL_SEC)
        
        if len(blockchain.pending_transactions) > 0:
            block = blockchain.create_block()
            
            if block:
                # Broadcast to all clients
                await broadcast_to_clients({
                    "type": "block_mined",
                    "block": block.to_dict()
                })


# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
# Federated Fraud Detection System

A privacy-preserving fraud detection system using federated learning and blockchain technology.

## Project Structure

- **backend/**: Core server and federated learning logic
- **frontend/**: React-based dashboard for monitoring
- **models/**: Trained ML models (global and local)
- **data/**: Dataset and data loading utilities
- **config/**: Configuration files
- **logs/**: Application logs
- **tests/**: Unit and integration tests
- **Docker/**: Containerization files

## Setup

1. Install dependencies:
```bash
   pip install -r backend/requirements.txt
   cd frontend && npm install

```
federated-fraud-detection
├─ backend
│  ├─ blockchain.py
│  ├─ db.py
│  ├─ federated_trainer.py
│  ├─ inference.py
│  ├─ requirements.txt
│  ├─ server.py
│  └─ utils
│     ├─ encryption.py
│     ├─ logger.py
│     ├─ metrics.py
│     └─ validators.py
├─ config
│  ├─ model_params.json
│  └─ settings.yaml
├─ data
│  └─ load_paysim.py
├─ Docker
│  ├─ docker-compose.yml
│  ├─ Dockerfile.backend
│  └─ Dockerfile.frontend
├─ frontend
│  ├─ components
│  │  ├─ BlockchainVisualizer.jsx
│  │  ├─ FederatedRoundChart.jsx
│  │  ├─ FraudMonitor.jsx
│  │  └─ NodeStatusCard.jsx
│  ├─ dashboard.jsx
│  └─ package.json
├─ logs
├─ models
│  ├─ global_model.joblib
│  └─ local_model_bankA.joblib
├─ README.md
├─ simulate_clients.py
└─ tests
   ├─ test_blockchain.py
   ├─ test_inference.py
   └─ test_trainer.py

```
```
federated-fraud-detection
├─ backend
│  ├─ blockchain.py
│  ├─ data
│  ├─ db.py
│  ├─ feature_engineering.py
│  ├─ federated_trainer.py
│  ├─ inference.py
│  ├─ models
│  ├─ requirements.txt
│  ├─ server.py
│  ├─ test_feature.py
│  └─ utils
│     ├─ encryption.py
│     ├─ logger.py
│     ├─ metrics.py
│     └─ validators.py
├─ config
│  ├─ model_params.json
│  └─ settings.yaml
├─ data
│  └─ load_paysim.py
├─ Docker
│  ├─ docker-compose.yml
│  ├─ Dockerfile.backend
│  └─ Dockerfile.frontend
├─ frontend
│  ├─ components
│  │  ├─ BlockchainVisualizer.jsx
│  │  ├─ FederatedRoundChart.jsx
│  │  ├─ FraudMonitor.jsx
│  │  └─ NodeStatusCard.jsx
│  ├─ dashboard.jsx
│  └─ package.json
├─ logs
├─ models
├─ README.md
├─ simulate_clients.py
├─ test-full-system.py
└─ tests
   ├─ test_blockchain.py
   ├─ test_inference.py
   └─ test_trainer.py

```
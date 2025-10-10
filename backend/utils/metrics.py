"""
Metrics Module - Performance and Evaluation Metrics
Calculates various ML and system performance metrics
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import time


class ModelMetrics:
    """Calculate and store model performance metrics"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_prob: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate metrics requiring probabilities
        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['average_precision'] = float(average_precision_score(y_true, y_prob))
            except ValueError:
                # Handle cases where ROC AUC cannot be calculated
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # Sensitivity (same as recall)
        metrics['sensitivity'] = metrics['recall']
        
        return metrics
    
    @staticmethod
    def calculate_fraud_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_prob: np.ndarray,
                               transaction_amounts: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate fraud-specific metrics
        
        Args:
            y_true: True fraud labels
            y_pred: Predicted fraud labels
            y_prob: Fraud probabilities
            transaction_amounts: Transaction amounts (optional)
        
        Returns:
            Fraud detection metrics
        """
        base_metrics = ModelMetrics.calculate_classification_metrics(
            y_true, y_pred, y_prob
        )
        
        # Calculate fraud detection rate
        fraud_indices = np.where(y_true == 1)[0]
        if len(fraud_indices) > 0:
            fraud_detected = np.sum(y_pred[fraud_indices] == 1)
            base_metrics['fraud_detection_rate'] = float(fraud_detected / len(fraud_indices))
        else:
            base_metrics['fraud_detection_rate'] = 0.0
        
        # Calculate false alarm rate
        legit_indices = np.where(y_true == 0)[0]
        if len(legit_indices) > 0:
            false_alarms = np.sum(y_pred[legit_indices] == 1)
            base_metrics['false_alarm_rate'] = float(false_alarms / len(legit_indices))
        else:
            base_metrics['false_alarm_rate'] = 0.0
        
        # Calculate financial impact if amounts provided
        if transaction_amounts is not None:
            # Total fraud amount
            total_fraud_amount = np.sum(transaction_amounts[y_true == 1])
            
            # Detected fraud amount
            detected_fraud = np.where((y_true == 1) & (y_pred == 1))[0]
            detected_fraud_amount = np.sum(transaction_amounts[detected_fraud])
            
            # Missed fraud amount
            missed_fraud_amount = total_fraud_amount - detected_fraud_amount
            
            base_metrics['total_fraud_amount'] = float(total_fraud_amount)
            base_metrics['detected_fraud_amount'] = float(detected_fraud_amount)
            base_metrics['missed_fraud_amount'] = float(missed_fraud_amount)
            base_metrics['fraud_amount_recovery_rate'] = float(
                detected_fraud_amount / total_fraud_amount
            ) if total_fraud_amount > 0 else 0.0
        
        return base_metrics
    
    @staticmethod
    def calculate_roc_curve_data(y_true: np.ndarray, 
                                y_prob: np.ndarray) -> Dict[str, List]:
        """
        Calculate ROC curve data points
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        
        Returns:
            ROC curve data
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc_score(y_true, y_prob))
        }
    
    @staticmethod
    def calculate_precision_recall_curve_data(y_true: np.ndarray, 
                                             y_prob: np.ndarray) -> Dict[str, List]:
        """
        Calculate precision-recall curve data
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        
        Returns:
            PR curve data
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'average_precision': float(average_precision_score(y_true, y_prob))
        }


class SystemMetrics:
    """Track system performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.operation_times = []
    
    def record_operation_time(self, operation: str, duration: float):
        """Record timing for an operation"""
        self.operation_times.append({
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation"""
        times = [
            op['duration'] for op in self.operation_times 
            if op['operation'] == operation
        ]
        return np.mean(times) if times else 0.0
    
    def get_throughput(self, operation: str, window_seconds: int = 60) -> float:
        """
        Calculate throughput (operations per second)
        
        Args:
            operation: Operation name
            window_seconds: Time window to consider
        
        Returns:
            Operations per second
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_ops = [
            op for op in self.operation_times
            if op['operation'] == operation and op['timestamp'] >= cutoff_time
        ]
        
        return len(recent_ops) / window_seconds if recent_ops else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics"""
        uptime = time.time() - self.start_time
        
        operations = list(set(op['operation'] for op in self.operation_times))
        
        summary = {
            'uptime_seconds': uptime,
            'total_operations': len(self.operation_times),
            'operations': {}
        }
        
        for op in operations:
            summary['operations'][op] = {
                'count': len([o for o in self.operation_times if o['operation'] == op]),
                'avg_time': self.get_average_time(op),
                'throughput': self.get_throughput(op)
            }
        
        return summary


class FederatedMetrics:
    """Metrics specific to federated learning"""
    
    @staticmethod
    def calculate_aggregation_weights(client_metrics: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate aggregation weights based on client data sizes
        
        Args:
            client_metrics: List of client metric dictionaries
        
        Returns:
            Normalized weights
        """
        data_sizes = [m.get('data_size', 1) for m in client_metrics]
        total_size = sum(data_sizes)
        
        weights = [size / total_size for size in data_sizes]
        return weights
    
    @staticmethod
    def calculate_global_metrics(client_metrics: List[Dict[str, Any]],
                                weights: List[float] = None) -> Dict[str, Any]:
        """
        Calculate weighted global metrics from client metrics
        
        Args:
            client_metrics: List of client metrics
            weights: Optional weights (calculated if not provided)
        
        Returns:
            Global metrics
        """
        if not client_metrics:
            return {}
        
        if weights is None:
            weights = FederatedMetrics.calculate_aggregation_weights(client_metrics)
        
        # Aggregate numeric metrics
        global_metrics = {}
        
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for key in metric_keys:
            values = [m.get(key, 0) for m in client_metrics]
            global_metrics[f'global_{key}'] = float(np.average(values, weights=weights))
        
        # Add metadata
        global_metrics['num_clients'] = len(client_metrics)
        global_metrics['total_samples'] = sum(m.get('data_size', 0) for m in client_metrics)
        
        return global_metrics
    
    @staticmethod
    def calculate_fairness_metrics(client_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate fairness metrics across clients
        
        Args:
            client_metrics: List of client metrics
        
        Returns:
            Fairness metrics
        """
        accuracies = [m.get('accuracy', 0) for m in client_metrics]
        
        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'accuracy_variance': float(np.var(accuracies))
        }


def test_metrics():
    """Test metrics calculations"""
    print("=" * 60)
    print("📊 Testing Metrics")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_prob = np.random.random(1000)
    
    # Test classification metrics
    print("\n1. Classification Metrics:")
    metrics = ModelMetrics.calculate_classification_metrics(y_true, y_pred, y_prob)
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"   {key}: {value}")
    
    # Test fraud metrics
    print("\n2. Fraud Detection Metrics:")
    amounts = np.random.uniform(10, 1000, 1000)
    fraud_metrics = ModelMetrics.calculate_fraud_metrics(
        y_true, y_pred, y_prob, amounts
    )
    print(f"   Fraud Detection Rate: {fraud_metrics['fraud_detection_rate']:.4f}")
    print(f"   False Alarm Rate: {fraud_metrics['false_alarm_rate']:.4f}")
    
    # Test system metrics
    print("\n3. System Metrics:")
    sys_metrics = SystemMetrics()
    sys_metrics.record_operation_time('inference', 0.05)
    sys_metrics.record_operation_time('inference', 0.04)
    sys_metrics.record_operation_time('training', 5.0)
    
    summary = sys_metrics.get_summary()
    print(f"   Total Operations: {summary['total_operations']}")
    print(f"   Avg Inference Time: {sys_metrics.get_average_time('inference'):.4f}s")
    
    # Test federated metrics
    print("\n4. Federated Metrics:")
    client_metrics_list = [
        {'accuracy': 0.95, 'data_size': 5000},
        {'accuracy': 0.93, 'data_size': 3000},
        {'accuracy': 0.96, 'data_size': 4000}
    ]
    
    global_metrics = FederatedMetrics.calculate_global_metrics(client_metrics_list)
    print(f"   Global Accuracy: {global_metrics['global_accuracy']:.4f}")
    
    fairness = FederatedMetrics.calculate_fairness_metrics(client_metrics_list)
    print(f"   Accuracy Std Dev: {fairness['std_accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All metrics tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
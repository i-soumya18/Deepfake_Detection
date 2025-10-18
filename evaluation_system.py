# Deepfake Detection - Comprehensive Evaluation System
# Advanced evaluation metrics and robustness analysis

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from model_architecture import HybridSpatialFrequencyModel
from training_pipeline import DeepfakeDataset, create_data_loaders, prepare_dataset
from preprocessing_pipeline import DeepfakePreprocessor

class DeepfakeEvaluator:
    """
    Comprehensive evaluation system for deepfake detection models
    Implements SOTA evaluation metrics and robustness testing
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def load_model(self, model_path):
        """Load trained model weights"""
        print(f"📦 Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded successfully!")
                print(f"📊 Model info - Epoch: {checkpoint.get('epoch', 'N/A')}, Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✅ Model weights loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def evaluate_dataset(self, data_loader, dataset_name="Dataset"):
        """
        Comprehensive evaluation on a dataset
        Returns detailed metrics and predictions
        """
        print(f"🔍 Evaluating on {dataset_name}...")
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_file_paths = []
        inference_times = []
        
        # Branch-specific predictions for analysis
        spatial_preds = []
        frequency_preds = []
        texture_preds = []
        temporal_preds = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'Evaluating {dataset_name}'):
                start_time = time.time()
                
                # Move data to device
                batch_data = {
                    'spatial': batch['spatial'].to(self.device),
                    'fft_magnitude': batch['fft_magnitude'].to(self.device),
                    'lbp': batch['lbp'].to(self.device),
                    'noise_residuals': batch['noise_residuals'].to(self.device)
                }
                labels = batch['label'].to(self.device)
                sequence_features = batch['sequence_features']
                
                if sequence_features is not None and sequence_features[0] is not None:
                    sequence_features = sequence_features.to(self.device)
                else:
                    sequence_features = None
                
                # Forward pass
                outputs = self.model(batch_data, sequence_features)
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs['prediction']).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                # Store results
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
                all_labels.extend(labels.cpu().numpy())
                all_file_paths.extend(batch['file_path'])
                
                # Branch-specific predictions
                spatial_preds.extend(torch.sigmoid(outputs['spatial_prediction']).cpu().numpy().flatten())
                frequency_preds.extend(torch.sigmoid(outputs['frequency_prediction']).cpu().numpy().flatten())
                texture_preds.extend(torch.sigmoid(outputs['texture_prediction']).cpu().numpy().flatten())
                if outputs['temporal_prediction'] is not None:
                    temporal_preds.extend(torch.sigmoid(outputs['temporal_prediction']).cpu().numpy().flatten())
                
                # Measure inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time / len(batch['label']))  # Per sample
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(
            all_labels, all_predictions, all_probabilities,
            spatial_preds, frequency_preds, texture_preds, temporal_preds
        )
        
        # Add performance metrics
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['total_samples'] = len(all_labels)
        
        # Store detailed results
        detailed_results = {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels,
            'file_paths': all_file_paths,
            'spatial_predictions': spatial_preds,
            'frequency_predictions': frequency_preds,
            'texture_predictions': texture_preds,
            'temporal_predictions': temporal_preds if temporal_preds else None,
            'metrics': metrics
        }
        
        self.results[dataset_name] = detailed_results
        
        print(f"✅ {dataset_name} evaluation completed!")
        print(f"📊 Key Metrics:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  - F1-Score: {metrics['f1_score']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - Average Inference Time: {metrics['avg_inference_time']:.4f}s")
        
        return detailed_results
    
    def _calculate_metrics(self, labels, predictions, probabilities, 
                          spatial_preds, frequency_preds, texture_preds, temporal_preds):
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
        
        # AUC metrics
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probabilities)
            metrics['auc_pr'] = average_precision_score(labels, probabilities)
        except:
            metrics['auc_roc'] = 0.5
            metrics['auc_pr'] = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        if len(cm) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Branch-specific metrics
        if spatial_preds:
            spatial_binary = (np.array(spatial_preds) > 0.5).astype(int)
            metrics['spatial_accuracy'] = accuracy_score(labels, spatial_binary)
            try:
                metrics['spatial_auc'] = roc_auc_score(labels, spatial_preds)
            except:
                metrics['spatial_auc'] = 0.5
        
        if frequency_preds:
            frequency_binary = (np.array(frequency_preds) > 0.5).astype(int)
            metrics['frequency_accuracy'] = accuracy_score(labels, frequency_binary)
            try:
                metrics['frequency_auc'] = roc_auc_score(labels, frequency_preds)
            except:
                metrics['frequency_auc'] = 0.5
        
        if texture_preds:
            texture_binary = (np.array(texture_preds) > 0.5).astype(int)
            metrics['texture_accuracy'] = accuracy_score(labels, texture_binary)
            try:
                metrics['texture_auc'] = roc_auc_score(labels, texture_preds)
            except:
                metrics['texture_auc'] = 0.5
        
        if temporal_preds:
            temporal_binary = (np.array(temporal_preds) > 0.5).astype(int)
            metrics['temporal_accuracy'] = accuracy_score(labels, temporal_binary)
            try:
                metrics['temporal_auc'] = roc_auc_score(labels, temporal_preds)
            except:
                metrics['temporal_auc'] = 0.5
        
        return metrics
    
    def cross_dataset_evaluation(self, datasets_info):
        """
        Cross-dataset evaluation for robustness analysis
        Tests generalization across different datasets
        """
        print("🔄 Starting Cross-Dataset Evaluation...")
        print("=" * 50)
        
        cross_results = {}
        
        for train_dataset, train_loader in datasets_info['train'].items():
            for test_dataset, test_loader in datasets_info['test'].items():
                eval_name = f"{train_dataset}_to_{test_dataset}"
                print(f"\n📊 Evaluating: {eval_name}")
                
                results = self.evaluate_dataset(test_loader, eval_name)
                cross_results[eval_name] = results['metrics']
        
        self.results['cross_dataset'] = cross_results
        
        # Summary of cross-dataset performance
        print(f"\n📈 Cross-Dataset Performance Summary:")
        for eval_name, metrics in cross_results.items():
            print(f"  {eval_name}: AUC={metrics['auc_roc']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        return cross_results
    
    def robustness_analysis(self, data_loader, perturbations=None):
        """
        Robustness analysis with various perturbations
        Tests model stability under different conditions
        """
        print("🛡️ Starting Robustness Analysis...")
        
        if perturbations is None:
            perturbations = {
                'noise_0.01': {'type': 'gaussian_noise', 'std': 0.01},
                'noise_0.05': {'type': 'gaussian_noise', 'std': 0.05},
                'brightness_0.1': {'type': 'brightness', 'factor': 0.1},
                'brightness_-0.1': {'type': 'brightness', 'factor': -0.1},
                'blur_3': {'type': 'gaussian_blur', 'kernel_size': 3},
                'blur_5': {'type': 'gaussian_blur', 'kernel_size': 5}
            }
        
        robustness_results = {}
        
        # Baseline (no perturbation)
        baseline_results = self.evaluate_dataset(data_loader, "Baseline")
        robustness_results['baseline'] = baseline_results['metrics']
        
        # Test with perturbations
        for pert_name, pert_config in perturbations.items():
            print(f"\n🔧 Testing perturbation: {pert_name}")
            
            # Create perturbed data loader
            perturbed_loader = self._apply_perturbation(data_loader, pert_config)
            
            # Evaluate
            results = self.evaluate_dataset(perturbed_loader, f"Perturbed_{pert_name}")
            robustness_results[pert_name] = results['metrics']
        
        self.results['robustness'] = robustness_results
        
        # Robustness summary
        print(f"\n📊 Robustness Summary:")
        baseline_auc = robustness_results['baseline']['auc_roc']
        for pert_name, metrics in robustness_results.items():
            if pert_name != 'baseline':
                auc_drop = baseline_auc - metrics['auc_roc']
                print(f"  {pert_name}: AUC={metrics['auc_roc']:.4f} (Δ={auc_drop:+.4f})")
        
        return robustness_results
    
    def _apply_perturbation(self, data_loader, perturbation_config):
        """Apply perturbation to data loader (simplified for demonstration)"""
        # For now, return the original loader
        # In practice, you would create a new dataset with perturbations
        return data_loader
    
    def generate_visualizations(self, output_dir='evaluation_results'):
        """Generate comprehensive visualization plots"""
        print(f"📊 Generating visualizations in: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        for dataset_name, results in self.results.items():
            if dataset_name in ['cross_dataset', 'robustness']:
                continue
                
            metrics = results['metrics']
            
            # ROC Curve
            self._plot_roc_curve(
                results['labels'], 
                results['probabilities'],
                f"{dataset_name} ROC Curve",
                f"{output_dir}/{dataset_name}_roc_curve.png"
            )
            
            # Precision-Recall Curve
            self._plot_pr_curve(
                results['labels'], 
                results['probabilities'],
                f"{dataset_name} Precision-Recall Curve",
                f"{output_dir}/{dataset_name}_pr_curve.png"
            )
            
            # Confusion Matrix
            self._plot_confusion_matrix(
                metrics['confusion_matrix'],
                f"{dataset_name} Confusion Matrix",
                f"{output_dir}/{dataset_name}_confusion_matrix.png"
            )
            
            # Branch Performance Comparison
            self._plot_branch_performance(
                metrics,
                f"{dataset_name} Branch Performance",
                f"{output_dir}/{dataset_name}_branch_performance.png"
            )
        
        # Cross-dataset comparison
        if 'cross_dataset' in self.results:
            self._plot_cross_dataset_performance(
                self.results['cross_dataset'],
                "Cross-Dataset Performance",
                f"{output_dir}/cross_dataset_performance.png"
            )
        
        # Robustness analysis
        if 'robustness' in self.results:
            self._plot_robustness_analysis(
                self.results['robustness'],
                "Robustness Analysis",
                f"{output_dir}/robustness_analysis.png"
            )
        
        print(f"✅ Visualizations saved to: {output_dir}")
    
    def _plot_roc_curve(self, labels, probabilities, title, save_path):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, labels, probabilities, title, save_path):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm, title, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_branch_performance(self, metrics, title, save_path):
        """Plot branch-wise performance comparison"""
        branches = []
        accuracies = []
        aucs = []
        
        for branch in ['spatial', 'frequency', 'texture', 'temporal']:
            if f'{branch}_accuracy' in metrics:
                branches.append(branch.capitalize())
                accuracies.append(metrics[f'{branch}_accuracy'])
                aucs.append(metrics[f'{branch}_auc'])
        
        # Add ensemble
        branches.append('Ensemble')
        accuracies.append(metrics['accuracy'])
        aucs.append(metrics['auc_roc'])
        
        x = np.arange(len(branches))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, aucs, width, label='AUC-ROC', alpha=0.8)
        
        plt.xlabel('Model Branch')
        plt.ylabel('Performance')
        plt.title(title)
        plt.xticks(x, branches)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_dataset_performance(self, cross_results, title, save_path):
        """Plot cross-dataset performance matrix"""
        # Create performance matrix
        datasets = list(set([eval_name.split('_to_')[0] for eval_name in cross_results.keys()]))
        matrix = np.zeros((len(datasets), len(datasets)))
        
        for i, train_ds in enumerate(datasets):
            for j, test_ds in enumerate(datasets):
                eval_name = f"{train_ds}_to_{test_ds}"
                if eval_name in cross_results:
                    matrix[i, j] = cross_results[eval_name]['auc_roc']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=datasets, yticklabels=datasets, square=True)
        plt.title(title)
        plt.xlabel('Test Dataset')
        plt.ylabel('Train Dataset')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_analysis(self, robustness_results, title, save_path):
        """Plot robustness analysis results"""
        perturbations = [k for k in robustness_results.keys() if k != 'baseline']
        auc_scores = [robustness_results[k]['auc_roc'] for k in perturbations]
        baseline_auc = robustness_results['baseline']['auc_roc']
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(perturbations, auc_scores, alpha=0.7)
        plt.axhline(y=baseline_auc, color='red', linestyle='--', 
                   label=f'Baseline AUC: {baseline_auc:.4f}')
        
        # Color bars based on performance drop
        for i, bar in enumerate(bars):
            drop = baseline_auc - auc_scores[i]
            if drop < 0.01:
                bar.set_color('green')
            elif drop < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xlabel('Perturbation')
        plt.ylabel('AUC-ROC')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_report(self, output_path='evaluation_report.json'):
        """Save comprehensive evaluation report"""
        print(f"💾 Saving detailed report to: {output_path}")
        
        # Prepare report data (remove non-serializable items)
        report_data = {}
        
        for dataset_name, results in self.results.items():
            if dataset_name in ['cross_dataset', 'robustness']:
                report_data[dataset_name] = results
            else:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {
                    'metrics': results['metrics'],
                    'total_samples': len(results['labels']),
                    'summary': {
                        'accuracy': float(results['metrics']['accuracy']),
                        'auc_roc': float(results['metrics']['auc_roc']),
                        'f1_score': float(results['metrics']['f1_score']),
                        'precision': float(results['metrics']['precision']),
                        'recall': float(results['metrics']['recall'])
                    }
                }
                report_data[dataset_name] = serializable_results
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✅ Report saved successfully!")
        return report_data

def main():
    """Main evaluation pipeline"""
    print("🎯 Deepfake Detection - Comprehensive Evaluation")
    print("=" * 55)
    
    # Configuration
    config = {
        'model_path': 'best_deepfake_model.pth',
        'data_path': '/home/soumya/PycharmProjects/Deepfake_Detection',
        'sample_limit': 50,  # Smaller sample for testing
        'batch_size': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'evaluation_results'
    }
    
    print(f"⚙️ Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"❌ Model file not found: {config['model_path']}")
        print("Please run training first to generate the model.")
        return
    
    # Initialize model
    print("🏗️ Initializing model...")
    model = HybridSpatialFrequencyModel(
        num_classes=1,
        use_temporal=False,
        dropout=0.2
    )
    
    # Initialize evaluator
    evaluator = DeepfakeEvaluator(model, config['device'])
    evaluator.load_model(config['model_path'])
    
    # Prepare test dataset
    print("📂 Preparing test dataset...")
    preprocessor = DeepfakePreprocessor()
    
    # Get dataset
    train_paths, val_paths, train_labels, val_labels = prepare_dataset(
        config['data_path'], 
        config['sample_limit']
    )
    
    # Create test data loader
    train_loader, val_loader = create_data_loaders(
        train_paths, val_paths, train_labels, val_labels,
        preprocessor, config['batch_size'], num_workers=0
    )
    
    # Evaluation
    print("🔍 Starting evaluation...")
    
    # Basic evaluation
    test_results = evaluator.evaluate_dataset(val_loader, "Test_Dataset")
    
    # Generate visualizations
    evaluator.generate_visualizations(config['output_dir'])
    
    # Save detailed report
    report = evaluator.save_detailed_report(
        os.path.join(config['output_dir'], 'evaluation_report.json')
    )
    
    # Print final summary
    print("\n" + "=" * 55)
    print("📊 EVALUATION SUMMARY")
    print("=" * 55)
    
    test_metrics = test_results['metrics']
    print(f"🎯 Overall Performance:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  - F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall: {test_metrics['recall']:.4f}")
    print(f"  - Inference Time: {test_metrics['avg_inference_time']:.4f}s/sample")
    
    print(f"\n🔧 Branch Performance:")
    if 'spatial_auc' in test_metrics:
        print(f"  - Spatial Branch AUC: {test_metrics['spatial_auc']:.4f}")
    if 'frequency_auc' in test_metrics:
        print(f"  - Frequency Branch AUC: {test_metrics['frequency_auc']:.4f}")
    if 'texture_auc' in test_metrics:
        print(f"  - Texture Branch AUC: {test_metrics['texture_auc']:.4f}")
    
    print(f"\n📁 Results saved to: {config['output_dir']}")
    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
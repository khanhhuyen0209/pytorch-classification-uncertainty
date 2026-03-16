"""
Automated Experiment Runner for Dirichlet Prior Testing
Tests alpha = evidence + prior for prior values from 0 to 4 with step 0.5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from train import train_model
from losses import edl_mse_loss
from data import get_mnist_dataloaders
from lenet import LeNet5
from helpers import get_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, 
        num_epochs=25,
        batch_size=32,
        learning_rate=0.001,
        num_classes=10,
    ):
        """Initialize experiment runner with hyperparameters"""
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.device = get_device()
        self.results = []
        
        # Create results directory
        self.results_dir = Path("./experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Experiment Runner initialized on device: {self.device}")
    
    def create_model_and_optimizer(self):
        """Create a fresh model and optimizer"""
        model = LeNet5(self.num_classes)
        model = model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        return model, optimizer
    
    def run_experiments(self, prior_values=None, uncertainty=True):
        """
        Run experiments with different prior values
        
        Args:
            prior_values: List of prior values to test (default: 0 to 4 with step 0.5)
            uncertainty: Whether to use uncertainty mode
        """
        if prior_values is None:
            # Generate prior values: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
            prior_values = [i * 0.5 for i in range(9)]
        
        logger.info(f"Starting experiments with prior values: {prior_values}")
        logger.info(f"Number of epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        
        total_experiments = len(prior_values)
        
        for idx, prior in enumerate(prior_values, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"[{idx}/{total_experiments}] Running experiment with PRIOR = {prior}")
            logger.info(f"{'='*70}")
            
            try:
                # Create fresh model for each experiment
                model, optimizer = self.create_model_and_optimizer()
                
                # Create modified loss function with custom prior
                def custom_loss(output, target, epoch_num, num_classes, annealing_step, device=None, prior=prior):
                    return edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device, prior)
                
                # Get dataloaders
                dataloaders = get_mnist_dataloaders(self.batch_size)
                
                # Train model
                model, losses, accuracy, evidences, exp_log = train_model(
                    model=model,
                    dataloaders=dataloaders,
                    num_classes=self.num_classes,
                    criterion=custom_loss,
                    optimizer=optimizer,
                    scheduler=None,
                    num_epochs=self.num_epochs,
                    device=self.device,
                    uncertainty=uncertainty,
                    prior=prior
                )
                
                # Save results
                self._save_experiment_results(prior, exp_log, losses, accuracy)
                
                # Add to summary
                self.results.append({
                    'prior': prior,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'final_accuracy': accuracy['accuracy'][-1] if accuracy['accuracy'] else None,
                    'final_loss': losses['loss'][-1] if losses['loss'] else None
                })
                
                logger.info(f"✓ Completed experiment with prior={prior}")
                
            except Exception as e:
                logger.error(f"✗ Failed experiment with prior={prior}: {str(e)}", exc_info=True)
                self.results.append({
                    'prior': prior,
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
        
        logger.info(f"\n{'='*70}")
        logger.info("All experiments completed!")
        logger.info(f"{'='*70}")
        
        # Save final summary
        self._save_summary()
    
    def _save_experiment_results(self, prior, exp_log, losses, accuracy):
        """Save detailed results for each experiment"""
        filename = self.results_dir / f"prior_{prior:.1f}.json"
        
        result_data = {
            'prior': prior,
            'timestamp': datetime.now().isoformat(),
            'experiment_log': exp_log,
            'losses': losses,
            'accuracy': accuracy
        }
        
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        logger.info(f"Saved detailed results to {filename}")
    
    def _save_summary(self):
        """Save summary of all experiments"""
        # Save as CSV
        summary_csv = self.results_dir / "summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            fieldnames = ['prior', 'status', 'timestamp', 'final_accuracy', 'final_loss', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    'prior': result['prior'],
                    'status': result['status'],
                    'timestamp': result['timestamp'],
                    'final_accuracy': result.get('final_accuracy', ''),
                    'final_loss': result.get('final_loss', ''),
                    'error': result.get('error', '')
                })
        
        logger.info(f"Saved CSV summary to {summary_csv}")
        
        # Save as JSON
        summary_json = self.results_dir / "summary.json"
        with open(summary_json, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Saved JSON summary to {summary_json}")
        
        # Print summary to console
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*70)
        for result in self.results:
            status_symbol = "✓" if result['status'] == 'completed' else "✗"
            logger.info(f"{status_symbol} Prior={result['prior']}: {result['status']}")
            if result.get('final_accuracy'):
                logger.info(f"   Final Accuracy: {result['final_accuracy']:.4f}")
            if result.get('error'):
                logger.info(f"   Error: {result['error']}")


def main():
    """Main entry point for running experiments""" 
    
    # Configuration
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001
    num_classes = 10
    
    # Prior values to test (0.1 to 2.0 with step 0.1)
    prior_values = [round(i * 0.1, 2) for i in range(1, 11)]   # 0.1 → 2.0 (skip 0.0 to avoid NaN in KL)
    
    # Create and run experiments
    runner = ExperimentRunner(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_classes=num_classes
    )
    
    runner.run_experiments(prior_values=prior_values, uncertainty=True)


if __name__ == "__main__":
    main()

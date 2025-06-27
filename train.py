"""
Main training script for DYNAMO.
Orchestrates the three-phase training pipeline.
"""

import os
import argparse
import torch
import wandb
from typing import Dict, Any, Optional

from model import DynamoModel
from training import run_phase1_training, run_phase2_training, run_phase3_training
from evaluation import create_evaluator, create_baseline_collection, create_routing_analyzer
from utils import Config, get_config, get_logger, set_seed

logger = get_logger(__name__)


def setup_wandb(config: Config) -> None:
    """Setup Weights & Biases logging."""
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            config=config.__dict__,
            tags=config.wandb_tags
        )
        logger.info("Weights & Biases initialized")


def load_checkpoint(model: DynamoModel, checkpoint_path: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return {}


def save_final_model(model: DynamoModel, config: Config) -> str:
    """Save the final trained model."""
    save_dir = os.path.join(config.checkpoint_dir, "final_model")
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_model(save_dir)
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    config.save(config_path)
    
    logger.info(f"Final model saved to {save_dir}")
    return save_dir


def run_evaluation(
    model: DynamoModel, 
    config: Config,
    phase_name: str = "final"
) -> Dict[str, Any]:
    """Run comprehensive evaluation."""
    logger.info(f"Running {phase_name} evaluation...")
    
    # Create evaluator
    evaluator = create_evaluator(model.task_names)
    
    # Load test datasets
    from data import DatasetLoader
    data_loader = DatasetLoader(config.__dict__)
    test_datasets = data_loader.create_datasets('test')
    test_dataloaders = data_loader.create_dataloaders(
        test_datasets,
        batch_size=config.evaluation.eval_batch_size,
        shuffle=False
    )
    
    # Evaluate DYNAMO model
    results = {}
    device = torch.device(config.device)
    
    for task_name, dataloader in test_dataloaders.items():
        logger.info(f"Evaluating on {task_name}...")
        task_results = evaluator.evaluate_model(model, dataloader, device)
        results[task_name] = task_results
    
    # Create and evaluate baselines
    baseline_collection = create_baseline_collection(model, config.__dict__)
    baseline_results = {}
    
    for baseline_name, baseline_model in baseline_collection.get_all_baselines().items():
        logger.info(f"Evaluating baseline: {baseline_name}")
        baseline_task_results = {}
        
        for task_name, dataloader in test_dataloaders.items():
            try:
                task_results = evaluator.evaluate_model(baseline_model, dataloader, device)
                baseline_task_results[task_name] = task_results
            except Exception as e:
                logger.warning(f"Failed to evaluate {baseline_name} on {task_name}: {e}")
                baseline_task_results[task_name] = {}
        
        baseline_results[baseline_name] = baseline_task_results
    
    # Routing analysis
    logger.info("Analyzing routing decisions...")
    analyzer = create_routing_analyzer(model, model.task_names)
    
    # Use a subset of test data for analysis
    analysis_dataloader = list(test_dataloaders.values())[0]  # Use first task's dataloader
    routing_analysis = analyzer.analyze_routing_decisions(
        analysis_dataloader, device, max_samples=500
    )
    
    # Generate visualizations
    viz_dir = os.path.join(config.log_dir, f"{phase_name}_visualizations")
    analyzer.visualize_routing_patterns(viz_dir)
    
    # Generate routing report
    report_path = os.path.join(config.log_dir, f"{phase_name}_routing_report.json")
    analyzer.generate_routing_report(routing_analysis, report_path)
    
    evaluation_results = {
        'dynamo_results': results,
        'baseline_results': baseline_results,
        'routing_analysis': routing_analysis,
        'parameter_efficiency': baseline_collection.compare_parameter_efficiency()
    }
    
    # Log to wandb
    if config.use_wandb:
        wandb.log({f"{phase_name}_evaluation": evaluation_results})
    
    logger.info(f"{phase_name} evaluation completed")
    return evaluation_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DYNAMO model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--phase", type=str, choices=["1", "2", "3", "all"], default="all",
                       help="Training phase to run")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only run evaluation, skip training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Set random seed
    set_seed(config.seed)
    
    # Setup logging
    logger.info("="*50)
    logger.info("DYNAMO Training Pipeline")
    logger.info("="*50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Seed: {config.seed}")
    
    # Setup wandb
    setup_wandb(config)
    
    # Initialize model
    logger.info("Initializing DYNAMO model...")
    model = DynamoModel(config.__dict__)
    
    # Print model info
    from utils.helpers import count_parameters
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(model, args.resume)
    
    # Training phases
    if not args.eval_only:
        training_metrics = {}
        
        if args.phase in ["1", "all"]:
            logger.info("\n" + "="*50)
            logger.info("PHASE 1: Individual LoRA Training")
            logger.info("="*50)
            
            phase1_metrics = run_phase1_training(config, model)
            training_metrics['phase1'] = phase1_metrics
            
            # Intermediate evaluation
            if config.evaluation.eval_after_each_phase:
                eval_results = run_evaluation(model, config, "phase1")
                training_metrics['phase1_evaluation'] = eval_results
        
        if args.phase in ["2", "all"]:
            logger.info("\n" + "="*50)
            logger.info("PHASE 2: Router Training")
            logger.info("="*50)
            
            phase2_metrics = run_phase2_training(config, model)
            training_metrics['phase2'] = phase2_metrics
            
            # Intermediate evaluation
            if config.evaluation.eval_after_each_phase:
                eval_results = run_evaluation(model, config, "phase2")
                training_metrics['phase2_evaluation'] = eval_results
        
        if args.phase in ["3", "all"]:
            logger.info("\n" + "="*50)
            logger.info("PHASE 3: Joint Fine-tuning")
            logger.info("="*50)
            
            phase3_metrics = run_phase3_training(config, model)
            training_metrics['phase3'] = phase3_metrics
        
        # Save training metrics
        metrics_path = os.path.join(config.log_dir, "training_metrics.pt")
        torch.save(training_metrics, metrics_path)
        logger.info(f"Training metrics saved to {metrics_path}")
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)
    
    final_evaluation = run_evaluation(model, config, "final")
    
    # Save final model
    if not args.eval_only:
        final_model_path = save_final_model(model, config)
        logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    
    if 'dynamo_results' in final_evaluation:
        for task_name, task_results in final_evaluation['dynamo_results'].items():
            if 'overall' in task_results:
                overall = task_results['overall']
                if 'avg_accuracy' in overall:
                    logger.info(f"{task_name}: {overall['avg_accuracy']:.3f} accuracy")
    
    if 'routing_analysis' in final_evaluation:
        routing = final_evaluation['routing_analysis']
        if 'routing_patterns' in routing and 'routing_accuracy' in routing['routing_patterns']:
            acc = routing['routing_patterns']['routing_accuracy']
            logger.info(f"Routing accuracy: {acc:.3f}")
    
    # Close wandb
    if config.use_wandb:
        wandb.finish()
    
    logger.info("DYNAMO training pipeline completed successfully!")


if __name__ == "__main__":
    main()


# DYNAMO: Dynamic Neural Adapter Multi-task Optimization

A PyTorch implementation of DYNAMO, a novel multi-task learning framework that uses dynamic routing to select task-specific LoRA adapters based on input characteristics.

## Overview

DYNAMO addresses the challenge of multi-task learning by combining:
- **Frozen RoBERTa backbone** for shared representations
- **Task-specific LoRA adapters** for efficient task specialization
- **Dynamic MLP router** for intelligent adapter selection
- **Three-phase training pipeline** for optimal performance

### Key Features

- 🎯 **Dynamic Routing**: Automatically selects the most appropriate adapters for each input
- 🔧 **Parameter Efficient**: Uses LoRA adapters instead of full fine-tuning
- 📊 **Multi-task Support**: Handles 5 different NLP tasks simultaneously
- 🎓 **Curriculum Learning**: Progressive training strategy for better convergence
- 📈 **Comprehensive Evaluation**: Includes baselines and routing analysis tools

## Architecture

```
Input Text
    ↓
RoBERTa Backbone (Frozen)
    ↓
Dynamic Router → Task-specific LoRA Adapters
    ↓           ↓         ↓         ↓
    └─→ Sentiment  QA  Summarization  Code Gen  Translation
```

## Supported Tasks

1. **Sentiment Analysis** (SST-2): Binary sentiment classification
2. **Question Answering** (SQuAD): Extractive question answering
3. **Summarization** (XSum): Abstractive text summarization
4. **Code Generation**: Python code generation from descriptions
5. **Translation**: English to other languages

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd dynamo_project

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional evaluation libraries
pip install rouge-score sacrebleu scikit-learn
```

## Project Structure

```
dynamo_project/
├── model/                      # Core model implementations
│   ├── __init__.py
│   ├── roberta_backbone.py     # Frozen RoBERTa backbone
│   ├── lora_adapters.py        # Task-specific LoRA adapters
│   ├── dynamic_router.py       # MLP-based routing network
│   └── dynamo_model.py         # Main DYNAMO model
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── losses.py               # Custom loss functions
│   ├── phase1_lora_training.py # Individual LoRA training
│   ├── phase2_router_training.py # Router training
│   └── phase3_joint_finetuning.py # Joint optimization
├── data/                       # Data loading and processing
│   ├── __init__.py
│   ├── dataset_loaders.py      # Task-specific datasets
│   └── mixed_task_dataset.py   # Mixed-task data generation
├── evaluation/                 # Evaluation and analysis
│   ├── __init__.py
│   ├── baselines.py            # Baseline model implementations
│   ├── metrics.py              # Task-specific metrics
│   └── analyzer.py             # Routing analysis tools
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── logger.py               # Logging utilities
│   └── helpers.py              # Helper functions
├── train.py                    # Main training script
├── example_usage.py            # Usage examples
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Configuration

Edit `config.yaml` to set your data paths and hyperparameters:

```yaml
# Data paths
data:
  sentiment_data_path: "./data/sst2"
  qa_data_path: "./data/squad"
  # ... other paths

# Training settings
training:
  num_epochs: 10
  batch_size: 16
  lora_lr: 5e-4
  router_lr: 1e-3
```

### 2. Training

Run the complete three-phase training pipeline:

```bash
# Train all phases
python train.py --config config.yaml --phase all

# Or train individual phases
python train.py --config config.yaml --phase 1  # LoRA training
python train.py --config config.yaml --phase 2  # Router training
python train.py --config config.yaml --phase 3  # Joint fine-tuning
```

### 3. Inference

Use the trained model for predictions:

```python
from example_usage import DynamoInference

# Load trained model
dynamo = DynamoInference("./checkpoints/final_model", "./config.yaml")

# Make predictions
result = dynamo.predict(
    "I love this movie!", 
    task_hint="sentiment",
    return_routing_info=True
)

print(f"Sentiment: {result['sentiment']['label']}")
print(f"Router predicted: {result['routing_info']['predicted_task']}")
```

## Training Pipeline

DYNAMO uses a three-phase training strategy:

### Phase 1: Individual LoRA Training
- Train each LoRA adapter independently on task-specific data
- Freeze the RoBERTa backbone
- Optimize task-specific performance

### Phase 2: Router Training
- Freeze trained LoRA adapters
- Train the dynamic router using mixed-task data
- Optimize routing decisions with complex loss function

### Phase 3: Joint Fine-tuning
- Joint optimization of router and adapters
- Curriculum learning strategy
- Final performance optimization

## Loss Functions

DYNAMO uses a sophisticated loss function combining:

- **Task-specific losses**: Standard losses for each task
- **Load balancing loss**: Encourages balanced adapter usage
- **Efficiency loss**: Promotes sparse routing decisions
- **Consistency loss**: Ensures stable routing for similar inputs
- **Regularization terms**: Entropy and temperature regularization

## Evaluation

### Baselines

The framework includes several baseline implementations:

- **Oracle Routing**: Perfect task knowledge (upper bound)
- **Single LoRA**: One large adapter for all tasks
- **Full Fine-tuning**: Traditional multi-task learning
- **Random Routing**: Random adapter selection
- **Uniform Routing**: Equal weights for all adapters

### Metrics

Task-specific metrics include:
- **Sentiment**: Accuracy, Precision, Recall, F1
- **QA**: Exact Match, F1, Start/End Accuracy
- **Generation**: ROUGE, BLEU, Exact Match
- **Routing**: Accuracy, Entropy, Load Balance

### Analysis Tools

Comprehensive routing analysis:
- Routing decision patterns
- Input-routing correlations
- Confidence analysis
- Task confusion matrices
- Visualization tools

## Configuration Options

### Model Configuration
```yaml
model:
  base_model_name: "roberta-base"  # Base model
  hidden_size: 768                 # Hidden dimension
  freeze_backbone: true            # Freeze backbone
```

### LoRA Configuration
```yaml
lora_configs:
  sentiment:
    rank: 16                       # LoRA rank
    alpha: 32                      # LoRA alpha
    dropout: 0.1                   # Dropout rate
    target_modules: ["query", "value"]  # Target modules
```

### Router Configuration
```yaml
router:
  hidden_sizes: [768, 512, 256]   # MLP hidden sizes
  dropout: 0.1                    # Dropout rate
  temperature: 1.0                # Initial temperature
```

### Training Configuration
```yaml
training:
  num_epochs: 10                  # Training epochs
  batch_size: 16                  # Batch size
  lora_lr: 5e-4                   # LoRA learning rate
  router_lr: 1e-3                 # Router learning rate
  load_balance_weight: 0.1        # Loss weights
```

## Advanced Usage

### Custom Tasks

To add a new task:

1. **Create dataset loader** in `data/dataset_loaders.py`
2. **Add LoRA configuration** in config file
3. **Implement task head** in `model/lora_adapters.py`
4. **Add metrics** in `evaluation/metrics.py`

### Custom Loss Functions

Implement custom losses in `training/losses.py`:

```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets, **kwargs):
        # Your loss implementation
        return loss
```

### Routing Analysis

Analyze routing decisions:

```python
from evaluation import create_routing_analyzer

analyzer = create_routing_analyzer(model, task_names)
analysis = analyzer.analyze_routing_decisions(dataloader)
analyzer.visualize_routing_patterns("./visualizations")
```

## Monitoring and Logging

### Weights & Biases Integration

Enable W&B logging in config:

```yaml
use_wandb: true
wandb_project: "dynamo"
wandb_tags: ["multi-task", "lora"]
```

### Local Logging

Logs are saved to:
- `./logs/`: Training logs and metrics
- `./checkpoints/`: Model checkpoints
- `./visualizations/`: Analysis plots

## Performance Tips

### Memory Optimization
- Use gradient accumulation for large effective batch sizes
- Enable mixed precision training
- Adjust batch size based on GPU memory

### Training Speed
- Use multiple GPUs with DataParallel
- Optimize data loading with multiple workers
- Use compiled models (PyTorch 2.0+)

### Hyperparameter Tuning
- Start with provided default values
- Tune LoRA rank based on task complexity
- Adjust loss weights based on task importance
- Use learning rate scheduling

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use smaller LoRA ranks

2. **Poor routing accuracy**
   - Increase router training epochs
   - Adjust temperature annealing
   - Check mixed-task data quality

3. **Task performance degradation**
   - Verify data preprocessing
   - Check loss weight balance
   - Ensure proper Phase 1 training

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use DYNAMO in your research, please cite:

```bibtex
@article{dynamo2024,
  title={DYNAMO: Dynamic Neural Adapter Multi-task Optimization},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers for the base models
- LoRA implementation inspired by Microsoft's LoRA
- Routing mechanisms based on mixture-of-experts literature

## Contact

For questions or issues, please:
- Open a GitHub issue
- Contact: [your-email@domain.com]

---

**Note**: This implementation is for research purposes. For production use, additional optimizations and testing may be required.


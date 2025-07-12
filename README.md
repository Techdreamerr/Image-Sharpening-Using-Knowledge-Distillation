# Image Sharpening using Knowledge Distillation

## Project Overview

This project implements an advanced image sharpening system using Knowledge Distillation (KD) technique. The system consists of a powerful teacher model and a lightweight student model that learns to perform high-quality image sharpening while maintaining computational efficiency.

## Problem Statement

Image sharpening is a fundamental computer vision task that enhances image quality by reducing blur and improving edge clarity. Traditional methods often struggle with complex degradation patterns, while deep learning approaches can be computationally expensive. This project addresses these challenges through:

- **Knowledge Distillation**: Transferring knowledge from a powerful teacher model to a lightweight student model
- **Multi-scale Processing**: Handling various blur levels and degradation patterns
- **Real-time Performance**: Optimized student model for practical applications
- **Quality Assurance**: Comprehensive evaluation using multiple metrics (SSIM, PSNR, LPIPS)

## Key Features

- **Teacher Model**: High-performance EDSR-based architecture for superior image sharpening
- **Student Model**: Lightweight CNN with depthwise separable convolutions and attention mechanisms
- **Knowledge Distillation**: Advanced loss functions combining pixel-wise, perceptual, and adversarial losses
- **Real-time GUI**: PyQt5-based application for interactive image processing
- **Comprehensive Evaluation**: Multi-dataset testing with detailed quality metrics
- **Modular Design**: Clean, maintainable codebase with configuration management

## Project Structure

```
image-sharpening-kd/
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   └── model_configs.yaml  # Model-specific configurations
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── utils/             # Utility functions
│   ├── data/              # Dataset and data loading
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation scripts
├── app/                   # GUI application
├── data/                  # Dataset storage
├── models/                # Saved model weights
├── results/               # Evaluation results
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
└── requirements.txt       # Dependencies
```

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Windows 10/11
- PowerShell

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-sharpening-kd
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup the project**:
   ```bash
   python setup.py
   ```

## Quick Start

### 1. Data Preparation
```bash
python scripts/prepare_data.py
```

### 2. Train Teacher Model
```bash
python src/training/train_teacher.py
```

### 3. Train Student Model (Knowledge Distillation)
```bash
python src/training/train_student.py
```

### 4. Evaluate Models
```bash
python src/evaluation/evaluate.py
```

### 5. Run GUI Application
```bash
python app/gui_app.py
```

## Usage Examples

### Training Pipeline
```python
from src.training.trainer import TeacherTrainer, StudentTrainer

# Train teacher model
teacher_trainer = TeacherTrainer(config)
teacher_trainer.train()

# Train student model with knowledge distillation
student_trainer = StudentTrainer(config)
student_trainer.train()
```

### Inference
```python
from src.models.teacher import TeacherModel
from src.models.student import StudentModel

# Load models
teacher = TeacherModel.load_pretrained('models/teacher.pth')
student = StudentModel.load_pretrained('models/student.pth')

# Process image
sharpened = student.enhance(blurry_image)
```

### GUI Application
```bash
python app/gui_app.py
```

## Configuration

The project uses YAML configuration files for easy parameter management:

- `configs/config.yaml`: Main configuration with all parameters
- `configs/model_configs.yaml`: Model-specific configurations

Key configuration sections:
- **Data**: Dataset paths, batch sizes, augmentation
- **Models**: Architecture parameters, model paths
- **Training**: Learning rates, epochs, optimization
- **Loss**: Loss function weights and parameters
- **Evaluation**: Metrics, test datasets
- **Hardware**: GPU settings, mixed precision

## Performance Metrics

The system is evaluated using multiple metrics:

- **SSIM (Structural Similarity Index)**: Target > 90%
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Lower is better

## Datasets

- **Training**: DIV2K, synthetic degraded images
- **Testing**: Set5, Set14, BSD100, Urban100, DIV2K validation
- **Custom**: User-provided images through GUI

## Model Architectures

### Teacher Model
- Based on EDSR (Enhanced Deep Super-Resolution)
- Multiple residual blocks with channel attention
- High capacity for superior performance

### Student Model
- Lightweight CNN with depthwise separable convolutions
- Optional attention mechanisms
- Optimized for speed while maintaining quality

## Knowledge Distillation

The knowledge distillation process includes:

1. **Feature Distillation**: Transfer intermediate features
2. **Output Distillation**: Match final outputs
3. **Perceptual Distillation**: Preserve perceptual quality
4. **Adversarial Distillation**: Improve realistic appearance

## Results

Expected performance on benchmark datasets:
- **SSIM**: > 90% on most datasets
- **PSNR**: > 30dB on average
- **LPIPS**: < 0.1 for high perceptual quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{image_sharpening_kd_2024,
  title={Image Sharpening using Knowledge Distillation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For questions and support, please open an issue on the GitHub repository or contact the maintainers.

## Acknowledgments

- DIV2K dataset creators
- EDSR paper authors
- PyTorch and PyQt5 communities 
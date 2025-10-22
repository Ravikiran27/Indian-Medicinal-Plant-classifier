# 🌿 Indian Medicinal Plant Classifier - Streamlit Web App

Professional web interface for classifying 80 Indian medicinal plant species using a fine-tuned EfficientNet-B3 deep learning model.

## Features

- **80 Species Recognition**: Identifies Indian medicinal plants with high accuracy
- **Real-time Inference**: Fast predictions using PyTorch trained model
- **Test-Time Augmentation (TTA)**: Optional accuracy boost through ensemble predictions
- **Interactive UI**: Clean, professional Streamlit interface with Plotly visualizations
- **Top-5 Predictions**: Shows confidence scores for most likely species
- **Confidence Visualization**: Horizontal bar charts and detailed probability distributions

## Prerequisites

- Python 3.8+
- Trained model files in `impc_outputs/` folder:
  - `best_model.pth` (PyTorch state dict)
  - `labels.json` (class index mapping)
  - `metrics.json` (optional, for model stats display)

## Installation

1. **Clone or download this project folder**

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify model files**:
   Ensure the `impc_outputs/` folder contains:
   ```
   impc_outputs/
   ├── best_model.pth
   ├── labels.json
   └── metrics.json (optional)
   ```

## Usage

### Run the Streamlit app

```powershell
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Upload Image**: Click "Browse files" and select a plant leaf image (JPG, PNG, BMP)
2. **View Predictions**: See the top predicted species with confidence score
3. **Enable TTA** (optional): Toggle in sidebar for more accurate predictions (slower)
4. **Explore Results**: 
   - View top 5 predictions with confidence bars
   - Expand details to see all class probabilities
   - Check model accuracy metrics in sidebar

## Model Information

- **Architecture**: EfficientNet-B3 (12M parameters)
- **Input Size**: 384×384 pixels
- **Training**: Fine-tuned on ImageNet pretrained weights
- **Dataset**: Indian Medicinal Plant Image Dataset (80 classes)
- **Accuracy**: 93%+ test accuracy (if using optimized training config)

## Configuration

Edit `app.py` constants if your setup differs:

```python
MODEL_DIR = Path("impc_outputs")      # Model folder path
IMG_SIZE = 384                         # Must match training config
```

## Troubleshooting

### Model not loading
- Verify `impc_outputs/best_model.pth` and `labels.json` exist
- Check PyTorch version compatibility: `pip install torch torchvision --upgrade`

### Out of memory
- Disable TTA option
- Reduce image size before upload
- Close other applications

### Slow predictions
- TTA is enabled (expected behavior)
- CPU inference is slower than GPU
- First prediction loads the model (cached afterward)

## Project Structure

```
Indian Medicinal Plant classifier/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README_STREAMLIT.md                 # This file
├── impc_outputs/                       # Model artifacts (from training)
│   ├── best_model.pth                 # Trained PyTorch model
│   ├── labels.json                    # Class name mapping
│   ├── metrics.json                   # Training/test metrics
│   └── ...
└── indian_medicinal_plant_classifier_kaggle.ipynb  # Training notebook
```

## Training the Model

To train your own model, use the included Kaggle notebook:
1. Upload `indian_medicinal_plant_classifier_kaggle.ipynb` to Kaggle
2. Add the Indian Medicinal Plant Dataset as input
3. Enable GPU accelerator
4. Run all cells
5. Download the `impc_outputs/` folder
6. Place it in this project directory

## Advanced Usage

### Custom Model Architecture

If you trained with a different model (e.g., ResNet50, EfficientNet-B0), update the `build_model` function in `app.py` to match your architecture.

### Batch Inference

For processing multiple images, modify `app.py` to accept a folder or multiple file uploads.

### API Deployment

Convert to FastAPI or Flask for REST API deployment:
- Load model once at startup
- Create `/predict` endpoint
- Return JSON predictions

## Performance Tips

- **GPU Acceleration**: If CUDA is available, model automatically uses GPU
- **Model Caching**: First load is slow; subsequent predictions are fast
- **Image Size**: Larger images take longer to preprocess
- **TTA Trade-off**: 3× slower but can improve accuracy by 1-2%

## License

This application uses the Indian Medicinal Plant Dataset and EfficientNet architecture. Ensure compliance with dataset and model licenses for commercial use.

## Credits

- **Model**: EfficientNet-B3 (Google Research)
- **Dataset**: Indian Medicinal Plant Image Dataset (Kaggle)
- **Framework**: PyTorch, Streamlit, Plotly

## Support

For issues or questions:
1. Check model files are correctly placed in `impc_outputs/`
2. Verify Python and package versions
3. Review error messages in terminal/console

---

**Built with ❤️ for medicinal plant conservation and education**

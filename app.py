"""
Indian Medicinal Plant Classifier - Streamlit Web App
=====================================================
Professional web interface for classifying 80 Indian medicinal plant species
using the fine-tuned EfficientNet model.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T, models as tvm
from PIL import Image
import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Indian Medicinal Plant Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = Path("impc_outputs")
MODEL_PATH = MODEL_DIR / "best_model.pth"
LABELS_PATH = MODEL_DIR / "labels.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
IMG_SIZE = 256  # Match training config (notebook uses 256)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@st.cache_resource
def load_model_and_labels():
    """Load the trained model and class labels with caching."""
    
    # Load labels
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    class_names = [labels[str(i)] for i in range(len(labels))]
    num_classes = len(class_names)
    
    # Build model architecture (match notebook - EfficientNet-B0)
    def build_model(num_classes):
        try:
            # Primary: EfficientNet-B0 (matches notebook config)
            model = tvm.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        except:
            # Fallback to ResNet50
            try:
                model = tvm.resnet50(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            except Exception as e:
                raise RuntimeError(f"Failed to build model architecture: {e}")
        return model
    
    model = build_model(num_classes)
    
    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load metrics if available
    metrics = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    
    return model, class_names, metrics


def get_transform(tta=False):
    """Get preprocessing transform pipeline."""
    if tta:
        # Test-time augmentation variants
        return [
            T.Compose([
                T.Resize(int(IMG_SIZE * 1.15)),
                T.CenterCrop(IMG_SIZE),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            T.Compose([
                T.Resize(int(IMG_SIZE * 1.15)),
                T.CenterCrop(IMG_SIZE),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            T.Compose([
                T.Resize(int(IMG_SIZE * 1.15)),
                T.CenterCrop(IMG_SIZE),
                T.RandomVerticalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
        ]
    else:
        return T.Compose([
            T.Resize(int(IMG_SIZE * 1.15)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


@torch.no_grad()
def predict(model, image_pil, class_names, use_tta=False):
    """Run inference on a PIL image."""
    model.eval()
    
    if use_tta:
        # Test-Time Augmentation
        transforms_list = get_transform(tta=True)
        all_probs = []
        for tfm in transforms_list:
            img_tensor = tfm(image_pil).unsqueeze(0)
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        avg_probs = torch.stack(all_probs).mean(0).squeeze(0).numpy()
    else:
        # Single prediction
        tfm = get_transform(tta=False)
        img_tensor = tfm(image_pil).unsqueeze(0)
        logits = model(img_tensor)
        avg_probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    
    # Get top 5 predictions
    top5_idx = np.argsort(avg_probs)[::-1][:5]
    top5_probs = avg_probs[top5_idx]
    top5_classes = [class_names[i] for i in top5_idx]
    
    return top5_classes, top5_probs, avg_probs


def plot_top5_predictions(classes, probs):
    """Create a horizontal bar chart for top 5 predictions."""
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=classes,
        orientation='h',
        marker=dict(
            color=probs * 100,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f'{p*100:.2f}%' for p in probs],
        textposition='auto',
    ))
    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Plant Species",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def main():
    # Header
    st.title("🌿 Indian Medicinal Plant Classifier")
    st.markdown("""
    **Professional AI-powered identification of 80 Indian medicinal plant species**  
    Upload an image of a medicinal plant leaf, and the model will predict the species with confidence scores.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        use_tta = st.checkbox(
            "Enable Test-Time Augmentation (TTA)",
            value=False,
            help="Improves accuracy by averaging predictions from multiple augmented views. Slower but more accurate."
        )
        
        show_all_classes = st.checkbox(
            "Show All Class Probabilities",
            value=False,
            help="Display probability distribution across all 80 classes."
        )
        
        st.markdown("---")
        st.header("📊 Model Info")
        
        try:
            model, class_names, metrics = load_model_and_labels()
            st.success(f"✅ Model loaded: {len(class_names)} classes")
            
            if metrics:
                st.metric("Test Accuracy", f"{metrics['test']['acc']*100:.2f}%")
                st.metric("Test F1-Score", f"{metrics['test']['f1']:.4f}")
                st.metric("Val Accuracy", f"{metrics['val']['acc']*100:.2f}%")
            
            st.markdown("---")
            st.markdown("**Model Architecture**: EfficientNet-B0")
            st.markdown(f"**Input Size**: {IMG_SIZE}×{IMG_SIZE}px")
            st.markdown("**Training**: Fine-tuned on ImageNet pretrained weights")
            st.markdown("**Epochs**: 10 with early stopping")
            st.markdown("**Optimizer**: AdamW + OneCycleLR")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info(f"Make sure `{MODEL_DIR}` folder exists with `best_model.pth` and `labels.json`")
            st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a medicinal plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.caption(f"Image size: {image.size[0]}×{image.size[1]}px")
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..." if not use_tta else "Running TTA analysis (this may take a moment)..."):
                try:
                    # Run prediction
                    top5_classes, top5_probs, all_probs = predict(
                        model, image, class_names, use_tta=use_tta
                    )
                    
                    # Display top prediction
                    pred_class = top5_classes[0]
                    pred_conf = top5_probs[0]
                    
                    if pred_conf > 0.7:
                        st.success(f"**Predicted Species**: {pred_class}")
                    elif pred_conf > 0.4:
                        st.warning(f"**Predicted Species**: {pred_class}")
                    else:
                        st.info(f"**Predicted Species**: {pred_class}")
                    
                    st.metric("Confidence", f"{pred_conf*100:.2f}%")
                    
                    # Top 5 bar chart
                    fig = plot_top5_predictions(top5_classes, top5_probs)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional details
                    with st.expander("📋 Top 5 Predictions Details"):
                        for i, (cls, prob) in enumerate(zip(top5_classes, top5_probs), 1):
                            st.write(f"{i}. **{cls}**: {prob*100:.2f}%")
                    
                    # Show all class probabilities if requested
                    if show_all_classes:
                        with st.expander("📊 All Class Probabilities"):
                            sorted_idx = np.argsort(all_probs)[::-1]
                            sorted_classes = [class_names[i] for i in sorted_idx]
                            sorted_probs = all_probs[sorted_idx]
                            
                            fig_all = go.Figure(go.Bar(
                                x=sorted_probs * 100,
                                y=sorted_classes,
                                orientation='h',
                                marker=dict(color='lightblue')
                            ))
                            fig_all.update_layout(
                                xaxis_title="Confidence (%)",
                                yaxis_title="Plant Species",
                                height=1200,
                                yaxis=dict(autorange="reversed")
                            )
                            st.plotly_chart(fig_all, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
        else:
            st.info("👆 Upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Indian Medicinal Plant Classifier</strong> | 80 Species Recognition</p>
        <p>Model: EfficientNet-B0 fine-tuned on Indian Medicinal Plant Dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

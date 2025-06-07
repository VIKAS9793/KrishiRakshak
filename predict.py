"""
Prediction script for crop disease classification.
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.models import get_model

def load_model(model_path, config_path, class_to_idx_path):
    """Load trained model and its configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load class to index mapping
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    
    # Initialize model
    model = get_model(
        model_name=config.get('model_name', 'efficientnet_b0'),
        num_classes=len(class_to_idx),
        pretrained=False
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create index to class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return model, idx_to_class, config

def preprocess_image(image_path, img_size=224):
    """Preprocess image for model inference."""
    # Define transforms
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return input_batch, image

def predict(model, input_batch, idx_to_class, topk=5):
    """Make prediction on input batch."""
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, k=topk)
    
    # Convert to lists
    top_probs = top_probs.numpy()
    top_indices = top_indices.numpy()
    
    # Get class labels and probabilities
    results = []
    for i in range(topk):
        class_idx = top_indices[i]
        class_name = idx_to_class[class_idx]
        prob = top_probs[i]
        results.append({
            'class': class_name,
            'probability': float(prob),
            'class_idx': int(class_idx)
        })
    
    return results

def display_prediction(image, results, class_names=None):
    """Display image with prediction results."""
    plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    classes = [r['class'] for r in results]
    probs = [r['probability'] for r in results]
    
    # Shorten class names for display
    if class_names is not None:
        display_classes = []
        for cls in classes:
            # Get the last part after the last underscore
            short_name = cls.split('_')[-1]
            # If it's a disease, get the disease name
            if '___' in cls:
                short_name = cls.split('___')[-1].replace('_', ' ')
            display_classes.append(short_name)
    else:
        display_classes = classes
    
    y_pos = np.arange(len(display_classes))
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, display_classes)
    plt.xlabel('Probability')
    plt.title('Prediction Results')
    plt.tight_layout()
    
    # Add probability values on the bars
    for i, v in enumerate(probs):
        plt.text(v + 0.01, i, f'{v:.2f}', color='black', fontweight='bold')
    
    plt.show()

def main():
    """Main prediction function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict crop disease from an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model-dir', type=str, default='outputs/experiment',
                        help='Directory containing the trained model')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return
    
    # Check if model directory exists
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        return
    
    # Define model and config paths
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    config_path = os.path.join(args.model_dir, 'config.json')
    class_to_idx_path = os.path.join(args.model_dir, 'class_to_idx.json')
    
    # Check if required files exist
    for path in [model_path, config_path, class_to_idx_path]:
        if not os.path.isfile(path):
            print(f"Error: Required file '{path}' not found in model directory.")
            return
    
    try:
        # Load model
        print("Loading model...")
        model, idx_to_class, config = load_model(model_path, config_path, class_to_idx_path)
        
        # Preprocess image
        print("Processing image...")
        input_batch, image = preprocess_image(args.image_path, img_size=config.get('img_size', 224))
        
        # Make prediction
        print("Making prediction...")
        results = predict(model, input_batch, idx_to_class, topk=args.topk)
        
        # Display results
        print("\nTop Predictions:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['class']}: {result['probability']:.4f}")
        
        # Display image and results
        display_prediction(image, results, class_names=list(idx_to_class.keys()))
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

if __name__ == '__main__':
    main()

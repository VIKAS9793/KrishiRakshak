import torch
import os
from src.models import get_model
import json

# Define the path to your best model and class_to_idx mapping
# Ensure this matches your actual output directory from the last training run
output_dir = "outputs/hackathon_demo/experiment_20250607_225122"
best_model_path = os.path.join(output_dir, "best_model.pth")
class_to_idx_path = os.path.join(output_dir, "class_to_idx.json")
onnx_model_path = os.path.join(output_dir, "best_model.onnx")

# Load class_to_idx to get num_classes
with open(class_to_idx_path, 'r') as f:
    class_to_idx = json.load(f)
num_classes = len(class_to_idx)

# Initialize the model architecture (EfficientNet-B0 as used in training)
# pretrained=False because we are loading weights from our saved model
model = get_model(model_name="efficientnet_b0", num_classes=num_classes, pretrained=False)

# Load the trained state_dict
# Ensure map_location is set to 'cpu' if you trained on GPU but are exporting on CPU, or vice versa
model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu'))['model_state_dict'])
model.eval() # Set model to evaluation mode

# Create a dummy input for ONNX export
# The shape should match your model's expected input (Batch_size, Channels, Height, Width)
# Use the img_size (128) you trained with
dummy_input = torch.randn(1, 3, 128, 128)

# Export the model to ONNX
print(f"Exporting model to ONNX at: {onnx_model_path}")
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, # Optional: for dynamic batch size
        opset_version=11 # A common opset version
    )
    print("Model exported to ONNX successfully!")
except Exception as e:
    print(f"Error during ONNX export: {e}")

print("ONNX export script completed.") 
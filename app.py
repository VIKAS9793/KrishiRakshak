"""
Krishi Rakshak - Crop Disease Detection
Minimal implementation for hackathon submission
"""
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Constants
CLASS_NAMES = ["Healthy", "Disease 1", "Disease 2"]  # Update with actual class names
MODEL_PATH = "model/model.pth"  # Update with actual model path

def load_model():
    """Load the pre-trained model"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the input image"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image):
    """Make prediction on the input image"""
    if image is None:
        return "Error: No image uploaded"
    
    try:
        # Convert to PIL Image
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Preprocess
        img_tensor = preprocess_image(img)
        
        # Load model and predict
        model = load_model()
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        # Get results
        predicted_class = CLASS_NAMES[preds[0]]
        confidence = float(confidence[preds[0]])
        
        return f"Prediction: {predicted_class}\nConfidence: {confidence:.1f}%"
        
    except Exception as e:
        return f"Error: {str(e)}"

def create_ui():
    """Create the main Gradio interface"""
    css = """
    .main-content {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    """
    
    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_classes=["main-content"]):
            gr.Markdown("# 🌱 Crop Disease Detection")
            
            with gr.Row():
                image_input = gr.Image(label="Upload Crop Image", type="numpy")
                
                with gr.Column():
                    predict_btn = gr.Button("Analyze")
                    output = gr.Textbox(label="Result")
                    gr.HTML("<h1 style='margin: 0; padding: 0; line-height: 1.2; font-size: 2.5em;'>🌱 Krishi Rakshak</h1>")
                    gr.HTML("<p style='margin: 5px 0 0 0; color: var(--text-secondary); font-size: 1.1em;'>AI-Powered Plant Disease Detection System</p>")
            
            gr.HTML('<div style="margin-top: 20px; border-bottom: 1px solid #e0e0e0; width: 100%;"></div>')  # Separator line
        
        # Language selector
        with gr.Row(elem_classes=["language-selector"]):
            language = gr.Radio(
                choices=["English", "हिंदी (Hindi)"],
                label="Select Language / भाषा चुनें",
                value="English",
                interactive=True
            )
        
        # Main content area
        with gr.Row(elem_classes=["main-content"]):
            # Left column - Image upload
            with gr.Column(scale=1, elem_classes=["upload-section"]):
                gr.Markdown("### Upload Plant Image")
                
                image_input = gr.Image(
                    type="filepath",
                    label="Choose an image of a plant leaf",
                    height=300,
                    elem_id="image-upload"
                )
                
                submit_btn = gr.Button(
                    "Analyze Image",
                    variant="primary",
                    elem_classes=["analyze-btn"]
                )
                
                gr.Markdown("*Upload a clear image of a plant leaf for disease detection*")
            
            # Right column - Results
            with gr.Column(scale=1, elem_classes=["results-section"]):
                gr.HTML('<h3 class="results-title">Analysis Results</h3>')
                
                output = gr.JSON(
                    label="",
                    show_label=False,
                    container=True
                )
                
                gr.Markdown("*Results will appear here after analysis*")
        
        # Event handlers
        language.change(
            fn=update_button_text,
            inputs=language,
            outputs=submit_btn
        )
        
        submit_btn.click(
            fn=predict,
            inputs=[image_input, language],
            outputs=output
        )
        
        # Auto-analyze on image upload (optional)
        image_input.change(
            fn=lambda img, lang: predict(img, lang) if img is not None else None,
            inputs=[image_input, language],
            outputs=output
        )
    
    return demo

def main():
    """Main function to run the application"""
    # Ensure assets directory exists
    os.makedirs("assets/logos", exist_ok=True)
    os.makedirs("assets/banners", exist_ok=True)
    
    # Try different ports
    ports_to_try = [8080, 8081, 8082, 8888, 5000]
    
    for port in ports_to_try:
        try:
            # Create the UI
            demo = create_ui()
            
            print(f"\n{'='*60}")
            print(f"🌱 Starting Krishi Rakshak on port {port}")
            print(f"📍 Local URL: http://localhost:{port}")
            try:
                network_ip = socket.gethostbyname(socket.gethostname())
                print(f"🌐 Network URL: http://{network_ip}:{port}")
            except:
                print("🌐 Network URL: Not available")
            print(f"{'='*60}\n")
            
            # Launch the application
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                inbrowser=True,
                quiet=False,
                show_error=True
            )
            
            # If we get here, launch was successful
            break
            
        except Exception as e:
            print(f"❌ Failed to start on port {port}: {str(e)}")
            if port == ports_to_try[-1]:  # If this was the last port to try
                print("\n❌ Error: Could not start the application on any port.")
                print(f"🔍 Please check if another application is using these ports: {', '.join(map(str, ports_to_try))}")
                print("💡 You can also try running with a different port by setting the PORT environment variable.")
                exit(1)

if __name__ == "__main__":
    main()
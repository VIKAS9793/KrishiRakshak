"""
Krishi Rakshak - Crop Disease Detection
A clean, functional implementation with proper structure
"""
import gradio as gr
import os
import socket
import base64
from pathlib import Path

# Global variables for logo
import os
from pathlib import Path

# Get absolute paths to assets
BASE_DIR = Path(__file__).parent.absolute()
LOGO_PATH = str(BASE_DIR / "assets" / "logos" / "logo.png")  # Absolute path to logo image

def predict(image, language):
    """
    Placeholder prediction function
    Replace this with your actual model prediction logic
    """
    if image is None:
        return {
            "error": "No image uploaded" if language == "English" else "कोई छवि अपलोड नहीं की गई",
            "status": "error"
        }
    
    # Placeholder prediction logic
    # Replace this with your actual model inference
    try:
        # Simulate processing time and prediction
        import time
        import random
        time.sleep(1)  # Simulate processing
        
        # Mock prediction results
        diseases = ["Healthy", "Leaf Blight", "Powdery Mildew", "Rust", "Bacterial Spot"]
        predicted_disease = random.choice(diseases)
        confidence = random.uniform(0.7, 0.95)
        
        if language == "English":
            result = {
                "status": "success",
                "prediction": predicted_disease,
                "confidence": f"{confidence:.1%}",
                "recommendation": "Consult with agricultural expert for proper treatment" if predicted_disease != "Healthy" else "Plant appears healthy!"
            }
        else:  # Hindi
            disease_translations = {
                "Healthy": "स्वस्थ",
                "Leaf Blight": "पत्ती का झुलसा रोग",
                "Powdery Mildew": "चूर्णी फफूंद",
                "Rust": "रतुआ रोग",
                "Bacterial Spot": "जीवाणु धब्बा"
            }
            result = {
                "status": "success",
                "prediction": disease_translations.get(predicted_disease, predicted_disease),
                "confidence": f"{confidence:.1%}",
                "recommendation": "उचित उपचार के लिए कृषि विशेषज्ञ से सलाह लें" if predicted_disease != "Healthy" else "पौधा स्वस्थ दिखता है!"
            }
        
        return result
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}" if language == "English" else f"भविष्यवाणी असफल: {str(e)}",
            "status": "error"
        }

def update_button_text(language):
    """Update button text based on selected language"""
    return gr.update(value="Analyze Image" if language == "English" else "छवि का विश्लेषण करें")

def create_ui():
    """Create the main Gradio interface"""
    
    # Custom CSS for styling
    custom_css = """
    :root {
        --primary-color: #4CAF50;
        --primary-dark: #388E3C;
        --primary-light: #C8E6C9;
        --background: #f5f5f5;
        --card-bg: #ffffff;
        --text-primary: #212121;
        --text-secondary: #757575;
        --shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    body {
        background: var(--background);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
    
    .header-section {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: var(--card-bg);
        border-radius: 12px;
        box-shadow: var(--shadow);
    }
    
    .header-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    
    .logo-image {
        border-radius: 8px;
    }
    
    .header-section h1 {
        color: var(--primary-dark);
        margin: 0 0 10px 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .header-section p {
        color: var(--text-secondary);
        margin: 0;
        font-size: 1.1em;
    }
    
    .main-content {
        display: flex;
        gap: 20px;
        margin-top: 20px;
    }
    
    .upload-section, .results-section {
        flex: 1;
        background: var(--card-bg);
        border-radius: 12px;
        padding: 25px;
        box-shadow: var(--shadow);
    }
    
    .analyze-btn {
        width: 100% !important;
        padding: 12px 20px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        background: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        margin-top: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .analyze-btn:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    .language-selector {
        margin-bottom: 20px;
        background: var(--card-bg);
        padding: 15px;
        border-radius: 8px;
        box-shadow: var(--shadow);
    }
    
    .results-title {
        color: var(--primary-dark);
        font-size: 1.3em;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-content {
            flex-direction: column;
        }
        
        .gradio-container {
            padding: 15px !important;
        }
        
        .header-section h1 {
            font-size: 2em;
        }
    }
    """
    
    # Create the Gradio interface
    with gr.Blocks(
        title="Krishi Rakshak - Plant Disease Detection",
        theme=gr.themes.Default(
            primary_hue="green",
            secondary_hue="lime",
            neutral_hue="stone",
            spacing_size="sm",
            radius_size="md"
        ),
        css=custom_css
    ) as demo:
        
        # Header section with logo and title
        with gr.Column(elem_classes=["header-section"]):
            with gr.Row(elem_classes=["header-row"]):
                # Larger logo
                if os.path.exists(LOGO_PATH):
                    logo = gr.Image(
                        value=LOGO_PATH,
                        visible=True,
                        show_label=False,
                        interactive=False,
                        height=120,  # Increased size
                        width=120,    # Increased size
                        container=False,
                        elem_classes=["logo-image"]
                    )
                
                # Title and subtitle
                with gr.Column():
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
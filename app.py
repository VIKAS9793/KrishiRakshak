"""
Krishi Rakshak - Crop Disease Detection
"""
import gradio as gr

# Sample disease information (will be replaced with actual model)
SAMPLE_DISEASES = {
    "Healthy": {
        "en": "Your plant looks healthy! 🎉",
        "hi": "आपका पौधा स्वस्थ दिख रहा है! 🎉"
    },
    "Diseased": {
        "en": "Possible disease detected. Please consult an expert.",
        "hi": "संभावित रोग का पता चला है। कृपया विशेषज्ञ से परामर्श लें।"
    }
}

def predict(image, language):
    """Placeholder prediction function"""
    # This will be replaced with actual model prediction
    result = SAMPLE_DISEASES["Healthy"] if image is not None else SAMPLE_DISEASES["Diseased"]
    
    return {
        "status": "Healthy" if image is not None else "Diseased",
        "confidence": "95%" if image is not None else "60%",
        "advice": result[language[:2].lower()]  # 'en' or 'hi'
    }

def create_ui():
    with gr.Blocks(
        title="Krishi Rakshak",
        theme=gr.themes.Soft(
            primary_hue="green",
            font=["Noto Sans", "Arial", "sans-serif"]
        )
    ) as demo:
        # Header
        gr.Markdown("""
        # 🌱 Krishi Rakshak
        ### स्वस्थ फसल, समृद्ध किसान  
        *Your AI-powered crop health guardian*
        """)
        
        # Language selector
        language = gr.Radio(
            ["English", "हिंदी (Hindi)"],
            label="Select Language",
            value="English"
        )
        
        # Main content
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    label="Upload Plant Leaf",
                    type="numpy"
                )
                
                # Example images (using placeholder text for now)
                gr.Markdown("\n*Upload an image to get started*")
                # We'll add real examples after we have sample images
            
            with gr.Column(scale=1):
                with gr.Group():
                    submit_btn = gr.Button("Analyze" if language.value == "English" else "विश्लेषण करें")
                    output = gr.JSON(label="Analysis Result")
        
        # Update button text when language changes
        def update_button(lang):
            return gr.update(value="Analyze" if lang == "English" else "विश्लेषण करें")
        
        language.change(update_button, inputs=language, outputs=submit_btn)
        
        # Connect the button
        submit_btn.click(
            fn=predict,
            inputs=[image_input, language],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()

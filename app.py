import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

try:
    model = tf.keras.models.load_model('tomato_classifier_model_vgg16_4_classes.h5')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

img_width, img_height = 224, 224

class_labels = {
    0: 'Early_Vegetative 🌿',
    1: 'Flowering_Initiation 🌼',
    2: 'Fruiting_and_Ripening 🍅',
    3: 'Germination_and_Seedling 🌱'
}

# Detailed care instructions for each growth stage
care_instructions = {
    0: {  # Early_Vegetative
        "title": "Early Vegetative Stage Care 🌿",
        "description": "Your tomato plant is in its early vegetative growth phase, focusing on developing strong stems and leaves.",
        "do": [
            "💧 Water regularly but avoid overwatering - soil should be moist, not soggy",
            "☀️ Provide 6-8 hours of direct sunlight daily",
            "🌡️ Maintain temperature between 65-75°F (18-24°C)",
            "💨 Ensure good air circulation around the plant",
            "🪴 Apply balanced fertilizer (10-10-10 or 14-14-14) every 2 weeks",
            "✂️ Start gentle pruning of lower leaves touching the soil",
            "🏗️ Begin staking or caging for future support"
        ],
        "dont": [
            "❌ Don't use high-nitrogen fertilizers exclusively",
            "❌ Avoid getting water on leaves (causes fungal diseases)",
            "❌ Don't transplant during extreme weather",
            "❌ Avoid heavy pruning at this stage",
            "❌ Don't let soil completely dry out between waterings"
        ]
    },
    1: {  # Flowering_Initiation
        "title": "Flowering Initiation Stage Care 🌼",
        "description": "Your plant is starting to flower! This is a critical stage that determines fruit production.",
        "do": [
            "🐝 Encourage pollination by gently shaking flowers daily",
            "💧 Water consistently - stress can cause flower drop",
            "🌡️ Maintain steady temperature (avoid sudden changes)",
            "🪴 Switch to low-nitrogen, high-phosphorus fertilizer",
            "✂️ Remove suckers (shoots between main stem and branches)",
            "🏗️ Ensure strong support structure is in place",
            "👀 Monitor for early pest signs (aphids, whiteflies)"
        ],
        "dont": [
            "❌ Don't over-fertilize with nitrogen (reduces flowering)",
            "❌ Avoid disturbing roots during this sensitive period",
            "❌ Don't let plants experience water stress",
            "❌ Avoid excessive pruning of flower clusters",
            "❌ Don't use pesticides that harm beneficial pollinators"
        ]
    },
    2: {  # Fruiting_and_Ripening
        "title": "Fruiting and Ripening Stage Care 🍅",
        "description": "Congratulations! Your plant is producing fruits. Focus on supporting healthy fruit development.",
        "do": [
            "💧 Maintain consistent, deep watering (1-2 inches per week)",
            "🪴 Use high-potassium fertilizer to support fruit development",
            "🏗️ Provide strong support for heavy fruit-laden branches",
            "☀️ Ensure fruits get adequate sunlight for proper ripening",
            "✂️ Prune lower leaves and non-productive branches",
            "👀 Monitor for fruit diseases (blight, cracking, blossom end rot)",
            "🔄 Harvest ripe fruits regularly to encourage continued production"
        ],
        "dont": [
            "❌ Don't let soil dry out completely (causes blossom end rot)",
            "❌ Avoid irregular watering patterns",
            "❌ Don't over-water (can cause fruit cracking)",
            "❌ Avoid high-nitrogen fertilizers (reduces fruit quality)",
            "❌ Don't harvest fruits too early - let them ripen on vine when possible"
        ]
    },
    3: {  # Germination_and_Seedling
        "title": "Germination and Seedling Stage Care 🌱",
        "description": "Your tomato is just starting its journey! This delicate stage requires gentle care and attention.",
        "do": [
            "💧 Keep soil consistently moist but not waterlogged",
            "🌡️ Maintain warm temperature (70-80°F / 21-27°C)",
            "💡 Provide bright, indirect light (14-16 hours daily)",
            "🌱 Use seed starting mix or well-draining potting soil",
            "🔄 Turn seedlings daily if using artificial lights",
            "🪴 Start diluted fertilizer once true leaves appear",
            "🌬️ Gradually introduce to outdoor conditions (hardening off)"
        ],
        "dont": [
            "❌ Don't expose to direct sunlight immediately",
            "❌ Avoid overwatering (causes damping-off disease)",
            "❌ Don't use regular garden soil for seedlings",
            "❌ Avoid disturbing roots unnecessarily",
            "❌ Don't transplant outdoors until night temperatures stay above 50°F (10°C)",
            "❌ Avoid strong fertilizers on young seedlings"
        ]
    }
}

def predict_growth_stage(img_file):
    """
    Predicts the growth stage of a tomato plant from an uploaded image.
    
    Args:
        img_file (str): The file path of the uploaded image.
        
    Returns:
        tuple: (prediction_text, confidence_html, care_instructions_html)
    """
    if not model_loaded:
        return "❌ Model not loaded. Please check if the model file exists.", "", ""
    
    if img_file is None:
        return "📤 Please upload an image first.", "", ""
    
    try:
        # Preprocess the uploaded image
        img = image.load_img(img_file, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0, 1]

        # Make a prediction using the model
        prediction = model.predict(img_array, verbose=0)
        
        # Get all confidence scores
        confidences = prediction[0]
        
        # Get the predicted class index (0, 1, 2, or 3)
        predicted_class_index = np.argmax(confidences)
        
        # Get the confidence of the prediction
        max_confidence = confidences[predicted_class_index]
        
        # Get the predicted label from the dictionary
        predicted_label = class_labels[predicted_class_index]
        
        # Create detailed results
        result_text = f"🎯 **Predicted Stage:** {predicted_label}\n📊 **Confidence:** {max_confidence*100:.1f}%"
        
        # Create confidence breakdown HTML
        confidence_html = "<div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;'>"
        confidence_html += "<h3 style='color: #2d5016; margin-bottom: 15px; text-align: center;'>📈 Confidence Analysis</h3>"
        
        for i, (class_idx, label) in enumerate(class_labels.items()):
            conf_pct = confidences[class_idx] * 100
            bar_width = max(5, conf_pct * 3)  # Minimum 5px, scale for visual bar
            
            if class_idx == predicted_class_index:
                bar_color = "linear-gradient(90deg, #4CAF50, #45a049)"
                text_color = "#2d5016"
                weight = "bold"
                border = "2px solid #4CAF50"
            else:
                bar_color = "linear-gradient(90deg, #e0e0e0, #d0d0d0)"
                text_color = "#666"
                weight = "normal"
                border = "1px solid #ddd"
            
            confidence_html += f"""
            <div style='margin: 15px 0; padding: 12px; border: {border}; border-radius: 8px; background: #fafafa;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='font-weight: {weight}; color: {text_color}; font-size: 16px;'>{label}</span>
                    <span style='font-weight: bold; color: {text_color}; font-size: 16px;'>{conf_pct:.1f}%</span>
                </div>
                <div style='width: 100%; height: 12px; background-color: #f0f0f0; border-radius: 6px; overflow: hidden;'>
                    <div style='width: {conf_pct}%; height: 100%; background: {bar_color}; transition: width 0.5s ease; border-radius: 6px;'></div>
                </div>
            </div>
            """
        
        confidence_html += "</div>"
        
        # Create care instructions HTML
        care_info = care_instructions[predicted_class_index]
        care_html = f"""
        <div style='background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 25px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.1); margin: 15px 0;'>
            <h2 style='color: #2d5016; margin-bottom: 15px; text-align: center; font-size: 24px;'>{care_info['title']}</h2>
            <p style='color: #555; font-size: 16px; text-align: center; margin-bottom: 25px; font-style: italic;'>{care_info['description']}</p>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;'>
                <div style='background: #d4edda; padding: 20px; border-radius: 12px; border-left: 5px solid #28a745;'>
                    <h3 style='color: #155724; margin-bottom: 15px; font-size: 18px;'>✅ What TO DO:</h3>
                    <ul style='color: #155724; line-height: 1.8; padding-left: 0; list-style: none;'>
        """
        
        for item in care_info['do']:
            care_html += f"<li style='margin: 8px 0; padding: 5px 0;'>• {item}</li>"
        
        care_html += f"""
                    </ul>
                </div>
                
                <div style='background: #f8d7da; padding: 20px; border-radius: 12px; border-left: 5px solid #dc3545;'>
                    <h3 style='color: #721c24; margin-bottom: 15px; font-size: 18px;'>🚫 What NOT TO DO:</h3>
                    <ul style='color: #721c24; line-height: 1.8; padding-left: 0; list-style: none;'>
        """
        
        for item in care_info['dont']:
            care_html += f"<li style='margin: 8px 0; padding: 5px 0;'>• {item}</li>"
        
        care_html += """
                    </ul>
                </div>
            </div>
        </div>
        """
        
        return result_text, confidence_html, care_html
    
    except Exception as e:
        return f"❌ An error occurred: {str(e)}", "", ""

# Enhanced Custom CSS for full width and modern design
custom_css = """
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 20px !important;
}

.main-container {
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
}

.upload-container {
    border: 3px dashed #4CAF50 !important;
    border-radius: 15px !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    transition: all 0.3s ease !important;
}

.upload-container:hover {
    border-color: #45a049 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2) !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 40px 20px;
    margin: -20px -20px 30px -20px;
    border-radius: 0 0 20px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.main-header h1 {
    font-size: 3em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.description-text {
    font-size: 18px;
    opacity: 0.9;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
}

.info-section {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 30px;
    border-radius: 20px;
    margin: 30px 0;
    box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}

footer {
    display: none !important;
}

.share-button {
    display: none !important;
}

.result-container {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin: 15px 0;
    border: 2px solid #e9ecef;
}

@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2em;
    }
    
    .description-text {
        font-size: 16px;
    }
    
    .care-grid {
        grid-template-columns: 1fr !important;
    }
}
"""

# Create the Gradio interface with enhanced styling
with gr.Blocks(css=custom_css, title="🍅 Advanced Tomato Growth Stage Classifier", theme=gr.themes.Soft()) as iface:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🍅 Advanced Tomato Plant Growth Stage Classifier</h1>
        <p class="description-text">
            Upload a clear image of your tomato plant to identify its current growth stage and receive detailed care instructions.<br>
            Our AI-powered system provides comprehensive guidance for optimal plant care at every stage.
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(
                type="filepath", 
                label="📸 Upload Your Tomato Plant Image",
                elem_classes="upload-container",
                sources=["upload"],
                height=400
            )
            
            # Predict button
            predict_btn = gr.Button(
                "🔍 Analyze Growth Stage & Get Care Instructions", 
                variant="primary",
                size="lg",
                elem_classes="btn-primary"
            )
        
        with gr.Column(scale=3):
            # Results
            prediction_output = gr.Markdown(
                label="🎯 Prediction Results",
                value="Upload an image and click 'Analyze Growth Stage' to get started!",
                elem_classes="result-container"
            )
            
            # Confidence breakdown
            confidence_output = gr.HTML(label="📊 Detailed Analysis")
    
    # Care instructions (full width)
    care_output = gr.HTML(label="🌱 Personalized Care Instructions")
    
    # Information section
    gr.HTML("""
    <div class="info-section">
        <h2 style='margin-top: 0; text-align: center; font-size: 28px;'>🌱 Tomato Growth Stages Overview</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 25px;'>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);'>
                <h3 style='color: white; margin-bottom: 10px;'>🌱 Germination & Seedling</h3>
                <p style='opacity: 0.9; line-height: 1.6;'>Initial sprouting stage with cotyledons and first true leaves developing.</p>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);'>
                <h3 style='color: white; margin-bottom: 10px;'>🌿 Early Vegetative</h3>
                <p style='opacity: 0.9; line-height: 1.6;'>Active growth of leaves and stems, building the plant's foundation.</p>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);'>
                <h3 style='color: white; margin-bottom: 10px;'>🌼 Flowering Initiation</h3>
                <p style='opacity: 0.9; line-height: 1.6;'>Formation of flower buds and early blooming phase.</p>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);'>
                <h3 style='color: white; margin-bottom: 10px;'>🍅 Fruiting & Ripening</h3>
                <p style='opacity: 0.9; line-height: 1.6;'>Development and maturation of tomato fruits.</p>
            </div>
        </div>
        <div style='text-align: center; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px;'>
            <h3 style='color: white; margin-bottom: 15px;'>💡 Expert Tips</h3>
            <p style='opacity: 0.9; font-size: 16px; line-height: 1.8;'>
                For best results, use clear, well-lit photos showing the overall plant structure. Our AI model analyzes leaf patterns, 
                stem development, presence of flowers or fruits, and overall plant morphology to provide accurate stage identification 
                and tailored care recommendations.
            </p>
        </div>
    </div>
    """)
    
    # Event handlers
    predict_btn.click(
        fn=predict_growth_stage,
        inputs=image_input,
        outputs=[prediction_output, confidence_output, care_output]
    )
    
    # Auto-predict on image upload
    image_input.change(
        fn=predict_growth_stage,
        inputs=image_input,
        outputs=[prediction_output, confidence_output, care_output]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
        inbrowser=True
    )
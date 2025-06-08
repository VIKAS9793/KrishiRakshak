"""
Krishi Rakshak - Crop Disease Detection
Minimal implementation for hackathon submission
"""
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import json
import sys
from pathlib import Path
import socket # Import the socket module

# Add project root to Python path to import from src
sys.path.append(str(Path(__file__).parent))
from src.models import get_model # Import your model loading utility

# Constants
# Dynamically get the output directory from the last training run
# IMPORTANT: Update this if your output directory changes!
OUTPUT_DIR = "outputs/hackathon_demo/experiment_20250607_225122"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
CLASS_TO_IDX_PATH = os.path.join(OUTPUT_DIR, "class_to_idx_reencoded.json")

# --- DIAGNOSTIC: Test JSON loading here ---
print(f"Attempting to load JSON from: {CLASS_TO_IDX_PATH}")
try:
    with open(CLASS_TO_IDX_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print("JSON loaded successfully in diagnostic block. First 5 keys:", list(test_data.keys())[:5])
except json.JSONDecodeError as e:
    print(f"DIAGNOSTIC ERROR: JSONDecodeError: {e}")
    sys.exit(1) # Exit if error for clear feedback
except FileNotFoundError:
    print(f"DIAGNOSTIC ERROR: File not found: {CLASS_TO_IDX_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"DIAGNOSTIC ERROR: An unexpected error occurred: {e}")
    sys.exit(1)
# --- END DIAGNOSTIC ---

# Load class names from the saved mapping
with open(CLASS_TO_IDX_PATH, 'r', encoding='utf-8') as f:
    CLASS_TO_IDX = json.load(f)
CLASS_NAMES = list(CLASS_TO_IDX.keys())
NUM_CLASSES = len(CLASS_NAMES)

# Print class names for debugging
print("Available class names:", CLASS_NAMES)

# Hindi translations for class names
HINDI_TRANSLATIONS = {
    "Apple___Apple_scab": "सेब - सेब स्कैब",
    "Apple___Black_rot": "सेब - काला सड़न",
    "Apple___Cedar_apple_rust": "सेब - देवदार सेब रस्ट",
    "Apple___healthy": "सेब - स्वस्थ",
    "Blueberry___healthy": "ब्लूबेरी - स्वस्थ",
    "Cherry_(including_sour)___Powdery_mildew": "चेरी - ख़स्ता फफूंदी",
    "Cherry_(including_sour)___healthy": "चेरी - स्वस्थ",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "मक्का - सर्कोस्पोरा पत्ती धब्बा ग्रे पत्ती धब्बा",
    "Corn_(maize)___Common_rust": "मक्का - सामान्य रस्ट",
    "Corn_(maize)___Northern_Leaf_Blight": "मक्का - उत्तरी पत्ती झुलसा",
    "Corn_(maize)___healthy": "मक्का - स्वस्थ",
    "Grape___Black_rot": "अंगूर - काला सड़न",
    "Grape___Esca_(Black_Measles)": "अंगूर - एस्का (काला खसरा)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "अंगूर - पत्ती झुलसा",
    "Grape___healthy": "अंगूर - स्वस्थ",
    "Orange___Haunglongbing_(Citrus_greening)": "संतरा - हुआंगलोंगबिंग (खट्टे का हरापन)",
    "Peach___Bacterial_spot": "आड़ू - बैक्टीरियल स्पॉट",
    "Peach___healthy": "आड़ू - स्वस्थ",
    "Pepper,_bell___Bacterial_spot": "शिमला मिर्च - बैक्टीरियल स्पॉट",
    "Pepper,_bell___healthy": "शिमला मिर्च - स्वस्थ",
    "Potato___Early_blight": "आलू - अगेती झुलसा",
    "Potato___Late_blight": "आलू - पछेती झुलसा",
    "Potato___healthy": "आलू - स्वस्थ",
    "Raspberry___healthy": "रसभरी - स्वस्थ",
    "Soybean___healthy": "सोयाबीन - स्वस्थ",
    "Squash___Powdery_mildew": "कद्दू - ख़स्ता फफूंदी",
    "Strawberry___Leaf_scorch": "स्ट्रॉबेरी - पत्ती झुलसा",
    "Strawberry___healthy": "स्ट्रॉबेरी - स्वस्थ",
    "Tomato___Bacterial_spot": "टमाटर - बैक्टीरियल स्पॉट",
    "Tomato___Early_blight": "टमाटर - अगेती झुलसा",
    "Tomato___Late_blight": "टमाटर - पछेती झुलसा",
    "Tomato___Leaf_Mold": "टमाटर - पत्ती मोल्ड",
    "Tomato___Septoria_leaf_spot": "टमाटर - सेप्टोरिया पत्ती धब्बा",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "टमाटर - स्पाइडर माइट्स दो-धब्बेदार स्पाइडर माइट",
    "Tomato___Target_Spot": "टमाटर - टारगेट स्पॉट",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "टमाटर - टमाटर पीला पत्ती कर्ल वायरस",
    "Tomato___Tomato_mosaic_virus": "टमाटर - टमाटर मोज़ेक वायरस",
    "Tomato___healthy": "टमाटर - स्वस्थ"
}

# Marathi translations for class names
MARATHI_TRANSLATIONS = {
    "Apple___Apple_scab": "सफरचंद - सफरचंद स्कॅब",
    "Apple___Black_rot": "सफरचंद - काळी कुज",
    "Apple___Cedar_apple_rust": "सफरचंद - सिडर सफरचंद गंज",
    "Apple___healthy": "सफरचंद - निरोगी",
    "Blueberry___healthy": "ब्लूबेरी - निरोगी",
    "Cherry_(including_sour)___Powdery_mildew": "चेरी - पावडरी मिल्ड्यू",
    "Cherry_(including_sour)___healthy": "चेरी - निरोगी",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "मका - सर्कोस्पोरा पान स्पॉट ग्रे पान स्पॉट",
    "Corn_(maize)___Common_rust": "मका - सामान्य गंज",
    "Corn_(maize)___Northern_Leaf_Blight": "मका - उत्तर पान ब्लाइट",
    "Corn_(maize)___healthy": "मका - निरोगी",
    "Grape___Black_rot": "द्राक्ष - काळी कुज",
    "Grape___Esca_(Black_Measles)": "द्राक्ष - एस्का (काळे मिसळ्स)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "द्राक्ष - पान ब्लाइट",
    "Grape___healthy": "द्राक्ष - निरोगी",
    "Orange___Haunglongbing_(Citrus_greening)": "संत्रा - हुआंगलोंगबिंग (लिंबू हिरवेपणा)",
    "Peach___Bacterial_spot": "पीच - जीवाणू स्पॉट",
    "Peach___healthy": "पीच - निरोगी",
    "Pepper,_bell___Bacterial_spot": "भोपळी मिरची - जीवाणू स्पॉट",
    "Pepper,_bell___healthy": "भोपळी मिरची - निरोगी",
    "Potato___Early_blight": "बटाटा - लवकर ब्लाइट",
    "Potato___Late_blight": "बटाटा - उशीरा ब्लाइट",
    "Potato___healthy": "बटाटा - निरोगी",
    "Raspberry___healthy": "रास्पबेरी - निरोगी",
    "Soybean___healthy": "सोयाबीन - निरोगी",
    "Squash___Powdery_mildew": "कोहळा - पावडरी मिल्ड्यू",
    "Strawberry___Leaf_scorch": "स्ट्रॉबेरी - पान स्कॉर्च",
    "Strawberry___healthy": "स्ट्रॉबेरी - निरोगी",
    "Tomato___Bacterial_spot": "टोमॅटो - जीवाणू स्पॉट",
    "Tomato___Early_blight": "टोमॅटो - लवकर ब्लाइट",
    "Tomato___Late_blight": "टोमॅटो - उशीरा ब्लाइट",
    "Tomato___Leaf_Mold": "टोमॅटो - पान मोल्ड",
    "Tomato___Septoria_leaf_spot": "टोमॅटो - सेप्टोरिया पान स्पॉट",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "टोमॅटो - स्पायडर माइट्स दोन-ठिपके स्पायडर माइट",
    "Tomato___Target_Spot": "टोमॅटो - टारगेट स्पॉट",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "टोमॅटो - टोमॅटो पिवळा पान कर्ल व्हायरस",
    "Tomato___Tomato_mosaic_virus": "टोमॅटो - टोमॅटो मोझेक व्हायरस",
    "Tomato___healthy": "टोमॅटो - निरोगी"
}

# Advisory information for each disease
ADVISORY_INFO = {
    "Apple___Apple_scab": {
        "en": "Advisory: Practice good sanitation by removing fallen leaves. Prune trees for better air circulation. Consider resistant varieties and apply fungicides preventatively if needed. More info: https://extension.umn.edu/plant-diseases/apple-scab",
        "hi": "सलाह: गिरी हुई पत्तियों को हटाकर अच्छी स्वच्छता बनाए रखें। बेहतर वायु संचार के लिए पेड़ों की छंटाई करें। प्रतिरोधी किस्मों पर विचार करें और आवश्यकता पड़ने पर निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: पडलेली पाने काढून चांगली स्वच्छता राखा. चांगल्या हवेच्या वहनासाठी झाडांची छाटणी करा. प्रतिरोधक जातींचा विचार करा आणि गरज पडल्यास निवारक फंगिसायड्स वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Apple___Black_rot": {
        "en": "Advisory: Practice rigorous orchard sanitation by removing mummified fruits and pruning out dead or diseased branches/cankers. Maintain tree health to reduce stress, as weakened trees are more susceptible. Fungicides can be used preventatively as part of a comprehensive spray program. More info: https://extension.umn.edu/plant-diseases/black-rot-apple",
        "hi": "सलाह: सड़े हुए फलों को हटाकर और मृत या रोगग्रस्त शाखाओं/कैंकरों की छंटाई करके सख्त बाग स्वच्छता का अभ्यास करें। तनाव कम करने के लिए पेड़ के स्वास्थ्य को बनाए रखें, क्योंकि कमजोर पेड़ अधिक संवेदनशील होते हैं। व्यापक स्प्रे कार्यक्रम के हिस्से के रूप में निवारक रूप से फफूंदनाशकों का उपयोग किया जा सकता है। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: सडलेल्या फळांना काढून आणि मृत किंवा रोगग्रस्त फांद्या/कँकरची छाटणी करून बागेची स्वच्छता काटेकोरपणे राखा. झाडांवरील ताण कमी करण्यासाठी त्यांचे आरोग्य चांगले ठेवा, कारण कमकुवत झाडे अधिक संवेदनशील असतात. व्यापक फवारणी कार्यक्रमाचा भाग म्हणून प्रतिबंधात्मक फंगिसायड्स वापरता येतात. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Apple___Cedar_apple_rust": {
        "en": "Advisory: Manage this disease by removing nearby cedar trees (the alternate host) within a few miles if possible. Plant resistant apple varieties. Apply fungicides effective against rust diseases from pink bud stage through third cover. More info: https://apples.extension.org/cedar-apple-rust/",
        "hi": "सलाह: यदि संभव हो तो कुछ मील के भीतर पास के देवदार के पेड़ों (वैकल्पिक मेजबान) को हटाकर इस बीमारी का प्रबंधन करें। प्रतिरोधी सेब की किस्में लगाएं। गुलाबी कली अवस्था से तीसरी कवर तक रस्ट रोगों के खिलाफ प्रभावी फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: शक्य असल्यास काही मैलांच्या आत जवळच्या देवदार वृक्षांना (पर्यायी होस्ट) काढून टाकून या रोगाचे व्यवस्थापन करा. प्रतिरोधक सफरचंदाच्या जाती लावा. गुलाबी कळी अवस्थेपासून तिसऱ्या कव्हरपर्यंत रस्ट रोगांवर प्रभावी फंगिसायड्स वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Apple___healthy": {
        "en": "Advisory: Your apple tree is healthy! Ensure at least 8 hours of direct sun daily. Plant a different apple variety or crabapple nearby for pollination. Maintain well-drained soil (pH 6-7). Prune annually in late winter for shape and air circulation. Provide 1 inch of water weekly during dry spells. Fertilize based on soil tests. Keep the base clear of weeds. More info: https://extension.umn.edu/fruit/growing-apples",
        "hi": "सलाह: आपका सेब का पेड़ स्वस्थ है! प्रतिदिन कम से कम 8 घंटे सीधी धूप सुनिश्चित करें। परागण के लिए पास में एक अलग सेब की किस्म या जंगली सेब लगाएं। अच्छी जल निकासी वाली मिट्टी (पीएच 6-7) बनाए रखें। आकार और वायु संचार के लिए देर सर्दियों में सालाना छंटाई करें। सूखे मंत्रों के दौरान प्रति सप्ताह 1 इंच पानी प्रदान करें। मिट्टी परीक्षण के आधार पर खाद डालें। आधार को खरपतवारों से साफ रखें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचे सफरचंदाचे झाड निरोगी आहे! दररोज कमीत कमी 8 तास थेट सूर्यप्रकाश असल्याची खात्री करा. परागकणासाठी जवळच वेगळी सफरचंद जात किंवा जंगली सफरचंद लावा. चांगल्या निचऱ्याची माती (pH 6-7) राखा. आकार आणि हवेच्या वहनासाठी दरवर्षी हिवाळ्यात छाटणी करा. कोरड्या काळात दर आठवड्याला 1 इंच पाणी द्या. माती परीक्षणानुसार खत द्या. तळाशी तणमुक्त ठेवा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Blueberry___healthy": {
        "en": "Advisory: Your blueberry plant is healthy! Ensure acidic soil (pH 4.0-5.5), full sun, consistent moisture (1-2 inches/week), and annually prune out old, weak, or dead wood for best production. More info: https://extension.umn.edu/fruit/growing-blueberries-home-garden",
        "hi": "सलाह: आपका ब्लूबेरी का पौधा स्वस्थ है! अम्लीय मिट्टी (पीएच 4.0-5.5), पूर्ण सूर्य, लगातार नमी (1-2 इंच/सप्ताह) सुनिश्चित करें, और सर्वोत्तम उत्पादन के लिए सालाना पुरानी, कमजोर या सूखी लकड़ी की छंटाई करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचा ब्लूबेरीचा रोपा निरोगी आहे! आम्ल माती (pH 4.0-5.5), पूर्ण सूर्यप्रकाश, सातत्यपूर्ण आर्द्रता (1-2 इंच/आठवडा) याची खात्री करा आणि उत्तम उत्पादनासाठी दरवर्षी जुनी, कमकुवत किंवा मृत लाकूड छाटणी करा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "en": "Advisory: Improve air circulation through pruning. Apply fungicides when symptoms first appear. Consider resistant varieties. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/powdery-mildew-trees-and-shrubs",
        "hi": "सलाह: छंटाई के माध्यम से वायु संचार में सुधार करें। लक्षण पहली बार दिखाई देने पर फफूंदनाशक लगाएं। प्रतिरोधी किस्मों पर विचार करें। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: छाटणीद्वारे हवेच्या वहनात सुधारणा करा. लक्षणे प्रथम दिसल्यावर फंगिसायड्स वापरा. प्रतिरोधक जातींचा विचार करा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Cherry_(including_sour)___healthy": {
        "en": "Advisory: Your cherry tree is healthy! Ensure full sun, good drainage, and protection from strong winds. Implement proper pruning for structure and light penetration, provide adequate watering, and fertilize annually based on soil tests. More info: https://extension.psu.edu/cherries-in-the-garden-and-the-kitchen",
        "hi": "सलाह: आपका चेरी का पेड़ स्वस्थ है! पूर्ण सूर्य, अच्छी जल निकासी, और तेज हवाओं से सुरक्षा सुनिश्चित करें। संरचना और प्रकाश के प्रवेश के लिए उचित छंटाई करें, पर्याप्त पानी दें, और मिट्टी परीक्षण के आधार पर सालाना खाद डालें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचे चेरीचे झाड निरोगी आहे! पूर्ण सूर्यप्रकाश, चांगला निचरा आणि जोरदार वाऱ्यापासून संरक्षण सुनिश्चित करा. संरचना आणि प्रकाश भेदनासाठी योग्य छाटणी करा, पुरेसे पाणी द्या आणि माती परीक्षणानुसार दरवर्षी खत द्या. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "en": "Advisory: Plant resistant hybrids. Practice crop rotation with non-hosts and manage corn residue (e.g., tillage) to reduce fungal inoculum. Apply fungicides preventatively if disease risk is high (susceptible hybrid, high humidity, early disease activity). More info: https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/gray-leaf-spot-on-corn/",
        "hi": "सलाह: प्रतिरोधी संकर लगाएं। फफूंद के संक्रमण को कम करने के लिए गैर-मेजबानों के साथ फसल चक्र अपनाएं और मकई के अवशेषों का प्रबंधन करें (उदाहरण के लिए, जुताई)। यदि रोग का जोखिम अधिक हो (संवेदनशील संकर, उच्च आर्द्रता, प्रारंभिक रोग गतिविधि) तो निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: प्रतिरोधक संकरित वाण लावा. बुरशीजन्य संसर्ग कमी करण्यासाठी गैर-होस्ट पिकांसोबत पीक फेरपालट करा आणि मक्याच्या अवशेषांचे व्यवस्थापन करा (उदा. नांगरणी). जर रोगाचा धोका जास्त असेल (संवेदनशील संकरित, उच्च आर्द्रता, प्रारंभिक रोग क्रियाकलाप) तर प्रतिबंधात्मक फंगिसायड्स वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Corn_(maize)___Common_rust": {
        "en": "Advisory: The best management practice is to use resistant corn hybrids. Fungicides can be beneficial, especially if applied early when few pustules have appeared. The disease favors cool, wet conditions. More info: https://extension.umn.edu/corn-pest-management/common-rust-corn",
        "hi": "सलाह: सबसे अच्छा प्रबंधन अभ्यास प्रतिरोधी मकई संकर का उपयोग करना है। फफूंदनाशक फायदेमंद हो सकते हैं, खासकर यदि कुछ फुंसी दिखाई देने पर जल्दी लगाए जाएं। यह रोग ठंडी, गीली परिस्थितियों को पसंद करता है। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: उत्तम व्यवस्थापन पद्धत म्हणजे प्रतिरोधक मक्याचे संकरित वाण वापरणे. फंगिसायड्स फायदेशीर ठरू शकतात, विशेषतः जेव्हा काही पुटिका दिसू लागताच लवकर वापरले जातात. हा रोग थंड, ओल्या परिस्थितीला अनुकूल असतो. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "en": "Advisory: Plant resistant hybrids (Ht genes or polygenic resistance). Practice crop rotation with non-hosts and manage corn residue through tillage to reduce inoculum. Consider fungicide applications on susceptible hybrids, especially if infection occurs before or at tasseling under favorable conditions. More info: https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight",
        "hi": "सलाह: प्रतिरोधी संकर लगाएं (Ht जीन या बहुजीनिक प्रतिरोध)। गैर-मेजबानों के साथ फसल चक्र अपनाएं और जुताई के माध्यम से मकई के अवशेषों का प्रबंधन करें ताकि संक्रमण कम हो सके। यदि अनुकूल परिस्थितियों में टेसलिंग से पहले या उसके दौरान संक्रमण होता है, तो संवेदनशील संकरों पर फफूंदनाशक अनुप्रयोगों पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: प्रतिरोधक संकरित वाण लावा (Ht जीन्स किंवा पॉलीजेनिक प्रतिरोध). संसर्ग कमी करण्यासाठी गैर-होस्ट पिकांसोबत पीक फेरपालट करा आणि नांगरणीद्वारे मक्याच्या अवशेषांचे व्यवस्थापन करा. संवेदनशील संकरित वाणांवर फंगिसायडचा वापर विचारात घ्या, विशेषतः जर अनुकूल परिस्थितीत टॅसलिंगपूर्वी किंवा त्यावेळी संसर्ग झाला असेल. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Corn_(maize)___healthy": {
        "en": "Advisory: Your corn plant is healthy! Plant in well-drained soil with full sun. Maintain proper spacing (30-36 inches between rows). Water regularly, especially during tasseling and silking. Apply balanced fertilizer based on soil tests. More info: https://extension.umn.edu/corn/growing-sweet-corn-home-garden",
        "hi": "सलाह: आपका मक्का का पौधा स्वस्थ है! अच्छी जल निकासी वाली मिट्टी में पूर्ण सूर्य के साथ लगाएं। उचित दूरी बनाए रखें (पंक्तियों के बीच 30-36 इंच)। नियमित रूप से पानी दें, विशेष रूप से टसलिंग और सिल्किंग के दौरान। मिट्टी परीक्षण के आधार पर संतुलित उर्वरक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचा मक्याचा रोपा निरोगी आहे! चांगल्या निचऱ्याच्या मातीत पूर्ण सूर्यप्रकाशासह लावा. योग्य अंतर राखा (ओळींमध्ये 30-36 इंच). नियमित पाणी द्या, विशेषतः टॅसलिंग आणि सिल्किंग दरम्यान. माती परीक्षणानुसार संतुलित खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Grape___Black_rot": {
        "en": "Advisory: Practice good canopy management for air circulation. Apply fungicides preventatively from bud break through fruit set. Remove and destroy infected plant parts. Consider resistant varieties. More info: https://extension.umn.edu/plant-diseases/black-rot-grapes",
        "hi": "सलाह: वायु संचार के लिए अच्छी छतरी प्रबंधन का अभ्यास करें। कली टूटने से फल सेट होने तक निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। प्रतिरोधी किस्मों पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: हवेच्या वहनासाठी चांगले छतरी व्यवस्थापन करा. कळी फुटल्यापासून फळ बांधल्यापर्यंत प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. प्रतिरोधक जातींचा विचार करा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Grape___Esca_(Black_Measles)": {
        "en": "Advisory: Prune during dry weather to prevent infection. Apply fungicides to pruning wounds. Remove and destroy infected vines. Consider trunk renewal. More info: https://extension.umn.edu/plant-diseases/esca-grapes",
        "hi": "सलाह: संक्रमण को रोकने के लिए सूखे मौसम में छंटाई करें। छंटाई के घावों पर फफूंदनाशक लगाएं। संक्रमित बेलों को हटा दें और नष्ट कर दें। तने की नवीनीकरण पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: संसर्ग टाळण्यासाठी कोरड्या हवामानात छाटणी करा. छाटणीच्या जखमांवर फंगिसायड्स वापरा. संसर्गित द्राक्षे काढून टाका. खोड नूतनीकरणाचा विचार करा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "en": "Advisory: Improve air circulation through canopy management. Apply fungicides when conditions favor disease development. Remove and destroy infected leaves. Consider resistant varieties. More info: https://extension.umn.edu/plant-diseases/leaf-blight-grapes",
        "hi": "सलाह: छतरी प्रबंधन के माध्यम से वायु संचार में सुधार करें। जब स्थितियां रोग विकास का पक्ष लेती हैं तो फफूंदनाशक लगाएं। संक्रमित पत्तियों को हटा दें और नष्ट कर दें। प्रतिरोधी किस्मों पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: छतरी व्यवस्थापनाद्वारे हवेच्या वहनात सुधारणा करा. रोग विकासास अनुकूल परिस्थिती असल्यावर फंगिसायड्स वापरा. संसर्गित पाने काढून टाका. प्रतिरोधक जातींचा विचार करा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Grape___healthy": {
        "en": "Advisory: Your grape vine is healthy! Plant in well-drained soil with full sun. Train vines on a trellis system. Prune annually in late winter. Water deeply but infrequently. Apply balanced fertilizer in early spring. More info: https://extension.umn.edu/fruit/growing-grapes-home-garden",
        "hi": "सलाह: आपकी अंगूर की बेल स्वस्थ है! अच्छी जल निकासी वाली मिट्टी में पूर्ण सूर्य के साथ लगाएं। बेलों को ट्रेलिस सिस्टम पर प्रशिक्षित करें। देर सर्दियों में सालाना छंटाई करें। गहराई से लेकिन कम बार पानी दें। शुरुआती वसंत में संतुलित उर्वरक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमची द्राक्षाची वेल निरोगी आहे! चांगल्या निचऱ्याच्या मातीत पूर्ण सूर्यप्रकाशासह लावा. वेलींना ट्रेलिस सिस्टमवर प्रशिक्षित करा. हिवाळ्यात दरवर्षी छाटणी करा. खोलवर पण कमी वेळा पाणी द्या. वसंत ऋतूच्या सुरुवातीला संतुलित खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "en": "Advisory: Control Asian citrus psyllid vectors. Remove infected trees. Use disease-free nursery stock. Apply systemic insecticides. More info: https://extension.umn.edu/plant-diseases/citrus-greening-huanglongbing",
        "hi": "सलाह: एशियाई साइट्रस साइलिड वैक्टर को नियंत्रित करें। संक्रमित पेड़ों को हटा दें। रोग मुक्त नर्सरी स्टॉक का उपयोग करें। प्रणालीगत कीटनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: आशियाई लिंबू सायलिड वेक्टर नियंत्रित करा. संसर्गित झाडे काढून टाका. रोगमुक्त नर्सरी स्टॉक वापरा. सिस्टेमिक कीडनाशके वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Peach___Bacterial_spot": {
        "en": "Advisory: Use resistant varieties. Apply copper-based bactericides during dormancy. Practice good sanitation. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/bacterial-spot-peach",
        "hi": "सलाह: प्रतिरोधी किस्मों का उपयोग करें। निष्क्रियता के दौरान तांबे-आधारित जीवाणुनाशक लगाएं। अच्छी स्वच्छता का अभ्यास करें। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: प्रतिरोधक जाती वापरा. निष्क्रिय काळात तांबे-आधारित जीवाणुनाशके वापरा. चांगली स्वच्छता राखा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Peach___healthy": {
        "en": "Advisory: Your peach tree is healthy! Plant in well-drained soil with full sun. Prune annually in late winter. Water deeply during dry periods. Apply balanced fertilizer in early spring. More info: https://extension.umn.edu/fruit/growing-stone-fruits-home-garden",
        "hi": "सलाह: आपका आड़ू का पेड़ स्वस्थ है! अच्छी जल निकासी वाली मिट्टी में पूर्ण सूर्य के साथ लगाएं। देर सर्दियों में सालाना छंटाई करें। सूखे मंत्रों के दौरान गहराई से पानी दें। शुरुआती वसंत में संतुलित उर्वरक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचे पीचचे झाड निरोगी आहे! चांगल्या निचऱ्याच्या मातीत पूर्ण सूर्यप्रकाशासह लावा. हिवाळ्यात दरवर्षी छाटणी करा. कोरड्या काळात खोलवर पाणी द्या. वसंत ऋतूच्या सुरुवातीला संतुलित खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Pepper,_bell___Bacterial_spot": {
        "en": "Advisory: Use disease-free seed and transplants. Practice crop rotation. Apply copper-based bactericides preventatively. Remove and destroy infected plants. More info: https://extension.umn.edu/plant-diseases/bacterial-spot-pepper",
        "hi": "सलाह: रोग मुक्त बीज और पौधों का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से तांबे-आधारित जीवाणुनाशक लगाएं। संक्रमित पौधों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त बियाणे आणि रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक तांबे-आधारित जीवाणुनाशके वापरा. संसर्गित रोपे काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Pepper,_bell___healthy": {
        "en": "Advisory: Your bell pepper plant is healthy! Plant in well-drained soil with full sun. Maintain consistent moisture. Apply balanced fertilizer. Stake plants for support. More info: https://extension.umn.edu/vegetables/growing-peppers",
        "hi": "सलाह: आपका शिमला मिर्च का पौधा स्वस्थ है! अच्छी जल निकासी वाली मिट्टी में पूर्ण सूर्य के साथ लगाएं। लगातार नमी बनाए रखें। संतुलित उर्वरक लगाएं। समर्थन के लिए पौधों को खूंटे से बांधें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचा भोपळी मिरचीचा रोपा निरोगी आहे! चांगल्या निचऱ्याच्या मातीत पूर्ण सूर्यप्रकाशासह लावा. सातत्यपूर्ण आर्द्रता राखा. संतुलित खते वापरा. समर्थनासाठी रोपांना खुंटी लावा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Potato___Early_blight": {
        "en": "Advisory: Practice crop rotation. Use certified disease-free seed potatoes. Apply fungicides preventatively. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/early-blight-potato",
        "hi": "सलाह: फसल चक्रण का अभ्यास करें। प्रमाणित रोग मुक्त बीज आलू का उपयोग करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: पीक फेरफार करा. प्रमाणित रोगमुक्त बटाटा बियाणे वापरा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Potato___Late_blight": {
        "en": "Advisory: Use certified disease-free seed potatoes. Practice crop rotation. Apply fungicides preventatively. Remove and destroy infected plants. More info: https://extension.umn.edu/plant-diseases/late-blight-potato",
        "hi": "सलाह: प्रमाणित रोग मुक्त बीज आलू का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: प्रमाणित रोगमुक्त बटाटा बियाणे वापरा. पीक फेरफार करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित रोपे काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Potato___healthy": {
        "en": "Advisory: Your potato plant is healthy! Plant certified seed potatoes in well-drained soil. Hill soil around plants as they grow. Water consistently. Apply balanced fertilizer. More info: https://extension.umn.edu/vegetables/growing-potatoes",
        "hi": "सलाह: आपका आलू का पौधा स्वस्थ है! प्रमाणित बीज आलू को अच्छी जल निकासी वाली मिट्टी में लगाएं। पौधों के बढ़ने के साथ उनके चारों ओर मिट्टी लगाएं। लगातार पानी दें। संतुलित उर्वरक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचा बटाट्याचा रोपा निरोगी आहे! प्रमाणित बटाटा बियाणे चांगल्या निचऱ्याच्या मातीत लावा. रोपे वाढत असताना त्यांच्या भोवती माती लावा. सातत्यपूर्ण पाणी द्या. संतुलित खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Raspberry___healthy": {
        "en": "Advisory: Your raspberry plant is healthy! Plant in well-drained soil with full sun. Prune annually to remove old canes. Maintain consistent moisture. Apply balanced fertilizer in early spring. More info: https://extension.umn.edu/fruit/growing-raspberries-home-garden",
        "hi": "सलाह: आपका रास्पबेरी का पौधा स्वस्थ है! अच्छी जल निकासी वाली मिट्टी में पूर्ण सूर्य के साथ लगाएं। पुरानी छड़ियों को हटाने के लिए सालाना छंटाई करें। लगातार नमी बनाए रखें। शुरुआती वसंत में संतुलित उर्वरक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचा रास्पबेरीचा रोपा निरोगी आहे! चांगल्या निचऱ्याच्या मातीत पूर्ण सूर्यप्रकाशासह लावा. जुन्या वेली काढण्यासाठी दरवर्षी छाटणी करा. सातत्यपूर्ण आर्द्रता राखा. वसंत ऋतूच्या सुरुवातीला संतुलित खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Soybean___healthy": {
        "en": "Advisory: Your soybean plant is healthy! Plant into warm (above 50°F) and well-drained soil. Ensure proper inoculation with Bradyrhizobium japonicum, as supplemental nitrogen is generally not needed. Provide consistent moisture, especially during pod development. Control weeds early to prevent yield loss. More info: https://extension.umn.edu/soybean/soybean-planting",
        "hi": "सलाह: आपका सोयाबीन का पौधा स्वस्थ है! गर्म (50°F से ऊपर) और अच्छी जल निकासी वाली मिट्टी में लगाएं। ब्रैडीराइजोबियम जैपोनिकम के साथ उचित टीकाकरण सुनिश्चित करें, क्योंकि पूरक नाइट्रोजन की आम तौर पर आवश्यकता नहीं होती है। विशेष रूप से फली के विकास के दौरान लगातार नमी प्रदान करें। उपज हानि को रोकने के लिए खरपतवारों को जल्दी नियंत्रित करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Squash___Powdery_mildew": {
        "en": "Advisory: Improve air circulation through proper spacing. Apply fungicides when symptoms first appear. Consider resistant varieties. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/powdery-mildew-vegetables",
        "hi": "सलाह: उचित दूरी के माध्यम से वायु संचार में सुधार करें। लक्षण पहली बार दिखाई देने पर फफूंदनाशक लगाएं। प्रतिरोधी किस्मों पर विचार करें। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: योग्य अंतराद्वारे हवेच्या वहनात सुधारणा करा. प्रतिरोधक जाती लावा. गरज पडल्यास निवारक फंगिसायड्स वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Strawberry___Leaf_scorch": {
        "en": "Advisory: Use disease-free plants. Practice crop rotation. Apply fungicides preventatively. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/leaf-scorch-strawberry",
        "hi": "सलाह: रोग मुक्त पौधों का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Strawberry___healthy": {
        "en": "Advisory: Ensure full sun and well-drained, slightly acidic soil (pH 5.5-6.5). Provide consistent moisture. Fertilize based on soil tests. More info: https://extension.umn.edu/fruit/growing-strawberries-home-garden",
        "hi": "सलाह: पूर्ण सूर्य और अच्छी जल निकासी वाली, थोड़ी अम्लीय मिट्टी (पीएच 5.5-6.5) सुनिश्चित करें। लगातार नमी प्रदान करें। मिट्टी परीक्षण के आधार पर खाद डालें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: पूर्ण सूर्यप्रकाश आणि चांगल्या निचऱ्याची, थोडी आम्ल माती (pH 5.5-6.5) सुनिश्चित करा. सातत्याने पाणी द्या. मातीच्या चाचणीनुसार खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Bacterial_spot": {
        "en": "Advisory: Use disease-free seed and transplants. Practice crop rotation. Apply copper-based bactericides preventatively. Remove and destroy infected plants. More info: https://extension.umn.edu/plant-diseases/bacterial-spot-tomato",
        "hi": "सलाह: रोग-मुक्त बीज और पौधे का उपयोग करें। फसल चक्रण का अभ्यास करें। ओवरहेड सिंचाई से बचें। निवारक रूप से तांबा-आधारित जीवाणुनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषी विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त बियाणे आणि रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक तांबे-आधारित जीवाणुनाशके वापरा. संसर्गित रोपे काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Early_blight": {
        "en": "Advisory: Use disease-free seed and transplants. Practice crop rotation. Apply fungicides preventatively. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/early-blight-tomato",
        "hi": "सलाह: रोग मुक्त बीज और पौधों का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त बियाणे आणि रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Late_blight": {
        "en": "Advisory: Use disease-free seed and transplants. Practice crop rotation. Apply fungicides preventatively. Remove and destroy infected plants. More info: https://extension.umn.edu/plant-diseases/late-blight-tomato",
        "hi": "सलाह: रोग मुक्त बीज और पौधों का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त बियाणे आणि रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित रोपे काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Leaf_Mold": {
        "en": "Advisory: Improve air circulation through proper spacing. Apply fungicides preventatively. Remove and destroy infected plant parts. Consider resistant varieties. More info: https://extension.umn.edu/plant-diseases/leaf-mold-tomato",
        "hi": "सलाह: उचित दूरी के माध्यम से वायु संचार में सुधार करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। प्रतिरोधी किस्मों पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: योग्य अंतराद्वारे हवेच्या वहनात सुधारणा करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. प्रतिरोधक जातींचा विचार करा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Septoria_leaf_spot": {
        "en": "Advisory: Use disease-free seed and transplants. Practice crop rotation. Apply fungicides preventatively. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/septoria-leaf-spot-tomato",
        "hi": "सलाह: रोग मुक्त बीज और पौधों का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त बियाणे आणि रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "en": "Advisory: Monitor plants regularly for early detection. Use insecticidal soaps or oils. Apply miticides when needed. Remove and destroy heavily infested plants. More info: https://extension.umn.edu/yard-and-garden-insects/spider-mites",
        "hi": "सलाह: शुरुआती पहचान के लिए पौधों की नियमित निगरानी करें। कीटनाशक साबुन या तेल का उपयोग करें। आवश्यकता पड़ने पर माइटिसाइड लगाएं। अत्यधिक संक्रमित पौधों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: लवकर शोधण्यासाठी रोपांची नियमित निरीक्षणे करा. कीडनाशक साबण किंवा तेले वापरा. गरज पडल्यास माइटिसायड्स वापरा. जास्त संसर्गित रोपे काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Target_Spot": {
        "en": "Advisory: Use disease-free seed and transplants. Practice crop rotation. Apply fungicides preventatively. Remove and destroy infected plant parts. More info: https://extension.umn.edu/plant-diseases/target-spot-tomato",
        "hi": "सलाह: रोग मुक्त बीज और पौधों का उपयोग करें। फसल चक्रण का अभ्यास करें। निवारक रूप से फफूंदनाशक लगाएं। संक्रमित पौधे के हिस्सों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: रोगमुक्त बियाणे आणि रोपे वापरा. पीक फेरफार करा. प्रतिबंधात्मक फंगिसायड्स वापरा. संसर्गित वनस्पती भाग काढून टाका. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "en": "Advisory: Control whitefly vectors. Use virus-resistant varieties. Remove and destroy infected plants. Practice good weed management. More info: https://extension.umn.edu/plant-diseases/tomato-yellow-leaf-curl-virus",
        "hi": "सलाह: व्हाइटफ्लाई वैक्टर को नियंत्रित करें। वायरस प्रतिरोधी किस्मों का उपयोग करें। संक्रमित पौधों को हटा दें और नष्ट कर दें। अच्छी खरपतवार प्रबंधन का अभ्यास करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: व्हाइटफ्लाय वेक्टर नियंत्रित करा. व्हायरस-प्रतिरोधक जाती वापरा. संसर्गित रोपे काढून टाका. चांगले तण व्यवस्थापन करा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___Tomato_mosaic_virus": {
        "en": "Advisory: Use virus-free seed and transplants. Control aphid vectors. Remove and destroy infected plants. Practice good sanitation. More info: https://extension.umn.edu/plant-diseases/tomato-mosaic-virus",
        "hi": "सलाह: वायरस मुक्त बीज और पौधों का उपयोग करें। एफिड वैक्टर को नियंत्रित करें। संक्रमित पौधों को हटा दें और नष्ट कर दें। अच्छी स्वच्छता का अभ्यास करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: व्हायरस-मुक्त बियाणे आणि रोपे वापरा. एफिड वेक्टर नियंत्रित करा. संसर्गित रोपे काढून टाका. चांगली स्वच्छता राखा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    },
    "Tomato___healthy": {
        "en": "Advisory: Your tomato plant is healthy! Plant in well-drained soil with full sun. Stake or cage plants for support. Water consistently at the base. Apply balanced fertilizer. More info: https://extension.umn.edu/vegetables/growing-tomatoes",
        "hi": "सलाह: आपका टमाटर का पौधा स्वस्थ है! अच्छी जल निकासी वाली मिट्टी में पूर्ण सूर्य के साथ लगाएं। समर्थन के लिए पौधों को खूंटे या पिंजरे से बांधें। आधार पर लगातार पानी दें। संतुलित उर्वरक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।",
        "mr": "सल्ला: तुमचा टोमॅटोचा रोपा निरोगी आहे! चांगल्या निचऱ्याच्या मातीत पूर्ण सूर्यप्रकाशासह लावा. समर्थनासाठी रोपांना खुंटी किंवा पिंजरा लावा. तळाशी सातत्यपूर्ण पाणी द्या. संतुलित खते वापरा. अधिक माहितीसाठी, तुमच्या स्थानिक कृषी विभागाशी संपर्क साधा."
    }
}

# Model initialization (load once at app startup)
# Global model variable to avoid reloading on each prediction
MODEL = None

def load_model():
    print("[DEBUG] Loading model...")
    try:
        model = get_model(num_classes=NUM_CLASSES)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # If checkpoint is a dict with 'model_state_dict', use that
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("[DEBUG] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        raise

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

def predict(image, language):
    print("[DEBUG] predict() called")
    try:
        if image is None:
            print("[DEBUG] No image provided.")
            return None, None, None, None
        global model
        if 'model' not in globals():
            print("[DEBUG] Model not loaded yet, loading now...")
            model = load_model()
        print("[DEBUG] Preprocessing image...")
        input_tensor = preprocess_image(image)
        print("[DEBUG] Running model inference...")
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        print(f"[DEBUG] Prediction: {predicted_class}, Confidence: {confidence}")
        if language == "English":
            translated_class = predicted_class
            advisory_key = "en"
        elif language == "हिंदी (Hindi)":
            translated_class = HINDI_TRANSLATIONS.get(predicted_class, predicted_class)
            advisory_key = "hi"
        elif language == "मराठी (Marathi)":
            translated_class = MARATHI_TRANSLATIONS.get(predicted_class, predicted_class)
            advisory_key = "mr"
        else:
            translated_class = predicted_class
            advisory_key = "en"
        advisory = ADVISORY_INFO.get(predicted_class, {}).get(advisory_key, "No advisory information available.")
        print("[DEBUG] predict() completed successfully.")
        return translated_class, confidence, advisory, predicted_class
    except Exception as e:
        print(f"[ERROR] Exception in predict(): {e}")
        import traceback
        traceback.print_exc()
        return "Error during prediction. Check logs.", None, None, None

def create_ui():
    with gr.Blocks(title="Krishi Rakshak - Crop Disease Detection", theme=gr.themes.Soft()) as demo:
        # Add custom CSS for better styling and theme
        
        
        gr.Markdown("""
        <style>
        html, body, .gradio-container, .gr-block, .gr-root, .gradio-app, .main, .footer-section {
            background: #f4f7f2 !important;
            color: #357a38 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 0 !important;
            background-color: #fffefb !important;
            box-shadow: 0 4px 16px rgba(60, 80, 40, 0.08) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        #header_section {
            background: linear-gradient(90deg, #e8f5e9 0%, #f4f7f2 100%) !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 12px 12px 0 0 !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: 100% !important;
        }
        #krishi_rakshak_logo img {
            display: block !important;
            margin: 0 auto 0.2rem auto !important;
            object-fit: contain !important;
            width: 110px !important;
            height: 110px !important;
            padding: 0 !important;
        }
        h1, h3, h4, h5, h6, label, .gradio-label, .gradio-title, .gradio-description {
            color: #357a38 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        h1 { margin-bottom: 4px !important; font-weight: 700 !important; letter-spacing: 1px !important; }
        h3 { margin-bottom: 10px !important; font-weight: 500 !important; color: #5d7c4a !important; }
        #krishi_rakshak_banner {
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        #krishi_rakshak_banner img {
            display: block !important;
            margin: 0 auto !important;
            max-width: 800px !important;
            width: 100% !important;
            height: auto !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(60, 80, 40, 0.07) !important;
        }
        .gr-row, .gradio-row, .gradio-block {
            margin-bottom: 5px !important;
            padding: 0 !important;
        }
        .gr-column, .gradio-column {
            padding: 5px !important;
            margin: 0 !important;
        }
        /* Upload/Preview fix */
        .custom-image-input .drop_zone {
            min-height: 200px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            position: relative !important;
            overflow: hidden !important;
            background: #f0f7ec !important;
            border: 2px dashed #b5c99a !important;
            border-radius: 8px !important;
        }
        .custom-image-input .image-preview {
            position: static !important;
            width: 100% !important;
            height: auto !important;
            object-fit: contain !important;
            margin-top: 10px !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 4px rgba(60, 80, 40, 0.05) !important;
        }
        .custom-image-input .clear_btn {
            margin-top: 5px !important;
        }
        /* Agrotech button theme */
        .gradio-button, button, .gr-button, .gradio-btn, .gr-button-primary {
            background: linear-gradient(90deg, #7cb342 0%, #388e3c 100%) !important;
            color: #fff !important;
            border: none !important;
            padding: 12px 25px !important;
            border-radius: 6px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: background 0.3s !important;
            box-shadow: 0 2px 8px rgba(60, 80, 40, 0.07) !important;
        }
        .gradio-button:hover, button:hover, .gr-button:hover, .gradio-btn:hover, .gr-button-primary:hover {
            background: linear-gradient(90deg, #558b2f 0%, #2e7d32 100%) !important;
        }
        /* Radio/inputs */
        .gr-text-input, .gr-radio-group, input[type="text"], input[type="radio"] {
            border: 1.5px solid #b5c99a !important;
            border-radius: 6px !important;
            padding: 10px !important;
            background: #f8fbe9 !important;
            color: #357a38 !important;
        }
        .gr-text-input:focus, .gr-radio-group:focus-within, input[type="text"]:focus {
            border-color: #7cb342 !important;
            box-shadow: 0 0 5px rgba(124, 179, 66, 0.3) !important;
            outline: none !important;
        }
        /* Footer */
        .footer-section {
            background: #e8f5e9 !important;
            padding: 10px !important;
            border-radius: 0 0 12px 12px !important;
            margin-top: 10px !important;
            color: #357a38 !important;
        }
        /* Accent lines and borders */
        hr, .gradio-separator, .gr-separator {
            border: 0;
            border-top: 2px solid #b5c99a !important;
            margin: 10px 0 !important;
        }
        </style>
        """)
        
        # Add logo and banner using gr.Image for proper display and sizing
        with gr.Column(elem_id="header_section", scale=1):
            gr.Image(value="assets/logos/logo.png", label=None, show_label=False, interactive=False, width=110, height=110, elem_id="krishi_rakshak_logo", container=False)
            gr.Markdown("<h1 style=\"color: #357a38; text-align: center; margin: 0 !important; padding: 0 !important; font-weight: 700; letter-spacing: 1px;\">🌱 Krishi Rakshak</h1>")
            gr.Markdown("<h3 style=\"color: #5d7c4a; text-align: center; margin: 0 !important; padding: 0 !important; font-weight: 500;\">स्वस्थ फसल, समृद्ध किसान | Healthy Crops, Prosperous Farmers | निरोगी पीक, समृद्ध शेतकरी</h3>")
            with gr.Row():
                gr.Image(value="assets/banners/banner.png", label=None, show_label=False, interactive=False, width=800, height=300, elem_id="krishi_rakshak_banner", container=False)
        
        # Add footer with instructions, moved higher for better visibility
        gr.Markdown("""
        <div class="footer-section" style="text-align: center;">
            <h4 style="color: #2E7D32;">How to Use / उपयोग कैसे करें / वापर कसा करावा</h4>
            <p>1. Upload a clear image of a crop leaf / फसल की पत्ती की स्पष्ट छवि अपलोड करें / पीक पानाची स्पष्ट प्रतिमा अपलोड करा</p>
            <p>2. Select your preferred language / अपनी पसंदीदा भाषा चुनें / तुमची पसंतीची भाषा निवडा</p>
            <p>3. Click 'Analyze Image' to get results / परिणाम प्राप्त करने के लिए 'छवि का विश्लेषण करें' पर क्लिक करें / परिणाम मिळविण्यासाठी 'प्रतिमेचे विश्लेषण करा' वर क्लिक करें</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Crop Image / फसल छवि अपलोड करें / पीक प्रतिमा अपलोड करा", elem_classes="custom-image-input")
                language = gr.Radio(
                    choices=["English", "हिंदी (Hindi)", "मराठी (Marathi)"],
                    value="English",
                    label="Select Language / भाषा चुनें / भाषा निवडा"
                )
                predict_btn = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column(scale=1):
                result = gr.Textbox(label="Prediction / अनुमान / अंदाज", interactive=False, elem_classes="gr-text-input")
                confidence = gr.Textbox(label="Confidence / आत्मविश्वास / आत्मविश्वास", interactive=False, elem_classes="gr-text-input")
                advisory = gr.Textbox(label="Advisory / सलाह / सल्ला", lines=5, interactive=False, elem_classes="gr-text-input")
                original_class = gr.Textbox(label="Original Class Name", visible=False)
        
        predict_btn.click(
            fn=predict,
            inputs=[image_input, language],
            outputs=[result, confidence, advisory, original_class]
        )
    
    return demo

def update_button_text(language):
    if language == "English":
        return gr.Button(value="Analyze Image")
    elif language == "हिंदी (Hindi)":
        return gr.Button(value="छवि का विश्लेषण करें")
    elif language == "मराठी (Marathi)":
        return gr.Button(value="प्रतिमेचे विश्लेषण करा")
    return gr.Button(value="Analyze Image")

def main():
    """Main function to run the application"""
    # Ensure assets directory exists
    os.makedirs("assets/logos", exist_ok=True)
    os.makedirs("assets/banners", exist_ok=True)

    # Try different ports sequentially for robustness
    ports_to_try = [7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870]
    app_started = False

    for port in ports_to_try:
        try:
            # Create the UI
            demo = create_ui()

            print(f"\n{'='*60}")
            print(f"🌱 Attempting to start Krishi Rakshak on port {port}...")
            print(f"{'='*60}\n")

            # Launch the application
            demo.queue() # Enable queuing for the Blocks app
            demo.launch(
                server_name="0.0.0.0",
                server_port=port, 
                share=True, # Allow sharing for a public URL
                inbrowser=True,
                quiet=False,
                show_error=True
            )

            # If we get here, launch was successful
            app_started = True
            break  # Exit loop if successful

        except OSError as e:
            print(f"❌ Port {port} is in use or unavailable: {e}")
            print("Trying next port...")
        except Exception as e:
            print(f"❌ An unexpected error occurred while starting on port {port}: {str(e)}")
            print("Trying next port...")

    if not app_started:
        print("\n❌ Critical Error: Could not start the application on any available port in the range 7860-7870.")
        print("💡 Please ensure no other applications are using these ports or restart your system.")
        exit(1)

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def find_available_port(start_port=7860, max_port=7870):
    for port in range(start_port, max_port + 1):
        if is_port_available(port):
            return port
    return None

# Define the Gradio interface
app = gr.Interface(fn=predict, inputs=[gr.Image(), gr.Radio(['English', 'Hindi', 'Marathi'])], outputs=gr.Label())

if __name__ == "__main__":
    main()
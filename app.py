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

# Hindi translations for class names (EXAMPLE - EXPAND AS NEEDED)
HINDI_TRANSLATIONS = {
    "Apple___Apple_scab": "सेब - सेब स्कैब",
    "Apple___Black_rot": "सेब - काला सड़न",
    "Apple___Cedar_apple_rust": "सेब - देवदार सेब रस्ट",
    "Apple___healthy": "सेब - स्वस्थ",
    "Blueberry___healthy": "ब्लूबेरी - स्वस्थ",
    "Cherry_(including_sour)___Powdery_mildew": "चेरी - ख़स्ता फफूंदी",
    "Cherry_(including_sour)___healthy": "चेरी - स्वस्थ",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "मक्का - सर्कोस्पोरा पत्ती धब्बा ग्रे पत्ती धब्बा",
    "Corn_(maize)___Common_rust_": "मक्का - सामान्य रस्ट",
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
    "Tomato___Spider_mites Two-spotted_spider_mite": "टमाटर - स्पाइडर माइट्स दो-धब्बेदार स्पाइडर माइट",
    "Tomato___Target_Spot": "टमाटर - टारगेट स्पॉट",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "टमाटर - टमाटर पीला पत्ती कर्ल वायरस",
    "Tomato___Tomato_mosaic_virus": "टमाटर - टमाटर मोज़ेक वायरस",
    "Tomato___healthy": "टमाटर - स्वस्थ"
}

# Advisory information for each disease
ADVISORY_INFO = {
    "Apple___Apple_scab": {
        "en": "Advisory: Practice good sanitation by removing fallen leaves. Prune trees for better air circulation. Consider resistant varieties and apply fungicides preventatively if needed. More info: https://extension.umn.edu/plant-diseases/apple-scab",
        "hi": "सलाह: गिरी हुई पत्तियों को हटाकर अच्छी स्वच्छता बनाए रखें। बेहतर वायु संचार के लिए पेड़ों की छंटाई करें। प्रतिरोधी किस्मों पर विचार करें और आवश्यकता पड़ने पर निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Apple___Black_rot": {
        "en": "Advisory: Practice rigorous orchard sanitation by removing mummified fruits and pruning out dead or diseased branches/cankers. Maintain tree health to reduce stress, as weakened trees are more susceptible. Fungicides can be used preventatively as part of a comprehensive spray program. More info: https://extension.umn.edu/plant-diseases/black-rot-apple",
        "hi": "सलाह: सड़े हुए फलों को हटाकर और मृत या रोगग्रस्त शाखाओं/कैंकरों की छंटाई करके सख्त बाग स्वच्छता का अभ्यास करें। तनाव कम करने के लिए पेड़ के स्वास्थ्य को बनाए रखें, क्योंकि कमजोर पेड़ अधिक संवेदनशील होते हैं। व्यापक स्प्रे कार्यक्रम के हिस्से के रूप में निवारक रूप से फफूंदनाशकों का उपयोग किया जा सकता है। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Apple___Cedar_apple_rust": {
        "en": "Advisory: Manage this disease by removing nearby cedar trees (the alternate host) within a few miles if possible. Plant resistant apple varieties. Apply fungicides effective against rust diseases from pink bud stage through third cover. More info: https://apples.extension.org/cedar-apple-rust/",
        "hi": "सलाह: यदि संभव हो तो कुछ मील के भीतर पास के देवदार के पेड़ों (वैकल्पिक मेजबान) को हटाकर इस बीमारी का प्रबंधन करें। प्रतिरोधी सेब की किस्में लगाएं। गुलाबी कली अवस्था से तीसरी कवर तक रस्ट रोगों के खिलाफ प्रभावी फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Apple___healthy": {
        "en": "Advisory: Your apple tree is healthy! Ensure at least 8 hours of direct sun daily. Plant a different apple variety or crabapple nearby for pollination. Maintain well-drained soil (pH 6-7). Prune annually in late winter for shape and air circulation. Provide 1 inch of water weekly during dry spells. Fertilize based on soil tests. Keep the base clear of weeds. More info: https://extension.umn.edu/fruit/growing-apples",
        "hi": "सलाह: आपका सेब का पेड़ स्वस्थ है! प्रतिदिन कम से कम 8 घंटे सीधी धूप सुनिश्चित करें। परागण के लिए पास में एक अलग सेब की किस्म या जंगली सेब लगाएं। अच्छी जल निकासी वाली मिट्टी (पीएच 6-7) बनाए रखें। आकार और वायु संचार के लिए देर सर्दियों में सालाना छंटाई करें। सूखे मंत्रों के दौरान प्रति सप्ताह 1 इंच पानी प्रदान करें। मिट्टी परीक्षण के आधार पर खाद डालें। आधार को खरपतवारों से साफ रखें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    # Add more advisory info for other diseases here
    "Blueberry___healthy": {
        "en": "Advisory: Your blueberry plant is healthy! Ensure acidic soil (pH 4.0-5.5), full sun, consistent moisture (1-2 inches/week), and annually prune out old, weak, or dead wood for best production. More info: https://extension.umn.edu/fruit/growing-blueberries-home-garden",
        "hi": "सलाह: आपका ब्लूबेरी का पौधा स्वस्थ है! अम्लीय मिट्टी (पीएच 4.0-5.5), पूर्ण सूर्य, लगातार नमी (1-2 इंच/सप्ताह) सुनिश्चित करें, और सर्वोत्तम उत्पादन के लिए सालाना पुरानी, कमजोर या सूखी लकड़ी की छंटाई करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "en": "Advisory: Improve air circulation through proper pruning. Manage irrigation to avoid excessive humidity. Remove root suckers. Apply preventative fungicides from shuck fall through harvest, rotating modes of action for resistance management. More info: https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
        "hi": "सलाह: उचित छंटाई से वायु संचार सुधारें। अत्यधिक नमी से बचने के लिए सिंचाई का प्रबंधन करें। जड़ से निकलने वाली शाखाओं को हटा दें। रोग प्रतिरोधक क्षमता के लिए फफूंदनाशक का उपयोग शुक फॉल से कटाई तक करें और उनकी कार्यप्रणाली बदलते रहें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Cherry_(including_sour)___healthy": {
        "en": "Advisory: Your cherry tree is healthy! Ensure full sun, good drainage, and protection from strong winds. Implement proper pruning for structure and light penetration, provide adequate watering, and fertilize annually based on soil tests. More info: https://extension.psu.edu/cherries-in-the-garden-and-the-kitchen",
        "hi": "सलाह: आपका चेरी का पेड़ स्वस्थ है! पूर्ण सूर्य, अच्छी जल निकासी, और तेज हवाओं से सुरक्षा सुनिश्चित करें। संरचना और प्रकाश के प्रवेश के लिए उचित छंटाई करें, पर्याप्त पानी दें, और मिट्टी परीक्षण के आधार पर सालाना खाद डालें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "en": "Advisory: Plant resistant hybrids. Practice crop rotation with non-hosts and manage corn residue (e.g., tillage) to reduce fungal inoculum. Apply fungicides preventatively if disease risk is high (susceptible hybrid, high humidity, early disease activity). More info: https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/gray-leaf-spot-on-corn/",
        "hi": "सलाह: प्रतिरोधी संकर लगाएं। फफूंद के संक्रमण को कम करने के लिए गैर-मेजबानों के साथ फसल चक्र अपनाएं और मकई के अवशेषों का प्रबंधन करें (उदाहरण के लिए, जुताई)। यदि रोग का जोखिम अधिक हो (संवेदनशील संकर, उच्च आर्द्रता, प्रारंभिक रोग गतिविधि) तो निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Corn_(maize)___Common_rust": {
        "en": "Advisory: The best management practice is to use resistant corn hybrids. Fungicides can be beneficial, especially if applied early when few pustules have appeared. The disease favors cool, wet conditions. More info: https://extension.umn.edu/corn-pest-management/common-rust-corn",
        "hi": "सलाह: सबसे अच्छा प्रबंधन अभ्यास प्रतिरोधी मकई संकर का उपयोग करना है। फफूंदनाशक फायदेमंद हो सकते हैं, खासकर यदि कुछ फुंसी दिखाई देने पर जल्दी लगाए जाएं। यह रोग ठंडी, गीली परिस्थितियों को पसंद करता है। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "en": "Advisory: Plant resistant hybrids (Ht genes or polygenic resistance). Practice crop rotation with non-hosts and manage corn residue through tillage to reduce inoculum. Consider fungicide applications on susceptible hybrids, especially if infection occurs before or at tasseling under favorable conditions. More info: https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight",
        "hi": "सलाह: प्रतिरोधी संकर लगाएं (Ht जीन या बहुजीनिक प्रतिरोध)। गैर-मेजबानों के साथ फसल चक्र अपनाएं और जुताई के माध्यम से मकई के अवशेषों का प्रबंधन करें ताकि संक्रमण कम हो सके। यदि अनुकूल परिस्थितियों में टेसलिंग से पहले या उसके दौरान संक्रमण होता है, तो संवेदनशील संकरों पर फफूंदनाशक अनुप्रयोगों पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Corn_(maize)___healthy": {
        "en": "Advisory: Your corn plant is healthy! Ensure full sun, deep, well-drained, fertile soil (pH 6.0-6.8), and consistent moisture (1 inch/week, especially before silking). Fertilize based on soil tests, and practice good pest management. More info: https://extension.unh.edu/resource/growing-sweet-corn-fact-sheet",
        "hi": "सलाह: आपका मकई का पौधा स्वस्थ है! पूर्ण सूर्य, गहरी, अच्छी जल निकासी वाली, उपजाऊ मिट्टी (पीएच 6.0-6.8) और लगातार नमी (1 इंच/सप्ताह, विशेष रूप से सिल्क आने से पहले) सुनिश्चित करें। मिट्टी परीक्षण के आधार पर खाद डालें, और अच्छी कीट प्रबंधन प्रथाओं का पालन करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Grape___Black_rot": {
        "en": "Advisory: Crucially, remove and destroy mummified berries and diseased canes from the vineyard. Ensure good air circulation through proper pruning and trellising. Apply fungicides preventatively from bud break through four to six weeks after bloom, rotating modes of action to prevent resistance. More info: https://extension.psu.edu/black-rot-on-grapes-in-home-gardens",
        "hi": "सलाह: महत्वपूर्ण रूप से, दाख की बारी से ममीकृत जामुन और रोगग्रस्त छड़ियों को हटा दें और नष्ट कर दें। उचित छंटाई और प्रशिक्षण के माध्यम से अच्छा वायु संचार सुनिश्चित करें। प्रतिरोध को रोकने के लिए, कली फूटने से लेकर फूल आने के चार से छह सप्ताह बाद तक निवारक रूप से फफूंदनाशक लगाएं, कार्रवाई के तरीकों को बदलते रहें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Grape___Esca_(Black_Measles)": {
        "en": "Advisory: Currently, there are no fully effective chemical treatments for Esca. Focus on cultural controls: avoid large pruning cuts, prune during dry weather, and consider wound protectants. Delay fruiting in young vines, and ensure proper planting and irrigation to reduce vine stress. More info: https://pnwhandbooks.org/plantdisease/host-disease/grape-vitis-spp-esca-young-esca-petri-disease",
        "hi": "सलाह: वर्तमान में, एस्का के लिए कोई पूरी तरह से प्रभावी रासायनिक उपचार नहीं हैं। सांस्कृतिक नियंत्रणों पर ध्यान दें: बड़े प्रूनिंग कट से बचें, शुष्क मौसम के दौरान प्रूनिंग करें, और घाव रक्षक पर विचार करें। युवा लताओं में फल लगने में देरी करें, और बेल के तनाव को कम करने के लिए उचित रोपण और सिंचाई सुनिश्चित करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "en": "Advisory: This disease is common in tropical and subtropical grapes and appears late in the season. Improve air circulation through proper pruning and trellising. Remove infected leaves and plant debris. Fungicides sprayed for other diseases during the season may help to reduce this disease. More info: https://plantvillage.psu.edu/topics/grape/infos",
        "hi": "सलाह: यह रोग उष्णकटिबंधीय और उपोष्णकटिबंधीय अंगूरों में आम है और देर से मौसम में दिखाई देता है। उचित छंटाई और प्रशिक्षण के माध्यम से हवा का संचार सुधारें। संक्रमित पत्तियों और पौधों के मलबे को हटा दें। मौसम के दौरान अन्य बीमारियों के लिए छिड़के गए फफूंदनाशक इस रोग को कम करने में मदद कर सकते हैं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Grape___healthy": {
        "en": "Advisory: Your grape plant is healthy! Ensure full sun (at least 6-8 hours direct sunlight), well-drained soil (pH 6.0-7.0 is ideal), and a sturdy trellis system for support. Prune annually in late winter or early spring while dormant, removing 80-90% of the previous year's growth to encourage new, fruitful shoots and good air circulation to prevent diseases. More info: https://extension.usu.edu/yardandgarden/research/grape-trellising-training-basics",
        "hi": "सलाह: आपका अंगूर का पौधा स्वस्थ है! पूर्ण सूर्य (कम से कम 6-8 घंटे सीधी धूप), अच्छी जल निकासी वाली मिट्टी (पीएच 6.0-7.0 आदर्श है), और समर्थन के लिए एक मजबूत ट्रेलिस प्रणाली सुनिश्चित करें। सुप्त अवस्था में देर सर्दियों या शुरुआती वसंत में सालाना छंटाई करें, पिछली फसल की 80-90% वृद्धि को हटा दें ताकि नई, फलदार टहनियों को बढ़ावा मिले और बीमारियों को रोकने के लिए अच्छा वायु संचार हो। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "en": "Advisory: Huanglongbing (HLB), or citrus greening, is a devastating bacterial disease with no known cure. Management focuses on preventing the spread of the Asian citrus psyllid (the vector), using certified disease-free nursery trees, and maintaining tree health with optimal nutrition and irrigation. Infected trees are typically removed to minimize further spread. More info: https://www.nifa.usda.gov/grants/programs/emergency-citrus-disease-research-extension-program/citrus-greening-huanglongbing-hlb",
        "hi": "सलाह: हुआंगलोंगबिंग (HLB), या खट्टे फलों का हरा होना, एक विनाशकारी जीवाणु रोग है जिसका कोई ज्ञात इलाज नहीं है। प्रबंधन एशियाई खट्टे सिल्लड (वाहक) के प्रसार को रोकने, प्रमाणित रोग-मुक्त नर्सरी पेड़ों का उपयोग करने, और इष्टतम पोषण और सिंचाई के साथ पेड़ के स्वास्थ्य को बनाए रखने पर केंद्रित है। आगे प्रसार को कम करने के लिए संक्रमित पेड़ों को आमतौर पर हटा दिया जाता है। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Peach___Bacterial_spot": {
        "en": "Advisory: Plant bacterial spot-resistant peach varieties where possible. Maintain tree vigor through proper fertilization and care. Practice timely pruning to improve air circulation and sunlight penetration. Minimize wind-blown sand, which can create wounds. Chemical sprays (e.g., copper or oxytetracycline) can be used preventatively; consult local extension for timing and products. More info: https://www.aces.edu/blog/topics/crop-production/bacterial-spot-treatment-in-peaches/",
        "hi": "सलाह: जहाँ संभव हो, बैक्टीरियल स्पॉट प्रतिरोधी आड़ू की किस्में लगाएं। उचित उर्वरक और देखभाल के माध्यम से पेड़ की शक्ति बनाए रखें। वायु संचार और सूर्य के प्रकाश के प्रवेश में सुधार के लिए समय पर छंटाई का अभ्यास करें। हवा से उड़ने वाली रेत को कम करें, जिससे घाव हो सकते हैं। रासायनिक स्प्रे (जैसे, तांबा या ऑक्सीटेट्रासाइक्लिन) का उपयोग निवारक रूप से किया जा सकता है; समय और उत्पादों के लिए स्थानीय कृषि विभाग से सलाह लें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Peach___healthy": {
        "en": "Advisory: Your peach tree is healthy! Ensure full sun (at least 8 hours/day), well-drained soil (pH 6-6.5), and good air drainage. Prune annually in late winter/early spring to maintain an open vase shape for light and air circulation. Provide consistent moisture, especially during fruit development, and fertilize based on soil tests. Thin fruit 3-4 weeks after bloom to 8-10 inches apart for optimal size. More info: https://ohioline.osu.edu/factsheet/hyg-1406",
        "hi": "सलाह: आपका आड़ू का पेड़ स्वस्थ है! पूर्ण सूर्य (कम से कम 8 घंटे/दिन), अच्छी जल निकासी वाली मिट्टी (पीएच 6-6.5), और अच्छे वायु संचार वाली जगह सुनिश्चित करें। प्रकाश और वायु संचार के लिए खुली फूलदान जैसी आकृति बनाए रखने के लिए देर सर्दियों/शुरुआती वसंत में सालाना छंटाई करें। फल विकास के दौरान लगातार नमी प्रदान करें, और मिट्टी परीक्षण के आधार पर खाद डालें। इष्टतम आकार के लिए फूल आने के 3-4 सप्ताह बाद फलों को 8-10 इंच की दूरी पर पतला करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Pepper,_bell___Bacterial_spot": {
        "en": "Advisory: Plant resistant bell pepper varieties. Use certified disease-free seeds or treat seeds with hot water. Practice a 3-year crop rotation with non-host plants and manage crop debris. Avoid overhead irrigation and working with wet plants. Apply copper-based bactericides preventatively, often combined with mancozeb. More info: https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper",
        "hi": "सलाह: प्रतिरोधी शिमला मिर्च की किस्में लगाएं। प्रमाणित रोग-मुक्त बीजों का उपयोग करें या बीजों को गर्म पानी से उपचारित करें। गैर-मेजबान पौधों के साथ 3 साल का फसल चक्र अपनाएं और फसल के मलबे का प्रबंधन करें। ओवरहेड सिंचाई और गीले पौधों के साथ काम करने से बचें। निवारक रूप से तांबा-आधारित जीवाणुनाशकों का प्रयोग करें, अक्सर मैनकोज़ेब के साथ मिलाकर। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Pepper,_bell___healthy": {
        "en": "Advisory: Your bell pepper plant is healthy! Ensure full sun (at least 6-8 hours/day) and well-drained soil with a pH of 5.8-6.6. Transplant after soil temperatures reach 60°F. Provide consistent moisture (1 inch/week) to prevent blossom end rot, especially during fruit development. Fertilize based on soil tests, and consider black plastic mulch for warmth and weed control. More info: https://extension.umn.edu/vegetables/growing-peppers",
        "hi": "सलाह: आपका शिमला मिर्च का पौधा स्वस्थ है! पूर्ण सूर्य (कम से कम 6-8 घंटे/दिन) और 5.8-6.6 पीएच वाली अच्छी जल निकासी वाली मिट्टी सुनिश्चित करें। मिट्टी का तापमान 60°F तक पहुंचने के बाद रोपण करें। विशेष रूप से फल विकास के दौरान, ब्लोसम एंड रॉट (Blossom end rot) को रोकने के लिए लगातार नमी (1 इंच/सप्ताह) प्रदान करें। मिट्टी परीक्षण के आधार पर खाद डालें, और गर्मी और खरपतवार नियंत्रण के लिए काली प्लास्टिक मल्च पर विचार करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Potato___Early_blight": {
        "en": "Advisory: Plant resistant potato varieties. Practice 2-3 year crop rotation with non-host crops. Remove and destroy infected plant debris and control volunteer potato plants and susceptible weeds. Avoid overhead irrigation, or water early in the day to allow foliage to dry. Apply fungicides preventatively starting at first sign of disease or after bloom, rotating active ingredients to prevent resistance. More info: https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
        "hi": "सलाह: प्रतिरोधी आलू की किस्में लगाएं। गैर-मेजबान फसलों के साथ 2-3 साल का फसल चक्र अपनाएं। संक्रमित पौधों के मलबे को हटा दें और नष्ट कर दें, और स्वयंसेवक आलू के पौधों और संवेदनशील खरपतवारों को नियंत्रित करें। ओवरहेड सिंचाई से बचें, या पत्तियों को सूखने देने के लिए दिन की शुरुआत में पानी दें। रोग के पहले लक्षण या फूल आने के बाद से ही निवारक रूप से फफूंदनाशक लगाएं, प्रतिरोध को रोकने के लिए सक्रिय सामग्री को बदलते रहें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Potato___Late_blight": {
        "en": "Advisory: Destroy all cull and volunteer potato plants. Plant certified late blight-free seed tubers. Avoid excessive and/or nighttime irrigation. Eliminate weed hosts like hairy nightshade. Scout fields regularly, especially in wet areas. Apply foliar fungicides preventatively on a regular schedule, adjusting based on disease forecasts. Quickly destroy any disease hotspots. Kill vines completely 2-3 weeks before harvest. More info: https://www.ndsu.edu/agriculture/extension/publications/late-blight-potato",
        "hi": "सलाह: सभी कटी हुई और स्वयंसेवक आलू के पौधों को नष्ट कर दें। प्रमाणित लेट ब्लाइट-मुक्त बीज कंद लगाएं। अत्यधिक और/या रात में सिंचाई से बचें। कांटेदार धतूरा जैसे खरपतवार मेजबानों को खत्म करें। विशेष रूप से गीले क्षेत्रों में, खेतों की नियमित रूप से निगरानी करें। रोग के पूर्वानुमान के आधार पर, नियमित रूप से फोलियर फफूंदनाशक लगाएं। किसी भी रोग के हॉटस्पॉट को तुरंत नष्ट करें। कटाई से 2-3 सप्ताह पहले बेलों को पूरी तरह से नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Potato___healthy": {
        "en": "Advisory: Your potato plant is healthy! Use certified disease-free seed tubers. Plant in well-drained, sandy, slightly acidic soil (pH 6-6.5). Hill soil around plants as they grow to protect tubers from greening. Provide consistent moisture (1 inch/week) especially during tuber enlargement. Fertilize based on soil tests, and practice crop rotation, avoiding planting in the same spot for 3-4 years. More info: https://extension.umn.edu/vegetables/growing-potatoes",
        "hi": "सलाह: आपका आलू का पौधा स्वस्थ है! प्रमाणित रोग-मुक्त बीज कंदों का उपयोग करें। अच्छी जल निकासी वाली, रेतीली, थोड़ी अम्लीय मिट्टी (पीएच 6-6.5) में लगाएं। कंदों को हरा होने से बचाने के लिए पौधों के बढ़ने पर उनके चारों ओर मिट्टी चढ़ाएं। विशेष रूप से कंद के बढ़ने के दौरान लगातार नमी (1 इंच/सप्ताह) प्रदान करें। मिट्टी परीक्षण के आधार पर खाद डालें, और फसल चक्र का अभ्यास करें, 3-4 साल तक एक ही जगह पर रोपण से बचें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Raspberry___healthy": {
        "en": "Advisory: Your raspberry plant is healthy! Ensure full sun (at least 6-8 hours/day) and well-drained soil rich in organic matter (pH 5.5-7.0). Provide 1-1.5 inches of water per week, especially from flowering until harvest. Prune annually to remove old fruited canes and thin new ones for good air circulation. Consider trellising for support to prevent canes from sprawling. Use certified disease-free plants. More info: https://extension.umn.edu/fruit/growing-raspberries-home-garden",
        "hi": "सलाह: आपका रास्पबेरी का पौधा स्वस्थ है! पूर्ण सूर्य (कम से कम 6-8 घंटे/दिन) और जैविक पदार्थ से भरपूर अच्छी जल निकासी वाली मिट्टी (पीएच 5.5-7.0) सुनिश्चित करें। विशेष रूप से फूल आने से लेकर कटाई तक प्रति सप्ताह 1-1.5 इंच पानी दें। अच्छी वायु संचार के लिए सालाना पुरानी फल देने वाली छड़ियों को हटा दें और नई छड़ियों को पतला करें। छड़ियों को फैलने से रोकने के लिए सहारा देने के लिए ट्रेलिसिंग पर विचार करें। प्रमाणित रोग-मुक्त पौधों का उपयोग करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Soybean___healthy": {
        "en": "Advisory: Your soybean plant is healthy! Plant into warm (above 50°F) and well-drained soil. Ensure proper inoculation with Bradyrhizobium japonicum, as supplemental nitrogen is generally not needed. Provide consistent moisture, especially during pod development. Control weeds early to prevent yield loss. More info: https://extension.umn.edu/soybean/soybean-planting",
        "hi": "सलाह: आपका सोयाबीन का पौधा स्वस्थ है! गर्म (50°F से ऊपर) और अच्छी जल निकासी वाली मिट्टी में लगाएं। ब्रैडीराइजोबियम जैपोनिकम के साथ उचित टीकाकरण सुनिश्चित करें, क्योंकि पूरक नाइट्रोजन की आम तौर पर आवश्यकता नहीं होती है। विशेष रूप से फली के विकास के दौरान लगातार नमी प्रदान करें। उपज हानि को रोकने के लिए खरपतवारों को जल्दी नियंत्रित करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Squash___Powdery_mildew": {
        "en": "Advisory: Improve air circulation through proper spacing and pruning. Plant resistant varieties. Apply fungicides preventatively if needed. More info: example.com/squash_mildew_en",
        "hi": "सलाह: उचित दूरी और छंटाई के माध्यम से हवा का संचार सुधारें। प्रतिरोधी किस्में लगाएं। आवश्यकता पड़ने पर निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Strawberry___Leaf_scorch": {
        "en": "Advisory: Remove infected leaves. Ensure good air circulation. Apply fungicides if necessary. More info: example.com/strawberry_scorch_en",
        "hi": "सलाह: संक्रमित पत्तियों को हटा दें। अच्छी हवा का संचार सुनिश्चित करें। आवश्यकता पड़ने पर फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Strawberry___healthy": {
        "en": "Advisory: Ensure full sun and well-drained, slightly acidic soil (pH 5.5-6.5). Provide consistent moisture. Fertilize based on soil tests. More info: example.com/strawberry_healthy_en",
        "hi": "सलाह: पूर्ण सूर्य और अच्छी जल निकासी वाली, थोड़ी अम्लीय मिट्टी (पीएच 5.5-6.5) सुनिश्चित करें। लगातार नमी प्रदान करें। मिट्टी परीक्षण के आधार पर खाद डालें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Bacterial_spot": {
        "en": "Advisory: Use certified disease-free seeds/transplants. Practice crop rotation. Avoid overhead irrigation. Apply copper-based bactericides preventatively. More info: example.com/tomato_bacterial_en",
        "hi": "सलाह: प्रमाणित रोग-मुक्त बीजों/पौधों का उपयोग करें। फसल चक्र का अभ्यास करें। ओवरहेड सिंचाई से बचें। निवारक रूप से तांबा-आधारित जीवाणुनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Early_blight": {
        "en": "Advisory: Plant resistant varieties. Practice crop rotation and remove plant debris. Apply fungicides preventatively. More info: example.com/tomato_early_en",
        "hi": "सलाह: प्रतिरोधी किस्में लगाएं। फसल चक्र का अभ्यास करें और पौधों के मलबे को हटा दें। निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Late_blight": {
        "en": "Advisory: Eliminate inoculum sources. Plant resistant varieties if available. Manage irrigation to reduce foliage wetness. Apply fungicides preventatively. More info: example.com/tomato_late_en",
        "hi": "सलाह: संक्रमण के स्रोतों को खत्म करें। यदि उपलब्ध हो तो प्रतिरोधी किस्में लगाएं। पत्तियों की नमी को कम करने के लिए सिंचाई का प्रबंधन करें। निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Leaf_Mold": {
        "en": "Advisory: Improve air circulation by proper spacing and pruning. Manage greenhouse humidity. Plant resistant varieties. More info: example.com/tomato_leaf_mold_en",
        "hi": "सलाह: उचित दूरी और छंटाई से हवा का संचार सुधारें। ग्रीनहाउस की नमी का प्रबंधन करें। प्रतिरोधी किस्में लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Septoria_leaf_spot": {
        "en": "Advisory: Practice crop rotation and remove plant debris. Avoid overhead irrigation. Apply fungicides preventatively. More info: example.com/tomato_septoria_en",
        "hi": "सलाह: फसल चक्र का अभ्यास करें और पौधों के मलबे को हटा दें। ओवरहेड सिंचाई से बचें। निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "en": "Advisory: Monitor plants regularly for mites. Use strong streams of water to dislodge mites. Consider insecticidal soaps or horticultural oils for control. Introduce natural predators if feasible. More info: example.com/tomato_spider_mites_en",
        "hi": "सलाह: पौधों पर नियमित रूप से मकड़ियों की निगरानी करें। मकड़ियों को हटाने के लिए पानी की तेज धाराओं का उपयोग करें। नियंत्रण के लिए कीटनाशक साबुन या बागवानी तेलों पर विचार करें। यदि संभव हो तो प्राकृतिक शिकारियों को पेश करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Target_Spot": {
        "en": "Advisory: Use resistant varieties if available. Practice crop rotation and remove plant debris. Avoid overhead irrigation. Apply fungicides preventatively. More info: example.com/tomato_target_en",
        "hi": "सलाह: यदि उपलब्ध हो तो प्रतिरोधी किस्मों का उपयोग करें। फसल चक्र का अभ्यास करें और पौधों के मलबे को हटा दें। ओवरहेड सिंचाई से बचें। निवारक रूप से फफूंदनाशक लगाएं। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Tomato_mosaic_virus": {
        "en": "Advisory: There is no cure for viral diseases. Remove and destroy infected plants immediately. Disinfect tools. Manage insect vectors. Use resistant varieties. More info: example.com/tomato_mosaic_en",
        "hi": "सलाह: वायरल रोगों का कोई इलाज नहीं है। संक्रमित पौधों को तुरंत हटा दें और नष्ट कर दें। औजारों को कीटाणुरहित करें। कीट वाहकों का प्रबंधन करें। प्रतिरोधी किस्मों का उपयोग करें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "en": "Advisory: This virus is transmitted by whiteflies. Control whitefly populations using insecticidal soaps, neem oil, or appropriate insecticides. Use resistant varieties or reflective mulches. Remove and destroy infected plants. More info: example.com/tomato_yellow_curl_en",
        "hi": "सलाह: यह वायरस सफेद मक्खियों द्वारा फैलता है। कीटनाशक साबुन, नीम के तेल, या उपयुक्त कीटनाशकों का उपयोग करके सफेद मक्खी आबादी को नियंत्रित करें। प्रतिरोधी किस्मों या परावर्तक मल्च का उपयोग करें। संक्रमित पौधों को हटा दें और नष्ट कर दें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    },
    "Tomato___healthy": {
        "en": "Advisory: Your tomato plant is healthy! Ensure full sun (6-8 hours/day) and well-drained, fertile soil (pH 6.0-6.8). Water consistently at the base of the plant to avoid foliage wetness. Support plants with stakes or cages. Fertilize based on soil tests. More info: example.com/tomato_healthy_en",
        "hi": "सलाह: आपका टमाटर का पौधा स्वस्थ है! पूर्ण सूर्य (6-8 घंटे/दिन) और अच्छी जल निकासी वाली, उपजाऊ मिट्टी (पीएच 6.0-6.8) सुनिश्चित करें। पत्तियों की नमी से बचने के लिए पौधे के आधार पर लगातार पानी दें। डंडों या पिंजरों से पौधों को सहारा दें। मिट्टी परीक्षण के आधार पर खाद डालें। अधिक जानकारी के लिए, अपने स्थानीय कृषि विभाग से संपर्क करें।"
    }
}

# Model initialization (load once at app startup)
# Global model variable to avoid reloading on each prediction
MODEL = None

def load_model():
    """Load the pre-trained model based on your training configuration."""
    global MODEL
    if MODEL is None:
        print(f"Loading model from: {MODEL_PATH}")
        # Initialize the model with the correct architecture and number of classes
        # Use pretrained=False because we are loading specific weights
        MODEL = get_model(model_name="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=False)
        
        # Load the trained state_dict
        # map_location='cpu' ensures it loads correctly even if trained on GPU
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'])
        MODEL.eval() # Set model to evaluation mode
        print("Model loaded successfully!")
    return MODEL

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
        confidence_score = float(confidence[preds[0]])
        
        # Return only the single most confident prediction
        if language == "हिंदी (Hindi)":
            translated_class_name = HINDI_TRANSLATIONS.get(predicted_class, predicted_class)
            advisory = ADVISORY_INFO.get(predicted_class, {"en": "No specific advisory available.", "hi": "कोई विशेष सलाह उपलब्ध नहीं है।"})
            return {
                "अनुमानित_रोग": translated_class_name,
                "आत्मविश्वास_स्कोर": f"{confidence_score:.1f}%",
                "सलाह": advisory["hi"]
            }
        else:
            advisory = ADVISORY_INFO.get(predicted_class, {"en": "No specific advisory available.", "hi": "कोई विशेष सलाह उपलब्ध नहीं है।"})
            return {
                "predicted_disease": predicted_class,
                "confidence_score": f"{confidence_score:.1f}%",
                "advisory": advisory["en"]
            }
        
    except Exception as e:
        return {"error": str(e)}

def create_ui():
    """Create the main Gradio interface"""
    css = """
    body {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        color: #212529; /* Darker default text color */
    }
    .main-content {
        max-width: 800px;
        margin: 0 auto;
        padding: 15px; /* Reduced padding */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 10px;
        background-color: #ffffff;
    }
    .gradio-container { background-color: #f0f2f5; } /* Light grey background for the whole page */
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px; /* Adjust as needed */
        margin-bottom: 15px; /* Reduced margin */
    }
    .banner {
        width: 100%;
        max-height: 200px; /* Set a max-height for the banner */
        object-fit: cover; /* Ensure the image covers the area without distortion */
        border-radius: 10px;
        margin-bottom: 15px; /* Reduced margin */
    }
    h1 { color: #1e7e34; text-align: center; font-size: 2.8em; margin-bottom: 8px; } /* Reduced margin */
    h3 { color: #004085; font-size: 1.8em; margin-top: 15px; } /* Reduced margin */
    p { color: #333333; font-size: 1.0em; }
    .analyze-btn {
        background-color: #28a745 !important;
        color: white !important;
        border-color: #28a745 !important;
    }
    .analyze-btn:hover { background-color: #218838 !important; }
    .language-selector label { font-weight: bold; }
    .upload-section, .results-section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px; /* Reduced padding */
        background-color: #fefefe;
    }
    .results-title { color: #004085; text-align: center; margin-bottom: 10px; } /* Reduced margin */
    .gr-json-display {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px; /* Reduced padding */
        font-family: 'Arial', sans-serif; /* Changed font to a more readable sans-serif */
        font-size: 1.0em; /* Slightly larger font size */
        line-height: 1.4; /* Reduced line height for tighter spacing */
        font-weight: normal; /* Ensure normal weight for readability, can be changed to bold if needed */
        color: #000000; /* Black color for analysis output */
        height: auto !important; /* Allow height to expand */
        overflow: visible !important; /* Prevent internal scrolling */
    }
    """
    
    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_classes=["main-content"]):
            # Optional Banner Image (if you have one)
            gr.Image("assets/banners/banner.png", label="", show_label=False, container=False, elem_classes=["banner"])

            # Logo and Title
            gr.Image("assets/logos/logo.png", label="", show_label=False, container=False, elem_classes=["logo"])
            gr.Markdown("# 🌱 Krishi Rakshak: AI-Powered Crop Disease Detection")
            gr.Markdown("**Upload an image of a plant leaf to get instant disease diagnosis and recommendations.**")
            
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
                    type="numpy",
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

# Helper function to update button text based on language selection
def update_button_text(language):
    if language == "English":
        return gr.Button(value="Analyze Image")
    elif language == "हिंदी (Hindi)":
        return gr.Button(value="छवि का विश्लेषण करें")
    return gr.Button(value="Analyze Image")

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def find_available_port(start_port=7860, max_port=7870):
    for port in range(start_port, max_port + 1):
        if is_port_available(port):
            return port
    return None

# Define the Gradio interface
app = gr.Interface(fn=predict, inputs=[gr.Image(), gr.Radio(['English', 'Hindi'])], outputs=gr.Label())

if __name__ == "__main__":
    main()
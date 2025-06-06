# Krishi Rakshak

> स्वस्थ फसल, समृद्ध किसान

An AI-powered crop health guardian that predicts crop diseases using RGB images and deep learning. Designed for early detection and advisory support to farmers in Hindi and English.

## Project Structure

```
KrishiRakshak/
├── data/                    # Dataset and processed data
├── models/                  # Model files (.pt, .tflite)
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and training code
│   ├── utils/             # Utility functions
│   └── app/               # Web UI and mobile app code
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/VIKAS9793/KrishiRakshak.git
   cd KrishiRakshak
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract it to the `data/` directory.

## Training

To train the model:

```bash
python src/train.py --data_dir data/plantvillage --model_name efficientnetb0 --epochs 50 --batch_size 32
```

## Web UI

Run the Gradio interface:

```bash
python src/app/web_ui.py
```

## Mobile App

See the `app/` directory for Flutter mobile app setup instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

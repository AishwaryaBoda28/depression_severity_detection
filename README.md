# 🧠 Depression Detection System

> An ethical, explainable and multilingual deep learning system for early detection and severity estimation of depression from social media text and facial images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25-brightgreen.svg)]()

---

## 📌 About The Project

Depression is one of the most common mental health disorders worldwide. Early detection is critical but often delayed due to social stigma and lack of access to mental health professionals.

This system leverages **social media text and facial images** to automatically detect and estimate the severity of depression using state-of-the-art deep learning models. The system goes beyond binary classification by predicting **four severity levels** and provides **explainable AI** outputs so users can understand why a prediction was made.

---

## 🎯 Key Features

- ✅ **4-Class Severity Detection** — Not Depressed, Mild, Moderate, Severe
- ✅ **Dual Transformer Encoder** — BERT + MuRIL for rich multilingual embeddings
- ✅ **BiLSTM Temporal Modeling** — captures emotional progression in text
- ✅ **Multimodal Analysis** — both text and facial image supported
- ✅ **3-Layer Explainability** — SHAP + LIME + Word Importance graphs
- ✅ **Multilingual Support** — auto-detects and translates any language to English
- ✅ **Keyword Safety Rules** — crisis keywords force appropriate severity levels
- ✅ **Continuous Severity Score** — 0 to 5 scale alongside categorical prediction
- ✅ **Real-time Web Interface** — Flask-based web application

---

## 📊 Results

Trained and evaluated on the **Reddit Depression Dataset**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Depressed | 0.94 | 0.98 | 0.96 | 47 |
| Mild | 0.98 | 0.92 | 0.95 | 48 |
| Moderate | 1.00 | 0.90 | 0.95 | 48 |
| Severe | 0.93 | 0.99 | 0.96 | 96 |
| **Macro Avg** | **0.96** | **0.95** | **0.95** | **239** |

### Overall Accuracy: **95%**

### Confusion Matrix
```
                Not Depressed  Mild  Moderate  Severe
Not Depressed  [     46         1       0        0  ]
Mild           [      1        44       0        3  ]
Moderate       [      1         0      43        4  ]
Severe         [      1         0       0       95  ]
```

> **Key Insight:** No Severe case was ever misclassified as Not Depressed — the model errs on the side of caution which is critical in a mental health context.

---

## 🏗️ Architecture
```
Text Input (any language)
        ↓
Language Detection + Translation
        ↓
┌─────────────────────────────┐
│  BERT-base-uncased (768-dim) │
│  +                          │
│  MuRIL (768-dim)            │
│  ↓                          │
│  Concatenated (1536-dim)    │
└─────────────────────────────┘
        ↓
BiLSTM (1536 → 256-dim)
        ↓
FC Layer (256 → 128-dim)
        ↓
┌──────────────┬──────────────┐
│ Classification│  Regression  │
│    Head       │    Head      │
│ (4 classes)   │  (0-5 score) │
└──────────────┴──────────────┘
        ↓
SHAP + LIME + Word Importance
      Explainability
```

**Image Path (Optional):**
```
Facial Image
     ↓
DeepFace Emotion Analysis
(happy, sad, fear, disgust,
 angry, neutral, surprise)
     ↓
Weighted Scoring → 4-class prediction
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| Text Encoders | BERT-base-uncased, MuRIL (HuggingFace) |
| Sequential Modeling | PyTorch BiLSTM |
| Image Analysis | DeepFace |
| Explainability | SHAP, LIME |
| Translation | GoogleTranslator, langdetect |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, Jinja2 |
| Dataset | Reddit Depression Dataset |

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/depression-detection-system.git
cd depression-detection-system
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Trained Model
Download `depression_model.pt` from Google Drive:
👉 [Download depression_model.pt](YOUR_GOOGLE_DRIVE_LINK_HERE)

Place it in the root directory of the project.

---

## 🚀 Run the Application
```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

---

## 📁 Project Structure
```
depression-detection-system/
│
├── app.py                  # Flask web application + prediction logic
├── model.py                # BERT + MuRIL + BiLSTM model architecture
├── main.py                 # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── templates/
│   └── index.html          # Frontend web interface
│
└── static/                 # Generated explanation graphs (auto-created)
    ├── lime_text.png        # Word importance graph
    ├── shap_text.png        # SHAP explanation graph
    ├── lime_text.html       # Interactive LIME explanation
    └── lime_image.png       # Facial emotion graph
```

---

## 🔍 How It Works

### Text Analysis
1. User enters text in any language
2. Language is auto-detected and translated to English
3. Text is encoded using BERT + MuRIL dual encoder (1536-dim)
4. BiLSTM captures sequential emotional patterns
5. Dual output layer predicts class + severity score
6. Keyword safety rules apply crisis-level overrides
7. SHAP, LIME and Word Importance graphs generated

### Image Analysis
1. User uploads a facial photo
2. DeepFace extracts 7 emotion probabilities
3. Weighted scoring maps emotions to depression severity
4. 4-class prediction + severity score returned
5. Emotion bar chart generated

---

## 💡 Explainability

The system provides **3 types of explanations** for every text prediction:

| Method | Description |
|--------|-------------|
| **Word Importance** | Custom masking — shows impact of each word by removing it |
| **SHAP** | Shapley values — mathematically rigorous marginal contribution per word |
| **LIME** | Local interpretable model — highlights supporting/opposing words interactively |

---

## 🌐 Multilingual Support

The system supports all major languages including:
- Hindi, Telugu, Tamil, Kannada, Malayalam
- Arabic, French, Spanish, German
- And many more via GoogleTranslator

---

## ⚠️ Disclaimer

This system is intended for **research and academic purposes only**. It is **not a clinical diagnostic tool** and should not be used as a substitute for professional mental health assessment. If you or someone you know is experiencing depression, please seek help from a qualified mental health professional.

---

## 👥 Team

| Name | Role |
|------|------|
| Mrs. V. Prathyusha | Project Guide, Assistant Professor |
| Pavani Ponnala | Developer |
| Boda Aishwarya | Developer |
| Gunti Adhithya | Developer |

---

## 🏫 Institution

**Department of Artificial Intelligence & Machine Learning**
CMR College of Engineering & Technology
Hyderabad, India

---

## 📄 Research Paper

*"Mining Social Media Data For Early Detection Of Depression Using Deep Learning"*
CMR College of Engineering & Technology, Department of AIML, 2026

---

## 📚 References

Key references used in this project:
- BERT: Devlin et al., 2019
- MuRIL: Khanuja et al., 2021
- SHAP: Lundberg & Lee, 2017
- LIME: Ribeiro et al., 2016
- DeepFace: Serengil & Ozpinar, 2020
- Reddit Depression Dataset

---

## 🔮 Future Work

- Joint feature fusion of text and image modalities
- Integration of audio modality for voice tone analysis
- Testing on larger real-world datasets
- Clinical validation with mental health professionals
- ResNet-18 based visual encoding
- Domain adaptation strategies

---

*Made with ❤️ by Team AIML, CMR College of Engineering & Technology*

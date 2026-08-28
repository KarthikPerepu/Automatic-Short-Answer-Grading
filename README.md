<div align="center">
  <h1>📝 Automated Short Answer Grading (ASAG)</h1>
  <p><i>An NLP model for automated student evaluation</i></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Model%20Hub-ffcc00.svg)](https://huggingface.co/sridhanush1208/asag-deberta-large)
  [![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io/)
</div>

---

## 📖 Overview
This repository contains a state-of-the-art **Automated Short Answer Grading (ASAG)** system. Built upon the powerful `microsoft/deberta-v3-large` architecture, this model automatically evaluates free-text student answers against a teacher's reference answer, providing instantaneous, highly accurate grades.

The system is designed to seamlessly replicate human grading intuition by leveraging deep semantic cross-encoding rather than simple keyword matching.

## ✨ Key Features
- **State-of-the-Art Architecture**: Utilizes `DeBERTa-v3-large` configured as a Cross-Encoder for deep bidirectional semantic comparison.
- **Massive Training Corpus**: Trained on an unprecedented aggregation of 8 distinct ASAG benchmarks (including Mohler, ScientsBank, Beetle, SemEval, and EngSAF), comprising over 21,000 unique student-reference pairs.
- **Auto-Calibrated Scoring**: Dynamically scales the grading out of any arbitrary maximum score (e.g., /10 or /100) using zero-shot normalization against a perfect match.
- **Interactive Web UI**: Comes with a fully functional Streamlit application supporting both single-answer testing and massive batch CSV grading.

## 🚀 Quick Start (Web UI)

You do not need to download massive model weights manually. The application is integrated with the Hugging Face Hub and will automatically download the 1.74GB trained weights (`sridhanush1208/asag-deberta-large`) into your system cache upon first run.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sridhanush1208/Automatic_ShortAnswer_Grading.git
   cd Automatic_ShortAnswer_Grading
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Model Architecture & Training

The model was meticulously trained on a High-Performance Computing (HPC) cluster using `bfloat16` precision to eliminate gradient overflow instability. 

- **Base Model**: `microsoft/deberta-v3-large` (434M Parameters)
- **Loss Function**: Mean Squared Error (MSE)
- **Primary Metric**: Quadratic Weighted Kappa (QWK) — *Scores 0.747+*
- **Training Strategy**: 5 Epochs, Gradient Checkpointing, and a 500-step Learning Rate Warmup to ensure pristine convergence.

If you wish to retrain the model from scratch on your own GPU/HPC:
```bash
python src/train.py
```
*(All advanced HPC hyperparameter configurations and dataset unification logic are contained entirely within `src/train.py`).*

## 📂 Repository Structure
```text
Automatic_ShortAnswer_Grading/
├── src/
│   ├── data_loader.py       # Data parsing, cleaning, and unification
│   ├── models.py            # DeBERTa-v3 Cross-Encoder wrapper
│   ├── train.py             # Hugging Face Trainer loop
│   └── evaluate.py          # Standalone metric evaluation (MSE, QWK)
├── app.py                   # Streamlit Frontend Web Application
├── requirements.txt         # Python package dependencies
└── README.md                # You are here
```

## 📊 Evaluation Metrics
* **Quadratic Weighted Kappa (QWK):** The primary metric used to evaluate ASAG tasks. It measures the agreement between human graders and the AI, heavily penalizing predictions that are wildly inaccurate.
* **Mean Squared Error (MSE):** Utilized strictly as the continuous loss function during neural network backpropagation.

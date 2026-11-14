# Nexus-NMT: English ⇄ Amharic Neural Machine Translation

Nexus-NMT is a custom, from-scratch Transformer model designed for bidirectional translation between English and Amharic.

## Model Details
- **Architecture:** Standard Transformer (6 Encoder Layers, 6 Decoder Layers, 8 Attention Heads).
- **Parameters:** ~105 Million (Bidirectional).
- **Vocabulary:** 40,000 Byte-Level BPE (trained specifically on the 3.2M dataset).
- **Dataset:** ~2.9 Million bilingual sentence pairs (5.8 Million bidirectional training examples).
- **Framework:** PyTorch

##  Repository Structure
- `model.py`: The PyTorch architecture for the Nexus-NMT Transformer.
- `web_app.py`: The Streamlit web interface for live translation.
- `requirements.txt`: Dependencies required for deployment.
- `stable_best_model.pt`: The trained model weights (Hosted on Hugging Face).

## 🛠️ How to Use the App
The live web app is built with Streamlit. It downloads the tokenizer and model weights directly from Hugging Face.

1. Install the requirements:
   ```bash
   pip install -r requirements.txt

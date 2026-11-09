import streamlit as st
import torch
import os
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
from model import NexusNMT

st.set_page_config(page_title="Nexus-NMT", page_icon="🌍")

# Fetch token from Streamlit secrets
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
if not hf_token:
    st.warning("⚠️ HF_TOKEN not found in Streamlit Secrets.")

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Tree_with_roots.png/800px-Tree_with_roots.png"
try:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(IMAGE_URL, use_container_width=True)
except Exception:
    pass

st.title("Nexus-NMT")
st.write("A custom Transformer built from scratch for English to Amharic translation.")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    repo_id = "samuel-nexus/nexus-nmt-eng-amh" 
    
    try:
        # 1. Load Tokenizer
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", token=hf_token)
        merges_path = hf_hub_download(repo_id=repo_id, filename="merges.txt", token=hf_token)
        tokenizer = ByteLevelBPETokenizer(tokenizer_path, merges_path)
        
        # 2. Initialize Model
        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
        
        # 3. Load Weights (Epoch 8)
        weights_path = hf_hub_download(repo_id=repo_id, filename="new_model_epoch8_batch_10000.pt", token=hf_token) 
        checkpoint = torch.load(weights_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

with st.spinner("Loading Nexus-NMT model from Hugging Face..."):
    model, tokenizer, device = load_model()

def translate(text):
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    unk_id = tokenizer.token_to_id("<unk>") or 3
    pad_id = tokenizer.token_to_id("<pad>") or 1
    
    # Encode input
    src_ids = tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        
        for _ in range(50): # Max output length
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()
            
            # 1. Ban Special Tokens
            logits[pad_id] = -float('inf')
            logits[unk_id] = -float('inf')
            logits[bos_id] = -float('inf')
            
            # 2. Repetition Penalty
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id]:
                    logits[token] -= 15.0 

            # 3. 🚀 TOP-K SAMPLING (Fixes the infinite loop)
            # Instead of picking the #1 word, pick randomly from the Top 5
            top_k = 5
            top_logits, top_indices = torch.topk(logits, top_k)
            
            # Convert to probabilities
            probs = torch.softmax(top_logits, dim=-1)
            
            # Sample from the top 5 words
            next_token = top_indices[torch.multinomial(probs, 1)].item()
            
            # Stop if we hit end of sentence
            if next_token == eos_id:
                break
                
            tgt_ids.append(next_token)

    # Decode output
    clean_text = tokenizer.decode(tgt_ids)
    
    # Remove special tokens
    for tag in ["<s>", "</s>", "<pad>", "<unk>"]:
        clean_text = clean_text.replace(tag, "")
        
    # Fix ByteLevel spacing (Ġ)
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    return clean_text if clean_text else "[Model generated empty response]"

user_input = st.text_area("English Text", "The president spoke to the nation.")
if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            result = translate(user_input)
            st.success("Translation Complete!")
            st.markdown(f"### Amharic Translation:\n**{result}**")

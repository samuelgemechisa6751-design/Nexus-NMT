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
st.write("A 60M parameter Multilingual Transformer built from scratch.")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    repo_id = "samuel-nexus/Nexus-NMT-Model"
    
    try:
        # 1. Load Tokenizer
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", token=hf_token)
        merges_path = hf_hub_download(repo_id=repo_id, filename="merges.txt", token=hf_token)
        tokenizer = ByteLevelBPETokenizer(tokenizer_path, merges_path)
        
        # 2. Initialize Model
        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
        
        # 3. Load Weights
        weights_path = hf_hub_download(repo_id=repo_id, filename="my_trained_model_final.pt", token=hf_token) 
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
    pad_id = tokenizer.token_to_id("<pad>") or 1
    
    # Collect all special tokens to ban them from being generated
    special_tokens = [bos_id, eos_id, pad_id]
    for tag in ["<2am>", "<2en>", "<2om>"]:
        tag_id = tokenizer.token_to_id(tag)
        if tag_id is not None:
            special_tokens.append(tag_id)

    # Encode input
    src_ids = tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        for _ in range(20): # Max length
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
            
            # Forward pass
            src_emb = model.positional_encoding(model.src_embedding(src_tensor) * model.scale)
            enc_out = src_emb
            for layer in model.encoder_layers: 
                enc_out = layer(enc_out, src_mask)
            
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * model.scale)
            dec_out = tgt_emb
            for layer in model.decoder_layers: 
                dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_out)[0, -1, :].clone()
            
            # 1. Ban special tokens from being generated
            for sp_id in special_tokens:
                logits[sp_id] = -float('inf')

            # 2. Strong repetition penalty
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id, pad_id]:
                    logits[token] -= 15.0 
            
            next_token = logits.argmax().item()
            
            # 3. STOP IMMEDIATELY if the model tries to output a special token
            if next_token in special_tokens:
                break
                
            tgt_ids.append(next_token)

    # Decode output
    clean_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    
    # 4. Forcefully remove any leftover garbage strings
    for tag in ["<s>", "</s>", "<pad>", "<2am>", "<2en>", "<2om>"]:
        clean_text = clean_text.replace(tag, "")
        
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    return clean_text if clean_text else "[Model generated empty response]"

user_input = st.text_area("English Text", "Hello, how are you?")
if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            result = translate(user_input)
            st.success("Translation Complete!")
            st.markdown(f"### Amharic Translation:\n**{result}**")

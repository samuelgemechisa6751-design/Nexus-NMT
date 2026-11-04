import streamlit as st
import torch
import torch.nn as nn
import math
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
import os
from model import NexusNMT  # Import the model architecture from model.py

st.set_page_config(page_title="Nexus-NMT", page_icon="🌍")

# Secure Token
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Tree_with_roots.png/800px-Tree_with_roots.png"
try:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(IMAGE_URL, use_container_width=True)
except Exception:
    pass

st.title("Nexus-NMT")
st.write("A 200M parameter Multilingual Transformer built from scratch.")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    repo_id = "samuel-nexus/Nexus-NMT-Model"
    
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", token=hf_token)
    merges_path = hf_hub_download(repo_id=repo_id, filename="merges.txt", token=hf_token)
    tokenizer = ByteLevelBPETokenizer(tokenizer_path, merges_path)
    
    vocab_size = tokenizer.get_vocab_size()
    model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
    
    # 🚨 ENSURE THIS MATCHES YOUR LATEST UPLOADED FILE NAME 🚨
    weights_path = hf_hub_download(repo_id=repo_id, filename="nexus_batch_54000_8bit.pt", token=hf_token) 
    
    state_dict_8bit = torch.load(weights_path, map_location=device)
    state_dict = {}
    
    for k, v in state_dict_8bit.items():
        if k.endswith("_scale"):
            continue
        scale = state_dict_8bit.get(k + "_scale", 1.0)
        state_dict[k] = (v.to(torch.float16) * scale)
        
    model.load_state_dict(state_dict)
    model = model.half()
    model.eval()
    return model, tokenizer, device

try:
    model, tokenizer, device = load_model()
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")
    st.stop()

def translate(text):
    # Matches the training tag format
    src_text = f"<2am> {text}"
    
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    tag_id = tokenizer.token_to_id("<2am>")
    
    if bos_id is None: bos_id = 0
    if eos_id is None: eos_id = 2
    if pad_id is None: pad_id = 1

    # Note: We do not add bos_id here because <2am> acts as the start
    src_ids = [tag_id] + tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        for _ in range(64):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
            
            src_emb = model.positional_encoding(model.src_embedding(src_tensor) * model.scale)
            enc_out = src_emb
            for layer in model.encoder_layers: 
                enc_out = layer(enc_out, src_mask)
            
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * model.scale)
            dec_out = tgt_emb
            for layer in model.decoder_layers: 
                dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_out)[0, -1, :].clone()
            
            # ONLY ban special tokens
            logits[bos_id] = -float('inf')
            logits[pad_id] = -float('inf')

            # Soft repetition penalty
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id]:
                    logits[token] -= 2.0 
            
            next_token = logits.argmax().item()
            if next_token == eos_id:
                break
            tgt_ids.append(next_token)

    clean_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    return clean_text.strip() if clean_text.strip() else "[Model generated empty response]"

user_input = st.text_area("English Text", "Hello, how are you?")
if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            result = translate(user_input)
            st.success("Translation Complete!")
            st.markdown(f"### Amharic Translation:\n**{result}**")

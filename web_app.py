import streamlit as st
import torch
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from model import NexusNMT

# Set page config
st.set_page_config(page_title="Nexus-NMT (Bidirectional)", page_icon="🌍")

# Fetch token from Streamlit secrets
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
if not hf_token:
    st.warning("⚠️ HF_TOKEN not found in Streamlit Secrets.")

# Optional: Add a nice header image
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Tree_with_roots.png/800px-Tree_with_roots.png"
try:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:import streamlit as st
import torch
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from model import NexusNMT

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Nexus-NMT | AI Translator", page_icon="🌐", layout="centered")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .title-text {
        font-size: 3.5rem; font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0px;
    }
    .subtitle-text { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 2rem; }
    .output-box {
        background-color: #1E1E1E; border-left: 5px solid #FF4B2B;
        padding: 20px; border-radius: 10px; font-size: 1.5rem; font-weight: 500;
        color: #FFFFFF; margin-top: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white; border: none; padding: 10px 24px; font-weight: bold;
        border-radius: 8px; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(255, 75, 43, 0.4); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER
# ==========================================
st.markdown('<h1 class="title-text">🌐 Nexus-NMT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Custom 40k Bidirectional Transformer built from scratch</p>', unsafe_allow_html=True)

hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))
if not hf_token:
    st.warning("⚠️ HF_TOKEN not found in Streamlit Secrets.")

# ==========================================
# 3. MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device("cpu") 
    repo_id = "samuel-nexus/nexus-nmt-final-bidir" 
    
    try:
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
        
        # 🚀 UPDATED TO BATCH 75000
        weights_path = hf_hub_download(repo_id=repo_id, filename="final_bidir_epoch1_batch_75000.pt", token=hf_token) 
        
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

with st.spinner("Loading Bidirectional Neural Network..."):
    model, tokenizer, device = load_model()

# ==========================================
# 4. TRANSLATION LOGIC
# ==========================================
def translate(text, direction):
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    unk_id = tokenizer.token_to_id("<unk>") or 3
    pad_id = tokenizer.token_to_id("<pad>") or 1
    
    tag_2am = tokenizer.token_to_id("<2am>")
    tag_2en = tokenizer.token_to_id("<2en>")
    
    if "English" in direction:
        tag_id = tag_2am
    else:
        tag_id = tag_2en
    
    src_ids = [tag_id] + tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        
        for step in range(50): 
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()
            
            logits[pad_id] = -float('inf')
            logits[unk_id] = -float('inf')
            logits[bos_id] = -float('inf')
            logits[tag_2am] = -float('inf')
            logits[tag_2en] = -float('inf')
            
            if step < 3:
                logits[eos_id] = -float('inf')
            
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id]:
                    logits[token] -= 15.0 

            next_token = logits.argmax().item()
            
            if next_token == eos_id:
                break
                
            tgt_ids.append(next_token)

    clean_text = tokenizer.decode(tgt_ids)
    
    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")
        
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    if not clean_text:
        return f"[Empty Response]"
        
    return clean_text

# ==========================================
# 5. UI INTERACTION
# ==========================================
st.divider()

direction = st.radio(
    "Select Translation Direction:",
    ["🇺🇸 English ➔ 🇪🇹 Amharic", "🇪🇹 Amharic ➔ 🇺🇸 English"],
    horizontal=True,
    label_visibility="collapsed"
)

placeholder_text = "Type your English sentence here..." if "English" in direction else "Type your Amharic sentence here..."
default_val = "The president spoke to the nation." if "English" in direction else "ፕሬዝዳንቱ ለህዝቡ ተናገሩ።"

user_input = st.text_area("Input Text", value=default_val, height=120, placeholder=placeholder_text)

if st.button("Translate Now ✨"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Analyzing and translating..."):
            result = translate(user_input, direction)
            st.success("Translation Complete!")
            st.markdown(f'<div class="output-box">{result}</div>', unsafe_allow_html=True)

# ==========================================
# 6. FOOTER METRICS
# ==========================================
st.divider()
st.markdown("<p style='text-align: center; color: gray; font-size: 0.9rem;'>Model Architecture & Stats</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: st.metric(label="Architecture", value="60M Param")
with col2: st.metric(label="Vocabulary", value="40k BPE")
with col3: st.metric(label="Training Data", value="3.2M Pairs")
        st.image(IMAGE_URL, use_container_width=True)
except Exception:
    pass

st.title("Nexus-NMT")
st.write("A custom 40k Transformer built from scratch for bidirectional translation.")

@st.cache_resource
def load_model():
    device = torch.device("cpu") 
    repo_id = "samuel-nexus/nexus-nmt-final-bidir" 
    
    try:
        # 1. Load 40k Tokenizer
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 2. Initialize Model
        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
        
        # 🚀 3. Load Weights (UPDATED TO BATCH 55000)
        weights_path = hf_hub_download(repo_id=repo_id, filename="final_bidir_epoch1_batch_55000.pt", token=hf_token) 
        
        # 🚀 MEMORY FIX: Load with weights_only=True for Streamlit
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

with st.spinner("Loading 40k Bidirectional model..."):
    model, tokenizer, device = load_model()

def translate(text, direction):
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    unk_id = tokenizer.token_to_id("<unk>") or 3
    pad_id = tokenizer.token_to_id("<pad>") or 1
    
    tag_2am = tokenizer.token_to_id("<2am>")
    tag_2en = tokenizer.token_to_id("<2en>")
    
    if direction == "English ➔ Amharic":
        tag_id = tag_2am
    else:
        tag_id = tag_2en
    
    src_ids = [tag_id] + tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        
        for step in range(50): 
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()
            
            logits[pad_id] = -float('inf')
            logits[unk_id] = -float('inf')
            logits[bos_id] = -float('inf')
            logits[tag_2am] = -float('inf')
            logits[tag_2en] = -float('inf')
            
            # Force at least 3 words
            if step < 3:
                logits[eos_id] = -float('inf')
            
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id]:
                    logits[token] -= 15.0 # Increased penalty slightly for better sentences

            next_token = logits.argmax().item()
            
            if next_token == eos_id:
                break
                
            tgt_ids.append(next_token)

    clean_text = tokenizer.decode(tgt_ids)
    
    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")
        
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    if not clean_text:
        return f"[Empty Response. Raw IDs: {tgt_ids}]"
        
    return clean_text

# 🚀 UI UPDATES: Add direction selector
direction = st.radio("Select Translation Direction:", ["English ➔ Amharic", "Amharic ➔ English"])

default_text = "The president spoke to the nation." if direction == "English ➔ Amharic" else "ፕሬዝዳንቱ ለህዝቡ ተናገሩ።"
user_input = st.text_area(f"{direction.split(' ➔ ')[0]} Text", default_text)

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            result = translate(user_input, direction)
            st.success("Translation Complete!")
            st.markdown(f"### {direction.split(' ➔ ')[1]} Translation:\n**{result}**")

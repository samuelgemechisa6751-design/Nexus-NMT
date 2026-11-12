import streamlit as st
import torch
import torch.nn.functional as F
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
        
        # 🚀 LOADING BATCH 80000
        weights_path = hf_hub_download(repo_id=repo_id, filename="final_bidir_epoch1_batch_80000.pt", token=hf_token) 
        
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

with st.spinner("Loading 80k Bidirectional Neural Network..."):
    model, tokenizer, device = load_model()

# ==========================================
# 4. BEAM SEARCH TRANSLATION LOGIC
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

    # Beam Search Parameters
    beam_width = 3
    max_len = 50
    min_len = 3 # Force at least 3 words

    with torch.no_grad():
        # Beam contains tuples of (log_prob, token_list)
        beam = [(0.0, [bos_id])]
        
        for step in range(max_len):
            new_beam = []
            for log_prob, tgt_ids in beam:
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
                logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()
                
                # 1. Ban Special Tokens & Tags
                logits[pad_id] = -float('inf')
                logits[unk_id] = -float('inf')
                logits[bos_id] = -float('inf')
                logits[tag_2am] = -float('inf')
                logits[tag_2en] = -float('inf')
                
                # 2. Enforce Minimum Length
                if step < min_len:
                    logits[eos_id] = -float('inf')
                
                # 3. Repetition Penalty (Prevents loops)
                for token in set(tgt_ids):
                    if token not in [bos_id, eos_id]:
                        logits[token] -= 15.0 
                
                # 4. Get Top-K Probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    new_log_prob = log_prob + topk_probs[i].item()
                    new_ids = tgt_ids + [topk_ids[i].item()]
                    new_beam.append((new_log_prob, new_ids))
            
            # Sort by cumulative probability
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]
            
            # Stop if all beams end with EOS
            if all(b[1][-1] == eos_id for b in beam):
                break
                
    # Pick the sentence with the highest overall probability
    best_sentence_ids = beam[0][1]
    
    # Decode
    clean_text = tokenizer.decode(best_sentence_ids)
    
    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")
        
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    if not clean_text:
        return "[Empty Response]"
        
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

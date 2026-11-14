import streamlit as st
import torch
import torch.nn.functional as F
import os
import html
import re
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from model import NexusNMT

# ==========================================
# 1. PAGE CONFIG & DESIGN SYSTEM
# ==========================================
st.set_page_config(page_title="Nexus-NMT | English ⇄ Amharic", page_icon="🌐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Noto+Sans+Ethiopic:wght@400;500;600;700&display=swap');

:root {
    --ink: #0F2A2E;
    --muted: #6A7F84;
    --mist: #EAF1F2;
    --line: rgba(15,42,46,0.08);
    --grad-start: #00B4D8;
    --grad-mid: #0077B6;
    --grad-end: #023047;
}

html, body, [class*="css"], .stApp, .stMarkdown, textarea, button {
    font-family: 'Sora', 'Noto Sans Ethiopic', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(900px 520px at 8% -8%, rgba(0,180,216,0.1), transparent 60%),
        radial-gradient(760px 480px at 100% 0%, rgba(2,48,71,0.08), transparent 60%),
        var(--mist) !important;
    color: var(--ink);
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1300px; padding-top: 3rem; padding-bottom: 3rem; }

.brand { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 3rem; flex-direction: column; }
.brand-mark { width: 85px; height: 85px; border-radius: 28px; background: linear-gradient(135deg, var(--grad-start) 0%, var(--grad-mid) 50%, var(--grad-end) 100%) !important; box-shadow: 0 15px 40px -10px rgba(0, 119, 182, 0.4); position: relative; display: flex; align-items: center; justify-content: center; }
.brand-mark::after { content: ""; width: 34px; height: 34px; border-radius: 50%; background: #ffffff !important; }
.brand-name { font-weight: 800; font-size: 2.4rem; letter-spacing: -0.05em; color: #023047 !important; margin-top: 10px; }

.lang-pill { display: flex; align-items: center; justify-content: center; gap: 10px; height: 64px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; border: 3px solid transparent; background: linear-gradient(#ffffff, #ffffff) padding-box, linear-gradient(90deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) border-box !important; color: var(--ink); box-shadow: 0 12px 30px -10px rgba(123, 97, 255, 0.30); }
.lang-pill .dot { width: 14px; height: 14px; border-radius: 50%; background: #00B4D8 !important; }
.lang-pill.target .dot { background: #7B61FF !important; }

.st-key-swap button { width: 64px !important; height: 64px !important; min-height: 64px !important; padding: 0 !important; margin: 0 auto !important; display: flex !important; align-items: center !important; justify-content: center !important; border-radius: 50% !important; font-size: 1.6rem !important; color: #ffffff !important; background: linear-gradient(135deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) !important; border: 3px solid #ffffff !important; box-shadow: 0 15px 30px -5px rgba(123, 97, 255, 0.45) !important; }

.stTextArea label { display: none; }
div[data-testid="stTextArea"] textarea {
    background: linear-gradient(135deg, #DDF6FB 0%, #E4EBFF 50%, #F4E6FF 100%) padding-box, linear-gradient(135deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) border-box !important;
    border: 3px solid transparent !important; border-radius: 24px !important; box-shadow: 0 18px 40px -12px rgba(123, 97, 255, 0.35) !important;
    font-size: 1.2rem !important; color: #1B1F5E !important; padding: 26px !important;
}

.out-panel { min-height: 240px; padding: 28px; border-radius: 24px; background: linear-gradient(135deg, #00B4D8 0%, #4361EE 50%, #7B2FF7 100%) !important; border: 3px solid rgba(255, 255, 255, 0.35); color: #FFFFFF !important; box-shadow: 0 18px 40px -10px rgba(67, 97, 238, 0.55); font-size: 1.2rem; line-height: 1.7; font-weight: 500; }
.out-panel.empty { background: linear-gradient(135deg, #DDF6FB 0%, #E4EBFF 50%, #F4E6FF 100%) !important; color: #7C86C4 !important; border: 3px dashed #A99BFF !important; box-shadow: none !important; }

.stButton > button[kind="primary"] { height: 64px; border-radius: 32px !important; border: none !important; font-weight: 700; color: #ffffff !important; background: linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%) !important; box-shadow: 0 15px 30px -5px rgba(2, 48, 71, 0.3) !important; }
.panel-label { font-size: 0.9rem; color: var(--muted); margin: 0 0 10px 6px; font-weight: 600; text-transform: uppercase; }

.side-card { background: #ffffff !important; border: 1px solid var(--line); border-radius: 20px; padding: 24px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.03); }
.side-title { font-size: 1rem; font-weight: 800; color: var(--grad-start); margin-bottom: 16px; text-transform: uppercase; }
.side-text { font-size: 0.95rem; color: var(--muted); margin-bottom: 10px; display: flex; justify-content: space-between; }
.side-val { font-weight: 700; color: var(--ink); }

.st-key-clearbtn button, .st-key-clrside button, .st-key-clrhist button { background: linear-gradient(135deg, #FF9E6D 0%, #FF6FB5 55%, #7B61FF 100%) !important; color: #ffffff !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (CLEAN FILE)
# ==========================================
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    repo_id = "samuel-nexus/nexus-nmt-v1-final"

    try:
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
        tokenizer = Tokenizer.from_file(tokenizer_path)

        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)

        # 🚀 USING THE CLEAN 240MB FILE
        weights_path = hf_hub_download(repo_id=repo_id, filename="nexus_nmt_132k.pt", token=hf_token)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model, tokenizer, device

    except Exception as e:
        st.error(f"Could not load the model: {e}")
        st.stop()

with st.spinner("Loading the model..."):
    model, tokenizer, device = load_model()

# ==========================================
# 3. TRANSLATION LOGIC (FIXED BEAM SEARCH)
# ==========================================
def translate(text, to_amharic):
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    unk_id = tokenizer.token_to_id("<unk>")
    pad_id = tokenizer.token_to_id("<pad>")
    tag_2am = tokenizer.token_to_id("<2am>")
    tag_2en = tokenizer.token_to_id("<2en>")
    
    tag_id = tag_2am if to_amharic else tag_2en

    src_ids = [tag_id] + tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    beam_width = 4
    max_len = 50

    with torch.no_grad():
        beam = [(0.0, [bos_id])]
        
        for step in range(max_len):
            new_beam = []
            for log_prob, tgt_ids in beam:
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
                logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()

                # Ban only structural tokens
                logits[pad_id] = -float('inf')
                logits[unk_id] = -float('inf')
                logits[bos_id] = -float('inf')
                logits[tag_2am] = -float('inf')
                logits[tag_2en] = -float('inf')

                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    new_token = topk_ids[i].item()
                    new_score = log_prob + topk_probs[i].item()
                    new_tgt_ids = tgt_ids + [new_token]
                    
                    if new_token == eos_id:
                        # Length normalization
                        normalized_score = new_score / len(new_tgt_ids)
                        new_beam.append((normalized_score, new_tgt_ids))
                    else:
                        new_beam.append((new_score, new_tgt_ids))

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]

            if all(b[1][-1] == eos_id for b in beam):
                break

    best_ids = beam[0][1]
    clean_text = tokenizer.decode(best_ids)
    
    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")
    
    clean_text = clean_text.replace("Ġ", " ").strip()
    return clean_text if clean_text else "[Empty response]"

# ==========================================
# 4. STATE & LAYOUT
# ==========================================
if "src" not in st.session_state: st.session_state.src = "en"
if "text" not in st.session_state: st.session_state.text = "The president spoke to the nation."
if "result" not in st.session_state: st.session_state.result = None

def swap_languages():
    st.session_state.src = "am" if st.session_state.src == "en" else "en"
    st.session_state.text = ""
    st.session_state.result = None

col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown('<div class="brand"><div class="brand-mark"></div><div class="brand-name">Nexus-NMT</div></div>', unsafe_allow_html=True)
    
    src_is_en = st.session_state.src == "en"
    src_label = '<span class="dot"></span>English' if src_is_en else '<span class="dot"></span><span class="ge">አማርኛ</span>'
    tgt_label = '<span class="dot"></span><span class="ge">አማርኛ</span>' if src_is_en else '<span class="dot"></span>English'

    c1, c2, c3 = st.columns([4, 1.5, 4], vertical_alignment="center")
    with c1: st.markdown(f'<div class="lang-pill">{src_label}</div>', unsafe_allow_html=True)
    with c2:
        with st.container(key="swap"): st.button("⇄", on_click=swap_languages)
    with c3: st.markdown(f'<div class="lang-pill target">{tgt_label}</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<p class="panel-label">Source Text</p>', unsafe_allow_html=True)
        user_input = st.text_area("Input", key="text", height=240, label_visibility="collapsed")

    with right:
        st.markdown('<p class="panel-label">Translation</p>', unsafe_allow_html=True)
        out_slot = st.empty()
        if st.session_state.result:
            out_slot.markdown(f'<div class="out-panel">{html.escape(st.session_state.result)}</div>', unsafe_allow_html=True)
        else:
            out_slot.markdown('<div class="out-panel empty">Translation appears here...</div>', unsafe_allow_html=True)

    st.write("")
    _, mid, clr, _ = st.columns([0.9, 2, 1, 0.9])
    with mid:
        go = st.button("Translate", type="primary", use_container_width=True)
    
    if go and user_input.strip():
        with st.spinner("Translating..."):
            st.session_state.result = translate(user_input, to_amharic=src_is_en)
            st.rerun()

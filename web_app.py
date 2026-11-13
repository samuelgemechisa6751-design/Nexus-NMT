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
    --teal: #0E7C86;
    --mist: #EAF1F2;
    --line: rgba(15,42,46,0.10);
    
    /* The Exact Gradient Colors */
    --grad-start: #00E5FF; /* Cyan */
    --grad-end: #FF00A8;   /* Magenta/Pink */
}

/* ---------- Canvas ---------- */
html, body, [class*="css"], .stApp, .stMarkdown, textarea, button {
    font-family: 'Sora', 'Noto Sans Ethiopic', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(900px 520px at 8% -8%, rgba(14,124,134,0.28), transparent 60%),
        radial-gradient(760px 480px at 100% 0%, rgba(242,169,59,0.26), transparent 60%),
        radial-gradient(900px 600px at 50% 115%, rgba(14,124,134,0.16), transparent 60%),
        var(--mist);
    color: var(--ink);
    filter: invert(0); 
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }

/* ---------- Brand bar ---------- */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 12px;
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    box-shadow: 0 6px 18px rgba(255, 0, 168, 0.35);
    position: relative;
}
.brand-mark::after {
    content: ""; position: absolute; inset: 9px; border-radius: 50%;
    background: var(--mist);
}
.brand-name { font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em; color: var(--ink); }
.brand-tag {
    margin-left: auto; font-size: 0.82rem; color: var(--muted);
    padding: 6px 14px; border: 1px solid var(--line); border-radius: 999px;
    background: rgba(255,255,255,0.55); backdrop-filter: blur(8px);
}

/* ---------- Hero ---------- */
.hero-title {
    font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; line-height: 1.08;
    letter-spacing: -0.03em; color: var(--ink); margin: 0 0 0.9rem 0;
}
.hero-sub { font-size: 1.05rem; color: var(--muted); margin-bottom: 2rem; line-height: 1.6; }

/* ---------- Language bar ---------- */
.lang-pill {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    height: 52px; border-radius: 16px; font-weight: 600; font-size: 1rem;
    background: #ffffff; border: 1px solid var(--line);
    backdrop-filter: blur(14px); color: var(--ink);
    box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset, 0 10px 30px -12px rgba(15,42,46,0.18);
}
.lang-pill .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--grad-start); }
.lang-pill.target .dot { background: var(--grad-end); }
.lang-pill .ge { font-family: 'Noto Sans Ethiopic', sans-serif; font-size: 1.1rem; }

/* ---------- Swap Button (Exact Gradient) ---------- */
.st-key-swap .stButton > button {
    width: 52px; height: 52px; min-height: 52px; padding: 0; margin: 0 auto; display: block;
    border-radius: 50%; font-size: 1.25rem; font-weight: 600; color: #ffffff !important;
    background: linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%) !important;
    border: none !important;
    box-shadow: 0 10px 26px -10px rgba(255, 0, 168, 0.6) !important;
    transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .2s ease !important;
}
.st-key-swap .stButton > button:hover { 
    transform: rotate(180deg) scale(1.05) !important; 
    box-shadow: 0 12px 30px -8px rgba(255, 0, 168, 0.8) !important;
}

/* ---------- Input panel (Solid White) ---------- */
.stTextArea label { display: none; }
.stTextArea > div > div,
.stTextArea > div > div > div,
.stTextArea [data-baseweb="textarea"], 
.stTextArea [data-baseweb="base-input"] {
    background-color: #ffffff !important; 
    background-image: none !important;
    border: 1px solid rgba(0, 229, 255, 0.4) !important; 
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
}
.stTextArea textarea {
    background-color: #ffffff !important;
    font-size: 1.15rem !important; 
    line-height: 1.6 !important; 
    color: #000000 !important; 
    padding: 20px 22px !important; 
}
.stTextArea textarea::placeholder { color: #888888 !important; }

/* ---------- Output panel (Colored Sentence) ---------- */
.out-panel {
    min-height: 220px; padding: 22px 24px; border-radius: 16px; position: relative;
    background: linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%) !important;
    color: #ffffff !important; /* Makes the whole sentence white/colored */
    box-shadow: 0 20px 40px -15px rgba(255, 0, 168, 0.4);
    font-size: 1.25rem; line-height: 1.6; font-weight: 500; overflow-wrap: anywhere;
    border: 1px solid rgba(255,255,255,0.2);
}
.out-panel.empty { 
    background: #ffffff !important; 
    color: var(--muted) !important; 
    border: 1px dashed rgba(0, 0, 0, 0.1) !important;
    box-shadow: none !important; 
    font-weight: 400; 
    font-size: 1.15rem; 
}
.out-panel.fresh { animation: reveal .5s cubic-bezier(.2,.8,.2,1); }
@keyframes reveal { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.panel-label { font-size: 0.85rem; color: var(--muted); margin: 0 0 8px 4px; font-weight: 600; }

/* ---------- Translate Button (Exact Gradient) ---------- */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    height: 56px; 
    border-radius: 16px; 
    border: none; 
    font-weight: 500; 
    font-size: 1.1rem; 
    color: #ffffff !important;
    background: linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%) !important;
    box-shadow: 0 10px 25px -5px rgba(255, 0, 168, 0.4) !important;
    transition: all .2s ease !important;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px); 
    box-shadow: 0 15px 35px -5px rgba(255, 0, 168, 0.6) !important; 
    color: #ffffff !important;
}

/* ---------- Sidebar Cards ---------- */
.side-card {
    background: #ffffff; border: 1px solid var(--line); border-radius: 16px;
    padding: 16px; margin-bottom: 16px; backdrop-filter: blur(12px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.side-title { font-size: 0.9rem; font-weight: 700; color: var(--grad-end); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em;}
.side-text { font-size: 0.85rem; color: var(--muted); margin-bottom: 8px; }
.side-val { font-weight: 700; color: var(--ink); }

/* Clear Input Button Fix */
.stButton > button:not([kind="primary"]) {
    background-color: #ffffff !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--grad-end) !important;
    color: var(--grad-end) !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    repo_id = "samuel-nexus/nexus-nmt-final-bidir"

    try:
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
        tokenizer = Tokenizer.from_file(tokenizer_path)

        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)

        weights_path = hf_hub_download(repo_id=repo_id, filename="final_bidir_epoch1_batch_80000.pt", token=hf_token)
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
# 3. TRANSLATION LOGIC
# ==========================================
def translate(text, to_amharic):
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    unk_id = tokenizer.token_to_id("<unk>") or 3
    pad_id = tokenizer.token_to_id("<pad>") or 1
    tag_2am = tokenizer.token_to_id("<2am>")
    tag_2en = tokenizer.token_to_id("<2en>")
    tag_id = tag_2am if to_amharic else tag_2en

    src_ids = [tag_id] + tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    beam_width = 3
    max_len = 50
    min_len = 3

    with torch.no_grad():
        beam = [(0.0, [bos_id])]
        for step in range(max_len):
            new_beam = []
            for log_prob, tgt_ids in beam:
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
                logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()
                
                logits[pad_id] = -float('inf')
                logits[unk_id] = -float('inf')
                logits[bos_id] = -float('inf')
                logits[tag_2am] = -float('inf')
                logits[tag_2en] = -float('inf')
                
                if step < min_len: logits[eos_id] = -float('inf')
                for token in set(tgt_ids):
                    if token not in [bos_id, eos_id]: logits[token] -= 15.0

                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    new_beam.append((log_prob + topk_probs[i].item(), tgt_ids + [topk_ids[i].item()]))
            
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]
            if all(b[1][-1] == eos_id for b in beam): break

    clean_text = tokenizer.decode(beam[0][1])
    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")
    return clean_text.replace("Ġ", " ").strip() or "[Empty response]"

# ==========================================
# 4. STATE INITIALIZATION
# ==========================================
SAMPLES = {"en": "The president spoke to the nation.", "am": "ፕሬዝዳንቱ ለህዝቡ ተናገሩ።"}

if "src" not in st.session_state:
    st.session_state.src = "en"
if "text" not in st.session_state:
    st.session_state.text = SAMPLES["en"]
if "result" not in st.session_state:
    st.session_state.result = None
if "fresh" not in st.session_state:
    st.session_state.fresh = False
if "history" not in st.session_state:
    st.session_state.history = []

def swap_languages():
    previous = st.session_state.result
    st.session_state.src = "am" if st.session_state.src == "en" else "en"
    st.session_state.text = previous if previous else SAMPLES[st.session_state.src]
    st.session_state.result = None
    st.session_state.fresh = False

def clear_input():
    st.session_state.text = ""
    st.session_state.result = None
    st.session_state.fresh = False

# ==========================================
# 5. MAIN LAYOUT (WITH SIDEBARS)
# ==========================================
col_left, col_main, col_right = st.columns([1, 2.5, 1], gap="large")

# --- LEFT SIDEBAR (History) ---
with col_left:
    st.markdown('<div class="side-card"><div class="side-title">🕒 History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<div class="side-text">No history yet.</div>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history[-3:]):
            src_str = str(item.get("src", ""))[:20]
            tgt_str = str(item.get("tgt", ""))[:20]
            st.markdown(f'<div class="side-text"><b>{html.escape(src_str)}...</b><br>{html.escape(tgt_str)}...</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- CENTER (Your Exact UI) ---
with col_main:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-name">Nexus-NMT</div>
        <div class="brand-tag">Built from scratch, no pretrained model</div>
    </div>
    <h1 class="hero-title">Translate between English and Amharic</h1>
    <p class="hero-sub">A 60M-parameter Transformer trained on 3.2 million sentence pairs. Type a sentence, press translate.</p>
    """, unsafe_allow_html=True)

    src_is_en = st.session_state.src == "en"
    src_label = '<span class="dot"></span>English' if src_is_en else '<span class="dot"></span><span class="ge">አማርኛ</span>'
    tgt_label = '<span class="dot"></span><span class="ge">አማርኛ</span>' if src_is_en else '<span class="dot"></span>English'

    c1, c2, c3 = st.columns([5, 1.1, 5], vertical_alignment="center")
    with c1:
        st.markdown(f'<div class="lang-pill">{src_label}</div>', unsafe_allow_html=True)
    with c2:
        with st.container(key="swap"):
            st.button("⇄", on_click=swap_languages, help="Swap languages")
    with c3:
        st.markdown(f'<div class="lang-pill target">{tgt_label}</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<p class="panel-label">Your text</p>', unsafe_allow_html=True)
        user_input = st.text_area(
            "Input text",
            key="text",
            height=200,
            max_chars=500,
            placeholder="Type an English sentence..." if src_is_en else "አማርኛ ዓረፍተ ነገር እዚህ ይጻፉ...",
        )
        char_count = len(user_input)
        st.markdown(f'<div style="text-align: right; font-size: 0.8rem; color: var(--muted); margin-top: -10px;">{char_count} / 500</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<p class="panel-label">Translation</p>', unsafe_allow_html=True)
        out_slot = st.empty()

        def render_output():
            result = st.session_state.result
            if result:
                cls = "out-panel fresh" if st.session_state.fresh else "out-panel"
                # This wraps the whole translated sentence in the colored gradient class
                out_slot.markdown(f'<div class="{cls}">{html.escape(str(result))}</div>', unsafe_allow_html=True)
            else:
                out_slot.markdown('<div class="out-panel empty">Your translation will appear here.</div>', unsafe_allow_html=True)

        render_output()

    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        go = st.button("Translate", type="primary", use_container_width=True)

    if go:
        if not user_input.strip():
            st.warning("Enter some text to translate.")
        else:
            has_ethiopic = bool(re.search(r'[\u1200-\u137F]', user_input))
            if has_ethiopic and src_is_en:
                st.session_state.src = "am"
                src_is_en = False
            
            with st.spinner("Translating..."):
                st.session_state.result = translate(user_input, to_amharic=src_is_en)
                st.session_state.fresh = True
                st.session_state.history.append({"src": user_input, "tgt": st.session_state.result})
            
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

# --- RIGHT SIDEBAR (Tools & Info) ---
with col_right:
    st.markdown('<div class="side-card"><div class="side-title">🛠️ Tools</div>', unsafe_allow_html=True)
    
    if st.session_state.result:
        st.download_button(
            label="📥 Download TXT",
            data=str(st.session_state.result),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.button("🗑️ Clear Input", on_click=clear_input, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="side-card">
        <div class="side-title">⚡ AI Status</div>
        <div class="side-text">Model: <span class="side-val">Nexus-NMT</span></div>
        <div class="side-text">Engine: <span class="side-val">Active</span></div>
        <div class="side-text">Languages: <span class="side-val">2</span></div>
    </div>
    """, unsafe_allow_html=True)

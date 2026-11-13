import streamlit as st
import torch
import torch.nn.functional as F
import os
import html
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from model import NexusNMT

# ==========================================
# 1. PAGE CONFIG & ULTIMATE NEON DESIGN SYSTEM
# ==========================================
st.set_page_config(page_title="Nexus-NMT | AI Studio", page_icon="🌌", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+Ethiopic:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #06060c;
    --panel-bg: rgba(20, 20, 35, 0.6);
    --neon-cyan: #00F0FF;
    --neon-magenta: #FF00E5;
    --text-main: #FFFFFF;
    --text-muted: #8B8B9F;
    --border-color: rgba(255, 255, 255, 0.08);
}

/* ---------- Canvas & Background ---------- */
html, body, [class*="css"], .stApp, .stMarkdown, textarea, button {
    font-family: 'Outfit', 'Noto Sans Ethiopic', sans-serif !important;
}

.stApp {
    background: 
        radial-gradient(circle at 15% 50%, rgba(0, 240, 255, 0.08), transparent 40%),
        radial-gradient(circle at 85% 30%, rgba(255, 0, 229, 0.08), transparent 40%),
        var(--bg-dark);
    color: var(--text-main);
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 4rem; }

/* ---------- Brand bar ---------- */
.brand { display: flex; align-items: center; gap: 14px; margin-bottom: 3rem; }
.brand-mark {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-magenta));
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
    position: relative;
    animation: pulse-glow 3s infinite alternate;
}
@keyframes pulse-glow {
    0% { box-shadow: 0 0 15px rgba(0, 240, 255, 0.3); }
    100% { box-shadow: 0 0 30px rgba(255, 0, 229, 0.6); }
}
.brand-name { font-weight: 700; font-size: 1.3rem; letter-spacing: -0.02em; color: var(--text-main); }
.brand-tag {
    margin-left: auto; font-size: 0.85rem; color: var(--neon-cyan);
    padding: 8px 16px; border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 999px;
    background: rgba(0, 240, 255, 0.05); backdrop-filter: blur(8px);
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}

/* ---------- Hero ---------- */
.hero-title {
    font-size: clamp(2.5rem, 6vw, 4rem); font-weight: 800; line-height: 1.1;
    letter-spacing: -0.03em; margin: 0 0 1rem 0; max-width: 20ch;
    background: linear-gradient(to right, #ffffff, #a0a0c0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 1.15rem; color: var(--text-muted); max-width: 50ch; margin-bottom: 2.5rem; line-height: 1.6; }

/* ---------- Language bar ---------- */
.lang-pill {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    height: 56px; border-radius: 16px; font-weight: 600; font-size: 1.1rem;
    background: var(--panel-bg); border: 1px solid var(--border-color);
    backdrop-filter: blur(16px); color: var(--text-main);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.lang-pill .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--neon-cyan); box-shadow: 0 0 10px var(--neon-cyan); }
.lang-pill.target .dot { background: var(--neon-magenta); box-shadow: 0 0 10px var(--neon-magenta); }

/* swap button */
.st-key-swap .stButton > button {
    width: 56px; height: 56px; min-height: 56px; padding: 0; margin: 0 auto; display: block;
    border-radius: 50%; font-size: 1.4rem; font-weight: 600;
    background: var(--panel-bg); color: var(--neon-cyan); border: 1px solid rgba(0, 240, 255, 0.4);
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
    transition: all .4s cubic-bezier(.2,.8,.2,1);
}
.st-key-swap .stButton > button:hover { 
    transform: rotate(180deg) scale(1.1); 
    border-color: var(--neon-magenta); color: var(--neon-magenta);
    box-shadow: 0 0 25px rgba(255, 0, 229, 0.4);
}

/* ---------- Input panel (text area) ---------- */
.stTextArea label { display: none; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] {
    background: transparent !important; border: none !important;
}
.stTextArea > div > div {
    background: var(--panel-bg); border: 1px solid var(--border-color) !important; border-radius: 20px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    transition: all .3s ease;
}
.stTextArea > div > div:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 240, 255, 0.15), 0 10px 30px -10px rgba(0,0,0,0.5);
}
.stTextArea textarea {
    font-size: 1.35rem !important; line-height: 1.6 !important; color: var(--text-main) !important;
    padding: 24px !important; background: transparent !important;
}
.stTextArea textarea::placeholder { color: #5A5A70 !important; }

/* ---------- Output panel ---------- */
.out-panel {
    min-height: 220px; padding: 24px; border-radius: 20px; position: relative;
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.1) 0%, rgba(255, 0, 229, 0.1) 100%);
    border: 1px solid rgba(255, 0, 229, 0.3);
    color: var(--text-main); box-shadow: 0 20px 40px -15px rgba(255, 0, 229, 0.3);
    font-size: 1.5rem; line-height: 1.6; font-weight: 500; overflow-wrap: anywhere;
}
.out-panel.empty { background: var(--panel-bg); color: #5A5A70; border: 1px dashed var(--border-color);
    box-shadow: none; font-weight: 400; font-size: 1.15rem; }
.panel-label { font-size: 0.9rem; color: var(--text-muted); margin: 0 0 12px 8px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;}

/* ---------- Translate button ---------- */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    height: 60px; border-radius: 16px; border: none; font-weight: 700; font-size: 1.2rem; color: #000;
    background: linear-gradient(90deg, var(--neon-cyan) 0%, #00B4D8 100%);
    box-shadow: 0 10px 30px -10px rgba(0, 240, 255, 0.6);
    transition: all .3s ease;
    letter-spacing: 1px;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-3px); 
    box-shadow: 0 15px 40px -10px rgba(0, 240, 255, 0.8);
    background: linear-gradient(90deg, var(--neon-magenta) 0%, #FF00AA 100%);
    color: #fff;
}

/* ---------- Footer stats ---------- */
.stats { display: flex; justify-content: center; gap: 40px; margin-top: 4rem; }
.stat { text-align: center; }
.stat b { display: block; font-size: 1.8rem; font-weight: 800; color: var(--neon-cyan); text-shadow: 0 0 15px rgba(0, 240, 255, 0.4); }
.stat span { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
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

        # 🚀 UPDATE THIS TO YOUR LATEST CHECKPOINT (110000)
        weights_path = hf_hub_download(repo_id=repo_id, filename="final_bidir_epoch1_batch_110000.pt", token=hf_token)
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

# ==========================================
# 3. BRAND + HERO
# ==========================================
st.markdown("""
<div class="brand">
    <div class="brand-mark"></div>
    <div class="brand-name">Nexus-NMT</div>
    <div class="brand-tag">Next-Gen Translation</div>
</div>
<h1 class="hero-title">Break the language barrier.</h1>
<p class="hero-sub">A custom 60M-parameter Transformer trained on 3.2 million sentence pairs. Accurate, bidirectional, and built entirely from scratch.</p>
""", unsafe_allow_html=True)

with st.spinner("Initializing Neural Network..."):
    model, tokenizer, device = load_model()

# ==========================================
# 4. TRANSLATION LOGIC
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

                if step < min_len:
                    logits[eos_id] = -float('inf')

                for token in set(tgt_ids):
                    if token not in [bos_id, eos_id]:
                        logits[token] -= 15.0

                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    new_log_prob = log_prob + topk_probs[i].item()
                    new_ids = tgt_ids + [topk_ids[i].item()]
                    new_beam.append((new_log_prob, new_ids))

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]

            if all(b[1][-1] == eos_id for b in beam):
                break

    best_sentence_ids = beam[0][1]
    clean_text = tokenizer.decode(best_sentence_ids)

    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")

    clean_text = clean_text.replace("Ġ", " ").strip()
    return clean_text or "[Empty response]"

# ==========================================
# 5. UI STATE + INTERACTION
# ==========================================
SAMPLES = {
    "en": "The president spoke to the nation.",
    "am": "ፕሬዝዳንቱ ለህዝቡ ተናገሩ።",
}

if "src" not in st.session_state:
    st.session_state.src = "en"
    st.session_state.text = SAMPLES["en"]
    st.session_state.result = None
    st.session_state.fresh = False

def swap_languages():
    previous = st.session_state.result
    st.session_state.src = "am" if st.session_state.src == "en" else "en"
    st.session_state.text = previous if previous else SAMPLES[st.session_state.src]
    st.session_state.result = None
    st.session_state.fresh = False

src_is_en = st.session_state.src == "en"
src_label = '<span class="dot"></span> English' if src_is_en else '<span class="dot"></span> አማርኛ (Amharic)'
tgt_label = '<span class="dot"></span> አማርኛ (Amharic)' if src_is_en else '<span class="dot"></span> English'

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
    st.markdown('<p class="panel-label">Source Text</p>', unsafe_allow_html=True)
    user_input = st.text_area(
        "Input text",
        key="text",
        height=240,
        max_chars=500,
        placeholder="Type an English sentence..." if src_is_en else "አማርኛ ዓረፍተ ነገር እዚህ ይጻፉ...",
    )

with right:
    st.markdown('<p class="panel-label">Translation</p>', unsafe_allow_html=True)
    out_slot = st.empty()

def render_output():
    result = st.session_state.result
    if result:
        cls = "out-panel fresh" if st.session_state.fresh else "out-panel"
        out_slot.markdown(f'<div class="{cls}">{html.escape(result)}</div>', unsafe_allow_html=True)
    else:
        out_slot.markdown('<div class="out-panel empty">Translation will appear here.</div>', unsafe_allow_html=True)

render_output()

st.write("")
_, mid, _ = st.columns([1, 2, 1])
with mid:
    go = st.button("TRANSLATE NOW ✨", type="primary", use_container_width=True)

if go:
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Processing..."):
            st.session_state.result = translate(user_input, to_amharic=src_is_en)
            st.session_state.fresh = True
        render_output()

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("""
<div class="stats">
    <div class="stat"><b>60M</b><span>Parameters</span></div>
    <div class="stat"><b>40k</b><span>Vocabulary</span></div>
    <div class="stat"><b>3.2M</b><span>Training Pairs</span></div>
</div>
""", unsafe_allow_html=True)

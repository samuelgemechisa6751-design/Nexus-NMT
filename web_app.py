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
    --teal-deep: #0A5E66;
    --saffron: #F2A93B;
    --mist: #EAF1F2;
    --glass: rgba(255,255,255,0.72);
    --line: rgba(15,42,46,0.10);
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
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
/* Increased max-width to fit sidebars */
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }

/* ---------- Brand bar ---------- */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 2.6rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 12px;
    background: conic-gradient(from 210deg, var(--teal), var(--saffron), var(--teal));
    box-shadow: 0 6px 18px rgba(14,124,134,0.35);
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
    font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 800; line-height: 1.08;
    letter-spacing: -0.03em; color: var(--ink); margin: 0 0 0.9rem 0; max-width: 16ch;
}
.hero-sub { font-size: 1.05rem; color: var(--muted); max-width: 46ch; margin-bottom: 2.2rem; line-height: 1.6; }

/* ---------- Language bar ---------- */
.lang-pill {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    height: 52px; border-radius: 16px; font-weight: 600; font-size: 1rem;
    background: var(--glass); border: 1px solid var(--line);
    backdrop-filter: blur(14px); color: var(--ink);
    box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset, 0 10px 30px -12px rgba(15,42,46,0.18);
}
.lang-pill .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--teal); }
.lang-pill.target .dot { background: var(--saffron); }
.lang-pill .ge { font-family: 'Noto Sans Ethiopic', sans-serif; font-size: 1.1rem; }

/* swap button */
.st-key-swap .stButton > button {
    width: 52px; height: 52px; min-height: 52px; padding: 0; margin: 0 auto; display: block;
    border-radius: 50%; font-size: 1.25rem; font-weight: 600;
    background: #fff; color: var(--teal); border: 1px solid var(--line);
    box-shadow: 0 10px 26px -10px rgba(14,124,134,0.5);
    transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .2s ease;
}
.st-key-swap .stButton > button:hover { transform: rotate(180deg); border-color: var(--teal); color: var(--teal-deep); }

/* ---------- Input panel (text area) ---------- */
.stTextArea label { display: none; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] {
    background: transparent !important; border: none !important;
}
.stTextArea > div > div {
    background: var(--glass); border: 1px solid var(--line) !important; border-radius: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset, 0 24px 50px -24px rgba(15,42,46,0.28);
    transition: border-color .2s ease, box-shadow .2s ease;
}
.stTextArea > div > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 4px rgba(14,124,134,0.14), 0 24px 50px -24px rgba(15,42,46,0.28);
}
.stTextArea textarea {
    font-size: 1.35rem !important; line-height: 1.6 !important; color: var(--ink) !important;
    padding: 22px 24px !important; background: transparent !important;
}

/* ---------- Output panel ---------- */
.out-panel {
    min-height: 220px; padding: 22px 24px; border-radius: 24px; position: relative;
    background: linear-gradient(160deg, var(--teal-deep) 0%, var(--teal) 100%);
    color: #fff; box-shadow: 0 30px 60px -26px rgba(10,94,102,0.75);
    font-size: 1.45rem; line-height: 1.6; font-weight: 500; overflow-wrap: anywhere;
}
.out-panel::before {
    content: ""; position: absolute; right: -50px; top: -50px; width: 180px; height: 180px; border-radius: 50%;
    background: radial-gradient(circle, rgba(242,169,59,0.55), transparent 70%);
    pointer-events: none;
}
.out-panel.empty { background: var(--glass); color: #9FB0B4; border: 1px dashed rgba(15,42,46,0.22);
    box-shadow: none; font-weight: 400; font-size: 1.15rem; }
.out-panel.empty::before { display: none; }
.out-panel.fresh { animation: reveal .5s cubic-bezier(.2,.8,.2,1); }
@keyframes reveal { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.panel-label { font-size: 0.82rem; color: var(--muted); margin: 0 0 8px 6px; }

/* ---------- Translate button ---------- */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    height: 56px; border-radius: 16px; border: none; font-weight: 700; font-size: 1.05rem; color: #fff;
    background: linear-gradient(120deg, var(--teal) 0%, var(--teal-deep) 100%);
    box-shadow: 0 16px 34px -14px rgba(14,124,134,0.8);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px); box-shadow: 0 22px 40px -14px rgba(14,124,134,0.9); color: #fff;
}

/* ---------- Footer stats ---------- */
.stats { display: flex; justify-content: center; gap: 0; margin-top: 3rem; flex-wrap: wrap; }
.stat { padding: 4px 34px; text-align: center; border-right: 1px solid var(--line); }
.stat:last-child { border-right: none; }
.stat b { display: block; font-size: 1.5rem; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
.stat span { font-size: 0.82rem; color: var(--muted); }

/* ---------- New Sidebar Cards ---------- */
.side-card {
    background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
    padding: 16px; margin-bottom: 16px; backdrop-filter: blur(12px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.side-title { font-size: 0.9rem; font-weight: 700; color: var(--teal); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em;}
.side-text { font-size: 0.85rem; color: var(--muted); margin-bottom: 8px; }
.side-val { font-weight: 700; color: var(--ink); }
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
# 4. UI STATE
# ==========================================
SAMPLES = {"en": "The president spoke to the nation.", "am": "ፕሬዝዳንቱ ለህዝቡ ተናገሩ።"}

if "src" not in st.session_state:
    st.session_state.src = "en"
    st.session_state.text = SAMPLES["en"]
    st.session_state.result = None
    st.session_state.fresh = False
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
# 5. MAIN LAYOUT WITH SIDEBARS
# ==========================================
col_left, col_main, col_right = st.columns([1, 3, 1], gap="large")

# --- LEFT SIDEBAR (History) ---
with col_left:
    st.markdown('<div class="side-card"><div class="side-title">🕒 History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown('<div class="side-text">No history yet.</div>', unsafe_allow_html=True)
    else:
        for i, item in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f'<div class="side-text"><b>{item["src"][:20]}...</b><br>{item["tgt"][:20]}...</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- CENTER (Your Exact Original UI) ---
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
        # Character Counter Feature
        char_count = len(user_input)
        st.markdown(f'<div style="text-align: right; font-size: 0.8rem; color: var(--muted); margin-top: -10px;">{char_count} / 500</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<p class="panel-label">Translation</p>', unsafe_allow_html=True)
        out_slot = st.empty()

        def render_output():
            result = st.session_state.result
            if result:
                cls = "out-panel fresh" if st.session_state.fresh else "out-panel"
                out_slot.markdown(f'<div class="{cls}">{html.escape(result)}</div>', unsafe_allow_html=True)
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
            # Auto-detect Language Feature
            has_ethiopic = bool(re.search(r'[\u1200-\u137F]', user_input))
            if has_ethiopic and src_is_en:
                st.session_state.src = "am"
                src_is_en = False
            
            with st.spinner("Translating..."):
                st.session_state.result = translate(user_input, to_amharic=src_is_en)
                st.session_state.fresh = True
                # Add to history
                st.session_state.history.append({"src": user_input, "tgt": st.session_state.result})
            st.rerun()

    # FOOTER
    st.markdown("""
    <div class="stats">
        <div class="stat"><b>60M</b><span>parameters</span></div>
        <div class="stat"><b>40k</b><span>BPE vocabulary</span></div>
        <div class="stat"><b>3.2M</b><span>training pairs</span></div>
    </div>
    """, unsafe_allow_html=True)

# --- RIGHT SIDEBAR (Tools & Info) ---
with col_right:
    st.markdown('<div class="side-card"><div class="side-title">🛠️ Tools</div>', unsafe_allow_html=True)
    
    # Download Feature
    if st.session_state.result:
        st.download_button(
            label="📥 Download TXT",
            data=st.session_state.result,
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Clear Feature
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

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

    /* NEW PALETTE: Midnight Emerald */
    --grad-start: #00B4D8; /* Bright Teal */
    --grad-mid: #0077B6;   /* Ocean Blue */
    --grad-end: #023047;   /* Deep Navy/Emerald */
}

/* ---------- Canvas ---------- */
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

/* ---------- Modern Brand Logo & Name ---------- */
.brand {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-bottom: 3rem;
    flex-direction: column;
}
.brand-mark {
    width: 85px;
    height: 85px;
    border-radius: 28px;
    background: linear-gradient(135deg, var(--grad-start) 0%, var(--grad-mid) 50%, var(--grad-end) 100%) !important;
    box-shadow: 0 15px 40px -10px rgba(0, 119, 182, 0.4),
                0 0 20px rgba(0, 180, 216, 0.2) !important;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.brand-mark::after {
    content: "";
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #ffffff !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.brand-name {
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.05em;
    color: #023047 !important;
    margin-top: 10px;
    text-align: center;
}

/* ---------- Language bar ---------- */
.lang-pill {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    height: 64px; border-radius: 20px; font-weight: 700; font-size: 1.1rem;
    /* white fill + neutral teal -> violet -> pink gradient border */
    border: 3px solid transparent;
    background:
        linear-gradient(#ffffff, #ffffff) padding-box,
        linear-gradient(90deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) border-box !important;
    color: var(--ink);
    box-shadow: 0 12px 30px -10px rgba(123, 97, 255, 0.30);
}
.lang-pill .dot { width: 14px; height: 14px; border-radius: 50%; background: #00B4D8 !important; box-shadow: 0 0 0 4px rgba(0, 180, 216, 0.18); }
.lang-pill.target .dot { background: #7B61FF !important; box-shadow: 0 0 0 4px rgba(123, 97, 255, 0.18); }
.lang-pill .ge { font-family: 'Noto Sans Ethiopic', sans-serif; font-size: 1.2rem; }

/* ---------- Swap Button ---------- */
.st-key-swap button {
    width: 64px !important;
    height: 64px !important;
    min-height: 64px !important;
    padding: 0 !important;
    margin: 0 auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 50% !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 15px 30px -5px rgba(123, 97, 255, 0.45) !important;
    transition: transform .3s ease, box-shadow .3s ease !important;
}
.st-key-swap button * {
    color: #ffffff !important;
    background-color: transparent !important;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.65);
}
.st-key-swap button:hover {
    transform: rotate(180deg) scale(1.1) !important;
    box-shadow: 0 20px 40px -5px rgba(255, 111, 181, 0.55) !important;
}

/* ---------- Modern Input Panel (NEW COLORS) ---------- */
.stTextArea label { display: none; }

/* wrapper layers are transparent (no dark layer), the textarea itself carries all the color */
div[data-testid="stTextArea"],
div[data-testid="stTextArea"] div,
div[data-testid="stTextArea"] [data-baseweb="textarea"],
div[data-testid="stTextArea"] [data-baseweb="base-input"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stTextArea"] textarea {
    /* soft rainbow fill + teal -> violet -> pink gradient border */
    background:
        linear-gradient(135deg, #DDF6FB 0%, #E4EBFF 50%, #F4E6FF 100%) padding-box,
        linear-gradient(135deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) border-box !important;
    border: 3px solid transparent !important;
    border-radius: 24px !important;
    box-shadow: 0 18px 40px -12px rgba(123, 97, 255, 0.35) !important;
    transition: box-shadow .3s ease, transform .3s ease;
    font-size: 1.2rem !important;
    line-height: 1.7 !important;
    color: #1B1F5E !important;
    -webkit-text-fill-color: #1B1F5E !important;
    caret-color: #7B61FF !important;
    padding: 26px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    box-shadow: 0 0 0 5px rgba(123, 97, 255, 0.18),
                0 20px 45px -10px rgba(0, 180, 216, 0.45) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #8A93C9 !important;
    -webkit-text-fill-color: #8A93C9 !important;
}

/* ---------- Modern Output Panel (NEW COLORS) ---------- */
.out-panel {
    min-height: 240px;
    padding: 28px;
    border-radius: 24px;
    position: relative;
    background: linear-gradient(135deg, #00B4D8 0%, #4361EE 50%, #7B2FF7 100%) !important;
    border: 3px solid rgba(255, 255, 255, 0.35);
    color: #FFFFFF !important;
    box-shadow: 0 18px 40px -10px rgba(67, 97, 238, 0.55);
    font-size: 1.2rem;
    line-height: 1.7;
    font-weight: 500;
    overflow-wrap: anywhere;
}
.out-panel.empty {
    background: linear-gradient(135deg, #DDF6FB 0%, #E4EBFF 50%, #F4E6FF 100%) !important;
    color: #7C86C4 !important;
    border: 3px dashed #A99BFF !important;
    box-shadow: 0 18px 40px -12px rgba(123, 97, 255, 0.25) !important;
    font-weight: 400;
}
.out-panel.fresh { animation: reveal .4s cubic-bezier(.2,.8,.2,1); }
@keyframes reveal { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.panel-label { font-size: 0.9rem; color: var(--muted); margin: 0 0 10px 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ---------- Translate Button ---------- */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    height: 64px;
    border-radius: 32px !important;
    border: none !important;
    font-weight: 700;
    font-size: 1.2rem;
    color: #ffffff !important;
    background: linear-gradient(90deg, var(--grad-start) 0%, var(--grad-end) 100%) !important;
    box-shadow: 0 15px 30px -5px rgba(2, 48, 71, 0.3) !important;
    transition: all .3s ease !important;
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 20px 40px -5px rgba(2, 48, 71, 0.5) !important;
}

/* ---------- Sidebar Cards ---------- */
.side-card {
    background: #ffffff !important; border: 1px solid var(--line); border-radius: 20px;
    padding: 24px; margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.03);
}
.side-title { font-size: 1rem; font-weight: 800; color: var(--grad-start); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; display:flex; align-items:center; gap:8px;}
.side-text { font-size: 0.95rem; color: var(--muted); margin-bottom: 10px; display:flex; justify-content:space-between;}
.side-val { font-weight: 700; color: var(--ink); }

/* Clear Input Button */
.stButton > button:not([kind="primary"]) {
    background-color: #ffffff !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    height: 48px;
    font-weight: 600;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--grad-start) !important;
    color: var(--grad-start) !important;
}

/* ---------- Colorful Translation History ---------- */
.hist-title {
    display: inline-block;
    margin: 2.5rem 0 1rem 6px;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
}
.hist-card {
    padding: 16px 22px;
    border-radius: 20px;
    border: 3px solid transparent;
    background:
        linear-gradient(135deg, #DDF6FB 0%, #E4EBFF 50%, #F4E6FF 100%) padding-box,
        linear-gradient(135deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) border-box;
    box-shadow: 0 12px 28px -12px rgba(123, 97, 255, 0.40);
    margin-bottom: 8px;
}
.hist-src { font-size: 0.95rem; color: #6A74B8; margin-bottom: 6px; overflow-wrap: anywhere; }
.hist-tgt { font-size: 1.1rem; font-weight: 700; color: #1B1F5E; overflow-wrap: anywhere; }
.hist-empty {
    text-align: center;
    padding: 28px;
    color: #7C86C4;
    font-weight: 500;
    background: linear-gradient(135deg, #DDF6FB 0%, #E4EBFF 50%, #F4E6FF 100%) !important;
    border: 3px dashed #A99BFF !important;
}
div[class*="st-key-hist_"] .stButton > button:not([kind="primary"]) {
    width: 56px !important;
    height: 56px !important;
    min-height: 56px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: none !important;
    font-size: 1.4rem !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #00B4D8 0%, #7B61FF 55%, #FF6FB5 100%) !important;
    box-shadow: 0 10px 22px -6px rgba(123, 97, 255, 0.55) !important;
    transition: transform .3s ease !important;
}
div[class*="st-key-hist_"] .stButton > button:not([kind="primary"]):hover {
    transform: rotate(-180deg) scale(1.1) !important;
    color: #ffffff !important;
}

/* ---------- Colorful Clear Buttons ---------- */
.st-key-clearbtn .stButton > button:not([kind="primary"]),
.st-key-clrside .stButton > button:not([kind="primary"]),
.st-key-clrhist .stButton > button:not([kind="primary"]) {
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    background: linear-gradient(135deg, #FF9E6D 0%, #FF6FB5 55%, #7B61FF 100%) !important;
    box-shadow: 0 14px 28px -8px rgba(255, 111, 181, 0.55) !important;
    transition: all .3s ease !important;
}
.st-key-clearbtn button *, .st-key-clrside button *, .st-key-clrhist button * {
    color: #ffffff !important;
}
.st-key-clearbtn .stButton > button:not([kind="primary"]) {
    height: 64px !important;
    border-radius: 32px !important;
    font-size: 1.1rem !important;
}
.st-key-clearbtn .stButton > button:not([kind="primary"]):hover,
.st-key-clrside .stButton > button:not([kind="primary"]):hover,
.st-key-clrhist .stButton > button:not([kind="primary"]):hover {
    transform: translateY(-3px) scale(1.03);
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 20px 36px -8px rgba(255, 111, 181, 0.7) !important;
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
    repo_id = "samuel-nexus/nexus-nmt-v1-final"

    try:
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
        tokenizer = Tokenizer.from_file(tokenizer_path)

        vocab_size = tokenizer.get_vocab_size()
        model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)

        # 🚀 FIXED: Using the exact filename you uploaded
        weights_path = hf_hub_download(repo_id=repo_id, filename="new_model_epoch1_batch_132000.pt", token=hf_token)
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

def load_history(i):
    item = st.session_state.history[i]
    st.session_state.src = "am" if re.search(r'[\u1200-\u137F]', item["src"]) else "en"
    st.session_state.text = item["src"]
    st.session_state.result = item["tgt"]
    st.session_state.fresh = False

def clear_history():
    st.session_state.history = []

# Common Afaan Oromoo words (Latin script) used to notice Oromo text automatically
OROMO_WORDS = {"fi", "kan", "akka", "irra", "ni", "hin", "dha", "ta'e", "naannoo", "oromoo", "afaan",
               "bakka", "waan", "yoo", "immoo", "isaa", "isaan", "kana", "sana", "nu", "ani", "ati",
               "inni", "isheen", "galatoomaa", "akkam", "nagaa", "maaloo", "hojii", "mana", "bishaan",
               "nyaata", "guyyaa", "barnoota", "magaalaa", "biyya", "jira", "jiru", "dhaa"}

def looks_like_oromo(text):
    if not text or re.search(r'[\u1200-\u137F]', text):
        return False
    words = re.findall(r"[a-z']+", text.lower())
    hits = sum(1 for w in words if w in OROMO_WORDS)
    ratio = hits / max(1, len(words))
    return (hits >= 2 and ratio >= 0.25) or (hits >= 1 and ratio >= 0.5 and len(words) <= 3)

# ==========================================
# 5. MAIN LAYOUT
# ==========================================
# Sidebar for tools and status
with st.sidebar:
    st.markdown("""
    <div class="side-card">
        <div class="side-title">⚡️ AI Status</div>
        <div class="side-text">Model <span class="side-val">Nexus-NMT</span></div>
        <div class="side-text">Engine <span class="side-val" style="color: #00B4D8;">Active</span></div>
        <div class="side-text">Languages <span class="side-val">2</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    st.markdown('<div class="side-title">🛠 Tools</div>', unsafe_allow_html=True)
    if st.session_state.result:
        st.download_button(
            label="📥 Download TXT",
            data=str(st.session_state.result),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )
    with st.container(key="clrside"):
        st.button("🗑 Clear Input", on_click=clear_input, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Main centered layout
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    # Centered Brand (New Emerald Style)
    st.markdown("""
    <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-name">Nexus-NMT</div>
    </div>
    """, unsafe_allow_html=True)

    src_is_en = st.session_state.src == "en"
    is_om = looks_like_oromo(st.session_state.get("text", ""))
    if is_om:
        src_label = '<span class="dot"></span>Afaan Oromoo'
    elif src_is_en:
        src_label = '<span class="dot"></span>English'
    else:
        src_label = '<span class="dot"></span><span class="ge">አማርኛ</span>'
    tgt_label = '<span class="dot"></span><span class="ge">አማርኛ</span>' if src_is_en else '<span class="dot"></span>English'

    # Language selector row
    c1, c2, c3 = st.columns([4, 1.5, 4], vertical_alignment="center")
    with c1:
        st.markdown(f'<div class="lang-pill">{src_label}</div>', unsafe_allow_html=True)
    with c2:
        with st.container(key="swap"):
            st.button("⇄", on_click=swap_languages, help="Swap languages")
    with c3:
        st.markdown(f'<div class="lang-pill target">{tgt_label}</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Text areas row
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<p class="panel-label">Source Text</p>', unsafe_allow_html=True)
        user_input = st.text_area(
            "Input text",
            key="text",
            height=240,
            max_chars=500,
            placeholder="Type your text here..." if src_is_en else "እዚህ ይጻፉ...",
            label_visibility="collapsed"
        )

    with right:
        st.markdown('<p class="panel-label">Translation</p>', unsafe_allow_html=True)
        out_slot = st.empty()

        def render_output():
            result = st.session_state.result
            if result:
                cls = "out-panel fresh" if st.session_state.fresh else "out-panel"
                out_slot.markdown(f'<div class="{cls}">{html.escape(str(result))}</div>', unsafe_allow_html=True)
            else:
                out_slot.markdown('<div class="out-panel empty">Translation appears here...</div>', unsafe_allow_html=True)

        render_output()

    st.write("")
    st.write("")

    # Button centered
    _, mid, clr, _ = st.columns([0.9, 2, 1, 0.9])
    with mid:
        go = st.button("Translate", type="primary", use_container_width=True)
    with clr:
        with st.container(key="clearbtn"):
            st.button("🗑 Clear", key="main_clear", on_click=clear_input, use_container_width=True)

    if go:
        if not user_input.strip():
            st.warning("Enter some text to translate.")
        elif looks_like_oromo(user_input):
            st.info("Afaan Oromoo was detected. This model translates English ⇄ Amharic only, so Oromo translation is not available yet.")
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

    # ---------- Colorful Translation History (always visible) ----------
    st.markdown('<div class="hist-title">🕘 Recent Translations</div>', unsafe_allow_html=True)
    hist = st.session_state.history
    if not hist:
        st.markdown('<div class="hist-card hist-empty">✨ Your translations will appear here. Translate something to get started!</div>', unsafe_allow_html=True)
    else:
        for i in reversed(range(max(0, len(hist) - 5), len(hist))):
            h_left, h_right = st.columns([9, 1], vertical_alignment="center")
            with h_left:
                st.markdown(
                    f'<div class="hist-card"><div class="hist-src">{html.escape(hist[i]["src"])}</div>'
                    f'<div class="hist-tgt">{html.escape(str(hist[i]["tgt"]))}</div></div>',
                    unsafe_allow_html=True
                )
            with h_right:
                with st.container(key=f"hist_{i}"):
                    st.button("↺", key=f"rebtn_{i}", on_click=load_history, args=(i,), help="Use this translation again")
        _, h_mid, _ = st.columns([2, 1, 2])
        with h_mid:
            with st.container(key="clrhist"):
                st.button("Clear history", key="clear_hist", on_click=clear_history, use_container_width=True)

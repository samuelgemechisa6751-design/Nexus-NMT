import streamlit as st
import torch
import torch.nn.functional as F
import os
import html
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from model import NexusNMT

# ==========================================
# 1. PAGE CONFIG & CLEAN MINIMALIST CSS
# ==========================================
st.set_page_config(page_title="Nexus-NMT | Translator", page_icon="🌐", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Ethiopic:wght@400;500;600&display=swap');

/* Global Styling */
html, body, [class*="css"], .stApp, .stMarkdown, textarea, button {
    font-family: 'Inter', 'Noto Sans Ethiopic', sans-serif !important;
}
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* Container */
.block-container { max-width: 900px; padding-top: 3rem; padding-bottom: 4rem; }

/* Hero Section */
.hero { text-align: center; margin-bottom: 3rem; }
.hero-title {
    font-size: 3rem; font-weight: 700; letter-spacing: -0.04em;
    color: #0F172A; margin-bottom: 1rem; line-height: 1.1;
}
.hero-sub {
    font-size: 1.15rem; color: #64748B; max-width: 500px;
    margin: 0 auto; line-height: 1.6;
}

/* Language Selectors */
.lang-row { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 2rem; }
.lang-box {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 12px 24px; font-weight: 600; color: #0F172A;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); width: 220px; text-align: center;
}

/* Text Area Styling */
.stTextArea label { display: none; }
.stTextArea > div > div {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    transition: border-color 0.2s ease;
}
.stTextArea > div > div:focus-within { border-color: #3B82F6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important; }
.stTextArea textarea {
    font-size: 1.25rem !important; padding: 20px !important; color: #0F172A !important;
}

/* Output Box */
.output-container {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
    padding: 24px; min-height: 200px; font-size: 1.35rem; font-weight: 500;
    color: #0F172A; line-height: 1.6; box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}
.output-container.empty { color: #94A3B8; font-weight: 400; font-size: 1.15rem; display: flex; align-items: center; justify-content: center; }

/* Button */
.stButton > button {
    background: #0F172A; color: #FFFFFF; border-radius: 12px;
    padding: 14px 28px; font-weight: 600; font-size: 1.05rem;
    border: none; width: 100%; transition: all 0.2s ease;
    box-shadow: 0 4px 6px rgba(15,23,42,0.1);
}
.stButton > button:hover {
    background: #1E293B; transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(15,23,42,0.15); color: #FFFFFF;
}

/* Stats Footer */
.stats-row { display: flex; justify-content: center; gap: 40px; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #E2E8F0; }
.stat-item { text-align: center; }
.stat-val { font-size: 1.5rem; font-weight: 700; color: #0F172A; }
.stat-lbl { font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

/* Hide the ugly Streamlit warning box */
.stAlert { display: none; }
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

        # 🚀 USING THE 80000 CHECKPOINT AS REQUESTED
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

# ==========================================
# 3. HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Nexus-NMT</h1>
    <p class="hero-sub">A custom 60-Million parameter Transformer trained from scratch on 3.2 Million sentence pairs.</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading..."):
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
# 5. UI INTERACTION
# ==========================================
if "src_is_en" not in st.session_state:
    st.session_state.src_is_en = True

def toggle_lang():
    st.session_state.src_is_en = not st.session_state.src_is_en

src_is_en = st.session_state.src_is_en
src_lang = "English" if src_is_en else "አማርኛ (Amharic)"
tgt_lang = "አማርኛ (Amharic)" if src_is_en else "English"

st.markdown(f'''
<div class="lang-row">
    <div class="lang-box">{src_lang}</div>
    <div class="lang-box">{tgt_lang}</div>
</div>
''', unsafe_allow_html=True)

# Hidden button to trigger swap
st.button("Swap Languages ⇄", on_click=toggle_lang, use_container_width=False)

col1, col2 = st.columns(2, gap="large")

with col1:
    user_input = st.text_area(
        "Input",
        value="The president spoke to the nation." if src_is_en else "ፕሬዝዳንቱ ለህዝቡ ተናገሩ።",
        height=200,
        max_chars=500,
    )

with col2:
    out_slot = st.empty()
    def render_output(val):
        if val:
            out_slot.markdown(f'<div class="output-container">{html.escape(val)}</div>', unsafe_allow_html=True)
        else:
            out_slot.markdown('<div class="output-container empty">Translation will appear here</div>', unsafe_allow_html=True)
    render_output(None)

st.write("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    if st.button("Translate", use_container_width=True):
        if not user_input.strip():
            pass # Hidden warning
        else:
            with st.spinner("Translating..."):
                result = translate(user_input, to_amharic=src_is_en)
                render_output(result)

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("""
<div class="stats-row">
    <div class="stat-item"><div class="stat-val">60M</div><div class="stat-lbl">Parameters</div></div>
    <div class="stat-item"><div class="stat-val">40k</div><div class="stat-lbl">Vocabulary</div></div>
    <div class="stat-item"><div class="stat-val">3.2M</div><div class="stat-lbl">Data Pairs</div></div>
</div>
""", unsafe_allow_html=True)

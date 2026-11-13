import streamlit as st
import torch
import torch.nn.functional as F
import os
import html
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from model import NexusNMT

# ==========================================
# 1. PAGE CONFIG & VIBRANT COLORFUL UI
# ==========================================
st.set_page_config(page_title="Nexus-NMT | AI Studio", page_icon="🌐", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Noto+Sans+Ethiopic:wght@400;500;600&display=swap');

/* Force everything to use our fonts and dark colored theme */
html, body, [class*="css"], .stApp, .stMarkdown, textarea, button {
    font-family: 'Outfit', 'Noto Sans Ethiopic', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0B0C10 0%, #1F2833 100%);
    color: #FFFFFF;
}

/* Hide standard Streamlit elements */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1000px; padding-top: 3rem; padding-bottom: 4rem; }

/* Hero Section */
.hero { text-align: center; margin-bottom: 3rem; }
.hero-title {
    font-size: 3.5rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg, #00F0FF, #FF00E5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem; line-height: 1.1;
}
.hero-sub {
    font-size: 1.15rem; color: #A0AAB5; max-width: 600px;
    margin: 0 auto; line-height: 1.6;
}

/* Language Selectors */
.lang-row { display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 2rem; }
.lang-box {
    background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.3); 
    border-radius: 12px; padding: 12px 24px; font-weight: 600; color: #00F0FF;
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.1); width: 220px; text-align: center;
    backdrop-filter: blur(10px);
}

/* Input Text Area */
.stTextArea label { display: none; }
.stTextArea > div > div {
    background: rgba(255, 255, 255, 0.03) !important; 
    border: 1px solid rgba(255, 0, 229, 0.4) !important;
    border-radius: 16px !important; 
    box-shadow: 0 4px 20px rgba(255, 0, 229, 0.1) !important;
    transition: all 0.3s ease;
}
.stTextArea > div > div:focus-within { 
    border-color: #FF00E5 !important; 
    box-shadow: 0 0 15px rgba(255, 0, 229, 0.3) !important; 
}
.stTextArea textarea {
    font-size: 1.25rem !important; padding: 20px !important; 
    color: #FFFFFF !important; /* FORCED WHITE TEXT */
    background: transparent !important;
}

/* Output Box */
.output-container {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), rgba(255, 0, 229, 0.1)); 
    border: 1px solid rgba(0, 240, 255, 0.5); border-radius: 16px;
    padding: 24px; min-height: 200px; font-size: 1.35rem; font-weight: 500;
    color: #FFFFFF; line-height: 1.6; box-shadow: 0 4px 20px rgba(0, 240, 255, 0.15);
    backdrop-filter: blur(10px);
}
.output-container.empty { 
    background: rgba(255, 255, 255, 0.03); color: #6A7F84; font-weight: 400; 
    font-size: 1.15rem; display: flex; align-items: center; justify-content: center; 
    border: 1px dashed rgba(255, 255, 255, 0.1); box-shadow: none;
}

/* Action Button */
.stButton > button {
    background: linear-gradient(90deg, #00F0FF, #FF00E5); 
    color: #000000; border-radius: 12px;
    padding: 14px 28px; font-weight: 800; font-size: 1.05rem;
    border: none; width: 100%; transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 0, 229, 0.4);
    text-transform: uppercase; letter-spacing: 0.05em;
}
.stButton > button:hover {
    transform: translateY(-2px); 
    box-shadow: 0 8px 25px rgba(0, 240, 255, 0.6); 
    color: #000000;
}

/* Stats Footer */
.stats-row { display: flex; justify-content: center; gap: 40px; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1); }
.stat-item { text-align: center; }
.stat-val { font-size: 1.8rem; font-weight: 800; color: #00F0FF; text-shadow: 0 0 10px rgba(0,240,255,0.4); }
.stat-lbl { font-size: 0.85rem; color: #A0AAB5; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
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

        # 🚀 USING THE 80000 CHECKPOINT
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
# 3. HEADER (Subtitle Removed)
# ==========================================
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Nexus-NMT</h1>
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

st.button("⇄ Swap Languages", on_click=toggle_lang, use_container_width=True)

st.write("")
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
if st.button("Translate Now", use_container_width=True):
    if user_input.strip():
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

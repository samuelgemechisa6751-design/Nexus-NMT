import streamlit as st
import torch
import torch.nn as nn
import math
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Nexus-NMT", page_icon="🌍")

# ==========================================
# SECRET TOKEN HANDLING
# ==========================================
# Securely fetch the Hugging Face token from Streamlit secrets.
# This prevents "File not found" errors if your HF repo is private.
hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

# ==========================================
# DISPLAY TREE IMAGE AT THE TOP
# ==========================================
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Tree_with_roots.png/800px-Tree_with_roots.png"

try:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(IMAGE_URL, use_container_width=True)
except Exception:
    st.warning("Could not load image.")

st.title("Nexus-NMT")
st.write("A 200M parameter Multilingual Transformer built from scratch.")

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model; self.num_heads = num_heads; self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        Q = self.W_q(q).view(q.size(0), q.size(1), self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(k.size(0), k.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(v.size(0), v.size(1), self.num_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: attn = attn.masked_fill(mask == 0, -1e4)
        out = torch.matmul(torch.softmax(attn, dim=-1), V)
        return self.W_o(out.transpose(1, 2).contiguous().view(q.size(0), q.size(1), self.d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff); self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout); self.relu = nn.ReLU()
    def forward(self, x): return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads); self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, enc_output, enc_output, src_mask)))
        return self.norm3(x + self.dropout3(self.feed_forward(x)))

class NexusNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, 768); self.tgt_embedding = nn.Embedding(tgt_vocab_size, 768)
        self.positional_encoding = PositionalEncoding(768)
        self.encoder_layers = nn.ModuleList([EncoderLayer(768, 12, 3072, 0.1) for _ in range(8)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(768, 12, 3072, 0.1) for _ in range(8)])
        self.fc_out = nn.Linear(768, tgt_vocab_size); self.dropout = nn.Dropout(0.1); self.scale = math.sqrt(768)
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        nopeak = (1 - torch.triu(torch.ones(1, tgt.size(1), tgt.size(1)), diagonal=1)).bool().to(tgt.device)
        return src_mask, tgt_mask & nopeak
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src) * self.scale))
        enc_out = src_emb
        for layer in self.encoder_layers: enc_out = layer(enc_out, src_mask)
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * self.scale))
        dec_out = tgt_emb
        for layer in self.decoder_layers: dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

# ==========================================
# 2. LOAD MODEL FROM HUGGING FACE
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    repo_id = "samuel-nexus/Nexus-NMT-Model"
    
    # Pass the token to prevent 401 Unauthorized errors for private repos
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", token=hf_token)
    merges_path = hf_hub_download(repo_id=repo_id, filename="merges.txt", token=hf_token)
    
    tokenizer = ByteLevelBPETokenizer(tokenizer_path, merges_path)
    
    vocab_size = tokenizer.get_vocab_size()
    model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
    
    # 🚨 ENSURE THIS MATCHES YOUR EXACT UPLOADED FILE 🚨
    weights_path = hf_hub_download(repo_id=repo_id, filename="nexus_batch_54000_8bit.pt", token=hf_token) 
    
    state_dict_8bit = torch.load(weights_path, map_location=device)
    state_dict = {}
    
    # Dequantize 8-bit weights back to float16
    for k, v in state_dict_8bit.items():
        if k.endswith("_scale"):
            continue
        scale = state_dict_8bit.get(k + "_scale", 1.0)
        state_dict[k] = (v.to(torch.float16) * scale)
        
    model.load_state_dict(state_dict)
    model = model.half()
    model.eval()
    return model, tokenizer, device

# Error handling for model loading
try:
    model, tokenizer, device = load_model()
except Exception as e:
    st.error(f"Failed to load model. If your repo is private, check your HF_TOKEN in Streamlit secrets. Error details: {e}")
    st.stop()

# ==========================================
# 3. TRANSLATION LOGIC (FIXED)
# ==========================================
def translate(text):
    # Ensure this exact prefix matches your training data
    src_text = f"<2am> {text}"
    
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    
    # Fallbacks if special tokens are missing
    if bos_id is None: bos_id = 0
    if eos_id is None: eos_id = 2
    if pad_id is None: pad_id = 1

    src_ids = [bos_id] + tokenizer.encode(src_text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        for _ in range(64):  # Max generation length
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
            
            # Forward pass through Encoder
            src_emb = model.positional_encoding(model.src_embedding(src_tensor) * model.scale)
            enc_out = src_emb
            for layer in model.encoder_layers: 
                enc_out = layer(enc_out, src_mask)
            
            # Forward pass through Decoder
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * model.scale)
            dec_out = tgt_emb
            for layer in model.decoder_layers: 
                dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_out)[0, -1, :].clone()
            
            # ONLY ban special tokens. Do NOT ban spaces or punctuation!
            logits[bos_id] = -float('inf')
            logits[pad_id] = -float('inf')

            # Soft repetition penalty (Only penalize slight repeats)
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id]:
                    logits[token] -= 2.0 
            
            next_token = logits.argmax().item()
            
            if next_token == eos_id:
                break
                
            tgt_ids.append(next_token)

    # Decode and clean up text (skip_special_tokens handles the <s> and </s> removal)
    clean_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    return clean_text.strip() if clean_text.strip() else "[Model generated empty response]"

# ==========================================
# 4. UI
# ==========================================
user_input = st.text_area("English Text", "Hello, how are you?")
if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            result = translate(user_input)
            st.success("Translation Complete!")
            st.markdown(f"### Amharic Translation:\n**{result}**")

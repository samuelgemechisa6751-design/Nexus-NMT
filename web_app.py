import streamlit as st
import torch
import torch.nn as nn
import math
from huggingface_hub import hf_hub_download
from tokenizers import ByteLevelBPETokenizer

st.set_page_config(page_title="Nexus-NMT Translator", page_icon="🌍")
st.title("🌍 Nexus-NMT: English to Amharic")
st.write("A 200M parameter Transformer built from scratch.")

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE (FLOAT16 SAFE)
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
# 2. LOAD MODEL FROM HUGGING FACE (LITE VERSION)
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    tokenizer_path = hf_hub_download(repo_id="samuel-nexus/Nexus-NMT-Model", filename="vocab.json")
    tokenizer = ByteLevelBPETokenizer(tokenizer_path, hf_hub_download(repo_id="samuel-nexus/Nexus-NMT-Model", filename="merges.txt"))
    
    vocab_size = tokenizer.get_vocab_size()
    model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size).to(device)
    
    # Load the compressed lite model
    weights_path = hf_hub_download(repo_id="samuel-nexus/Nexus-NMT-Model", filename="nexus_batch_16000_lite.pt") 
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.half()  # Convert to half-precision
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# ==========================================
# 3. TRANSLATION LOGIC (ANTI-LOOP)
# ==========================================
def translate(text):
    src_text = f"<2am> {text}"
    src_ids = [tokenizer.token_to_id("<s>")] + tokenizer.encode(src_text).ids + [tokenizer.token_to_id("</s>")]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [tokenizer.token_to_id("<s>")]
        for _ in range(64):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
            
            src_emb = model.positional_encoding(model.src_embedding(src_tensor) * model.scale)
            enc_out = src_emb
            for layer in model.encoder_layers: enc_out = layer(enc_out, src_mask)
            
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * model.scale)
            dec_out = tgt_emb
            for layer in model.decoder_layers: dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_out)[0, -1, :].clone()
            
            # 🚨 BAN PUNCTUATION AND SPECIAL TOKENS 🚨
            logits[tokenizer.token_to_id("<s>")] = -float('inf')
            logits[tokenizer.token_to_id("<pad>")] = -float('inf')
            
            # Ban common punctuation tokens
            for punct in [",", ".", " ", "\n", "።", "፣"]:
                token_id = tokenizer.token_to_id(punct)
                if token_id is not None:
                    logits[token_id] = -float('inf')

            # 🚨 HEAVY REPETITION PENALTY 🚨
            for token in set(tgt_ids):
                if token not in [tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")]:
                    logits[token] -= 10.0  # Very strong penalty
            
            next_token = logits.argmax().item()
            tgt_ids.append(next_token)
            if next_token == tokenizer.token_to_id("</s>"): break

    return tokenizer.decode(tgt_ids)

# ==========================================
# 4. UI
# ==========================================
user_input = st.text_area("English Text", "Hello, how are you?")
if st.button("Translate"):
    with st.spinner("Translating..."):
        result = translate(user_input)
        st.success("Translation Complete!")
        st.markdown(f"### Amharic Translation:\n**{result}**")

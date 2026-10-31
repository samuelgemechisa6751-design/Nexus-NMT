
import torch
from tokenizers import ByteLevelBPETokenizer
from model import NexusNMT
import os

def translate(sentence, direction="en-am"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
    vocab_size = tokenizer.get_vocab_size()
    
    model = NexusNMT(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=1024).to(device)
    
    # Load the NEW fixed weights
    model.load_state_dict(torch.load("nexus_nmt_fixed.pt", map_location=device))
    
    tag = "<2am>" if direction == "en-am" else "<2en>"
    src_text = f"{tag} {sentence}"
    src_ids = [tokenizer.token_to_id("<s>")] + tokenizer.encode(src_text).ids + [tokenizer.token_to_id("</s>")]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    output_ids = model.generate(src_tensor, tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>"), tokenizer.token_to_id("<pad>"))
    return tokenizer.decode(output_ids)

if __name__ == "__main__":
    test_sentences = [
        "Hello, how are you?",
        "The President of Ethiopia is visiting the capital.",
        "I love learning new languages."
    ]
    
    for text in test_sentences:
        print(f"\nEN: {text}")
        am = translate(text, "en-am")
        print(f"AM: {am}")

def translate(text, direction):
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    unk_id = tokenizer.token_to_id("<unk>") or 3
    pad_id = tokenizer.token_to_id("<pad>") or 1
    
    tag_2am = tokenizer.token_to_id("<2am>")
    tag_2en = tokenizer.token_to_id("<2en>")
    
    # Add the correct tag based on the selected direction
    if direction == "English ➔ Amharic":
        tag_id = tag_2am
    else:
        tag_id = tag_2en
    
    # Encode input with the tag at the beginning
    src_ids = [tag_id] + tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        
        for step in range(50): # Max output length
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(src_tensor, tgt_tensor)[0, -1, :].clone()
            
            # 1. Ban Special Tokens & Tags
            logits[pad_id] = -float('inf')
            logits[unk_id] = -float('inf')
            logits[bos_id] = -float('inf')
            logits[tag_2am] = -float('inf')
            logits[tag_2en] = -float('inf')
            
            # 🚀 2. PREVENT EARLY STOPPING (Force at least 2 words)
            if step < 2:
                logits[eos_id] = -float('inf')
            
            # 3. Repetition Penalty
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id]:
                    logits[token] -= 10.0 

            # 4. Greedy Decoding
            next_token = logits.argmax().item()
            
            # Stop if we hit end of sentence
            if next_token == eos_id:
                break
                
            tgt_ids.append(next_token)

    # Decode output
    clean_text = tokenizer.decode(tgt_ids)
    
    # Remove special tokens
    for tag in ["<s>", "</s>", "<pad>", "<unk>", "<2am>", "<2en>"]:
        clean_text = clean_text.replace(tag, "")
        
    # Fix ByteLevel spacing (Ġ)
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    # 🚀 Debug output if empty
    if not clean_text:
        return f"[Empty Response. Raw IDs: {tgt_ids}]"
        
    return clean_text

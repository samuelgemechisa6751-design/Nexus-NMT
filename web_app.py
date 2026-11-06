def translate(text):
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    tag_2am = tokenizer.token_to_id("<2am>")
    
    if bos_id is None: bos_id = 0
    if eos_id is None: eos_id = 2
    if pad_id is None: pad_id = 1
    if tag_2am is None: tag_2am = 3

    # ❌ OLD: src_ids = [tag_2am] + tokenizer.encode(text).ids + [eos_id]
    # ✅ NEW: The tokenizer ALREADY adds the <s> token. We don't add tag_2am manually.
    src_ids = tokenizer.encode(text).ids + [eos_id]
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        for _ in range(30): # Increased max length slightly
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
            
            src_emb = model.positional_encoding(model.src_embedding(src_tensor) * model.scale)
            enc_out = src_emb
            for layer in model.encoder_layers: 
                enc_out = layer(enc_out, src_mask)
            
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * model.scale)
            dec_out = tgt_emb
            for layer in model.decoder_layers: 
                dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_out)[0, -1, :].clone()
            
            # Ban special tokens
            logits[bos_id] = -float('inf')
            logits[pad_id] = -float('inf')
            logits[tag_2am] = -float('inf')

            # 🚀 FIX: Stronger repetition penalty. 
            # We increase it from 2.0 to 4.0 to stop the "enenenenen" loop.
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id, pad_id]:
                    logits[token] -= 4.0 
            
            next_token = logits.argmax().item()
            
            # Stop if EOS or PAD is generated
            if next_token == eos_id or next_token == pad_id:
                break
                
            tgt_ids.append(next_token)

    # Decode the output. 
    # Note: ByteLevelBPETokenizer handles spaces via the 'Ġ' character.
    # skip_special_tokens=True handles removing <s> and </s>
    clean_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    
    # Clean up any leftover byte-level artifacts if necessary
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    return clean_text if clean_text else "[Model generated empty response]"

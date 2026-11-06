def translate(text):
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    pad_id = tokenizer.token_to_id("<pad>") or 1
    
    # We will ban ALL potential special tokens
    special_tokens = [bos_id, eos_id, pad_id]
    for tag in ["<2am>", "<2en>", "<2om>"]:
        tag_id = tokenizer.token_to_id(tag)
        if tag_id is not None:
            special_tokens.append(tag_id)

    # Encode input
    src_ids = tokenizer.encode(text).ids + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        tgt_ids = [bos_id]
        for _ in range(30):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            src_mask, tgt_mask = model.generate_mask(src_tensor, tgt_tensor)
            
            # Forward pass
            src_emb = model.positional_encoding(model.src_embedding(src_tensor) * model.scale)
            enc_out = src_emb
            for layer in model.encoder_layers: 
                enc_out = layer(enc_out, src_mask)
            
            tgt_emb = model.positional_encoding(model.tgt_embedding(tgt_tensor) * model.scale)
            dec_out = tgt_emb
            for layer in model.decoder_layers: 
                dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            logits = model.fc_out(dec_out)[0, -1, :].clone()
            
            # 🚀 CRITICAL FIX: Ban every single special token from ever being generated
            for sp_id in special_tokens:
                logits[sp_id] = -float('inf')

            # 🚀 CRITICAL FIX: Massive repetition penalty to break the "enedenen" loop
            for token in set(tgt_ids):
                if token not in [bos_id, eos_id, pad_id]:
                    logits[token] -= 15.0 # Increased from 4.0 to 15.0
            
            next_token = logits.argmax().item()
            
            if next_token == eos_id or next_token == pad_id:
                break
                
            tgt_ids.append(next_token)

    # Decode output
    clean_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
    
    # 🚀 CLEANUP: Forcefully remove any special tags that slip through
    for tag in ["<s>", "</s>", "<pad>", "<2am>", "<2en>", "<2om>"]:
        clean_text = clean_text.replace(tag, "")
        
    # Clean up the weird ByteLevel 'Ġ' artifacts
    clean_text = clean_text.replace("Ġ", " ").strip()
    
    return clean_text if clean_text else "[Model generated empty response]"

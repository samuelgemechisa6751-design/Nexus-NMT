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
            
            # Only ban the special tokens, let the model speak naturally
            logits[tokenizer.token_to_id("<s>")] = -float('inf')
            logits[tokenizer.token_to_id("<pad>")] = -float('inf')

            # Repetition penalty (standard)
            for token in set(tgt_ids):
                if token not in [tokenizer.token_to_id("<s>"), tokenizer.token_to_id("</s>")]:
                    logits[token] -= 2.0
            
            next_token = logits.argmax().item()
            tgt_ids.append(next_token)
            if next_token == tokenizer.token_to_id("</s>"): break

    return tokenizer.decode(tgt_ids)

import torch

def handle(inputs, probs, mean_logits, std_logits, text, tokenizer, custom_metrics, verbose):
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    predicted_classes = torch.argmax(mean_logits, dim=-1)

    results = []
    for i in range(len(text)):
        token_preds = []
        for j in range(probs.shape[1]):
            token_info = {
                "token_id": int(inputs["input_ids"][i][j]),
                "predicted_class": predicted_classes[i][j].item(),
                "entropy": entropy[i][j].item(),
                "probs": probs[i][j].tolist(),
                "std_dev": std_logits[i][j].tolist()
            }
            token_preds.append(token_info)
        results.append({
            "input": text[i],
            "token_predictions": token_preds
        })

        if verbose:
            print(f"[{i}] Token-level prediction with {len(token_preds)} tokens.")

    return results

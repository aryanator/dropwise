import torch

def handle(inputs, probs, mean_logits, std_logits, text, tokenizer, custom_metrics, verbose):
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    predicted_classes = torch.argmax(mean_logits, dim=-1)
    confidence = probs.max(dim=-1).values
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    margin = (sorted_probs[:, 0] - sorted_probs[:, 1])

    results = []
    for i in range(len(text)):
        result = {
            "input": text[i],
            "predicted_class": predicted_classes[i].item(),
            "entropy": entropy[i].item(),
            "confidence": confidence[i].item(),
            "confidence_margin": margin[i].item(),
            "std_dev": std_logits[i].tolist(),
            "probs": probs[i].tolist()
        }
        for name, func in custom_metrics.items():
            result[name] = func(probs[i])
        results.append(result)

        if verbose:
            print(f"[{i}] Pred: {result['predicted_class']}, "
                  f"Conf: {result['confidence']:.3f}, "
                  f"Entropy: {result['entropy']:.3f}, "
                  f"Margin: {result['confidence_margin']:.3f}")

    return results

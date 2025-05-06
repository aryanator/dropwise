def handle(inputs, probs, mean_logits, std_logits, text, tokenizer, custom_metrics, verbose):
    results = []
    for i in range(len(text)):
        score = mean_logits[i].item()
        std = std_logits[i].item()
        result = {
            "input": text[i],
            "predicted_score": score,
            "uncertainty": std
        }
        results.append(result)

        if verbose:
            print(f"[{i}] Score: {score:.3f} ± {std:.3f}")

    return results

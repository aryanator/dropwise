import torch

def handle(inputs, probs, mean_logits, std_logits, text, tokenizer, custom_metrics, verbose):
    start_logits = mean_logits[:, :, 0]
    end_logits = mean_logits[:, :, 1]

    predicted_starts = torch.argmax(start_logits, dim=-1)
    predicted_ends = torch.argmax(end_logits, dim=-1)

    results = []
    for i in range(len(text)):
        start = predicted_starts[i].item()
        end = predicted_ends[i].item() + 1
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i][start:end])
        answer = tokenizer.convert_tokens_to_string(tokens)
        result = {
            "input": text[i],
            "start": start,
            "end": end,
            "answer": answer,
            "start_logits": start_logits[i].tolist(),
            "end_logits": end_logits[i].tolist()
        }
        results.append(result)

        if verbose:
            print(f"[{i}] Answer: '{answer}'")

    return results

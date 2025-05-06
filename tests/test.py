import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dropwise import DropwisePredictor

def test_dropwise_predictor():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    predictor = DropwisePredictor(model, tokenizer, num_passes=5)

    text = "This movie was unexpectedly great!"
    result = predictor.predict(text)

    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert result["probs"].shape[-1] == 2
    assert torch.all(result["std_dev"] >= 0)

    print("Test passed.")

if __name__ == "__main__":
    test_dropwise_predictor()
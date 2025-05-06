import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Union

class DropwisePredictor:
    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer, 
                 num_passes: int = 20, task_type: str = "sequence-classification", use_cuda: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.num_passes = num_passes
        self.task_type = task_type
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        self.model.to(self.device)
        self.model.eval()
        self._enable_mc_dropout()

    def _enable_mc_dropout(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def predict(self, text: Union[str, list]):
        if isinstance(text, str):
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        else:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)

        all_logits = []

        with torch.no_grad():
            for _ in range(self.num_passes):
                outputs = self.model(**inputs)
                logits = outputs.logits
                all_logits.append(logits)

        stacked_logits = torch.stack(all_logits)  # [num_passes, batch, num_classes]
        mean_logits = stacked_logits.mean(dim=0)
        std_logits = stacked_logits.std(dim=0)
        probs = F.softmax(mean_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        predicted_class = torch.argmax(mean_logits, dim=-1)

        return {
            "mean_logits": mean_logits,
            "std_dev": std_logits,
            "entropy": entropy,
            "predicted_class": predicted_class,
            "probs": probs
        }

# === Example usage ===
if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    predictor = DropwisePredictor(model, tokenizer, num_passes=30)
    result = predictor.predict("This is definitely worth watching!")

    print("Predicted class:", result['predicted_class'].item())
    print("Entropy:", result['entropy'].item())
    print("Confidence:", result['probs'].max().item())
    print("Per-class std dev:", result['std_dev'])

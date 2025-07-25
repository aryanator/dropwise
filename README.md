# Dropwise

**Dropwise**[https://pypi.org/project/dropwise/] is a lightweight PyTorch/HuggingFace wrapper for performing Monte Carlo Dropout–based uncertainty estimation in Transformers. It enables confidence-aware decision making by revealing how certain a model is about its predictions — all with just a few lines of code.

---

## 🚀 Features

- ✅ Enable dropout during inference for **Bayesian-like uncertainty** estimation  
- ✅ Compute **predictive entropy**, **confidence**, and **per-class standard deviation**  
- ✅ Modular support for **classification, QA, token tagging, and regression**  
- ✅ Works seamlessly with **Hugging Face Transformers** and **PyTorch**  
- ✅ Supports **batch inference**, **CPU/GPU**, and customizable `num_passes`  
- ✅ Cleanly packaged and extensible for research or production

---

## 🤖 Supported Tasks

| Task Type               | Example Model                                 |
|------------------------|------------------------------------------------|
| `sequence-classification` | `distilbert-base-uncased-finetuned-sst-2-english`  
| `token-classification`    | `dslim/bert-base-NER`  
| `question-answering`      | `deepset/bert-base-cased-squad2`  
| `regression`              | `roberta-base` (with custom regression head)

> ⚠️ Your model must contain dropout layers for MC sampling to work (most HF models do).

---

## 📦 Installation

```bash
pip install dropwise
```

Or install from source:

```bash
git clone https://github.com/aryanator01/dropwise.git
cd dropwise
pip install -e .
```

---

## 🧠 Example Usage (Per Task)

### 📘 Sequence Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dropwise.predictor import DropwisePredictor

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

predictor = DropwisePredictor(model, tokenizer, task_type="sequence-classification", num_passes=20)
results = predictor(["The movie was fantastic!"])

print(results[0])
```

---

### 🏷️ Token Classification (NER)

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from dropwise.predictor import DropwisePredictor

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

predictor = DropwisePredictor(model, tokenizer, task_type="token-classification", num_passes=15)
results = predictor(["Hugging Face is based in New York City."])

print(results[0]['token_predictions'])
```

---

### ❓ Question Answering

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from dropwise.predictor import DropwisePredictor

model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

question = "Where is Hugging Face based?"
context = "Hugging Face Inc. is a company based in New York City."
qa_input = f"{question} [SEP] {context}"

predictor = DropwisePredictor(model, tokenizer, task_type="question-answering", num_passes=10)
results = predictor([qa_input])

print(results[0]['answer'])
```

---

### 📈 Regression

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dropwise.predictor import DropwisePredictor

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

predictor = DropwisePredictor(model, tokenizer, task_type="regression", num_passes=20)
results = predictor(["The child is very young."])

print(results[0]['predicted_score'], "+/-", results[0]['uncertainty'])
```

---

## 📊 Output Dictionary (per sample)

| Field               | Description                                      |
|--------------------|--------------------------------------------------|
| `predicted_class`  | Index of most probable class (classification)    |
| `predicted_score`  | Scalar prediction (regression only)              |
| `confidence`       | Highest softmax probability                      |
| `entropy`          | Predictive entropy (higher = less confident)     |
| `std_dev`          | Standard deviation across MC passes              |
| `probs`            | Class-wise softmax probabilities                 |
| `margin`           | Confidence gap between top-2 predictions         |
| `answer`           | Predicted answer span (QA only)                  |
| `token_predictions`| List of per-token dicts (token classification)   |

---

## 🧪 Run Tests

```bash
python tests/test_predictor.py
```

Covers all task types with preloaded models.

---

## 📂 Folder Structure

```
dropwise/
├── predictor.py
├── tasks/
│   ├── __init__.py
│   ├── sequence_classification.py
│   ├── token_classification.py
│   ├── question_answering.py
│   └── regression.py
tests/
└── test_predictor.py
```

---

## 📝 License

MIT License

---

Built with ❤️ for robust, explainable, uncertainty-aware AI systems.

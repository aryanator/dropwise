from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification
)
from dropwise.predictor import DropwisePredictor


def test_sequence_classification():
    print("\n🔍 Testing Sequence Classification")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictor = DropwisePredictor(model, tokenizer, task_type="sequence-classification", num_passes=10)
    results = predictor(["The movie was excellent!", "Worst experience ever."], verbose=True)
    for res in results:
        print(res)


def test_token_classification():
    print("\n🔍 Testing Token Classification")
    model_name = "dslim/bert-base-NER"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictor = DropwisePredictor(model, tokenizer, task_type="token-classification", num_passes=10)
    results = predictor(["Hugging Face is based in New York City."], verbose=True)
    for res in results:
        print(res)


def test_question_answering():
    print("\n🔍 Testing Question Answering")
    model_name = "deepset/bert-base-cased-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    context = "Hugging Face Inc. is a company based in New York City. It’s known for Transformers library."
    question = "Where is Hugging Face based?"
    qa_input = f"{question} [SEP] {context}"

    predictor = DropwisePredictor(model, tokenizer, task_type="question-answering", num_passes=10)
    results = predictor([qa_input], verbose=True)
    for res in results:
        print(res)


def test_regression():
    print("\n🔍 Testing Regression")
    model_name = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictor = DropwisePredictor(model, tokenizer, task_type="regression", num_passes=10)
    results = predictor(["She is a young adult.", "The old man was walking slowly."], verbose=True)
    for res in results:
        print(res)



if __name__ == "__main__":
    test_sequence_classification()
    test_token_classification()
    test_question_answering()
    test_regression()

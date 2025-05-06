from dropwise.tasks.sequence_classification import handle as sequence_classification
from dropwise.tasks.token_classification import handle as token_classification
from dropwise.tasks.question_answering import handle as question_answering
from dropwise.tasks.regression import handle as regression

def get_task_handler(task_type: str):
    if task_type == "sequence-classification":
        return sequence_classification
    elif task_type == "token-classification":
        return token_classification
    elif task_type == "question-answering":
        return question_answering
    elif task_type == "regression":
        return regression
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

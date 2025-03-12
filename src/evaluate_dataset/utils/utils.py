import sklearn.metrics
from utils.utils import *

def calculate_metrics(ref_statements, responses):
    """
    Calculate accuracy, micro, macro and weighted f1 scores of predicted labels in responses compared to reference assessments in reference_path.
    The results are saved to output_path.

    Args:
    ref_statements (List): List of reference statements.
    responses (List): List of responses.
    """

    # get labels from responses
    pred_labels = [
        item["label"].lower()
        for ref in ref_statements 
        if (item := find_by_id(ref['id'], responses)) is not None
    ]    
    # get labels from reference
    ref_labels = [ref["assessment"].lower() for ref in ref_statements]

    # calculate metrics
    f1_micro = sklearn.metrics.f1_score(ref_labels, pred_labels, average="micro")
    f1_macro = sklearn.metrics.f1_score(ref_labels, pred_labels, average="macro")
    f1_weighted = sklearn.metrics.f1_score(ref_labels, pred_labels, average="weighted")
    precision = sklearn.metrics.precision_score(ref_labels, pred_labels, average="weighted")
    recall = sklearn.metrics.recall_score(ref_labels, pred_labels, average="weighted")
    accuracy = sklearn.metrics.accuracy_score(ref_labels, pred_labels)

    return f1_micro, f1_macro, f1_weighted, accuracy, precision, recall

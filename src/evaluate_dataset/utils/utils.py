import sklearn.metrics
from utils.utils import *

def calculate_metrics(ref_statements, responses, allowed_labels):
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
    return sklearn.metrics.classification_report(ref_labels, pred_labels, output_dict=True, labels=allowed_labels)


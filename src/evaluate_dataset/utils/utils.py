import sklearn.metrics
import numpy as np
import pprint
from utils.utils import *

def calculate_metrics(ref_statements, responses, allowed_labels):
    """
    Calculate accuracy, micro, macro and weighted f1 scores of predicted labels in responses compared to reference labels in reference_path.
    The results are saved to output_path.

    Args:
    ref_statements (List): List of reference statements.
    responses (List): List of responses.
    """

    # get labels from responses
    pred_labels = []
    for ref in ref_statements:
        pred_labels.append(next((i['label'] for i in responses if str(i['id']) == str(ref['id'])), None))

    # get labels from reference
    ref_labels = [ref["label"].lower() for ref in ref_statements]

    # calculate metrics
    return sklearn.metrics.classification_report(ref_labels, pred_labels, output_dict=True, labels=allowed_labels, zero_division=np.nan)

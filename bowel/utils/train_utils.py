import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score


def get_classes_amount(y):
    """Calculates classes balance.

    Args:
        y (ndarray): Array of {0,1} classes.

    Returns:
        (int, int, float): Tuple of amount of 0s, amount of 1s and ratio of 0s to all.
    """
    zeros = np.count_nonzero(y == 0)
    ones = np.count_nonzero(y == 1)
    return zeros, ones, zeros / (zeros + ones)


def get_confusion_matrix(y_true, y_pred, threshold):
    """Calculates confusion matrix.

    Args:
        y_true (ndarray): Array of ground truth classes.
        y_pred (ndarray): Array of predicted classes probabilities.
        threshold (float): Threshold in range [0,1] above which, set probabilities to 1 class.

    Returns:
        ndarray: 2D array with confusion matrix values.
    """
    y_pred = (y_pred >= threshold)
    return confusion_matrix(y_true.flatten(), (y_pred >= threshold).flatten())


def get_score(y_true, y_pred, threshold=0.5):
    """Calculates various metrics.

    Args:
        y_true (ndarray): Array of ground truth classes.
        y_pred (ndarray): Array of predicted classes probabilities.
        threshold (float): Threshold in range [0,1] above which, set probabilities to 1 class. Defaults to 0.5.

    Returns:
        dict: Dictionary with calculated metrics.
    """
    c = get_confusion_matrix(y_true, y_pred, threshold)
    zeros, ones, ratio = get_classes_amount(y_true)
    precision = c[1, 1] / (c[1, 1] + c[0, 1])
    recall = c[1, 1] / (c[1, 1] + c[1, 0])
    return {1: ones, 0: zeros, 'ratio': ratio,
            'TN': c[0][0], 'FP': c[0][1], 'FN': c[1][0], 'TP': c[1][1],
            'accuracy': (c[0, 0] + c[1, 1]) / (c[0, 0] + c[0, 1] + c[1, 0] + c[1, 1]),
            'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall),
            'specificity': c[0, 0] / (c[0, 0] + c[0, 1]), 'auc_pr': average_precision_score(y_true.flatten(), y_pred.flatten())}

def get_scores_mean(scores):
    """Calculates average metrics from list of metrics from crossvalidation.

    Args:
        scores (list[dict]): List of metrics.

    Returns:
        dict: Dictionary with averaged metrics.
    """
    mean_score = {}
    for key in scores[0].keys():
        mean_score[key] = sum(d[key] for d in scores) / len(scores)
    return mean_score

def get_times(df, offset, duration):
    """Gets times when sounds occured from data frame with annotations.

    Args:
        df (pd.DataFrame): Data frame with columns: start, end, category.
        offset (float): Time in seconds from which times are taken.
        duration (float): Duration in seconds to take times.

    Returns:
        list[dict]: List of dicts with start and end values in seconds.
    """
    df = df[(df['end'] > offset) & (df['start'] <
                                    offset + duration) & (df['category'] != 'n')]
    df_times = df[['start', 'end']] - offset
    return df_times.to_dict('records')

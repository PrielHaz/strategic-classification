import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from tabulate import tabulate


def metrics_to_tabular_string(metrics):
    formatted_metrics = "\nClassification Metrics\n"
    metrics_data = [
        [metric, value]
        for metric, value in metrics.items()
        if metric not in ["confusion_matrix", "classification_report"]
    ]
    formatted_metrics += tabulate(
        metrics_data, headers=["Metric", "Value"], tablefmt="grid"
    )

    cm = metrics["confusion_matrix"]
    cm_labels = [["TN", "FP"], ["FN", "TP"]]
    formatted_cm = [
        [f"{label}={value}" for label, value in zip(row_labels, row)]
        for row_labels, row in zip(cm_labels, cm)
    ]
    formatted_metrics += "\n\nConfusion Matrix\n"
    formatted_metrics += tabulate(formatted_cm, tablefmt="grid")

    cr_dict = metrics["classification_report"]
    cr_table = []
    for class_name, class_metrics in cr_dict.items():
        if isinstance(class_metrics, dict):
            row = [class_name] + list(class_metrics.values())
            cr_table.append(row)

    formatted_metrics += "\n\nClassification Report\n"
    cr_headers = ["Class"] + list(cr_dict["0"].keys())
    formatted_metrics += tabulate(cr_table, headers=cr_headers, tablefmt="grid")
    formatted_metrics += "\n"
    return formatted_metrics


def calculate_metrics(y_true, logits, threshold):
    y_pred = (logits > threshold).astype(int)
    auc = roc_auc_score(y_true, logits)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cr = classification_report(
        y_true, y_pred, target_names=["0", "1"], output_dict=True
    )
    metrics = {
        "threshold_used": threshold,
        "auc": auc,
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "mcc": mcc,
        "classification_report": cr,
    }
    return metrics


def find_best_threshold_for_mcc(y_true, logits):
    best_threshold = 0.0
    best_mcc = -1
    thresholds = np.arange(0.0, 1.01, 0.01)
    for threshold in thresholds:
        y_pred = (logits > threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return best_threshold, best_mcc


if __name__ == "__main__":
    example_metrics = {
        "threshold_used": 0.5,
        "auc": 0.623659054699947,
        "accuracy": 0.8667763157894737,
        "f1_score": 0.25688073394495414,
        "confusion_matrix": np.array([[513, 25], [56, 14]]),
        "mcc": 0.2000089789569144,
        "classification_report": {
            "0": {
                "precision": 0.9015817223198594,
                "recall": 0.9535315985130112,
                "f1-score": 0.9268292682926829,
                "support": 538.0,
            },
            "1": {
                "precision": 0.358974358974359,
                "recall": 0.2,
                "f1-score": 0.25688073394495414,
                "support": 70.0,
            },
            "accuracy": 0.8667763157894737,
            "macro avg": {
                "precision": 0.6302780406471092,
                "recall": 0.5767657992565056,
                "f1-score": 0.5918550011188185,
                "support": 608.0,
            },
            "weighted avg": {
                "precision": 0.8391104798294236,
                "recall": 0.8667763157894737,
                "f1-score": 0.8496970357197536,
                "support": 608.0,
            },
        },
    }
    print(metrics_to_tabular_string(example_metrics))

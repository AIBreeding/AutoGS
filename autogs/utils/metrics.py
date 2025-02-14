from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, 
    mean_squared_log_error, explained_variance_score
)


class Metric(object):
    def __init__(self, is_clf):
        self.is_clf = is_clf

    def _clf_scores(self, y_true, y_pred):
        scores = {
            "roc_auc_score": roc_auc_score(y_true, y_pred[:, 1]) if len(y_pred.shape) > 1 else None,  # 支持二分类
            "accuracy_score": accuracy_score(y_true, y_pred.argmax(axis=1) if len(y_pred.shape) > 1 else y_pred),
            "precision_score": precision_score(y_true, y_pred.argmax(axis=1) if len(y_pred.shape) > 1 else y_pred, average='weighted'),
            "recall_score": recall_score(y_true, y_pred.argmax(axis=1) if len(y_pred.shape) > 1 else y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred.argmax(axis=1) if len(y_pred.shape) > 1 else y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(y_true, y_pred.argmax(axis=1) if len(y_pred.shape) > 1 else y_pred)
        }
        return {metric: score for metric, score in scores.items() if score is not None}

    def _reg_scores(self, y_true, y_pred):
        scores = {
            "r2_score": r2_score(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "mean_squared_log_error": mean_squared_log_error(y_true, y_pred),
            "explained_variance_score": explained_variance_score(y_true, y_pred)
        }
        return scores

    def score(self, y_true, y_pred):
        if self.is_clf:
            return self._clf_scores(y_true, y_pred)
        else:
            return self._reg_scores(y_true, y_pred)
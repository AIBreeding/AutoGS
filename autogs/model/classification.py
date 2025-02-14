from .model import AutoGS
from .classifiers import classifier_selector
from dataclasses import dataclass
from typing import Callable, List
from sklearn.metrics import accuracy_score, precision_score


@dataclass
class ClassAutoGS(AutoGS):
    """
    AutoGS classification usage
    Optional base classifiers: 
    --------------------
    'Dummy', 'LightGBM', 'XGBoost', 'CatBoost', 'AdaBoost', 
    'GBDT', 'HistGradientBoosting', 'KNN', 'SGD', 'Bagging', 'SVC', 'RF'
    
    """
    __doc__ += AutoGS.__doc__
    
    metric_optimise: Callable = accuracy_score
    models_optimize: List[str] = None
    models_assess: List[str] = None
    _ml_objective: str = 'classification'
    
    def __post_init__(self):
        
        self._stratify = self.y
        super().__post_init__()
        
        if self.metric_assess is None:
            precision_score_macro = [lambda y_true, y_pred: precision_score(y_true, y_pred, average = 'macro')]
            self.metric_assess: List[Callable] = [accuracy_score, precision_score_macro]

        if self.models_optimize is None: 
            self.models_optimize: List[str] = ['svc', 'sgd', 'histgradientboost']
            
        if self.models_assess is None:
            self.models_assess: List[str] = self.models_optimize
            
        n_classes = len(set(self.y_train)) + 1
        self._models_optimize: List[Callable] = classifier_selector(classifier_names=self.models_optimize,
                                                        random_state=self.random_state,  n_classes = n_classes)
        self._models_assess: List[Callable] = classifier_selector(classifier_names=self.models_assess,
                                                      random_state=self.random_state,  n_classes = n_classes)
        
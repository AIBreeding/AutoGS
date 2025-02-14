from .model import AutoGS
from .regressors import regressor_selector
from dataclasses import dataclass
from typing import Callable, List
from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as np


def pearson_correlation(y, y_pred):
    corr, _ = pearsonr(y, y_pred)
    return corr

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@dataclass
class RegAutoGS(AutoGS):
    """
    AutoGS regression usage
    Optional base regressors: 
    --------------------
   'Dummy', 'LightGBM', 'XGBoost', 'CatBoost', 'BayesianRidge', 'LassoLARS', 'AdaBoost', 
   'GBDT', 'HistGradientBoosting', 'KNN', 'SGD', 'Bagging', 'SVR', 'ElasticNet', 'RF'
    
    """
    __doc__ += AutoGS.__doc__
    
    metric_optimise: Callable = r2_score
    models_optimize: List[str] = None
    models_assess: List[str] = None
    _ml_objective: str = 'regression'
    
    def __post_init__(self):
        
        super().__post_init__()
        
        if self.metric_assess is None:
            self.metric_assess: List[Callable] =  [median_absolute_error,r2_score,pearsonr,mean_squared_error,
                                                   pearson_correlation, root_mean_squared_error]

        if self.models_optimize is None: 
            self.models_optimize: List[str] = ['lassolars', 'bayesianridge', 'histgradientboost']
            
        if self.models_assess is None:
            self.models_assess: List[str] = self.models_optimize
            
        self._models_optimize: List[Callable] = regressor_selector(regressor_names=self.models_optimize,
                                                        random_state=self.random_state)
        self._models_assess: List[Callable] = regressor_selector(regressor_names=self.models_assess,
                                                      random_state=self.random_state)
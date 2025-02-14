from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from typing import Union
from .helper import FuncHelper


def decorator_report(variable, to_return_self: bool = False):
    def wrap_function(f):
        def wrap_arguments(args):
            if isinstance(getattr(args, variable), type(None)):
                return args if to_return_self else None
            else:
                return f(args)

        return wrap_arguments

    return wrap_function


class Chooser:
    def __init__(self, arg: any, func: callable, transformer: str, trial=None, **kwargs):  # change to args
        self.arg = arg
        self.func = func
        self.transformer = transformer
        self.trial = trial
        self.func_fitted = None
        self.__dict__.update(kwargs)

    def fit(self):
        self.func_fitted = FuncHelper.run_with_argument(self.func, self.arg)
        return self.func_fitted

    @decorator_report("func_fitted")
    def _report_trial(self):
        self.trial.suggest_categorical(self.transformer, [self.func_fitted.get_params()])
        return self

    def fit_report_trial(self) -> callable:
        self.fit()
        self._report_trial()
        return self.func_fitted

class PcaChooser(Chooser):
    def __init__(self, pca_value: Union[int, float, dict] = None, trial=None):
        super().__init__(arg=pca_value, func=PCA, transformer='pca_value', trial=trial)

class PolyChooser(Chooser):
    def __init__(self, poly_value: Union[int, float, dict] = None, trial=None):
        super().__init__(arg=poly_value, func=PolynomialFeatures, transformer='poly_value', trial=trial)


class SplineChooser(Chooser):
    def __init__(self, spline_value: Union[int, float, dict], trial=None):
        super().__init__(arg=spline_value, func=SplineTransformer, transformer='spline_value', trial=trial)


class ScalerChooser:
    def __init__(self, arg: str = '', transformer: str = 'scaler', trial=None, **kwargs):
        self.arg = arg
        self.transformer = transformer
        self.trial = trial
        self.__dict__.update(kwargs)

    def suggest_trial(self):
        self.arg = self.trial.suggest_categorical(self.transformer, [None, "minmax", "standard", "robust"])
        return self

    def string_to_func(self):
        if self.arg == "minmax":
            self.func = MinMaxScaler()
        elif self.arg == "standard":
            self.func = StandardScaler()
        elif self.arg == "robust":
            self.func = RobustScaler()
        else:
            self.func = None
        return self.func

    def suggest_fit(self):
        self.suggest_trial()
        return self.string_to_func()


class TransformerChooser:
    def __init__(self, n_quantiles: int = None, trial=None, random_state: int = 42):
        self.n_quantiles = n_quantiles
        self.trial = trial
        self.random_state = random_state
        self.func = QuantileTransformer
        self.func_fitted = None

    def suggest_trial(self):
        transform_type = self.trial.suggest_categorical("transformers", [None, 'quantile_trans'])
        if not transform_type == None:
            self.n_quantiles = self.trial.suggest_int('n_quantiles', 100, 4000, step=100)
        return self

    @decorator_report("n_quantiles")
    def fit(self):
        self.func_fitted = FuncHelper.run_with_argument(self.func().set_params, {'n_quantiles': self.n_quantiles,
                                                                                 'random_state': self.random_state})
        return self.func_fitted

    def suggest_and_fit(self):
        self.suggest_trial()
        self.fit()
        return self.func_fitted


class CategoricalChooser:
    def __init__(self, ordinal_columns, categorical_columns):
        self.ordinal_columns = ordinal_columns
        self.categorical_columns = categorical_columns


    def fit(self):
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ordinal_encoder = OrdinalEncoder()
        
        if type(self.ordinal_columns) == type(self.categorical_columns) == list:
            transformers = [
                ("ordinal", ordinal_encoder, self.ordinal_columns),
                ("nominal", one_hot_encoder, self.categorical_columns),
            ]
        elif  type(self.ordinal_columns) != type(self.categorical_columns) == list:
            transformers = [
                ("nominal", one_hot_encoder, self.categorical_columns),
            ]
        elif  type(self.categorical_columns) != type(self.ordinal_columns) == list:
            transformers = [
               ("ordinal", ordinal_encoder, self.ordinal_columns),
            ]
        else:
            self.func_fitted = None
            return self.func_fitted
        
        self.func_fitted = ColumnTransformer(
                transformers=transformers,
                remainder="passthrough", verbose_feature_names_out=False,
            )
        
        return self.func_fitted


#TODO make this less hacky, fix nonsensical "fit" method and resolve returning None
class FourrierExpansion(BaseEstimator, TransformerMixin):
    def __init__(self, fourrier_value: int):
        self.fourrier_value = fourrier_value

    def fit(self, X=None, y=None):
        # Custom transformers that don't need to learn anything
        # from the data can leave the fit method empty.
        if self.fourrier_value == None:
            return None
        else:
            return self
        
    def transform(self, X):
        if self.fourrier_value == None:
            return None
        else:
            x_normalized = 2 * (X-X.min())/(X.max()-X.min()) -1
            x_expand = X
            for i in range(self.fourrier_value):
                x_expand = np.concatenate([x_expand, np.cos(i * x_normalized)], axis = 1)
                x_expand = np.concatenate([x_expand, np.sin(i * x_normalized)], axis = 1)
            return x_expand

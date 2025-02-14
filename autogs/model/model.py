from __future__ import annotations
import os
import sys
import time
import json
import shutil
import pickle
import random
import optuna
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from optuna.samplers import TPESampler
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from ..utils.transformers import *
from ..utils.helper import FuncHelper
from typing import Callable, Union, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.ensemble import StackingRegressor, StackingClassifier

@dataclass
class AutoGS:
    """
    An automated machine learning framework for predicting genotype-by-environment interactions.

    Parameters:
    -----------
    y: pandas.DataFrame
        Phenotype (target) values of shape (n_samples, 1).
    X: pandas.DataFrame
        Features of shape (n_samples, n_features).
    test_size: float, optional (default=0.2)
        Size of the test set.
    timeout: int, optional (default=600)
        Timeout in seconds for optimization of hyperparameters.
    n_trial: int, optional (default=100)
        Number of trials for optimization of hyperparameters.
    n_weak_models: int, optional (default=0)
        Number of models to train AutoGS model on in addition to best model. For each specified
        model the best performing and randomly selected n_weak_models are used for stacking.
        E.g. if n_weak_models = 2 for 'RF', the best performing 'RF' model is used for stacking
        in addition to 2 other 'RF' models. Setting this parameter to non-zero allows the AutoGS model
        to include (unique) additional information from the additional models, despite them performing worse
        independly than the best model
    n_jobs: int, optional (default=1)
        number of simoultaneous threads to run optimisation on. 
    cross_validation: callable, optional (default=KFold with 5 splits and shuffling, random_state=2024)
        The cross-validation object to use for evaluation of models.
    sampler: callable, optional (default=TPESampler with seed=random_state)
        The sampler object to use for optimization of hyperparameters.
    pruner: callable, optional (default=HyperbandPruner with min_resource=1, max_resource='auto', reduction_factor=3)
        The pruner object to prune unpromising training trials.
    poly_value: int, float, dict, optional (default=None)
        The polynomial transformation to apply to the data, if any. E.g. {'degree': 2, 'interaction_only'= False} or 2
    spline_value: int, float, dict, optional (default=None)
        The spline transformation to apply to the data, if any. {'n_knots': 5, 'degree':3} or 5
    fourrier_value: int,
        DEPRECIATED
    pca_value: int, float, dict, optional (default=None).
        The PCA transformation to apply to the data, if any. E.g. {'n_components': 0.95, 'whiten'=False}
    metric_optimise: callable, optional (default=median_absolute_error for regression, accuracy_score for classification)
        The metric optimizing hyperparameters. 
    metric_assess: list of callabloptimization_objectivees, optional (default=[median_absolute_error, r2_score])
        Model evaluation metrics.
    optimization_objective: str, optional (default='minimize')
        Hyperparameter optimization objective ('minimize' or 'maximize').
    write_folder: str, optional (default='/results/' in the current working directory)
        The folder where to write the results.
    reload_study: bool, optional (default=False)
        Whether to continue study if previous study exists in write_folder.
    reload_trial: bool, optional (default=False)
        Upper bound on number of trials if new trials are permitted on reloaded study. For example, if n_trials = 20 and reloaded
        study already performed 15 trials, the new study will at most perform 5 additional trials
    models_optimize: list of str, optional (default=['LightGBM', 'XGBoost', 'GBDT', 'RF'])
        List of model names to optimize.
    models_assess: list of str, optional (default=None)
        The list of names of models to assess. If None, uses the same as `list_optimise`.
    early_stopping_rounds: int, optional (default=None)
        Number of early stopping rounds for 'LightGBM', 'XGBoost' and 'CatBoost'.
    nominal_columns: list of Union[int, float, string)]
        Column headers of input DataFrame. These columns will be treated as containing nominal categorical columns
        Nominal columns contain unranked categories. For example, the Agricultural Management Practices category.
    ordinal_columns: list of Union[int, float, string)]
        Column headers of input DataFrame. These columns will be treated as containing ordinal categorical columns.
        Ordinal columns contain ranked categories, such as hours of the day
    fit_size: list of float, optional (default=[0.1, 0.2, 0.3, 0.4, 0.6, 1])
        The list of sizes of the data to use for fitting the models.
    random_state: int
        The random seed to use, default is 2024.
    warning_verbosity: str
        The warning verbosity to use, default is 'ignore'.

    Methods
    -------
    model_hyperoptimise:
        Performs hyperparameter optimization using the Optuna library. The method contains several
        nested functions and follows a pipeline for training and evaluating a regressor. The method starts by
        preparing the study for hyperparameter optimization and loops through each regressor in the list
        "regressors_2_optimise", optimizes its hyperparameters, and saves the final study iteration as a pickle file.

    model_select_best:
        This method is used to create estimator pipelines for all the regressors specified in models_assess
        attribute and store them in the estimators attribute of the class instance.

    model_evaluate:
        Evaluates performance of selected models. I first trains the models on the training dataset and
        then stacks the models and assesses performance on test fraction of dataset. 
        
    SHAP Calculation:
        Explains feature importance of AutoGS model using SHAP.  

    train:
        applies in correct order 'model_hyperoptimize', 'model_select_best' and 'model_evaluate' methods.

    Returns
    -------
    None

    """

    y: pd.DataFrame
    X: pd.DataFrame
    test_size: float = 0.2
    timeout: int = 600
    n_trial: int = 100
    n_weak_models: int = 0
    n_jobs: int = 1
    cross_validation: callable = None
    n_splits: int = 5
    sampler: callable = None
    pruner: callable = None
    poly_value: Union[int, float, dict, type(None)] = None
    spline_value: Union[int, float, dict, type(None)] = None
    pca_value: Union[int, float, dict, type(None)] = None
    fourrier_value: int = None
    metric_optimise: Callable = None
    metric_assess: List[Callable] = None
    optimization_objective: str = 'maximize'
    write_folder: str = os.getcwd() + '/result/'
    reload_study: bool = False
    reload_trial: bool = False
    overwrite: bool = False
    early_stopping_rounds: int = None
    nominal_columns: Union[List[str], type(None)] = None
    ordinal_columns: Union[List[str], type(None)] = None
    fit_size: List[float] = None
    random_state: Union[int, type(None)] = 2024
    warning_verbosity: str = 'ignore'
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: pd.DataFrame = None
    y_test: pd.DataFrame = None
    train_index: Any = None
    test_index: Any = None
    estimators: List[Callable] = None
    y_pred: Any = None
    summary: Dict[str, List[float]] = None

    _models_optimize: List[Callable] = None
    _models_assess: List[Callable] = None
    _ml_objective: str = None
    _shuffle: bool = True
    _stratify: pd.DataFrame = None
    _model_final = None

    def __post_init__(self):

        self.cross_validation = self.cross_validation if 'split' in dir(self.cross_validation) else \
            KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.sampler = self.sampler if 'optuna.samplers' in type(self.sampler).__module__ else \
            TPESampler(seed=self.random_state)

        self.pruner = self.pruner if 'optuna.pruners' in type(self.pruner).__module__ else \
            optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)

        self.fit_size = [0.1, 0.2, 0.3, 0.4, 0.6, 1] if self.fit_size is None else self.fit_size

        self.create_dir()
        self.split_train_test(shuffle=self._shuffle, stratify=self._stratify)

    def create_dir(self):
        if self.write_folder[-1] != "/": self.write_folder = self.write_folder + "/"

        self.write_folder_sampler = self.write_folder+"samplers/"
        self.write_folder_shap = self.write_folder+"shap/"
        self.write_folder_estimators = self.write_folder+"estimators/"

        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)

        if not os.path.exists(self.write_folder_sampler):
            os.makedirs(self.write_folder_sampler)
            
        if not os.path.exists(self.write_folder_shap):
            os.makedirs(self.write_folder_shap)
            
        if not os.path.exists(self.write_folder_estimators):
            os.makedirs(self.write_folder_estimators)
            
        return self

    # clear log
    def _clean_log(self):
        import os
        import shutil
        cwd = os.getcwd()
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)
        catboost_folders = []
        for f in os.listdir(cwd):
            if f.startswith("catboost"):
                catboost_folders += [f]
        for f in catboost_folders:
            shutil.rmtree(os.path.join(cwd, f))

    def split_train_test(self, shuffle: bool = True, stratify: pd.DataFrame = None ):
        """
        Split the data into training and test sets.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data before splitting, by default True

        Returns
        -------
        None

        The data is split and stored in class attributes.
        """

        if type(self.y) == np.ndarray: self.y = pd.DataFrame(self.y)
        if type(self.X) == np.ndarray: self.X = pd.DataFrame(self.X)
        if type(self.y) == pd.core.series.Series: self.y = self.y.to_frame()
        if type(self.X) == pd.core.series.Series: self.X = self.X.to_frame()
        self.y.columns = self.y.columns.astype(str)
        self.X.columns = self.X.columns.astype(str)

        non_numeric_columns = self.X.apply(lambda x: not pd.api.types.is_numeric_dtype(x))
        non_numeric_column_names = non_numeric_columns[non_numeric_columns].index.to_list()


        if type(self.nominal_columns) == type(self.ordinal_columns) == list:
            submitted_non_numeric = set(self.nominal_columns + self.ordinal_columns)
        elif type(self.nominal_columns) == type(self.ordinal_columns) == type(None):
            submitted_non_numeric = set([])
        elif type(self.nominal_columns) == type(None):
            submitted_non_numeric = set(self.ordinal_columns)
        elif type(self.ordinal_columns) == type(None):
            submitted_non_numeric = set(self.nominal_columns)

        non_numeric_difference = list(set(non_numeric_column_names) ^ submitted_non_numeric)
        if non_numeric_difference != []:
            print(f"Possible ordinal or nominal columns not specified as either: {non_numeric_difference})")

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                        random_state=self.random_state, shuffle=shuffle)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_index = X_train.index.values
        self.test_index = X_test.index.values

        return self


    @FuncHelper.method_warning_catcher
    def model_hyperoptimise(self) -> AutoGS:
        """
        Performs hyperparameter optimization on the models specified in `self.models_assess` using Optuna.
        The optimization is performed on the training data and the final study is saved to disk.
        
        Returns:
            AutoGS: The instance of the class with the updated study information.
            
        Raises:
            CatBoostError: If `catboost` is one of the models in `self.models_assess`, the optimization process
            may raise this error if there is an issue with the `catboost` library.
        """

        def _optimise():
            """
            Optimizes the models specified in the `self.models_assess` dictionary using Optuna.
            The study direction, sampler, and pruner are specified in the `self.optimization_objective`, `self.sampler`, 
            and `self.pruner` attributes respectively. 
            
            The method uses the `_create_objective` function to create the objective function that is optimized by Optuna.
            The final study iteration is saved using joblib.
            """

            if 'catboost' in list(self._models_optimize.keys()):
                import catboost
                catch = (catboost.CatBoostError,)
            else:
                catch = ( )

            best_params_estimator = {}
            
            for model_name, (model, create_params) in self._models_optimize.items():

                dir_study_db = f"{self.write_folder}{model_name}.db"
                dir_study_db_url = f"sqlite:///{dir_study_db}"
                dir_sampler = f"{self.write_folder_sampler}{model_name}_sampler.pkl"

                if os.path.exists(dir_sampler):
                    study_sampler = pickle.load(open(dir_sampler, "rb"))

                    if not self.reload_study:
                        message = [f"Study `{self._ml_objective}_{model_name}` already exists but `reload_study == False` -- > " +
                              "model skipped. \nSet `reload_study = True` to continue on existing study."]

                        FuncHelper.function_warning_catcher(
                            lambda x: print(x, flush=True),
                            message,
                            new_warning_verbosity = 'default',
                            old_warning_verbosity = 'ignore',
                            new_std_error = sys.__stdout__
                            )
                        continue

                else:
                    study_sampler = self.sampler
                    create_engine(dir_study_db_url)

                study = optuna.create_study(study_name=f"{self._ml_objective}_{model_name}",
                                            direction=self.optimization_objective,
                                            sampler=study_sampler,
                                            pruner=self.pruner,
                                            storage = dir_study_db_url,
                                            load_if_exists = self.reload_study)

                if (self.reload_study) & (self.reload_trial):
                    n_trials = self.n_trial - len(study.trials)
                    if n_trials <= 0:
                        continue
                else:
                    n_trials = self.n_trial

                study.optimize(_create_objective(study, create_params, model, model_name, dir_sampler),
                                      n_trials=n_trials, timeout=self.timeout, catch=catch, n_jobs=self.n_jobs)

                best_params_estimator[model_name] = study.best_params
            with open(os.path.join(self.write_folder, "best_params.json"), "w") as f:
                json.dump(best_params_estimator, f, indent=4)
            
            return


        def _create_objective(study, create_params, model, model_name, dir_sampler):
            """
            Method creates the objective function that is optimized by Optuna. The objective function first saves
            the Optuna study and instantiates the scaler for the independent variables. Then, it determines if the
            feature combinations improve the results, and if so, fits the SplineChooser and PolyChooser. Next, it
            instantiates PCA compression and the transformer for the dependent variables. Finally, the method tunes
            the estimator algorithm and creates the model.
            """

            def _objective(trial):
               
                scaler = ScalerChooser(trial=trial).suggest_fit()

                optionals_included = any([bool(i) for i in [self.spline_value, self.poly_value]])

                spline_input = None
                poly_input = None
                
                if optionals_included:
                    feature_combo = trial.suggest_categorical("feature_combo", [False, True])
                    if feature_combo:
                        spline_input = self.spline_value
                        poly_input = self.poly_value

                spline = SplineChooser(spline_value=spline_input, trial=trial).fit_report_trial()
                poly = PolyChooser(poly_value=poly_input, trial=trial).fit_report_trial()
                pca = PcaChooser(pca_value=self.pca_value, trial=trial).fit_report_trial()
                param = create_params(trial)
                model_with_parameters = model().set_params(**param)
                categorical = CategoricalChooser(self.ordinal_columns, self.nominal_columns).fit()
                if self._ml_objective == 'regression':
                    transformer = TransformerChooser(random_state=self.random_state, trial=trial).suggest_and_fit()

                    model_final = TransformedTargetRegressor(
                        regressor=model_with_parameters,
                        transformer=transformer
                    )
                elif self._ml_objective == 'classification':
                    model_final = model_with_parameters

                pipeline = Pipeline([
                    ('categorical', categorical),
                    ('poly', poly),
                    ('spline', spline),
                    ('scaler', scaler),
                    ('pca', pca),
                    ('model', model_final)
                    ])

                performance = _model_performance(trial, model_name, pipeline)

                with open(dir_sampler, "wb") as sampler_state:
                    pickle.dump(study.sampler, sampler_state)

                return performance

            return _objective

        def _model_performance(trial, model_name, pipeline) -> float:
            """
            Evaluates the performance of the `pipeline` model. The performance is evaluated by splitting the data into
            K-folds and iteratively training and assessing the model using an increasing fraction of the training and
            test folds. If the performance for the first iterations is poor, the model is pruned.
            """
            indexes_train_kfold = list(self.cross_validation.split(self.X_train))

            fractions = [self.fit_size[-1]] if trial.number == 0 else self.fit_size

            result_folds_fracs = []
            result_folds_stds = []

            for idx_fraction, partial_fit_size in enumerate(fractions):

                min_samples = int(np.ceil(len(self.X_train) * partial_fit_size * 1 / self.cross_validation.n_splits))
                if min_samples < 20:
                    continue

                result_folds = []

                for idx_fold, fold in enumerate(indexes_train_kfold):

                    fold_X_train = self.X_train.iloc[fold[0]]
                    fold_X_test = self.X_train.iloc[fold[1]]
                    fold_y_train = self.y_train.iloc[fold[0]]
                    fold_y_test = self.y_train.iloc[fold[1]]

                    idx_partial_fit_train = fold_X_train.sample(frac=partial_fit_size,
                                                                              random_state=self.random_state).index
                    idx_partial_fit_test = fold_X_test.sample(frac=partial_fit_size,
                                                                            random_state=self.random_state).index

                    fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
                    fold_X_test_size = fold_X_test.loc[idx_partial_fit_test]
                    fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
                    fold_y_test_size = fold_y_test.loc[idx_partial_fit_test]

                    early_stopping_permitted = bool(
                        set([model_name]) & set(['xgboost', 'catboost'])) 

                    if early_stopping_permitted: 

                        pipeline[:-1].fit_transform(fold_X_train_frac)

                        fold_X_test_size_transformed = pipeline[:-1].transform(fold_X_test_size)

                        pipeline.fit(fold_X_train_frac, fold_y_train_frac,
                                      model__eval_set=[(fold_X_test_size_transformed, fold_y_test_size)],
                                      model__early_stopping_rounds = self.early_stopping_rounds)

                    else:

                        pipeline.fit(fold_X_train_frac, fold_y_train_frac)

                    try:
                        prediction = pipeline.predict(fold_X_test_size)

                        result_fold = self.metric_optimise(fold_y_test_size, prediction)
                        pass

                    except Exception as e:
                        print(e)
                        result_fold = np.nan
                        pass

                    result_folds.append(result_fold)
                    
                # result_file_path = f"{self.write_folder}result_folds_results.csv" #改
                # result_folds = pd.DataFrame(result_folds)
                # result_folds.to_csv(result_file_path, index=True) #改              
                result_folds_frac = np.mean(result_folds)
                result_folds_std = np.std(result_folds)

                result_folds_fracs.append(result_folds_frac)
                result_folds_stds.append(result_folds_std)

                if partial_fit_size < 1.0:

                    trial.report(result_folds_frac, idx_fraction)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

            performance = result_folds_fracs[-1]
            
            return performance

        if bool(self._models_optimize):
            _optimise()

            return 

    def model_select_best(self, random_state_model_selection=None) -> AutoGS:
        """
        This method is used to create estimator pipelines for all the models specified in models_assess
        attribute and store them in the estimators attribute of the class instance.

        The method loads the study result for each model from the file with name "{model_name}.pkl" in
        write_folder directory. Then it instantiates objects of SplineChooser, PolyChooser, PcaChooser, ScalerChooser
        and TransformerChooser classes using the best parameters obtained from the study result. Next, it creates a
        pipeline using the Pipeline class from scikit-learn library. Each pipeline per model is added to a list of
        pipelines, which is then assigned to the estimators attribute of the class instance.

        Returns
        -------
        class instance.
        """

        # prepare all estimators for stacking
        estimators = []
        for model_name in list(self._models_assess.keys()):

            # set randomness parameters for randomly selecting models (if self.n_weak_models > 0)
            if type(random_state_model_selection) == type(None):
                random_state_model_selection = self.random_state
            random.seed(random_state_model_selection)

            # reload relevant study. Sampler not reloaded here as no additional studies are performed
            study = optuna.create_study(
                study_name=f"{self._ml_objective}_{model_name}",
                direction=self.optimization_objective,
                storage=f"sqlite:///{self.write_folder}{model_name}.db",
                load_if_exists=True)

            # select parameters corresponding to model
            list_params = list(study.best_params)
            list_params_not_model = ['scaler', 'pca_value', 'spline_value', 'poly_value', 'feature_combo',
                                         'transformers', 'n_quantiles']
            list_params_model = set(list_params).difference(set(list_params_not_model))

            # select all trials associated with model
            df_trials = study.trials_dataframe()
            df_trials_non_pruned = df_trials[df_trials.state == 'COMPLETE']

            # ensure that selected number of weak models does not exceed `total completed trials` - `best trial`
            n_weak_models = self.n_weak_models
            if self.n_weak_models > len(df_trials_non_pruned) -1:

                message = ["Number of unique weak models less than requested number of weak models: " +
                           f"{len(df_trials_non_pruned) -1} < {self.n_weak_models} \n" +
                           "n_weak_models set to total number of weak models instead."]
                print(message[0], flush=True)

                n_weak_models = len(df_trials_non_pruned) -1

            # select best models
            if self.optimization_objective == 'maximize':
                idx_best = df_trials_non_pruned.index[df_trials_non_pruned.value.argmax()]
            elif self.optimization_objective == 'minimize':
                idx_best = df_trials_non_pruned.index[df_trials_non_pruned.value.argmin()]

            # add additional models
            idx_remaining = df_trials_non_pruned.number.values.tolist()
            idx_remaining.remove(idx_best)
            idx_models = [idx_best] + random.sample(idx_remaining, n_weak_models)

            # name best and weaker models
            selected_models = [model_name+'_best']  + [model_name+'_'+str(i) for i in idx_models[1:]]

            # create estimator for best and additional weaker models
            for i, idx_model in enumerate(idx_models):

                model_params = study.trials[idx_model].params
                parameter_dict = {k: model_params[k] for k in model_params.keys() & set(list_params_model)}

                # select all the pipeline steps corresponding to input settings or best trial
                categorical = CategoricalChooser(self.ordinal_columns, self.nominal_columns).fit()
                spline = SplineChooser(spline_value=model_params.get('spline_value')).fit()
                poly = PolyChooser(poly_value=model_params.get('poly_value')).fit()
                pca = PcaChooser(pca_value=model_params.get('pca_value')).fit()
                scaler = ScalerChooser(arg=model_params.get('scaler')).string_to_func()

                model_with_parameters = self._models_assess[model_name][0](**parameter_dict)

                # Create transformed regressor
                if self._ml_objective == 'regression':
                    transformer = TransformerChooser(model_params.get('n_quantiles'), self.random_state).fit()

                    model_final = TransformedTargetRegressor(
                        regressor=model_with_parameters,
                        transformer=transformer
                    )
                # or normal classification model
                elif self._ml_objective == 'classification':
                    model_final = model_with_parameters

                pipe_single_study = Pipeline([
                    ('categorical', categorical),
                    ('poly', poly),
                    ('spline', spline),
                    ('scaler', scaler),
                    ('pca', pca),
                    ('model', model_final)]
                )
                # Fit the pipeline
                pipe_single_study.fit(self.X_train, self.y_train)
                
                estimators.append((selected_models[i], pipe_single_study))
                
                write_file_estimator = self.write_folder_estimators + f"{selected_models[i]}_model.joblib"

                # if file doesn't exist, write it
                if not os.path.isfile(write_file_estimator):
                    joblib.dump(pipe_single_study, write_file_estimator)
                else:
                    answer = input(f"File {write_file_estimator} already exists. Overwrite (y/n)?")
                    if answer == 'y':
                        joblib.dump(pipe_single_study, write_file_estimator)
                    elif answer == 'n':
                        pass
                    else:
                        print("Invalid input: file not saved")

        self.estimators = estimators
        self.list_all_models_assess = [estimator[0] for estimator in estimators]

        return


    def model_evaluate(self) -> AutoGS:
        """
        Model evaluation method of an estimator.

        This method will evaluate the model performance of the estimators specified in 'models_assess' by
        splitting the test data into folds according to the cross-validation specified, training the estimators on the
        training data and evaluating the predictions on the test data. The performance will be stored in a dictionary
        per metric per estimator. If the estimator is the AutoGS model, it will be saved to disk.

        Returns
        -------
        class instance
        """
        
        # check whether split_train_test method has been performed, else perform it
        if getattr(self, 'X_train') is None: self.split_train_test()

        # split data according to cross validation for assessment
        indexes_test_cv = list(self.cross_validation.split(self.X_test))

        # determine names of models to assess
        models_assess = self.list_all_models_assess + ['AutoGS']

        # create an empty dictionary to populate with performance while looping over models
        summary = dict([(model, list()) for model in models_assess])

        # Dynamically generate metric names based on the metric_assess functions
        metric_names = [metric.__name__ for metric in self.metric_assess]
        
        result_file_path = f"{self.write_folder}evaluation_results.txt"
        with open(result_file_path, "w", encoding="utf-8") as f:
            for i, model in enumerate(models_assess):
                estimator_temp = self.estimators[i:i + 1]

                # the final model is the AutoGS model
                if i == len(self.estimators):
                    estimator_temp = self.estimators

                    # create a scorer compatible with Cross Validated Ridge
                    greater_is_better = self.optimization_objective == 'maximize'
                    scoring = make_scorer(
                        self.metric_optimise, 
                        greater_is_better = greater_is_better
                        )

                    # fit AutoGS model while catching warnings
                    if self._ml_objective == 'regression':
                        self._model_final = StackingRegressor(
                            estimators=estimator_temp,
                            final_estimator=RidgeCV(scoring=scoring),
                            cv=self.cross_validation
                            )

                    elif self._ml_objective == 'classification':
                        self._model_final = StackingClassifier(
                            estimators=estimator_temp,
                            final_estimator=RidgeClassifierCV(scoring=scoring),
                            cv=self.cross_validation
                            )

                    FuncHelper.function_warning_catcher(self._model_final.fit, [self.X_train, self.y_train],
                                                        self.warning_verbosity)

                    # predict on the whole testing dataset
                    self.y_pred = self._model_final.predict(self.X_test)

                    # store AutoGS model, if file already exists, confirm overwrite
                    write_file_AutoGS_model = self.write_folder + "AutoGS_model.joblib"

                    if os.path.isfile(write_file_AutoGS_model):
                        question = "AutoGS model already exists in directory. Overwrite ? (y/n):"
                        user_input = input(len(question) * '_' + '\n' + question + '\n' + len(question) * '_' + '\n')

                        if user_input != 'n':
                            response = "AutoGS model overwritten"
                            joblib.dump(self._model_final, write_file_AutoGS_model)
                        else:
                            response = "AutoGS model not saved"

                        print(len(response) * '_' + '\n' + response + '\n' + len(response) * '_'  + '\n')

                    # if file doesn't exist, write it
                    if not os.path.isfile(write_file_AutoGS_model):
                        joblib.dump(self._model_final, write_file_AutoGS_model)

                else:
                    self._model_final = estimator_temp[0][1]
                    FuncHelper.function_warning_catcher(self._model_final.fit, [self.X_train, self.y_train],
                                                        self.warning_verbosity)

                # Initialize metric_performance_dict for each model
                metric_performance_dict = {
                    f'metric_{i}': [metric, []] for i, metric in enumerate(self.metric_assess)
                }
                    
                # Print and save model results
                f.write(f"\n🔍 Model: {model}\n" + "-"*50 + "\n")
                print(f"\n🔍 Model: {model}\n" + "-"*50)
                pred_results = pd.DataFrame()
                for idx_fold, fold in enumerate(indexes_test_cv):
                    # Select the fold indexes
                    fold_test = fold[1]
                    # Predict on the TEST data fold
                    prediction = self._model_final.predict(self.X_test.iloc[fold_test, :])
                    true_values = self.y_test.iloc[fold_test]
                    prediction = np.squeeze(prediction)
                    true_values = np.squeeze(true_values)

                    fold_result = pd.DataFrame({
                        'SampleID': self.X_test.iloc[fold_test].index,  # 假设样本ID是测试集索引
                        'TrueValues': true_values,
                        'Predictions': prediction,
                        'Fold': idx_fold + 1  # 折痕编号
                    })
                    
                    pred_results = pd.concat([pred_results, fold_result], ignore_index=True)                    
                    
                    # Assess prediction per metric and store per-fold performance in dictionary
                    fold_results = {}
                    f.write(f"  Fold {idx_fold + 1}:\n")
                    print(f"  Fold {idx_fold + 1}:")
                    for metric_name, key in zip(metric_names, metric_performance_dict.keys()):
                        fold_value = metric_performance_dict[key][0](true_values, prediction)
                        fold_results[metric_name] = fold_value
                        f.write(f"    {metric_name:<25}: {fold_value:.4f}\n")
                        print(f"    {metric_name:<25}: {fold_value:.4f}")

                        # Add fold results to list
                        metric_performance_dict[key][1].append(fold_value)

                # Collect the mean and std deviation for each metric
                f.write(f"\n  Summary for {model}:\n")
                print(f"\n  Summary for {model}:")
                for metric_name, key in zip(metric_names, metric_performance_dict.keys()):
                    mean_value = np.mean(metric_performance_dict[key][1])
                    std_value = np.std(metric_performance_dict[key][1])
                    # Use emoji or symbols to enhance clarity
                    status_icon = "✅" if mean_value < 1 else "✅"
                    f.write(f"    {metric_name:<25}: {mean_value:.4f} ± {std_value:.4f} {status_icon}\n")
                    print(f"    {metric_name:<25}: {mean_value:.4f} ± {std_value:.4f} {status_icon}")
                    
                results_file_path = f"{self.write_folder}pred_results.csv"
                pred_results.to_csv(results_file_path, index=False)                  

                 
                # store mean and standard deviation of performance over folds per model
                # summary[model] = [
                #     [f"{metric_name}±STD: {np.mean(metric_performance_dict[key][1]):.4f}±{np.std(metric_performance_dict[key][1]):.4f}" for metric_name, key in zip(metric_names, metric_performance_dict.keys())]
                # ]

            # self.summary = summary

        return self
    

    def train(self):
            start_time = time.time()
            self.model_hyperoptimise()
            self.model_select_best()
            self.model_evaluate()
            end_time = time.time()
            print(f"Total time taken: {end_time - start_time:.4f} seconds")

            return

    
    def CalSHAP(self, n_train_points = 200, n_test_points = 200, cluster = True):
        """

        Parameters
        ----------
        n_train_points: int, default=200
            number of training observations (or clusters) for to use in explaining the model
        n_test_points: int, default=200
            number of test observations for which to assess feature importance
        cluster: bool, default=True
            whether to cluster the training data. If not individual points are chosen to create explainer 
            
        Returns
        -------
        shap_values:
            Shapely values calculated for data
        data:
            Subset of test data on which shapely values are calculated
        
        
        """

        import shap
        
        # reload the final model if it exists
        if type(self._model_final) is type(None):
            try:
                self._model_final = joblib.load(f"{self.write_folder}AutoGS_model.joblib")
            except:
                raise Exception(f"No trained model available in write_folder: {self.write_folder}")

        # cluster the training data to speed up, otherwise randomly sample training data
        if cluster:
            print('Clustering...')
            kmeans = KMeans(n_clusters = n_train_points, n_init = 10).fit(self.X_train)
            X_train_summary = kmeans.cluster_centers_
            
        else:
            X_train_summary = self.X_train.sample(n = n_train_points, random_state = self.random_state)
            
        
        # create explainer based on clustered training data
        explainer = FuncHelper.function_warning_catcher(shap.SamplingExplainer, [self._model_final.predict, X_train_summary],#LinearExplainer,KernelExplainer,self._model_final.predict
                                            self.warning_verbosity)
        
        # select subset of test data 
        data = self.X_test.sample(n = n_test_points, random_state = self.random_state)
        feature_names = list(self.X_test.keys())
        label = self.y_test.sample(n = n_test_points, random_state = self.random_state)
 
        # calculate SHAP values
        print('Calculating Shapely values...', flush=True)
        shap_values = FuncHelper.function_warning_catcher(explainer.shap_values, [data],
                                                          self.warning_verbosity)

        data_file_path = os.path.join(self.write_folder_shap, 'fore_data.csv')
        label_file_path = os.path.join(self.write_folder_shap, 'fore_data_label.csv')
        shap_values_file_path = os.path.join(self.write_folder_shap, 'shap_values.npy')
        feature_names_path = os.path.join(self.write_folder_shap, 'feature_names.npy')

        data.to_csv(data_file_path, index=True)
        label.to_csv(label_file_path, index=True)
        np.save(shap_values_file_path, shap_values)
        np.save(feature_names_path, feature_names)
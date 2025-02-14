def classifier_selector(classifier_names, n_classes , random_state = None, ):
    """
    Function to load classifier only when selected by the user. 
    This prevents the entire function from becoming unuseable when one or multiple of the classifier or not installed
    
    Parameters
    ----------
    classifier_names : list of str, classifier names with the following options:
        'Dummy', 'LightGBM', 'XGBoost', 'CatBoost', 'AdaBoost', 
        'GBDT', 'HistGradientBoosting', 'KNN', 'SGD', 'Bagging', 'SVC', 'RF'
    n_classes
    random_state : int, reproduceability state for classifier with randomization options
    
    Returns
    -------
    classifier: dict, classifier names as keys corresponding to tuple of classifier function and hyperparameter optimization function
    
    """

    classifiers = ['Dummy', 'LightGBM', 'XGBoost', 'CatBoost', 'AdaBoost','GBDT', 'HistGradientBoosting','KNN', 'SGD', 'Bagging', 'SVC', 'RF']
    selected_classifiers = list(set(classifier_names) & set(classifiers))
    if selected_classifiers == []:
        return print('no valid classifier names provided')
    classifiers_dict = {}
    classifiers_dict['Dummy'] = dummy_loader() if 'Dummy' in selected_classifiers else None
    classifiers_dict['LightGBM'] = lightgbm_loader(n_classes = n_classes, random_state = random_state) if 'LightGBM' in selected_classifiers else None
    classifiers_dict['XGBoost'] = xgboost_loader(random_state = random_state) if 'XGBoost' in selected_classifiers else None
    classifiers_dict['CatBoost'] = catboost_loader(random_state = random_state) if 'CatBoost' in selected_classifiers else None
    classifiers_dict['AdaBoost'] = adaboost_loader(random_state = random_state) if 'AdaBoost' in selected_classifiers else None
    classifiers_dict['GBDT'] = gradientboost_loader(random_state = random_state) if 'GBDT' in selected_classifiers else None
    classifiers_dict['HistGradientBoosting'] = histgradientboost_loader(random_state = random_state) if 'HistGradientBoosting' in selected_classifiers else None
    classifiers_dict['KNN'] = knn_loader() if 'KNN' in selected_classifiers else None
    classifiers_dict['SGD'] = sgd_loader(random_state = random_state) if 'SGD' in selected_classifiers else None
    classifiers_dict['Bagging'] = bagging_loader(random_state = random_state) if 'Bagging' in selected_classifiers else None
    classifiers_dict['SVC'] = svc_loader(random_state = random_state) if 'SVC' in selected_classifiers else None
    classifiers_dict['RF'] = randomforest_loader(random_state=random_state) if 'RF' in selected_classifiers else None    

    method_dict_none_rem = {k: v for k, v in classifiers_dict.items() if v is not None}
    
    index_map = {v: i for i, v in enumerate(classifier_names)}
    method_dict_none_rem_sorted = dict(sorted(method_dict_none_rem.items(), key=lambda pair: index_map[pair[0]]))
    
    return method_dict_none_rem_sorted
      

def dummy_loader():

    from sklearn.dummy import DummyClassifier
    def dummyHParams(trial):
        param_dict = {}
        param_dict['strategy'] = trial.suggest_categorical("strategy", ['most_frequent', 'stratified', 'uniform'])
        return param_dict
    
    return (DummyClassifier, dummyHParams)

def lightgbm_loader(random_state, n_classes):
    
    from lightgbm import LGBMClassifier
    
    def lightgbmHParams(trial):
        param_dict = {}
        if n_classes > 1:
            param_dict['objective'] = trial.suggest_categorical("objective", ['multiclass'])
        elif n_classes == 1:
            param_dict['objective'] = trial.suggest_categorical("objective", ['binary'])
        param_dict['num_classes'] = trial.suggest_categorical("num_classes", [n_classes])
        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 20)
        param_dict['n_estimators'] = trial.suggest_int('n_estimators', 50, 2000, log = True)
        param_dict['min_split_gain'] = trial.suggest_float("min_split_gain", 0, 15)  # boosts speed, decreases performance though
        param_dict['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 10.0, log = True)
        param_dict['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 10.0, log = True)
        param_dict['num_leaves'] = trial.suggest_int('num_leaves', 2, 256)
        param_dict['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
        param_dict['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 100)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        param_dict['verbosity'] = trial.suggest_categorical("verbosity", [-1])
        return param_dict
    
    return (LGBMClassifier, lightgbmHParams)


def xgboost_loader(random_state):
    
    from xgboost import XGBClassifier
    
    def xgboostHParams(trial):
        param_dict = {}
        param_dict['booster'] = trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart'])
        param_dict['lambda'] = trial.suggest_float("lambda", 1e-8, 10.0, log = True)
        param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 10.0, log = True)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        param_dict['verbosity'] = trial.suggest_categorical("verbosity", [0])
        
        if (param_dict['booster'] == 'gbtree') or (param_dict['booster'] == 'dart') :
            
            param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 14, log = False)  
            
            if (param_dict['max_depth'] >= 12) :
                max_n_estimators = 200; min_eta = 1e-2
            elif (param_dict['max_depth'] >= 10) :
                max_n_estimators = 300; min_eta = 1e-3 
            else :
                max_n_estimators = 400; min_eta = 1e-4
                
            param_dict['n_estimators'] = trial.suggest_int("n_estimators", 20, max_n_estimators, log=False) 
            param_dict['eta'] = trial.suggest_float("eta", min_eta, 1.0, log = True)   
            param_dict['min_child_weight'] = trial.suggest_float("min_child_weight", 0, 10, log = False)
            param_dict['gamma'] = trial.suggest_float("gamma", 0, 10, log = False)
            param_dict['subsample'] = trial.suggest_float("subsample", 0.1, 1.0, log = False)
            param_dict['colsample_bytree'] = trial.suggest_float("colsample_bytree", 0.1, 1.0, log = False)
            param_dict['max_bin'] = trial.suggest_categorical("max_bin", [64, 128, 256, 512, 1024])    # performance boost when power of 2 (NOT -1)
        
            if (param_dict['booster'] == 'dart') :
                param_dict['sample_type'] = trial.suggest_categorical("sample_type", ['uniform', 'weighted'])
                param_dict['normalize_type'] = trial.suggest_categorical("normalize_type", ['tree', 'forest'])
                param_dict['rate_drop'] = trial.suggest_float("rate_drop", 0., 1.0, log = False)
                param_dict['one_drop'] = trial.suggest_categorical("one_drop", [0, 1])
        return param_dict
    
    return (XGBClassifier, xgboostHParams)


def catboost_loader(random_state):
    
    from catboost import CatBoostClassifier
    
    def catboostHParams(trial):
        param_dict = {}
        param_dict['depth'] = trial.suggest_int("depth", 1, 10, log = False) 
        
        if (param_dict['depth'] >= 8) :
            max_iterations = 300; min_learning_rate = 1e-2
        elif (param_dict['depth'] >= 6) :
            max_iterations = 400; min_learning_rate = 5e-3
        else :
            max_iterations = 500; min_learning_rate = 1e-3
                
        param_dict['iterations'] = trial.suggest_int("iterations", 20, max_iterations, log = True)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", min_learning_rate, 1e0, log = True)  
        param_dict['l2_leaf_reg'] = trial.suggest_float("l2_leaf_reg", 1e-2, 1e1, log = True)
        param_dict['rsm'] = trial.suggest_float("rsm", 1e-2, 1e0, log = False)
        param_dict['logging_level'] = trial.suggest_categorical("logging_level", ['Silent'])
        param_dict['random_seed'] = trial.suggest_categorical("random_seed", [random_state])
     
        auto_class_weights = trial.suggest_categorical("auto_class_weights", [None, 'Balanced', 'SqrtBalanced'])
        if auto_class_weights != None:
            param_dict['auto_class_weights'] = auto_class_weights
        
        
        return param_dict
    
    return (CatBoostClassifier, catboostHParams)
    

def adaboost_loader(random_state):
    
    from sklearn.ensemble import AdaBoostClassifier
    
    def adaBoostHParams(trial):
        param_dict = {}
        param_dict['n_estimators'] =trial.suggest_int("n_estimators", 10, 200)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-2, 1e0, log = True)
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    return (AdaBoostClassifier, adaBoostHParams)


def gradientboost_loader(random_state):
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    def gradBoostHParams(trial):
        param_dict = {}
        
        param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 10)
        if (param_dict['max_depth'] >= 8) :
            max_iterations = 200; min_learning_rate = 5e-2
        elif (param_dict['max_depth'] >= 5) :
            max_iterations = 300; min_learning_rate = 1e-2
        else :
            max_iterations = 400; min_learning_rate = 5e-3
            
        param_dict['n_estimators'] =trial.suggest_int("n_estimators", 10, max_iterations)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", min_learning_rate, 1e0, log = True)
        param_dict['subsample'] = trial.suggest_float("subsample", 1e-2, 1.0, log = False)
        param_dict['criterion'] = trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error'])
        param_dict['n_iter_no_change'] = trial.suggest_categorical("n_iter_no_change", [20])
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    return (GradientBoostingClassifier, gradBoostHParams)


def histgradientboost_loader(random_state):
    
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    def histGradBoostHParams(trial):
        param_dict = {}
        param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 20, log = False)
        param_dict['max_iter'] = trial.suggest_int("max_iter", 10, 500, log = True)
        param_dict['max_leaf_nodes'] = trial.suggest_int("max_leaf_nodes", 2, 100)
        param_dict['min_samples_leaf'] = trial.suggest_int("min_samples_leaf", 2, 200)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1.0, log = True)
        param_dict['n_iter_no_change'] = trial.suggest_categorical("n_iter_no_change", [20]) 
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    return (HistGradientBoostingClassifier, histGradBoostHParams)


def knn_loader():
    
    from sklearn.neighbors import KNeighborsClassifier
    
    def KNearNeighboursHParams(trial):
        param_dict = {}
        param_dict['n_neighbors'] = trial.suggest_int("n_neighbors", 1, 101, step = 5)
        param_dict['weights'] = trial.suggest_categorical("weights", ['uniform', 'distance'])
        return param_dict
    
    return (KNeighborsClassifier, KNearNeighboursHParams)


def sgd_loader(random_state):
    
    from sklearn.linear_model import SGDClassifier
    
    def sgdHParams(trial):
        param_dict = {}
        param_dict['loss'] =  trial.suggest_categorical("loss", ['squared_error', 
                                                                 'huber', 'epsilon_insensitive', 
                                                                 'squared_epsilon_insensitive',
                                                                 'modified_huber', 'squared_hinge', 
                                                                 'perceptron', 
                                                                 ])
        param_dict['penalty'] = trial.suggest_categorical("penalty", ['l2', 'l1', 'elasticnet'])
        param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
        if (param_dict['penalty'] == 'elasticnet') :
            param_dict['l1_ratio'] = trial.suggest_float("l1_ratio", 0, 1, log = False)
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        
        return param_dict
    
    return (SGDClassifier, sgdHParams)


def bagging_loader(random_state):
    
    from sklearn.ensemble import BaggingClassifier
    
    def baggingHParams(trial):
        param_dict = {}
        param_dict['n_estimators'] =  trial.suggest_int("n_estimators", 1, 101, step = 5)
        param_dict['max_samples'] = trial.suggest_float("max_samples", 1e-1, 1.0, step = 0.1)
        param_dict['max_features'] = trial.suggest_float("max_features", 1e-1, 1.0, step = 0.1)
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        param_dict['bootstrap'] =  trial.suggest_categorical("bootstrap", [True, False])
        param_dict['bootstrap_features'] =  trial.suggest_categorical("bootstrap_features", [True, False])
        
        return param_dict
    
    return (BaggingClassifier, baggingHParams)


def svc_loader(random_state):
    
    from sklearn.svm import LinearSVC
    
    def svcHParams(trial):
        param_dict = {}
        param_dict['loss'] = trial.suggest_categorical("loss", ['hinge', 'squared_hinge'])
        param_dict['dual'] = trial.suggest_categorical("dual", ['auto'])
        
        param_dict['penalty'] = trial.suggest_categorical("penalty", ['l2'])
        param_dict['C'] = trial.suggest_float("C", 1e-5, 1e2, log = True)
        param_dict['tol'] = trial.suggest_float("tol", 1e-8, 1e2, log = True)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    return (LinearSVC, svcHParams)
    


def randomforest_loader(random_state):
    from sklearn.ensemble import RandomForestClassifier

    def randomForestHParams(trial):
        param_dict = {}
        param_dict['n_estimators'] = trial.suggest_int('n_estimators', 10, 101, step = 5)
        param_dict['max_depth'] = trial.suggest_int('max_depth', 2, 32, step = 2)
        param_dict['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20, step = 2)
        param_dict['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 20, step = 2)
        param_dict['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    return (RandomForestClassifier, randomForestHParams)

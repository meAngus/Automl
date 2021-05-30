from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, r2_score, average_precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from dabl.preprocessing import detect_types
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from collections import defaultdict
import optuna
import warnings

warnings.filterwarnings('ignore')


class auto_model:

    def __init__(self,
                 X, y, model=None, params=None, problem_type=None, n_trials=10, timeout=600, gc_after_trial=True):
        '''initialize model

        Args:
            X: input values for training
            y: predicted values
            model: model name to train (xgboost, random forest, lightgbm, ebm, catboost)
            params: defined parameters for training
            problem_type: user define problem type, auto detection will be applied if None is set
            n_trials: number of trials
            timeout: stop study after the given number of seconds.
            gc_after_trial: Flag to execute garbage collection at the end of each trial
        '''

        self._X = X
        self._y = y
        self._model_name = model
        self.n_trials = n_trials
        self.timeout = timeout
        self.gc = gc_after_trial

        self._problem_type = problem_type
        self.estimator = self.__build_function

    @property
    def model_name(self):

        if not self._model_name:
            self._model_name = 'random forest'
        return self._model_name.lower()

    @property
    def problem_type(self):

        if not self._problem_type:
            self._problem_type = type_of_target(self._y)

        if self._problem_type == 'regression':
            self._problem_type = 'continuous'
        return self._problem_type

    @property
    def __build_function(self):

        def objective_xgboost(trial):
            X, y = self._X, self._y

            train_x, valid_x, train_y, valid_y = train_test_split(X.values, y.values, test_size=0.25)

            param = {
                "verbosity": 0,
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 3),
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
                "colsample_bylevel:": trial.suggest_float("colsample_bylevel", 0.2, 0.9),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 0.9),
            }

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

            if self.problem_type == 'multiclass':

                bst = XGBClassifier(**param, objective='multi:softprob').fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)
                score = roc_auc_score(valid_y, pred, average='weighted', multi_class='ovr')

            elif self.problem_type == 'binary':

                bst = XGBClassifier(**param, objective='binary:logistic').fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)[:, 1]
                score = roc_auc_score(valid_y, pred)

            else:

                bst = XGBRegressor(**param, objective='reg:squarederror').fit(train_x, train_y)
                score = -mean_squared_error(valid_y, bst.predict(valid_x), squared=False)

            return score

        def objective_lightgbm(trial):
            train_x, valid_x, train_y, valid_y = train_test_split(self._X, self._y, test_size=0.25)

            param = {
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 0.9),
            }

            if self.problem_type == 'multiclass':

                bst = LGBMClassifier(**param, objective='multi:softmax').fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)
                score = roc_auc_score(valid_y, pred, average='weighted', multi_class='ovr')

            elif self.problem_type == 'binary':

                bst = LGBMClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)[:, 1]
                score = roc_auc_score(valid_y, pred)

            else:

                bst = LGBMRegressor(**param).fit(train_x, train_y)
                score = -mean_squared_error(valid_y, bst.predict(valid_x), squared=False)

            return score

        def objective_catboost(trial):

            train_x, valid_x, train_y, valid_y = train_test_split(self._X, self._y, test_size=0.25)

            param = {
                "verbose": 0,
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "depth": trial.suggest_int("depth", 1, 12),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            }

            if param["bootstrap_type"] == "Bayesian":
                param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

            elif param["bootstrap_type"] == "Bernoulli":
                param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            if self.problem_type == 'multiclass':

                param["eval_metric"] = "AUC"
                bst = CatBoostClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)
                score = roc_auc_score(valid_y, pred, average='weighted', multi_class='ovr')

            elif self.problem_type == 'binary':

                param["eval_metric"] = "AUC"
                bst = CatBoostClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)[:, 1]
                score = roc_auc_score(valid_y, pred)

            else:

                bst = CatBoostRegressor(**param).fit(train_x, train_y)
                score = -mean_squared_error(valid_y, bst.predict(valid_x), squared=False)

            return score

        def objective_rf(trial):
            X, y = self._X, self._y
            X.fillna(0, inplace=True)

            train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)

            param = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                # "criterion": trial.suggest_categorical("criterion",["gini","entropy"]),
                "max_depth": trial.suggest_int("max_depth", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 25),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1),
                "max_features": trial.suggest_float("max_features", 0.5, 1),
            }

            if self.problem_type == 'multiclass':

                bst = RandomForestClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)
                score = roc_auc_score(valid_y, pred, average='weighted', multi_class='ovr')

            elif self.problem_type == 'binary':

                bst = RandomForestClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)[:, 1]
                score = roc_auc_score(valid_y, pred)

            if self.problem_type == 'continuous':
                bst = RandomForestRegressor(**param).fit(train_x, train_y)
                score = -mean_squared_error(valid_y, bst.predict(valid_x), squared=False)

            return score

        def objective_ebm(trial):
            X, y = self._X, self._y
            X.fillna(0, inplace=True)

            train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)

            param = {
                "max_rounds": trial.suggest_int("max_rounds", 100, 10000),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_leaves": trial.suggest_int("max_leaves", 2, 20),
                "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1)}

            if self.problem_type == 'multiclass':

                bst = ExplainableBoostingClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)
                score = roc_auc_score(valid_y, pred, average='weighted', multi_class='ovr')

            elif self.problem_type == 'binary':

                bst = ExplainableBoostingClassifier(**param).fit(train_x, train_y)
                pred = bst.predict_proba(valid_x)[:, 1]
                score = roc_auc_score(valid_y, pred)

            else:
                bst = ExplainableBoostingRegressor(**param).fit(train_x, train_y)
                score = -mean_squared_error(valid_y, bst.predict(valid_x), squared=False)

            return score

        objectives = {'xgboost': objective_xgboost,
                      'lightgbm': objective_lightgbm,
                      'catboost': objective_catboost,
                      'random forest': objective_rf,
                      'ebm': objective_ebm}

        study = optuna.create_study(direction="maximize")
        study.optimize(objectives[self.model_name], n_trials=self.n_trials, timeout=self.timeout,
                       gc_after_trial=self.gc)

        if self.problem_type in ('multiclass', 'binary'):
            '''refit process'''

            if self.model_name == "xgboost":
                model = XGBClassifier(**study.best_trial.params).fit(self._X.values, self._y.values)

            elif self.model_name == "lightgbm":
                model = LGBMClassifier(**study.best_trial.params).fit(self._X, self._y)

            elif self.model_name == 'catboost':
                model = CatBoostClassifier(**study.best_trial.params, verbose=0).fit(self._X, self._y)

            elif self.model_name == 'random forest':
                X, y = self._X, self._y
                X.fillna(0, inplace=True)
                model = RandomForestClassifier(**study.best_trial.params).fit(X, y)

            elif self.model_name == 'ebm':
                X, y = self._X, self._y
                X.fillna(0, inplace=True)
                model = ExplainableBoostingClassifier(**study.best_trial.params).fit(X, y)

        elif self.problem_type == 'continuous':

            if self.model_name == "xgboost":
                model = XGBRegressor(**study.best_trial.params).fit(self._X.values, self._y.values)

            elif self.model_name == "lightgbm":
                model = LGBMRegressor(**study.best_trial.params).fit(self._X, self._y)

            elif self.model_name == 'catboost':
                model = CatBoostRegressor(**study.best_trial.params, verbose=0).fit(self._X, self._y)

            elif self.model_name == 'random forest':
                X, y = self._X, self._y
                X.fillna(0, inplace=True)
                model = RandomForestRegressor(**study.best_trial.params).fit(X, y)

            elif self.model_name == 'ebm':
                X, y = self._X, self._y
                X.fillna(0, inplace=True)
                model = ExplainableBoostingRegressor(**study.best_trial.params).fit(X, y)

        return model

    def predict(self, X):
        '''prediction on X

        Args:
            X: instances to predict

        returns:
            predicted values
        '''

        if self.model_name in ('ebm', 'random forest'):
            X = X.fillna(0)

        if self.model_name == 'xgboost':
            X = X.values

        return self.estimator.predict(X)

    def predict_proba(self, X):
        '''prediction on X

        Args:
            X: instances to predict

        returns:
            predicted values
        '''

        if self.model_name in ('ebm', 'random forest'):
            X = X.fillna(0)

        if self.model_name == 'xgboost':
            X = X.values

        return self.estimator.predict_proba(X)
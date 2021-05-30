from automl import auto_model
from explainer import model_explainer
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def test_regressors():
    df = pd.read_csv('~/hnw_51')
    df = df.astype('float')
    df = df.sample(1000)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    X = df.drop(['TOT_NON_RB_LQD_AST_AMT'], axis=1)
    y = df['TOT_NON_RB_LQD_AST_AMT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    ebm = auto_model(X_train, y_train, 'ebm', problem_type='regression', n_trials=1)
    xgb = auto_model(X_train, y_train, 'xgboost', problem_type='regression', n_trials=1)
    lgb = auto_model(X_train, y_train, 'lightgbm', problem_type='regression', n_trials=1)
    rf = auto_model(X_train, y_train, 'random forest', problem_type='regression', n_trials=1)
    cat = auto_model(X_train, y_train, 'catboost', problem_type='regression', n_trials=1)
    default = auto_model(X_train, y_train, problem_type='regression', n_trials=1)

    assert isinstance(ebm.estimator, ExplainableBoostingRegressor)
    assert isinstance(xgb.estimator, XGBRegressor)
    assert isinstance(lgb.estimator, LGBMRegressor)
    assert isinstance(rf.estimator, RandomForestRegressor)
    assert isinstance(cat.estimator, CatBoostRegressor)
    assert isinstance(default.estimator, RandomForestRegressor)

    assert ebm.problem_type == 'continuous'
    assert ebm.problem_type == 'continuous'
    assert ebm.problem_type == 'continuous'
    assert ebm.problem_type == 'continuous'
    assert ebm.problem_type == 'continuous'
    assert ebm.problem_type == 'continuous'


def test_classifiers():
    df = pd.read_csv('~/hnw_51')
    df = df.astype('float')
    df = df.sample(1000)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    X = df.drop(['TOT_NON_RB_LQD_AST_AMT'], axis=1)
    labels = [2, 1, 0]
    splits = [-1, 5000.0, 10000.0, float("inf")]
    y = pd.cut(df['TOT_NON_RB_LQD_AST_AMT'], bins=splits, labels=labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33, stratify=y)

    ebm = auto_model(X_train, y_train, 'ebm', n_trials=1)
    xgb = auto_model(X_train, y_train, 'xgboost', n_trials=1)
    lgb = auto_model(X_train, y_train, 'lightgbm', n_trials=1)
    rf = auto_model(X_train, y_train, 'random forest', n_trials=1)
    cat = auto_model(X_train, y_train, 'catboost', n_trials=1)
    default = auto_model(X_train, y_train, n_trials=1)

    assert isinstance(ebm.estimator, ExplainableBoostingClassifier)
    assert isinstance(xgb.estimator, XGBClassifier)
    assert isinstance(lgb.estimator, LGBMClassifier)
    assert isinstance(rf.estimator, RandomForestClassifier)
    assert isinstance(cat.estimator, CatBoostClassifier)
    assert isinstance(default.estimator, RandomForestClassifier)

    assert ebm.problem_type == 'multiclass'
    assert xgb.problem_type == 'multiclass'
    assert lgb.problem_type == 'multiclass'
    assert rf.problem_type == 'multiclass'
    assert cat.problem_type == 'multiclass'
    assert default.problem_type == 'multiclass'


def test_explainers():
    df = pd.read_csv('~/hnw_51')
    df = df.astype('float')
    df = df.sample(1000)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    X = df.drop(['TOT_NON_RB_LQD_AST_AMT'], axis=1)
    y = df['TOT_NON_RB_LQD_AST_AMT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    ebm = auto_model(X_train, y_train, 'ebm', problem_type='regression', n_trials=1)
    xgb = auto_model(X_train, y_train, 'xgboost', problem_type='regression', n_trials=1)
    lgb = auto_model(X_train, y_train, 'lightgbm', problem_type='regression', n_trials=1)
    rf = auto_model(X_train, y_train, 'random forest', problem_type='regression', n_trials=1)
    cat = auto_model(X_train, y_train, 'catboost', problem_type='regression', n_trials=1)

    cat_explainer = model_explainer(cat)
    rf_explainer = model_explainer(rf)
    xgb_explainer = model_explainer(xgb)
    ebm_explainer = model_explainer(ebm)
    lgb_explainer = model_explainer(lgb)

    assert ebm_explainer.check_fill_zero == True
    assert rf_explainer.check_fill_zero == True
    assert lgb_explainer.check_fill_zero == False
    assert cat_explainer.check_fill_zero == False
    assert xgb_explainer.check_fill_zero == False

    assert 'RMSE Score' in cat_explainer.perf(X_test, y_test)
    assert 'AUC Score' not in cat_explainer.perf(X_test, y_test)
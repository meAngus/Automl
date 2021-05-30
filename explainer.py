from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from interpret.blackbox import ShapKernel, LimeTabular, PartialDependence
from interpret import show
from interpret.perf import ROC, RegressionPerf
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, r2_score, average_precision_score, \
    classification_report
import numpy as np
import pandas as pd
from interpret.data import Marginal, ClassHistogram
from alibi.explainers.ale import ALE, plot_ale
from sklearn.utils.multiclass import type_of_target
from collections import defaultdict
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

set_visualize_provider(InlineProvider())
import warnings

warnings.filterwarnings('ignore')


def combined_features_rank(explainers, key=0, num_toshow=3):
    ''' get top features'''

    hm = defaultdict(int)

    for explainer in explainers:

        s = list(zip(explainer.data(key=key)['names'], explainer.data(key=key)['scores']))

        s.sort(key=lambda x: -abs(x[1]))

        new_s = list(filter(lambda x: x[1] > 0, s))

        for i, v in enumerate(new_s):
            hm[v[0]] += len(X_train) - i

    ret = list()

    for i, v in sorted(hm.items(), key=lambda d: -d[1]):
        ret.append(i)

    return ret[:num_toshow]


class model_explainer:
    ''' provide model explanation'''

    def __init__(self, model):

        '''initialize explainer

        Args:
            model: the model to be explained
        '''

        self.model = model.estimator or model
        self.X = model._X
        self.y = model._y
        self.problem_type = model.problem_type

    @property
    def check_fill_zero(self):

        autofill_models = (
        XGBRegressor, XGBClassifier, LGBMClassifier, LGBMRegressor, CatBoostClassifier, CatBoostRegressor)

        if isinstance(self.model, autofill_models):
            return False
        return True

    def explain_local(self, X_instance, y_instance, method='shap', visualize=True):
        ''' local explanation of the model

        Args:
            X_instance: Sample data for local explanation
            y_instance: sample lables for local explanation
            method: The methodology applied for explanation(shap,lime and ebm are supported)

        '''
        X = self.X.fillna(0)
        X_instance = X_instance.fillna(0)

        if (isinstance(self.model, (ExplainableBoostingClassifier, ExplainableBoostingRegressor))):
            ebm_local = self.model.explain_local(X_instance, y_instance, name='EBM')

            if visualize:
                show(ebm_local)

            return ebm_local

        if self.problem_type in ('binary', 'multiclass'):
            predict_fn = self.model.predict_proba

        else:
            predict_fn = self.model.predict

        if method.lower() == 'shap':
            background_val = np.median(X, axis=0).reshape(1, -1)
            shap = ShapKernel(predict_fn=predict_fn, data=background_val, feature_names=self.X.columns)
            shap_local = shap.explain_local(X_instance, y_instance, name='SHAP')

            if visualize:
                show(shap_local)

            return shap_local

        elif method.lower() == 'lime':
            lime = LimeTabular(predict_fn=predict_fn, data=X, random_state=1)
            lime_local = lime.explain_local(X_instance, y_instance, name='LIME')

            if visualize:
                show(lime_local)

            return lime_local

        else:
            raise ValueError('Currently only ebm, lime and shap are supported for local explanation')

    def explain_global(self, method='ale', ale_figsize=[10, 20]):
        ''' global explanation of the model

        Args:
        method: The methodology applied for the global explanation(pdp, ale are supported),
                and ebm explains itself.
        ale_figsize: the figure size of ale plots for better visualizations

        '''

        X = self.X.fillna(0)

        if (isinstance(self.model, (ExplainableBoostingClassifier, ExplainableBoostingRegressor))):
            ebm_global = self.model.explain_global(name='EBM')
            show(ebm_global)
            return

        if self.problem_type in ('binary', 'multiclass'):
            predict_fn = self.model.predict_proba

        else:
            predict_fn = self.model.predict

        if method.lower() == 'pdp':
            pdp = PartialDependence(predict_fn=predict_fn, data=X)
            pdp_global = pdp.explain_global(name='Partial Dependence')
            show(pdp_global)

        elif method.lower() == 'ale':
            ale = ALE(predict_fn, feature_names=self.X.columns, target_names=['target'])
            exp = ale.explain(np.array(X))
            plot_ale(exp, fig_kw={'figwidth': ale_figsize[0], 'figheight': ale_figsize[1]})

        else:
            raise ValueError('Currently only pdp, ale and ebm are supported for global explanation')

    def perf(self, x_test, y_test):
        ''' Performance evaluation of the model

        Args:
        x_test: test data to evaluate performance
        y_test: test labels to evaluate performance

        '''

        if self.check_fill_zero:
            x_test = x_test.fillna(0)

        if self.problem_type == 'binary':
            return {'AUC Score': roc_auc_score(y_test, self.model.predict_proba(x_test)[:, 1], average='weighted'),
                    'MAP Score': average_precision_score(y_test, self.model.predict(x_test))}

        elif self.problem_type == 'multiclass':
            return {'AUC Score': roc_auc_score(y_test, self.model.predict_proba(x_test), average='weighted',
                                               multi_class='ovr'),
                    'MAP Score': average_precision_score(y_test, self.model.predict(x_test), average='weighted')}

        else:
            return {'RMSE Score': mean_squared_error(y_test, self.model.predict(x_test), squared=False)}


def voted_top_features(explainers, key=0, num_toshow=3):
    ''' get top features'''

    hm = defaultdict(int)

    for explainer in explainers:

        s = list(zip(explainer.data(key=key)['names'], explainer.data(key=key)['scores']))

        s.sort(key=lambda x: -abs(x[1]))

        new_s = list(filter(lambda x: x[1] > 0, s))

        for i, v in enumerate(new_s):
            hm[v[0]] += len(explainer.data(key=0)['values']) - i

    ret = list()

    for i, v in sorted(hm.items(), key=lambda d: -d[1]):
        ret.append(i)

    return ret[:num_toshow]
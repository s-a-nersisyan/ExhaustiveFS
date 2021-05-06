import numpy as np

from sklearn.model_selection import \
    StratifiedKFold, \
    GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC

from core.regression.regressors import CoxRegression
from core.utils import check_if_func_accepts_arg


class Model:
    def __init__(self, model, kwargs, random_state):
        """Class constructor

          Parameters
          ----------
          model : sklearn-like
              Model class, should have fit and
              predict methods. Most of the sklearn models will be suitable.
          kwargs : dict
              Dict of keyword arguments for model initialization.
          random_state : int
              Random seed (set to an arbitrary integer for reproducibility).
        """
        self.model = model
        self.model_kwargs = kwargs
        if check_if_func_accepts_arg(self.model.__init__, 'random_state'):
            self.model_kwargs['random_state'] = random_state

        self.random_state = random_state

    def get_best_cv_model(
        self,
        X_train,
        y_train,
        scoring_functions,
        main_scoring_function,
        cv_ranges,
        cv_folds,
    ):
        """Search for best model considered passed cross-validation parameters

        Returns
        -------
        tuple
            Best model and its parameters.
        """
        # For ROC AUC and SVM we should pass probability=True argument
        if 'ROC_AUC' in scoring_functions and self.model == SVC:
            self.model_kwargs['probability'] = True

        if cv_ranges:
            model = self.model(**self.model_kwargs)

            splitter = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            scoring = {
                s: make_scorer(
                    scoring_functions[s],
                    needs_proba=self.check_if_method_needs_proba(s),
                )
                for s in scoring_functions
            }
            searcher = GridSearchCV(
                model,
                cv_ranges,
                scoring=scoring,
                cv=splitter,
                refit=False
            )
            searcher.fit(X_train, y_train)

            all_params = searcher.cv_results_['params']
            mean_test_scorings = {
                s: searcher.cv_results_['mean_test_' + s]
                for s in scoring_functions
            }
            best_ind = np.argmax(mean_test_scorings[main_scoring_function])
            best_params = {
                param: all_params[best_ind][param]
                for param in all_params[best_ind]
            }
        else:
            best_params = {}

        # Refit model with best parameters
        model = self.model(**self.model_kwargs, **best_params)

        return model, best_params

    @staticmethod
    def check_if_method_needs_proba(method):
        """Check if method needs special treatment like probability prediction.

        Returns
        -------
        bool
        """
        return method in ['ROC_AUC']

    def check_if_model_needs_numpy(self):
        """Check if model fit accepts numpy instead of DataFrame.

        Returns
        -------
        bool
        """

        return self.model not in [CoxRegression]

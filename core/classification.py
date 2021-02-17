import pandas as pd
import numpy as np

from multiprocessing import Pool
import time
import math
import itertools

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer


class ExhaustiveClassification:
    def __init__(
        self, df, ann, n_k,
        feature_pre_selector, feature_pre_selector_kwargs,
        feature_selector, feature_selector_kwargs,
        preprocessor, preprocessor_kwargs,
        classifier, classifier_kwargs,
        classifier_CV_ranges, classifier_CV_folds,
        scoring_functions, main_scoring_function, main_scoring_threshold,
        n_processes=1, random_state=None, verbose=True
    ):
        """Class constructor
        
        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame whose rows represent samples
            and columns represent features.
        ann : pandas.DataFrame
            DataFrame with annotation of samples. Three columns are mandatory:
            Class (binary labels), Dataset (dataset identifiers) and 
            Dataset type (Training, Filtration, Validation).
        n_k : pandas.DataFrame
            DataFrame with columns n and k defining a grid
            for exhaustive feature selection: n is a number
            of selected features, k is a length of each
            features subset.
        feature_pre_selector : callable
            Function for feature pre-selection. For examples, see
            feature_pre_selectors.py.
        feature_pre_selector_kwargs : dict
            Dict of keyword arguments for feature pre-selector.
        feature_selector : callable
            Function for feature selection. For examples, see
            feature_selectors.py.
        feature_selector_kwargs : dict
            Dict of keyword arguments for feature selector.
        preprocessor : sklearn.preprocessing-like
            Class for data preprocessing, should have fit and
            transform methods. Any method from sklearn.preprocessing
            will be suitable.
        preprocessor_kwargs : dict
            Dict of keyword arguments for preprocessor initialization.
        classifier : sklearn.classifier-like
            Class for classification, should have fit and
            predict methods. Any sklearn classifier will be suitable.
        classifier_kwargs : dict
            Dict of keyword arguments for classifier initialization.
        classifier_CV_ranges : dict
            Dict defining classifier parameters which should be
            cross-validated. Keys are parameter names, values are
            iterables for grid search.
        classifier_CV_folds : int
            Number of fold for K-Folds cross-validation.
        scoring_functions : dict
            Dict with scoring functions which will be calculated
            for each classifier. Keys are names (arbitrary strings),
            values are sklearn.metrics-like callables (should accept
            y_true, y_pred arguments and return a score).
        main_scoring_function : str
            Key from scoring_functions dict defining the "main" scoring
            function which will be optimized during cross-validation
            and will be used for classifier filtering.
        main_scoring_threshold : float
            A number defining threshold for classifier filtering: 
            classifiers with score below this threshold on 
            training/filtration sets will not be further evaluated.
        n_processes : int
            Number of processes.
        random_state : int
            Random seed (set to an arbitrary integer for reproducibility).
        verbose : bool
            If True, print running time for each pair of n, k.
        """

        self.df = df
        self.ann = ann
        self.n_k = n_k

        self.n_processes = n_processes
        self.random_state = random_state
        self.verbose = verbose

        self.feature_pre_selector = feature_pre_selector
        self.feature_pre_selector_kwargs = feature_pre_selector_kwargs

        self.feature_selector = feature_selector
        self.feature_selector_kwargs = feature_selector_kwargs

        self.preprocessor = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs

        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs
        self.classifier_CV_ranges = classifier_CV_ranges
        self.classifier_CV_folds = classifier_CV_folds

        self.scoring_functions = scoring_functions
        self.main_scoring_function = main_scoring_function
        self.main_scoring_threshold = main_scoring_threshold

    def exhaustive_run(self):
        """Run the pipeline for classifier construction 
        using exhaustive feature selection.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with constructed classifiers and their
            quality scores.
        """

        # Pre-select features
        if self.feature_pre_selector:
            pre_selected_features = self.feature_pre_selector(
                self.df, 
                self.ann, 
                **self.feature_pre_selector_kwargs
            )
        else:
            pre_selected_features = self.df.columns.to_list()

        # Iterate over n, k pairs
        all_result_dfs = []
        for n, k in zip(self.n_k["n"], self.n_k["k"]):
            # Do feature selection
            features = self.feature_selector(
                self.df[pre_selected_features], 
                self.ann, 
                n, 
                **self.feature_selector_kwargs
            )
            
            # Split feature subsets to chunks for multiprocessing
            feature_subsets = list(itertools.combinations(features, k))
            chunk_size = math.ceil(len(feature_subsets) / self.n_processes)
            process_args = []
            for i in range(self.n_processes):
                start = chunk_size * i
                end = chunk_size * (i + 1) if i < self.n_processes - 1 else len(feature_subsets)
                process_args.append(feature_subsets[start:end])
            
            # Run exhaustive search in multiple processes
            start_time = time.time()
            with Pool(self.n_processes) as p:
                process_results = p.map(self.exhaustive_run_over_chunk, process_args, chunksize=1)
            end_time = time.time()
            
            if self.verbose:
                print("Pipeline iteration finished in {} seconds for n={}, k={} (n_processes = {})".format(end_time - start_time, n, k, self.n_processes))
            
            # Merge results
            df_n_k_results = pd.concat(process_results, axis=0)
            df_n_k_results["n"] = n
            df_n_k_results["k"] = k
            all_result_dfs.append(df_n_k_results)

        res = pd.concat(all_result_dfs, axis=0)
        res.index.name = "features"
        res["n"] = res["n"].astype(int)
        res["k"] = res["k"].astype(int)
        return res

    def exhaustive_run_over_chunk(self, args):
        """Run the pipeline for classifier construction
        using exhaustive feature selection over chunk of
        feature subsets
        
        Parameters
        ----------
        args : tuple
            Two-element tuple, containing DataFrame with
            n features (i.e. after feature selection) and 
            a list of feature subsets.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with constructed classifiers and their
            quality scores.
        """

        feature_subsets = args

        results = []
        for features_subset in feature_subsets:
            features_subset = list(features_subset)
            classifier, best_params, preprocessor = self.fit_classifier(features_subset)
            scores, filtration_passed = self.evaluate_classifier(classifier, preprocessor, features_subset)

            item = {"Features subset": features_subset, "Best parameters": best_params, "Scores": scores}
            if filtration_passed:
                results.append(item)
        
        score_cols = ["{};{}".format(dataset, s) for dataset in np.unique(self.ann["Dataset"]) for s in self.scoring_functions]
        parameter_cols = list(self.classifier_CV_ranges)

        df_results = pd.DataFrame(columns=score_cols + parameter_cols)
        for item in results:
            index = ";".join(item["Features subset"])
            for dataset in item["Scores"]:
                for s in item["Scores"][dataset]:
                    df_results.loc[index, "{};{}".format(dataset, s)] = item["Scores"][dataset][s]
            
            for parameter in parameter_cols:
                df_results.loc[index, parameter] = item["Best parameters"][parameter]

        return df_results
    
    def fit_classifier(self, features_subset):
        """Fit classifier given features subset
        
        Parameters
        ----------
        features_subset : list
            list of features which should be used for
            classifier fitting.
        
        Returns
        -------
        sklearn.classifier-like, sklearn.preprocessing-like
            Classifier and preprocessor fitted on the
            training set.
        """
        # Extract training set
        X_train = self.df.loc[self.ann["Dataset type"] == "Training", features_subset].to_numpy()
        y_train = self.ann.loc[self.ann["Dataset type"] == "Training", "Class"].to_numpy()

        # Fit preprocessor and transform training set
        if self.preprocessor:
            preprocessor = self.preprocessor(**self.preprocessor_kwargs)
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
        else:
            preprocessor = None

        # Fit classifier with CV search of unknown parameters
        classifier = self.classifier(random_state=self.random_state, **self.classifier_kwargs)

        splitter = StratifiedKFold(
                n_splits=self.classifier_CV_folds, 
                shuffle=True, 
                random_state=self.random_state
        )
        searcher = GridSearchCV(
            classifier,
            self.classifier_CV_ranges,
            scoring={s: make_scorer(self.scoring_functions[s]) for s in self.scoring_functions},
            cv=splitter,
            refit=False
        )
        searcher.fit(X_train, y_train)

        all_params = searcher.cv_results_["params"]
        mean_test_scorings = {s: searcher.cv_results_["mean_test_" + s] for s in self.scoring_functions}
        best_ind = np.argmax(mean_test_scorings[self.main_scoring_function])
        best_params = {param: all_params[best_ind][param] for param in all_params[best_ind]}

        # Refit classifier with best parameters
        classifier = self.classifier(random_state=self.random_state, **self.classifier_kwargs, **best_params)
        classifier.fit(X_train, y_train)

        return classifier, best_params, preprocessor
    
    def evaluate_classifier(self, classifier, preprocessor, features_subset):
        """Evaluate classifier given features subset
        
        Parameters
        ----------
        classifier : sklearn.classifier-like
            Fitted classifier object with a method predict(X).
        preprocessor : sklearn.preprocessing-like
            Fitted preprocessing object with a method transform(X) .
        features_subset : list
            list of features which should be used for
            classifier evaluation.
        
        Returns
        -------
        dict, bool
            Dict with scores for each dataset and
            boolean value indicating whether a 
            classifier passed given threshold on
            training and filtration sets.
        """
        scores = {}
        filtration_passed = True
        for dataset, dataset_type in self.ann[["Dataset", "Dataset type"]].drop_duplicates().to_numpy():
            X_test = self.df.loc[self.ann["Dataset"] == dataset, features_subset].to_numpy()
            y_test = self.ann.loc[self.ann["Dataset"] == dataset, "Class"].to_numpy()

            # Normalize dataset using preprocessor fitted on training set
            if preprocessor:
                X_test = preprocessor.transform(X_test)
            # Make predictions
            y_pred = classifier.predict(X_test)
            
            if 'ROC_AUC' in self.scoring_functions.keys():
                y_score = classifier.predict_proba(X_test)

            scores[dataset] = {}
            for s in self.scoring_functions:
                if ( s == 'ROC_AUC' ):
                    scores[dataset][s] = self.scoring_functions[s](y_test, y_score[:, 1])
                else:
                    scores[dataset][s] = self.scoring_functions[s](y_test, y_pred)
                    
            if (
                dataset_type in ["Training", "Filtration"] and
                scores[dataset][self.main_scoring_function] < self.main_scoring_threshold
            ):
                filtration_passed = False
        
        return scores, filtration_passed
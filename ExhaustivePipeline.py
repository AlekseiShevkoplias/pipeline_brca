import itertools
import sklearn.preprocessing
import numpy as np

class ExhaustivePipeline:
    def __init__(
        self, df, n_k, n_threads,
        feature_pre_selector, feature_pre_selector_kwargs,
        feature_selector, feature_selector_kwargs,
        preprocessor, preprocessor_kwargs,
        classifier, classifier_kwargs
    ):
        '''
        df: pandas dataframe. Rows represent samples, columns represent features (e.g. genes).
        df should also contain three columns:
            -  "Class": binary values associated with target variable;
            -  "Dataset": id of independent dataset;
            -  "Dataset type": "Training", "Filtration" or "Validation".

        n_k: pandas dataframe. Two columns must be specified:
            -  "n": number of features for feature selection
            -  "k": tuple size for exhaustive search
        '''

        self.df = df
        self.n_k = n_k
        self.n_threads = n_threads

        self.feature_pre_selector = feature_pre_selector
        self.feature_pre_selector_kwargs = feature_pre_selector_kwargs

        self.feature_selector = feature_selector
        self.feature_selector_kwargs = feature_selector_kwargs

        self.preprocessor = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs

        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs

    def run(self):
        # First, pre-select features
        features = self.feature_pre_selector(self.df, **self.feature_pre_selector_kwargs)
        df_pre_selected = self.df[features + ["Class", "Dataset", "Dataset type"]]
        datasets = pd.unique(self.df.Dataset)
        # Start iterating over n, k pairs
        for n, k in zip(self.n_k["n"], self.n_k["k"]):
            features = self.feature_selector(df_pre_selected, n, **self.feature_selector_kwargs)
            df_selected = df_pre_selected[features + ["Class", "Dataset", "Dataset type"]]

            # TODO: this loop should be run in multiple processes
            for feature_subset in itertools.combinations(features, k):
                df_train = df_selected.loc[df_selected["Dataset type"] == "Training", feature_subset + ["Class"]]
                X_train = df_train.drop(columns=["Class"]).to_numpy()
                y_train = df_train["Class"].to_numpy()

                self.preprocessor.fit(X_train, **self.preprocessor_kwargs)
                X_train = self.preprocessor.transform(X_train)

                clf = SVC(kernel="linear", class_weight="balanced")
                best_scores, best_params = utils.grid_cv(clf, X_train, y_train, tuned_parameters, scoring, 1)

                clf = SVC(kernel="linear", class_weight="balanced", C=best_params[0], probability=True)
                clf.fit(X_train, y_train)
                hits = 0
                for dataset in datasets:
                    df_test = df_selected.loc[df_selected["Dataset"] == dataset, feature_subset + ["Class"]]
                    X_test = df_train.drop(columns=["Class"]).to_numpy()
                    y_test = df_train["Class"].to_numpy()
                    X_test = self.preprocessor.transform(X_test)

                    results = utils.test_classifier(clf, X_test, y_test)
                    if len(list(filter(lambda x: x >= 0.65, results))) == 4:
                        hits += 1

                    print("\t".join([data[i][3]] + ["{:.2f}".format(v) for v in results]), file=f_out)

                print("\t".join(
                    ["Summary"] + gs_subset + [str(hits)] + pr_subset + ["{:.2f}".format(v) for v in clf.coef_[0]]),
                      file=f_out)
                print("", file=f_out, flush=True)

                # TODO: train classifier (with cross-validated parameters)
                # TODO: evaluate performance on EACH dataset and store results


def feature_pre_selector_template(df, **kwargs):
    '''
    Input expression dataframe, return list of features
    TODO: special function which load genes from specified file
    '''
    pass


def feature_selector_template(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    pass




class Preprocessor_template:
    '''
    This class should have three methods:
        -  __init__
        -  fit
        -  transform
    Any sklearn classifier preprocessor be suitable
    '''

    def __init__(self, **kwargs):
        self.scaler = preprocessing.StandardScaler(**kwargs)

    def fit(self, X):
        return self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)


class Classifier_template:
    '''
    This class should have three methods:
        -  __init__
        -  fit
        -  predict
    Any sklearn classifier will be suitable
    '''
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

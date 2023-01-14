from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def rf_algo(x_train, y_train, x_test, y_test, test_df, process):
    rf = RfClassifier(x_train, y_train, process)
    clf = rf.test_model(x_test, y_test)
    if process == "file":
        return rf.predict_data_file(clf, test_df)
    else:
        return rf.predict_data_row(clf, test_df)


def use_random_search(clf, x_train, y_train):
    max_depth = [int(x) for x in np.linspace(2, 100, num=15)]
    max_depth.append(None)
    random_grid = {
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=9000, num=20)],
        "max_features": ["log2", "sqrt"],
        "max_depth": max_depth,
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        "bootstrap": [True, False],
    }
    pprint(random_grid)
    # Use the random grid to search for best hyperparameters
    rf_random = RandomizedSearchCV(
        estimator=clf,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(x_train, y_train)

    pprint(rf_random.best_params_)
    # First result with randomizedSearchCV
    # {
    # 'bootstrap': False,
    #  'max_depth': None,
    #  'max_features': 'sqrt',
    #  'min_samples_leaf': 1,
    #  'min_samples_split': 2,
    #  'n_estimators': 9000
    #  }


def use_gridSearch(clf, x_train, y_train):
    param_grid = {
        "bootstrap": [False],
        "max_depth": [None],
        "max_features": ["sqrt"],
        "min_samples_leaf": [1, 2, 3, 4],
        "min_samples_split": [1, 2, 3, 4],
        "n_estimators": [8900, 9000, 8950, 9050],
    }
    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2
    )
    grid_search.fit(x_train, y_train)
    pprint(grid_search.best_params_)
    # Best params
    # {'bootstrap': False,
    #  'max_depth': None,
    #  'max_features': 'sqrt',
    #  'min_samples_leaf': 1,
    #  'min_samples_split': 2,
    #  'n_estimators': 8950}


class RfClassifier:
    def __init__(self, x_train, y_train, process):
        self.x_train = x_train
        self.y_train = y_train
        self.process = process

    def test_model(self, x_test, y_test):
        clf = RandomForestClassifier(
            bootstrap=False,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            n_estimators=8950,
            random_state=42
        )
        clf = clf.fit(self.x_train, self.y_train)
        # use_random_search(clf, self.x_train, self.y_train)
        # use_gridSearch(clf, self.x_train, self.y_train)
        # Prediction for the model
        y_pred = clf.predict(x_test)
        if self.process == "file":
            print("Testing model for file data preprocessing")
        else:
            print("Testing model for row data preprocessing")
        print("Accuracy of random forest model:  %.3f" % accuracy_score(y_test, y_pred))
        f, ax = plt.subplots(1, 1, figsize=(30, 30))
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            normalize=None,
        )
        plt.title("Confusion matrix", fontsize=28)
        plt.ylabel("True label", fontsize=25)
        plt.xlabel("Predicted label", fontsize=25)
        tick_marks = np.arange(20)
        plt.xticks(tick_marks, range(1, 21))
        plt.yticks(tick_marks, range(1, 21))
        plt.savefig(f"./images/confusion_matrix_rf_{self.process}.png", dpi=75, format="png")
        plt.show()
        return clf

    def predict_data_file(self, clf, test_df):
        # We use the "clf" classifier and we predict the "data".
        # We temporarily remove the "file" column during predictions
        predictions = clf.predict(test_df[test_df.columns.drop(["file"])])
        # We have the predictions for each row.
        # We will now attach the predictions array to the dataframe,
        # and then output everything.
        test_df["predictions"] = predictions.tolist()
        # We will return "data"
        out = pd.DataFrame()
        out["id"] = test_df["file"]
        out["class"] = test_df["predictions"]
        return out

    def predict_data_row(self, clf, test_df):
        # We use the "clf" classifier and we predict the "data".
        # We temporarily remove the "file" column during predictions
        predictions = clf.predict(test_df[test_df.columns.drop(["file"])])
        # We have the predictions for each row.
        # We will now attach the predictions array to the dataframe,
        # and then output everything.
        test_df["predictions"] = predictions.tolist()
        # We will use the max instances as discriminator here
        # We will return "data"
        res = (
            test_df.groupby("file")["predictions"]
                .value_counts()
                .rename_axis(["file", "most_freq"])
                .reset_index(name="freq")
                .drop_duplicates("file")
        )
        out = pd.DataFrame()
        out["id"] = res["file"]
        out["class"] = res["most_freq"]
        return out

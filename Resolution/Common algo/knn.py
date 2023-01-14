import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def knn_algo(x_train, y_train, x_test, y_test, test_df, k, process):
    knn = KnnClassifier(x_train, y_train, process)
    clf = knn.test_model(x_test, y_test, k)
    if process == "file":
        return knn.predict_data_file(clf, test_df)
    else:
        return knn.predict_data_row(clf, test_df)


class KnnClassifier:
    def __init__(self, x_train, y_train, process):
        self.x_train = x_train
        self.y_train = y_train
        self.process = process

    def test_model(self, x_test, y_test, k: int):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf = clf.fit(self.x_train, self.y_train)
        # Prediction for the model
        y_pred = clf.predict(x_test)
        if self.process == "file":
            print("Testing model for file data preprocessing")
        else:
            print("Testing model for row data preprocessing")
        print("Accuracy of knn model:  %.3f" % accuracy_score(y_test, y_pred))
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
        plt.savefig(f"./images/confusion_matrix_knn{k}_{self.process}.png", dpi=75, format="png")
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

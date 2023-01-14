import logging

import keras.initializers.initializers_v2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

logging.basicConfig(level=logging.INFO)

# Change here is we want to use kfold or not
CROSS_VALIDATION = False


class SmartphoneDataframe:
    def __init__(self, train_location, test_location, label_file):
        self.train_location = train_location
        self.test_location = test_location
        self.train_df = pd.DataFrame()
        self.pred_df = pd.DataFrame()
        self.train_labels = pd.read_csv(label_file, sep=",")

    def create_train_df(self):
        for line in self.train_labels.index:
            temp = pd.read_csv(
                self.train_location + str(self.train_labels["id"][line]) + ".csv",
                header=None,
            )
            # Transform the 3 colmuns and X lines into one line and X columns so we have one line / file
            temp = temp.stack().to_frame().T
            # Rename the name of the column in order to better identify XYZ vectors
            temp.columns = ["{}_{}".format(*number) for number in temp.columns]
            # Add a class column
            temp["class"] = str(self.train_labels["class"][line])
            # concatenate previous DF with current one by adding it at the next row
            self.train_df = pd.concat([self.train_df, temp], axis=0, join="outer")
        # Replace all NAN by 0 because not all files have the same length
        self.train_df = self.train_df.fillna(0)

    def create_pred_df(self):
        for ind in range(10000, 24001):
            try:
                temp = pd.read_csv(self.test_location + str(ind) + ".csv", header=None)
                temp = temp.stack().to_frame().T
                temp.columns = ["{}_{}".format(*number) for number in temp.columns]
                temp["file"] = ind
                # After preparation, concat the dataframe to the FULL test dataframe
                self.pred_df = pd.concat([self.pred_df, temp], axis=0, join="outer")
            except FileNotFoundError as error:
                logging.debug(error)
        # Replace all NAN by 0 because not all files have the same length
        self.pred_df = self.pred_df.fillna(0)

    def prepare_df(self):
        # I concat the two DF in order to get the same amount of columns
        train_pred_df = pd.concat([self.train_df, self.pred_df])
        # Replace NaN by 0
        train_pred_df = train_pred_df.fillna(0)
        # Split to get back pred_df and train_df with the added columns
        self.train_df = train_pred_df.iloc[: np.shape(self.train_df)[0], :]
        self.pred_df = train_pred_df.iloc[np.shape(self.train_df)[0]:, :]
        self.pred_df = self.pred_df.drop(["class"], axis=1)
        self.train_df = self.train_df.drop(["file"], axis=1)
        # Reorder column to have the "class" and "file" column at the end of the DF
        new_col_order = [col for col in self.train_df.columns if col != "class"] + [
            "class"
        ]
        self.train_df = self.train_df[new_col_order]
        new_col_order = [col for col in self.pred_df.columns if col != "file"] + [
            "file"
        ]
        self.pred_df = self.pred_df[new_col_order]

    def prepare_data(self, scaler_used):
        # Remove 1 to labels because we are going to use categorical which needs to begin from 0
        # /!\ DO NOT forget to add one to final results
        y_train_df = self.train_labels["class"] - 1

        # Drop class and file for scaling
        x_train = self.train_df.drop(labels=["class"], axis=1)
        x_pred = self.pred_df.drop(labels=["file"], axis=1)

        # Can be StandardScaler, MinMaxScaler ...
        scaler = scaler_used
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_pred = scaler.transform(x_pred)

        # Reshape for cnn
        # I use x_train shape but can also use x_pred, they have the same shape
        # Divide by 3 because we have 3 features (x,y,z)
        number_rows = int(np.shape(x_train)[1] / 3)
        x_train_df = x_train.reshape(-1, number_rows, 3)
        x_pred = x_pred.reshape(-1, number_rows, 3)

        # encode labels to 0-1 values
        y_train_df = to_categorical(y_train_df, num_classes=20)

        return x_train_df, y_train_df, x_pred, self.pred_df


def _show_diagram(history, arg1, arg2, arg3):
    plt.plot(history.history[arg1])
    plt.plot(history.history[arg2])
    plt.title(arg3)
    plt.ylabel(arg1)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


class CnnModel:
    """
    Define the CNN model and train it.
    """

    def __init__(self):
        tf.random.set_seed(42)
        self.model = Sequential(
            [
                Conv1D(128, 5, activation="relu", input_shape=(159, 3)),
                Conv1D(128, 5, activation="relu"),
                MaxPooling1D(3),
                Dropout(0.25),
                Conv1D(256, 5, activation="relu"),
                Conv1D(256, 5, activation="relu"),
                MaxPooling1D(3),
                Dropout(0.25),
                GlobalAveragePooling1D(),
                Flatten(),
                Dropout(0.5),
                Dense(128, activation="relu", kernel_initializer=keras.initializers.initializers_v2.VarianceScaling,
                      kernel_regularizer="l2", activity_regularizer="l2"),
                Dense(64, activation="relu", kernel_initializer=keras.initializers.initializers_v2.VarianceScaling,
                      kernel_regularizer="l2", activity_regularizer="l2"),
                Dense(20, activation="softmax"),
            ]
        )
        # I use a checkpoint to prevent overfitting when using epoch so the model will
        # Only save the best epoch
        checkpoint_path = (
            "./tmp/checkpoint/test.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5"
        )
        self.model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
        )
        self.callback = [
            EarlyStopping(patience=8, restore_best_weights="True", monitor="val_loss"),
            self.model_checkpoint,
        ]
        self.model.compile(
            optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model.summary()

    def fit_model(self, x_train_df, y_train_df):
        x_train, x_test, y_train, y_test = train_test_split(
            x_train_df, y_train_df, test_size=0.3, random_state=42
        )
        history = self.model.fit(
            x_train,
            y_train,
            epochs=256,
            validation_data=(x_test, y_test),
            verbose=2,
            batch_size=128,
            shuffle=True,
            callbacks=self.callback,
        )

        _show_diagram(history, "accuracy", "val_accuracy", "model accuracy")
        _show_diagram(history, "loss", "val_loss", "model loss")
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print("====================================")
        print(f"Test accuracy: {accuracy * 100} %\n")
        print(f"Test loss: {loss} %\n")

    def use_kfold(self, n, x_train_df, y_train_df):
        kfold = KFold(n_splits=n, shuffle=True, random_state=42)
        model_history = []
        for train, test in kfold.split(x_train_df, y_train_df):
            print("Start new fold")
            results = self.model.fit(
                x_train_df[train],
                y_train_df[train],
                epochs=256,
                batch_size=128,
                validation_data=(x_train_df[test], y_train_df[test]),
                callbacks=self.callback,
            )
            loss, accuracy = self.model.evaluate(x_train_df[test], y_train_df[test])
            print("====================================")
            print(f"Test accuracy: {accuracy * 100} %\n")
            print(f"Test loss: {loss} %\n")
            model_history.append(results)
            print("=" * 10, end="\n\n\n")
        plt.title = "Kfold accuracy"
        plt.plot(
            model_history[0].history["accuracy"],
            label="Train Accuracy Fold 1",
            color="orange",
        )
        plt.plot(
            model_history[0].history["val_accuracy"],
            label="Val Accuracy Fold 1",
            color="orange",
            linestyle="dashdot",
        )
        plt.plot(
            model_history[1].history["accuracy"],
            label="Train Accuracy Fold 2",
            color="red",
        )
        plt.plot(
            model_history[1].history["val_accuracy"],
            label="Val Accuracy Fold 2",
            color="red",
            linestyle="dashdot",
        )
        plt.plot(
            model_history[2].history["accuracy"],
            label="Train Accuracy Fold 3",
            color="blue",
        )
        plt.plot(
            model_history[2].history["val_accuracy"],
            label="Val Accuracy Fold 3",
            color="blue",
            linestyle="dashdot",
        )
        plt.plot(
            model_history[3].history["accuracy"],
            label="Train Accuracy Fold 4",
            color="black",
        )
        plt.plot(
            model_history[3].history["val_accuracy"],
            label="Val Accuracy Fold 4",
            color="black",
            linestyle="dashdot",
        )
        plt.plot(
            model_history[4].history["accuracy"],
            label="Train Accuracy Fold 5",
            color="green",
        )
        plt.plot(
            model_history[4].history["val_accuracy"],
            label="Val Accuracy Fold 5",
            color="green",
            linestyle="dashdot",
        )
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig("./images/kfold_results_2.png", format="png", dpi=1200)
        plt.show()

    def conf_matrix(self, x_train_df, y_train_df):
        best_model = load_model(
            "tmp/checkpoint/kfold5.epoch23-loss0.0489V6.hdf5"
        )
        # Create a new split to create a confusion matrix for the best model
        x_train_not_used, x_test_matrix, y_train_not_used, y_test_matrix = train_test_split(
            x_train_df, y_train_df, test_size=0.3, random_state=42
        )
        f, ax = plt.subplots(1, 1, figsize=(30, 30))
        test_predictions = best_model.predict(x_test_matrix)
        metrics.ConfusionMatrixDisplay.from_predictions(
            np.argmax(y_test_matrix, axis=1),
            np.argmax(test_predictions, axis=1),
            ax=ax,
            normalize=None,
        )
        plt.title("Confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        tick_marks = np.arange(20)
        plt.xticks(tick_marks, range(1, 21))
        plt.yticks(tick_marks, range(1, 21))
        plt.savefig("./images/confusion_matrix_cnn_v6.png", dpi=1200, format="png")
        plt.show()


    def predict_data(self, x_pred, pred_df):
        best_model = load_model(
            "tmp/checkpoint/kfold5.epoch23-loss0.0489V6.hdf5"
        )
        predictions_categorical = best_model.predict(x_pred)
        predictions = np.argmax(predictions_categorical, axis=1)
        out = pd.DataFrame()
        pred_df["file"] = pred_df["file"].astype("int")
        out["id"] = pred_df["file"]
        out["class"] = predictions.tolist()
        # We add one because of the categorical function used before
        out["class"] = out["class"] + 1

        out.to_csv(
            "./csv/cnn_flatten_standard_adam_kfold5_v6.csv", index=False, sep=","
        )


if __name__ == "__main__":
    data = SmartphoneDataframe(
        "../../train/train/", "../../test/test/", "../../train_labels.csv"
    )
    data.create_train_df()
    data.create_pred_df()
    data.prepare_df()

    # also return x_pred for the data we want to predict and pred_df to get the file name
    x_train_df, y_train_df, x_pred, pred_df = data.prepare_data(
        StandardScaler()
    )
    model = CnnModel()
    if CROSS_VALIDATION:
        model.use_kfold(5, x_train_df, y_train_df)
    else:
        model.fit_model(x_train_df, y_train_df)

    # Before using matrix and predict functions, train the model for the first time in order to get the best model.
    # Then, change the model path in matrix and predict functions
    # model.conf_matrix(x_train_df, y_train_df)
    # model.predict_data(x_pred, pred_df)

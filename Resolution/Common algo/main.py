import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dt import dt_algo
from knn import knn_algo
from rf import rf_algo
from xgb import xgboost_algo

logging.basicConfig(level=logging.INFO)

train = "../../train/train/"
test = "../../test/test/"


def main():
    # use "file" for process with one row per file or use "row" to process with one row per row
    # But I recommend using the file param because it has a better accuracy.

    # Uncomment this if you use the row per row data preprocessing
    x_train, x_test, y_train, y_test, test_df = load_files_array_row()
    launch_algorithms(x_train, x_test, y_train, y_test, test_df, "row")
    x_train, x_test, y_train, y_test, test_df = load_files_array_file()
    launch_algorithms(x_train, x_test, y_train, y_test, test_df, "file")


def load_files_array_file():
    full_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_labels_loc = "../../train_labels.csv"
    train_labels = pd.read_csv(train_labels_loc, sep=",")

    for line in train_labels.index:
        temp = pd.read_csv(train + str(train_labels["id"][line]) + ".csv", header=None)
        # Transform all the 3 colmuns and X of lines into one line and X columns
        temp = temp.stack().to_frame().T
        # Just rename the name of the column in order to better identify XYZ vectors
        temp.columns = ["{}_{}".format(*number) for number in temp.columns]
        # Add a class column
        temp["class"] = str(train_labels["class"][line])
        # concatenate previous DF with current one by adding it at the next row
        full_df = pd.concat([full_df, temp], axis=0, join="outer")
    # Replace all NAN by 0 because not all files have the same length
    full_df = full_df.fillna(0)

    for ind in range(10000, 24001):
        try:
            temp = pd.read_csv(test + str(ind) + ".csv", header=None)
            temp = temp.stack().to_frame().T
            temp.columns = ["{}_{}".format(*number) for number in temp.columns]
            temp["file"] = ind
            # After preparation, concat the dataframe to the FULL test dataframe
            test_df = pd.concat([test_df, temp], axis=0, join="outer")
        except FileNotFoundError as error:
            logging.debug(error)
    test_df = test_df.fillna(0)

    # I check the difference of amount of columns between each DF in order to be able to transform them with the scaler.
    # Because I need to have the same shape of column to transform the DF so I add some columns with 0 as value.
    if np.shape(full_df)[1] - 1 > np.shape(test_df)[1]:
        diff = (np.shape(full_df)[1] - 1) - (np.shape(test_df)[1])
        for _ in range(int(diff / 3)):
            last_column = test_df.columns[-1]
            id_last_column = last_column.split("_")[0]
            for i in range(3):
                test_df[f"{int(id_last_column) + 1}_{i}"] = float(0)

    elif np.shape(full_df)[1] - 1 < np.shape(test_df)[1]:
        diff = (np.shape(test_df)[1]) - (np.shape(full_df)[1] - 1)
        for _ in range(int(diff / 3)):
            last_column = (
                full_df.columns[-1]
                if full_df.columns[-1] != "class"
                else full_df.columns[-2]
            )

            id_last_column = last_column.split("_")[0]
            for i in range(3):
                full_df[f"{int(id_last_column) + 1}_{i}"] = float(0)

    x_train, x_test, y_train, y_test = train_test_split(
        full_df[full_df.columns.drop("class")],
        full_df["class"],
        test_size=0.3,
        random_state=42,
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # Do not transform the column with the file id
    test_df.loc[:, test_df.columns != "file"] = scaler.transform(
        test_df.loc[:, test_df.columns != "file"]
    )

    return x_train, x_test, y_train, y_test, test_df


def load_files_array_row():
    train_labels_loc = "../../train_labels.csv"
    train_labels = pd.read_csv(train_labels_loc, sep=",")
    train_files = sorted(os.listdir("../../train/train"))
    test_files = sorted(os.listdir("../../test/test"))
    full_df = [
        np.genfromtxt(f"../../train/train/{file}", delimiter=",") for file in train_files
    ]

    test_df = [
        np.genfromtxt(f"../../test/test/{file}", delimiter=",") for file in test_files
    ]
    # Apply 0 post padding
    full_df = tf.keras.preprocessing.sequence.pad_sequences(
        full_df, padding="post", dtype="float32"
    )
    test_df = tf.keras.preprocessing.sequence.pad_sequences(
        test_df, padding="post", dtype="float32"
    )

    mean_train = [np.mean(file, axis=0) for file in full_df]
    mean_test = [np.mean(file, axis=0) for file in test_df]

    full_df = pd.DataFrame(mean_train, columns=["X", "Y", "Z"])
    test_df = pd.DataFrame(mean_test, columns=["X", "Y", "Z"])

    full_df = pd.concat((full_df, train_labels["class"]), axis=1)
    files_id = [[x.split(".")[0]] for x in test_files]
    files_id = pd.DataFrame(files_id, columns=["file"])
    test_df = pd.concat((test_df, files_id), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        full_df[full_df.columns.drop("class")],
        full_df["class"],
        test_size=0.3,
        random_state=42,
    )

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    test_df.loc[:, test_df.columns != "file"] = scaler.transform(
        test_df.loc[:, test_df.columns != "file"]
    )
    return x_train, x_test, y_train, y_test, test_df


def launch_algorithms(x_train, x_test, y_train, y_test, test_df, process):
    if process == "file":
        result_knn = knn_algo(x_train, y_train, x_test, y_test, test_df, 7, process)
        result_knn.to_csv(f"./Predictions by file/" + "_knn_7.csv", index=False, sep=",")
        result_rf = rf_algo(x_train, y_train, x_test, y_test, test_df, process)
        result_rf.to_csv(
            "./Predictions by file/" + "_rf_minMax_hyperParam.csv",
            index=False,
            sep=",",
        )
        result_dt = dt_algo(x_train, y_train, x_test, y_test, test_df, process)
        result_dt.to_csv(
            "./Predictions by file/" + "_dt_default.csv", index=False, sep=","
        )

        # DO NOT run the 2 differents types of data preprocessing for xgboost at the same time, you will get error !
        # result_xgb = xgboost_algo(x_train, y_train, x_test, y_test, test_df, process)
        # result_xgb.to_csv(
        #     "./Predictions by file/" + "_xgb_default.csv", index=False, sep=","
        # )

    elif process == "row":
        result_knn = knn_algo(x_train, y_train, x_test, y_test, test_df, 7, process)
        result_knn.to_csv("./Predictions by rows/" + "_knn_7.csv", index=False, sep=",")
        result_rf = rf_algo(x_train, y_train, x_test, y_test, test_df, process)
        result_rf.to_csv(
            "./Predictions by rows/" + "_rf_minMax_hyperParam.csv", index=False, sep=","
        )
        result_dt = dt_algo(x_train, y_train, x_test, y_test, test_df, process)
        result_dt.to_csv(
            "./Predictions by rows/" + "_dt_default.csv", index=False, sep=","
        )
        result_xgb = xgboost_algo(x_train, y_train, x_test, y_test, test_df, process)
        result_xgb.to_csv(
            "./Predictions by rows/" + "_xgb_default.csv", index=False, sep=","
        )


if __name__ == "__main__":
    main()

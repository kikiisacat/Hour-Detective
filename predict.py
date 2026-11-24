import catboost
import os
import numpy
import pandas

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
REMOVED_FEATURES = []

seeds = [42, 123, 456, 789, 101112]

PARAMS = {
    "iterations": 2680,
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 9,
    "loss_function": "RMSE",
    "verbose": False,
    "task_type": "CPU",
    "allow_writing_files": False
}

training_data = pandas.read_csv("train.csv")

DROP_COLUMNS = ["project_id", "hours_spent", "log_hours", "hours_bucket"] + REMOVED_FEATURES

x = training_data.drop(columns=DROP_COLUMNS, errors="ignore")
y = numpy.log1p(training_data["hours_spent"])

cat_cols = x.select_dtypes(include=["object", "bool", "category", "string"]).columns.to_list()

def preprocess(df):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("string")
        df[col] = df[col].fillna("missing").astype(str)
    return df

x = preprocess(x)
print(len(x))
print(x.shape[1])
print(len(cat_cols))

final_iterations = 2680

testing = os.path.exists("test.csv")

if testing:
    test_data = pandas.read_csv("test.csv")
    x_test = preprocess(test_data[x.columns])
    all_predictions = []

    for seed in seeds:
        final_params = PARAMS.copy()
        final_params["random_seed"] = seed
        final_params["iterations"] = final_iterations

        final_model = CatBoostRegressor(**final_params)
        final_model.fit(x, y, cat_features=cat_cols)
        all_predictions.append(final_model.predict(x_test))

    predictions = numpy.maximum(
        numpy.expm1(numpy.mean(all_predictions, axis=0)), 0
    )

    submission = pandas.DataFrame({
        "project_id": test_data["project_id"],
        "hours_spent": predictions
    })

    submission.to_csv("predictions.csv", index=False)
import catboost
import os
import numpy
import pandas

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

REMOVED_FEATURES = []

PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.02,
    "depth": 6,
    "l2_leaf_reg": 5,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 100,
    "task_type": "CPU"
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

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=training_data["hours_bucket"])
model = CatBoostRegressor(**PARAMS)
model.fit(x_train, y_train, cat_features=cat_cols, eval_set=(x_val, y_val), early_stopping_rounds=200, use_best_model=True)
val_predictions = numpy.maximum(model.predict(x_val), 0)
RMSLE = numpy.sqrt(numpy.mean((val_predictions - y_val) ** 2))
print(f"Validation RMSLE: {RMSLE}")
print(f"Best iteration: {model.get_best_iteration()}")

importances = pandas.Series(model.get_feature_importance(), index=x.columns).sort_values(ascending=False)
print(importances)

testing = False

if testing == True:
    test = pandas.read_csv("test.csv")
    x_test = preprocess(test[x.columns])
    final_params = PARAMS.copy()
    final_params["iterations"] = max(1, model.get_best_iteration() + 1)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(x, y, cat_features=cat_cols)
    predictions = numpy.clip(numpy.expm1(final_model.predict(x_test)), 0, 140.7)
    submission = pandas.DataFrame({"project_id": test["project_id"], "hours_spent": predictions})
    submission.to_csv("prediction.csv", index=False)
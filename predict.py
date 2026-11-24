import catboost
import os
import numpy
import pandas

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
REMOVED_FEATURES = []

seeds = [42]

PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 9,
    "loss_function": "RMSE",
    "verbose": 100,
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

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
best_iterations = []
out_of_fold_predictions = numpy.zeros(len(x))

for fold, (train_idx, val_idx) in enumerate(kf.split(x, training_data["hours_bucket"])):
    print(f"Fold {fold + 1}")
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    seed_predictions = []

    for seed in seeds: 
        print(seed)
        params = PARAMS.copy()
        params["random_seed"] = seed
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train, cat_features=cat_cols, eval_set=(x_val, y_val), early_stopping_rounds=300, use_best_model=True)
        seed_predictions.append(model.predict(x_val))
        best_iterations.append(model.get_best_iteration()+1)

    val_predictions = numpy.maximum(numpy.mean(seed_predictions, axis=0), 0)
    out_of_fold_predictions[val_idx] = val_predictions
    RMSLE = numpy.sqrt(numpy.mean((val_predictions - y_val) ** 2))
    scores.append(RMSLE)
    print(fold, RMSLE)

print("Mean RMSLE:", numpy.mean(scores))
print("Fold scores:", scores)
print("Overall RMSLE:", numpy.sqrt(numpy.mean((out_of_fold_predictions - y) ** 2)))
final_iterations = int(numpy.median(best_iterations))
print("Final iterations:", final_iterations)

testing = os.path.exists("test.csv")

if testing == True:
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
    predictions = numpy.maximum(numpy.expm1(numpy.mean(all_predictions, axis=0)), 0)
    submission = pandas.DataFrame({"project_id": test_data["project_id"], "hours_spent": predictions})
    submission.to_csv("predictions.csv", index=False)
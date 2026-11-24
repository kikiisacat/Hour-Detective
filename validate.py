# For testing, AI helped move the code over into here and add some improvements compared to predict.py

import numpy
import pandas

from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor

REMOVED_FEATURES = []

seeds = [42, 123, 456, 789, 101112]

PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 9,
    "loss_function": "RMSE",
    "verbose": False,
    "task_type": "CPU",
    "allow_writing_files": False
}

training_data = pandas.read_csv("train.csv")

DROP_COLUMNS = [
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
] + REMOVED_FEATURES

x = training_data.drop(columns=DROP_COLUMNS, errors="ignore")
y = numpy.log1p(training_data["hours_spent"])

cat_cols = x.select_dtypes(
    include=["object", "bool", "category", "string"]
).columns.to_list()


def preprocess(df):
    df = df.copy()

    for col in cat_cols:
        df[col] = df[col].astype("string")
        df[col] = df[col].fillna("missing").astype(str)

    return df


x = preprocess(x)

print("Training projects:", len(x))
print("Features:", x.shape[1])
print("Categorical features:", len(cat_cols))

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []
best_iterations = []
feature_importances = []

out_of_fold_predictions = numpy.zeros(len(x))

for fold, (train_idx, val_idx) in enumerate(
    kf.split(x, training_data["hours_bucket"])
):
    print(f"\nFold {fold + 1}/5")

    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    seed_predictions = []

    for seed in seeds:
        params = PARAMS.copy()
        params["random_seed"] = seed

        model = CatBoostRegressor(**params)

        model.fit(
            x_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(x_val, y_val),
            early_stopping_rounds=300,
            use_best_model=True
        )

        pred = model.predict(x_val)

        seed_predictions.append(pred)
        best_iterations.append(model.get_best_iteration() + 1)
        feature_importances.append(model.get_feature_importance())

        seed_rmsle = numpy.sqrt(
            numpy.mean((numpy.maximum(pred, 0) - y_val) ** 2)
        )

        print(
            f"Seed {seed}: {seed_rmsle:.8f} | "
            f"Iterations: {model.get_best_iteration() + 1}"
        )

    val_predictions = numpy.maximum(
        numpy.mean(seed_predictions, axis=0), 0
    )

    out_of_fold_predictions[val_idx] = val_predictions

    RMSLE = numpy.sqrt(
        numpy.mean((val_predictions - y_val) ** 2)
    )

    scores.append(RMSLE)

    print(f"Fold {fold + 1} Ensemble RMSLE: {RMSLE:.8f}")


overall_RMSLE = numpy.sqrt(
    numpy.mean((out_of_fold_predictions - y) ** 2)
)

final_iterations = int(numpy.median(best_iterations))

importances = pandas.Series(
    numpy.mean(feature_importances, axis=0),
    index=x.columns
).sort_values(ascending=False)

print("\n========== RESULTS ==========")
print("Removed features:", REMOVED_FEATURES)
print("Fold scores:", scores)
print("Mean RMSLE:", numpy.mean(scores))
print("Standard deviation:", numpy.std(scores))
print("Overall RMSLE:", overall_RMSLE)
print("Final iterations:", final_iterations)

print("\nAverage feature importance:")
print(importances.to_string())

print("\nParameters:")
print(PARAMS)

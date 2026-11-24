# Hour Detective

A CatBoost regression model that predicts approved project development hours using repository statistics and project metadata.

## Model Configuration

- Iterations: 5000
- Depth: 7
- Learning rate: 0.02
- L2 regularization: 9
- Loss function: RMSE on logarithmically transformed hours
- Validation RMSLE: 0.6570757

## Running the model

### Install dependencies:

pip install catboost numpy pandas scikit-learn

### Place train.csv and test.csv in the project directory and run:

python predict.py

The script trains the model and generates predictions.csv.

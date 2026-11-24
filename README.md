# Hour Detective

A CatBoost regression model that predicts approved project development hours using repository statistics and project metadata.

## Model Configuration

- Iterations: 5000
- Depth: 7
- Learning rate: 0.02
- L2 regularization: 9
- Loss function: RMSE on logarithmically transformed hours
- Validation RMSLE: 0.6570757

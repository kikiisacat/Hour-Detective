import catboost
import os
import numpy
import pandas

PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.02,
    "depth": 6,
    "l2_leaf_reg": 5,
    "loss_function": "RM

}
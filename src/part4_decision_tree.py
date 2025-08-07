'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LR_TRAIN_IN = os.path.join(DATA_DIR, "df_arrests_train_lr.csv")
LR_TEST_IN  = os.path.join(DATA_DIR, "df_arrests_test_lr.csv")
DT_TEST_OUT = os.path.join(DATA_DIR, "df_arrests_test_dt.csv")

def run_decision_tree():
    df_train = pd.read_csv(LR_TRAIN_IN)
    df_test  = pd.read_csv(LR_TEST_IN)

    features = ["num_fel_arrests_last_year", "current_charge_felony"]
    target = "y"

    X_train, y_train = df_train[features].values, df_train[target].values
    X_test,  y_test  = df_test[features].values,  df_test[target].values

    param_grid_dt = {"max_depth": [3, 5, 10]}
    dt_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=10)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs_cv_dt = GridSearchCV(dt_model, param_grid=param_grid_dt, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs_cv_dt.fit(X_train, y_train)

    best_depth = gs_cv_dt.best_params_["max_depth"]
    if best_depth == min(param_grid_dt["max_depth"]):
        interp = "most regularization (shallowest)"
    elif best_depth == max(param_grid_dt["max_depth"]):
        interp = "least regularization (deepest)"
    else:
        interp = "in the middle"

    print("What was the optimal value for max_depth?")
    print(f"Answer: {best_depth}")
    print("Did it have the most or least regularization? Or in the middle?")
    print(f"Answer: {interp}")

    best_dt = gs_cv_dt.best_estimator_
    df_test = df_test.copy()
    df_test["pred_dt"] = best_dt.predict_proba(X_test)[:, 1]
    df_test.to_csv(DT_TEST_OUT, index=False)
    print(f"[DT] Saved: {DT_TEST_OUT}")
    return df_train, df_test
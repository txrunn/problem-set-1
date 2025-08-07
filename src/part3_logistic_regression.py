'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARRESTS_IN = os.path.join(DATA_DIR, "df_arrests.csv")
LR_TRAIN_OUT = os.path.join(DATA_DIR, "df_arrests_train_lr.csv")
LR_TEST_OUT  = os.path.join(DATA_DIR, "df_arrests_test_lr.csv")

def run_logistic():
    df = pd.read_csv(ARRESTS_IN, parse_dates=["arrest_date_univ"])
    features = ["num_fel_arrests_last_year", "current_charge_felony"]
    target = "y"

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, shuffle=True, stratify=y, random_state=42
    )

    param_grid = {"C": [0.1, 1.0, 10.0]}
    lr_model = LogisticRegression(solver="liblinear", max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs_cv = GridSearchCV(lr_model, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs_cv.fit(X_train, y_train)

    best_C = gs_cv.best_params_["C"]
    interp = {0.1: "most regularization", 1.0: "in the middle", 10.0: "least regularization"}[best_C]
    print("What was the optimal value for C?")
    print(f"Answer: {best_C}")
    print("Did it have the most or least regularization? Or in the middle?")
    print(f"Answer: {interp}")

    best_lr = gs_cv.best_estimator_
    df_train = pd.DataFrame(X_train, columns=features)
    df_train[target] = y_train

    df_test = pd.DataFrame(X_test, columns=features)
    df_test[target] = y_test
    df_test["pred_lr"] = best_lr.predict_proba(X_test)[:, 1]

    df_train.to_csv(LR_TRAIN_OUT, index=False)
    df_test.to_csv(LR_TEST_OUT, index=False)
    print(f"[LR] Saved: {LR_TRAIN_OUT}")
    print(f"[LR] Saved: {LR_TEST_OUT}")
    return df_train, df_test



'''
You will run this problem set from main.py, so set things up accordingly
'''

import os
import numpy as np
import pandas as pd
import part1_etl as etl
import part2_preprocessing as preprocessing
import part3_logistic_regression as logistic_regression
import part4_decision_tree as decision_tree
import part5_calibration_plot as calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    print("\n=== PART 1: ETL ===")
    etl.run_etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    print("\n=== PART 2: Preprocessing ===")
    df_arrests = preprocessing.run_preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression
    print("\n=== PART 3: Logistic Regression ===")
    df_train_lr, df_test_lr = logistic_regression.run_logistic()

    # PART 4: Call functions/instanciate objects from decision_tree
    print("\n=== PART 4: Decision Tree ===")
    df_train_dt, df_test_dt = decision_tree.run_decision_tree()

    # PART 5: Call functions/instanciate objects from calibration_plot
    print("\n=== PART 5: Calibration-light ===")
    # Align test frame (decision_tree output includes pred_dt and pred_lr from previous step file)
    df_test = df_test_dt.copy()
    y_true = df_test["y"].values
    y_lr   = df_test["pred_lr"].values
    y_dt   = df_test["pred_dt"].values

    ece_lr = calibration_plot(y_true, y_lr, n_bins=5, label="Logistic Regression")
    ece_dt = calibration_plot(y_true, y_dt, n_bins=5, label="Decision Tree")

    which = "Logistic Regression" if ece_lr < ece_dt else "Decision Tree"
    print("Which model is more calibrated?")
    print(f"Answer: {which} (smaller average calibration error)")

    # ----- Extra Credit -----
    try:
        print("\n=== Extra Credit ===")
        # PPV@50 for each model
        def ppv_at_k(y, p, k=50):
            idx = np.argsort(-p)[:k]
            return y[idx].mean()

        ppv50_lr = ppv_at_k(y_true, y_lr, k=50)
        ppv50_dt = ppv_at_k(y_true, y_dt, k=50)
        print(f"PPV@50 (LR): {ppv50_lr:.3f}")
        print(f"PPV@50 (DT): {ppv50_dt:.3f}")

        # AUCs
        auc_lr = roc_auc_score(y_true, y_lr)
        auc_dt = roc_auc_score(y_true, y_dt)
        print(f"AUC (LR): {auc_lr:.3f}")
        print(f"AUC (DT): {auc_dt:.3f}")

        better_ppv = "LR" if ppv50_lr > ppv50_dt else "DT"
        better_auc = "LR" if auc_lr > auc_dt else "DT"
        agree = "agree" if better_ppv == better_auc else "do not agree"
        print("Do both metrics agree that one model is more accurate than the other?")
        print(f"Answer: They {agree} (PPV@50 says {better_ppv}, AUC says {better_auc}).")
    except Exception as e:
        print(f"[Extra Credit] Skipped due to error: {e}")

if __name__ == "__main__":
    main()
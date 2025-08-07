'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10, label="Model"):
    """
    Create a calibration plot with a 45-degree dashed line and
    return a simple average |mean_pred - frac_positive| (ECE-like).
    """
    # sklearn returns: (prob_true, prob_pred)
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    # Seaborn styling
    sns.set_theme(style="whitegrid")

    # Plot: x = mean predicted probability, y = fraction of positives
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, marker="o", label=label)

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Plot ({label})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Return an ECE-like scalar for comparison in main.py
    ece = float(np.mean(np.abs(prob_pred - prob_true)))
    return ece
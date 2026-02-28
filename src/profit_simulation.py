import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def simulate_profit(y_true, y_probs,
                    retention_cost=500,
                    annual_revenue=6000,
                    save_rate=0.6):

    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        saved_revenue = tp * annual_revenue * save_rate
        cost = (tp + fp) * retention_cost
        net_profit = saved_revenue - cost

        results.append({
            "threshold": t,
            "TP": tp,
            "FP": fp,
            "net_profit": net_profit
        })

    return pd.DataFrame(results)
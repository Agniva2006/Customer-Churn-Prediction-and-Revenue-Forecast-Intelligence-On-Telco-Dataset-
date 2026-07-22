"""Model explainability module utilizing SHAP (SHapley Additive exPlanations).

Provides feature attributions for individual customer predictions and global model insights.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP package not installed — falling back to model feature importances.")


def get_pipeline_classifier_and_features(pipeline: Any) -> Tuple[Any, List[str], Any]:
    """Extract fitted classifier, feature names, and preprocessed matrix from a pipeline."""
    # Check if pipeline is a scikit-learn Pipeline
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier")
        
        # Unpack CalibratedClassifierCV if present
        if hasattr(classifier, "calibrated_classifiers_"):
            base_estimator = classifier.calibrated_classifiers_[0].estimator
        else:
            base_estimator = classifier

        return base_estimator, preprocessor
    return pipeline, None


def explain_prediction_heuristic(
    customer_dict: Dict[str, Any], top_n: int = 3
) -> List[Dict[str, Any]]:
    """Heuristic feature attribution fallback when SHAP tree explainer is unavailable."""
    drivers = []
    
    contract = customer_dict.get("Contract", "")
    if contract == "Month-to-month":
        drivers.append({"feature": "Contract", "value": contract, "impact": "High risk (+ Month-to-month)"})
    
    tenure = customer_dict.get("tenure", 0)
    if tenure <= 12:
        drivers.append({"feature": "tenure", "value": f"{tenure} months", "impact": "High risk (Short tenure)"})
        
    tech = customer_dict.get("TechSupport", "")
    if tech == "No":
        drivers.append({"feature": "TechSupport", "value": tech, "impact": "Medium risk (No Tech Support)"})
        
    payment = customer_dict.get("PaymentMethod", "")
    if payment == "Electronic check":
        drivers.append({"feature": "PaymentMethod", "value": payment, "impact": "Medium risk (Electronic check)"})

    return drivers[:top_n]


def get_top_churn_drivers(
    model_pipeline: Any,
    customer_df: pd.DataFrame,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """Extract the top features driving a customer's churn risk.

    Parameters
    ----------
    model_pipeline : scikit-learn Pipeline or model
        Trained model object.
    customer_df : pd.DataFrame
        Single row DataFrame containing customer inputs.
    top_n : int
        Number of top drivers to return.

    Returns
    -------
    list of dict
        Each dict contains feature name, input value, and contribution description.
    """
    row_dict = customer_df.iloc[0].to_dict()
    
    if not HAS_SHAP:
        return explain_prediction_heuristic(row_dict, top_n=top_n)

    try:
        # Check if pipeline has preprocessor step
        if hasattr(model_pipeline, "named_steps"):
            # Transform raw features up to classifier
            preproc_pipeline = model_pipeline[:-1]
            transformed_matrix = preproc_pipeline.transform(customer_df)
            classifier = model_pipeline.named_steps["classifier"]
            
            # Extract underlying tree model from CalibratedClassifierCV if needed
            if hasattr(classifier, "calibrated_classifiers_") and len(classifier.calibrated_classifiers_) > 0:
                tree_model = classifier.calibrated_classifiers_[0].estimator
            else:
                tree_model = classifier
                
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(transformed_matrix)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # Binary positive class
            else:
                shap_vals = shap_values[0]

            # Get feature names if transformer available
            feature_names = None
            if hasattr(preproc_pipeline[-1], "get_feature_names_out"):
                feature_names = list(preproc_pipeline[-1].get_feature_names_out())
            elif hasattr(transformed_matrix, "columns"):
                feature_names = list(transformed_matrix.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

            top_indices = np.argsort(shap_vals)[::-1][:top_n]
            drivers = []
            for idx in top_indices:
                feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                impact_val = float(shap_vals[idx])
                if impact_val > 0:
                    drivers.append({
                        "feature": feat_name,
                        "shap_value": round(impact_val, 4),
                        "impact": f"Increases churn risk by ~{impact_val:.2f}",
                    })

            if drivers:
                return drivers
    except Exception as exc:
        logger.warning("SHAP explanation failed (%s), falling back to heuristic.", exc)

    return explain_prediction_heuristic(row_dict, top_n=top_n)

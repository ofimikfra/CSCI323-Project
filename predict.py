import joblib
import numpy as np
import pandas as pd


# KNN Classifier
# Classify High / Low Waste (threshold = mean waste % of entire dataset)

def predict_knn(material_type: str, area: float, thickness: float) -> dict:
    """
    Predict waste level (High / Low) using the trained KNN model.

    Parameters
    ----------
    material_type : 'GI' or 'PI'
    area          : area in sq/m
    thickness     : thickness in mm

    Returns
    -------
    dict
        prediction      - 'High Waste' or 'Low Waste'
        confidence      - probability of the predicted class
        low_waste_prob  - probability of Low Waste
        high_waste_prob - probability of High Waste
    """
    knn           = joblib.load("models/knn_model.pkl")
    scaler        = joblib.load("models/knn_scaler.pkl")
    label_encoder = joblib.load("models/knn_label_encoder.pkl")

    try:
        mat_enc = label_encoder.transform([material_type])[0]
    except ValueError:
        raise ValueError(f"Unknown material type '{material_type}'. Use 'GI' or 'PI'.")

    X = pd.DataFrame(
        [[area, thickness, mat_enc]],
        columns=["Area (sq/m)", "Thickness (mm)", "Material Type Encoded"]
    )
    X_sc  = scaler.transform(X)
    pred  = knn.predict(X_sc)[0]
    probs = knn.predict_proba(X_sc)[0]

    return {
        "prediction":      "High Waste" if pred == 1 else "Low Waste",
        "confidence":      float(probs[pred]),
        "low_waste_prob":  float(probs[0]),
        "high_waste_prob": float(probs[1]),
    }


# Logistic Regression Classifier
# Classify High / Low Waste Risk (threshold = 10% waste percentage)

def predict_logistic(material_type: str, area: float, thickness: float) -> dict:
    """
    Predict waste risk (High / Low) using the trained Logistic Regression model.

    Parameters
    ----------
    material_type : 'GI' or 'PI'
    area          : area in sq/m
    thickness     : thickness in mm

    Returns
    -------
    dict
        prediction     - 'High Risk' or 'Low Risk'
        confidence     - probability of the predicted class
        low_risk_prob  - probability of Low Risk
        high_risk_prob - probability of High Risk
    """
    model  = joblib.load("models/lr_model.pkl")
    scaler = joblib.load("models/lr_scaler.pkl")

    mat_num = {"GI": 0, "PI": 1}.get(material_type.upper())
    if mat_num is None:
        raise ValueError(f"Unknown material type '{material_type}'. Use 'GI' or 'PI'.")

    X = pd.DataFrame(
        [[mat_num, area, thickness]],
        columns=["Material_Num", "Area (sq/m)", "Thickness (mm)"]
    )
    # Only the continuous columns were scaled during training
    X[["Area (sq/m)", "Thickness (mm)"]] = scaler.transform(
        X[["Area (sq/m)", "Thickness (mm)"]]
    )

    pred  = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "prediction":     "High Risk" if pred == 1 else "Low Risk",
        "confidence":     float(probs[pred]),
        "low_risk_prob":  float(probs[0]),
        "high_risk_prob": float(probs[1]),
    }


# Random Forest Regressor
# Predict Waste Percentage (continuous)

def predict_random_forest(
    material_type: str,
    area: float,
    thickness: float,
    material_cost_aed: float,
    unit_cost: float,
) -> dict:
    """
    Predict waste percentage using the trained Random Forest pipeline.

    Parameters
    ----------
    material_type     : 'GI' or 'PI'
    area              : area in sq/m
    thickness         : thickness in mm
    material_cost_aed : total material cost in AED
    unit_cost         : average unit cost for the material type

    Returns
    -------
    dict
        predicted_waste_percentage - e.g. 0.0823 (= 8.23%)
        waste_category             - 'High Waste' (>10%) or 'Low Waste'
    """
    pipeline = joblib.load("models/rf_pipeline.pkl")

    cost_per_sqm = material_cost_aed / area if area != 0 else 0.0

    X = pd.DataFrame([{
        "Material_Type":     material_type.upper(),
        "Area_sq/m":         area,
        "Thickness_mm":      thickness,
        "Material_Cost_AED": material_cost_aed,
        "Unit_Cost":         unit_cost,
        "Cost_per_sqm":      cost_per_sqm,
    }])

    pred = float(pipeline.predict(X)[0])

    return {
        "predicted_waste_percentage": round(pred, 4),
        "waste_category":             "High Waste" if pred > 0.10 else "Low Waste",
    }


# SVM Waste Level Classifier
# Classify High / Low Waste Level (threshold = median waste percentage)

def predict_svm_waste_level(
    material_type: str,
    area: float,
    thickness: float,
    material_cost_aed: float,
) -> dict:
    """
    Predict waste level (High / Low) using the trained SVM model.

    Parameters
    ----------
    material_type     : 'GI' or 'PI'
    area              : area in sq/m
    thickness         : thickness in mm
    material_cost_aed : total material cost in AED

    Returns
    -------
    dict
        prediction      - 'High Waste' or 'Low Waste'
        confidence      - probability of the predicted class
        low_waste_prob  - probability of Low Waste
        high_waste_prob - probability of High Waste
    """
    model = joblib.load("models/svm_waste_model.pkl")
    le    = joblib.load("models/svm_label_encoder.pkl")
    meta  = joblib.load("models/svm_meta.pkl")

    mat_enc       = le.transform([material_type.upper()])[0]
    avg_item_cost = meta["gi_avg_cost"] if material_type.upper() == "GI" else meta["pi_avg_cost"]
    raw_count     = meta["gi_raw_count"] if material_type.upper() == "GI" else meta["pi_raw_count"]

    cost_per_sqm   = material_cost_aed / area if area != 0 else 0.0
    thickness_area = thickness * area

    X = np.array([[
        area, thickness, material_cost_aed,
        cost_per_sqm, thickness_area,
        avg_item_cost, raw_count, mat_enc,
    ]])

    pred  = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "prediction":      "High Waste" if pred == 1 else "Low Waste",
        "confidence":      float(probs[pred]),
        "low_waste_prob":  float(probs[0]),
        "high_waste_prob": float(probs[1]),
    }



# SVM Material Type Classifier
# Predict material type — GI or PI


def predict_svm_material_type(
    area: float,
    thickness: float,
    material_cost_aed: float,
) -> dict:
    """
    Predict material type (GI / PI) using the trained SVM model.

    Parameters
    ----------
    area              : area in sq/m
    thickness         : thickness in mm
    material_cost_aed : total material cost in AED

    Returns
    -------
    dict
        prediction - 'GI' or 'PI'
        confidence - probability of the predicted class
        gi_prob    - probability of GI
        pi_prob    - probability of PI
    """
    model = joblib.load("models/svm_material_model.pkl")
    le    = joblib.load("models/svm_label_encoder.pkl")

    cost_per_sqm   = material_cost_aed / area if area != 0 else 0.0
    thickness_area = thickness * area

    X = np.array([[
        area, thickness, material_cost_aed,
        cost_per_sqm, thickness_area,
    ]])

    pred  = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]

    return {
        "prediction": label,
        "confidence": float(probs[pred]),
        "gi_prob":    float(probs[0]),
        "pi_prob":    float(probs[1]),
    }


# formatting

def header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def row(label, value):
    print(f"  {label:<32} {value}")


# test runners

def run_knn_tests(SIMPLE_CASES):
    header("MODEL 1 — KNN  (High / Low Waste)")
    for c in SIMPLE_CASES:
        r = predict_knn(c["material_type"], c["area"], c["thickness"])
        print()
        row("Material Type",    c["material_type"])
        row("Area (sq/m)",      c["area"])
        row("Thickness (mm)",   c["thickness"])
        row("Prediction",       r["prediction"])
        row("Confidence",       f"{r['confidence']:.1%}")
        row("Low / High prob",  f"{r['low_waste_prob']:.1%}  /  {r['high_waste_prob']:.1%}")


def run_lr_tests(SIMPLE_CASES):
    header("MODEL 2 — Logistic Regression  (High / Low Waste Risk)")
    for c in SIMPLE_CASES:
        r = predict_logistic(c["material_type"], c["area"], c["thickness"])
        print()
        row("Material Type",    c["material_type"])
        row("Area (sq/m)",      c["area"])
        row("Thickness (mm)",   c["thickness"])
        row("Prediction",       r["prediction"])
        row("Confidence",       f"{r['confidence']:.1%}")
        row("Low / High prob",  f"{r['low_risk_prob']:.1%}  /  {r['high_risk_prob']:.1%}")


def run_rf_tests(EXTENDED_CASES):
    header("MODEL 3 — Random Forest  (Waste Percentage)")
    for c in EXTENDED_CASES:
        r = predict_random_forest(
            material_type     = c["material_type"],
            area              = c["area"],
            thickness         = c["thickness"],
            material_cost_aed = c["material_cost"],
            unit_cost         = c["unit_cost"],
        )
        print()
        row("Material Type",       c["material_type"])
        row("Area (sq/m)",         c["area"])
        row("Thickness (mm)",      c["thickness"])
        row("Material Cost (AED)", c["material_cost"])
        row("Unit Cost (AED)",     c["unit_cost"])
        row("Predicted Waste %",   f"{r['predicted_waste_percentage']:.2%}")
        row("Waste Category",      r["waste_category"])


def run_svm_tests(EXTENDED_CASES):
    header("MODEL 4a — SVM  (Waste Level: High / Low)")
    for c in EXTENDED_CASES:
        r = predict_svm_waste_level(
            material_type     = c["material_type"],
            area              = c["area"],
            thickness         = c["thickness"],
            material_cost_aed = c["material_cost"],
        )
        print()
        row("Material Type",       c["material_type"])
        row("Area (sq/m)",         c["area"])
        row("Thickness (mm)",      c["thickness"])
        row("Material Cost (AED)", c["material_cost"])
        row("Prediction",          r["prediction"])
        row("Confidence",          f"{r['confidence']:.1%}")
        row("Low / High prob",     f"{r['low_waste_prob']:.1%}  /  {r['high_waste_prob']:.1%}")

    header("MODEL 4b — SVM  (Material Type: GI / PI)")
    for c in EXTENDED_CASES:
        r = predict_svm_material_type(
            area              = c["area"],
            thickness         = c["thickness"],
            material_cost_aed = c["material_cost"],
        )
        print()
        row("Area (sq/m)",         c["area"])
        row("Thickness (mm)",      c["thickness"])
        row("Material Cost (AED)", c["material_cost"])
        row("Predicted Material",  r["prediction"])
        row("Confidence",          f"{r['confidence']:.1%}")
        row("GI / PI prob",        f"{r['gi_prob']:.1%}  /  {r['pi_prob']:.1%}")
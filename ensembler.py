from predict import (
    predict_knn,
    predict_logistic,
    predict_svm_waste_level,
    header, row
)


def ensemble_waste_level(
    material_type: str,
    area: float,
    thickness: float,
    material_cost_aed: float,
    strategy: str = "soft_vote",
) -> dict:

    if strategy not in ("soft_vote", "majority_vote"):
        raise ValueError("strategy must be 'soft_vote' or 'majority_vote'")

    # ── Collect individual predictions ───────────────────────────────────────
    knn_result = predict_knn(material_type, area, thickness)
    lr_result  = predict_logistic(material_type, area, thickness)
    svm_result = predict_svm_waste_level(
        material_type, area, thickness, material_cost_aed
    )

    # Normalise LR labels ("High Risk" -> "High Waste" etc.) so all three
    # models share the same label space for voting.
    def _normalise(label: str) -> str:
        return "High Waste" if "High" in label else "Low Waste"

    models = {
        "KNN": {
            "prediction":      _normalise(knn_result["prediction"]),
            "low_waste_prob":  knn_result["low_waste_prob"],
            "high_waste_prob": knn_result["high_waste_prob"],
        },
        "Logistic Regression": {
            "prediction":      _normalise(lr_result["prediction"]),
            "low_waste_prob":  lr_result["low_risk_prob"],
            "high_waste_prob": lr_result["high_risk_prob"],
        },
        "SVM": {
            "prediction":      _normalise(svm_result["prediction"]),
            "low_waste_prob":  svm_result["low_waste_prob"],
            "high_waste_prob": svm_result["high_waste_prob"],
        },
    }

    votes = {name: m["prediction"] for name, m in models.items()}
    high_votes = sum(1 for v in votes.values() if v == "High Waste")
    low_votes  = 3 - high_votes

    # ── Agreement summary ────────────────────────────────────────────────────
    if high_votes == 3 or low_votes == 3:
        agreement = "unanimous"
    else:
        agreement = "majority"

    # ── Strategy: Majority Vote ───────────────────────────────────────────────
    if strategy == "majority_vote":
        if high_votes > low_votes:
            final = "High Waste"
            confidence = high_votes / 3
        else:
            final = "Low Waste"
            confidence = low_votes / 3

        return {
            "prediction":  final,
            "confidence":  round(confidence, 4),
            "strategy":    "majority_vote",
            "votes":       votes,
            "agreement":   agreement,
            "models":      models,
        }

    # ── Strategy: Soft Vote (averaged probabilities) ─────────────────────────
    avg_low  = sum(m["low_waste_prob"]  for m in models.values()) / 3
    avg_high = sum(m["high_waste_prob"] for m in models.values()) / 3

    final      = "High Waste" if avg_high >= avg_low else "Low Waste"
    confidence = avg_high if final == "High Waste" else avg_low

    return {
        "prediction":       final,
        "confidence":       round(confidence, 4),
        "strategy":         "soft_vote",
        "votes":            votes,
        "agreement":        agreement,
        "low_waste_prob":   round(avg_low,  4),
        "high_waste_prob":  round(avg_high, 4),
        "models":           models,
    }



def run_ensemble_tests(EXTENDED_CASES):
    for strategy in ("soft_vote", "majority_vote"):
        header(f"ENSEMBLE  (Waste Level — {strategy.replace('_', ' ').title()})")
        for c in EXTENDED_CASES:
            r = ensemble_waste_level(
                material_type     = c["material_type"],
                area              = c["area"],
                thickness         = c["thickness"],
                material_cost_aed = c["material_cost"],
                waste_cost_aed    = c["waste_cost"],
                strategy          = strategy,
            )

            print("\n" + "=" * 60)
            print()
            row("Material Type",       c["material_type"])
            row("Area (sq/m)",         c["area"])
            row("Thickness (mm)",      c["thickness"])
            row("Material Cost (AED)", c["material_cost"])
            row("Waste Cost (AED)",    c["waste_cost"])
            print(f"  {'─' * 44}")
            row("KNN vote",            r["votes"]["KNN"])
            row("Logistic Reg vote",   r["votes"]["Logistic Regression"])
            row("SVM vote",            r["votes"]["SVM"])
            row("Agreement",           r["agreement"])
            print(f"  {'─' * 44}")
            row("Final Prediction",    r["prediction"])
            row("Confidence",          f"{r['confidence']:.1%}")
            if strategy == "soft_vote":
                row("Avg Low / High prob",
                    f"{r['low_waste_prob']:.1%}  /  {r['high_waste_prob']:.1%}")
                

def run_ensemble_tests(EXTENDED_CASES):
    for strategy in ("soft_vote", "majority_vote"):
        header(f"ENSEMBLE  (Waste Level — {strategy.replace('_', ' ').title()})")
        for c in EXTENDED_CASES:
            r = ensemble_waste_level(
                material_type     = c["material_type"],
                area              = c["area"],
                thickness         = c["thickness"],
                material_cost_aed = c["material_cost"],
                strategy          = strategy,
            )
            print("\n" + "=" *60)
            print()
            row("Material Type",       c["material_type"])
            row("Area (sq/m)",         c["area"])
            row("Thickness (mm)",      c["thickness"])
            row("Material Cost (AED)", c["material_cost"])
            print(f"  {'─' * 44}")
            row("KNN vote",            r["votes"]["KNN"])
            row("Logistic Reg vote",   r["votes"]["Logistic Regression"])
            row("SVM vote",            r["votes"]["SVM"])
            row("Agreement",           r["agreement"])
            print(f"  {'─' * 44}")
            row("Final Prediction",    r["prediction"])
            row("Confidence",          f"{r['confidence']:.1%}")
            if strategy == "soft_vote":
                row("Avg Low / High prob",
                    f"{r['low_waste_prob']:.1%}  /  {r['high_waste_prob']:.1%}")

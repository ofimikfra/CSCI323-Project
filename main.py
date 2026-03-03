from predict import (
    run_knn_tests,
    run_lr_tests,
    run_rf_tests,
    run_svm_tests
)

from ensembler import run_ensemble_tests

# Simple cases: just material type, area, thickness (KNN + Logistic Regression)
SIMPLE_CASES = [
    {"material_type": "GI", "area": 54.0,  "thickness": 1.2,  "label": "GI – large area / thin"},
    {"material_type": "PI", "area": 14.0,  "thickness": 25.0, "label": "PI – small area / thick"},
    {"material_type": "GI", "area": 8.5,   "thickness": 0.8,  "label": "GI – small area / very thin"},
    {"material_type": "PI", "area": 120.0, "thickness": 50.0, "label": "PI – very large / very thick"},
]

# Extended cases: include cost data (Random Forest + SVM)
EXTENDED_CASES = [
    {
        "material_type": "GI",  "area": 54.0,  "thickness": 1.2,
        "material_cost": 1200.0, "unit_cost": 22.5,
        "label": "GI – large area / thin",
    },
    {
        "material_type": "PI",  "area": 14.0,  "thickness": 25.0,
        "material_cost": 3500.0, "unit_cost": 250.0,
        "label": "PI – small area / thick",
    },
    {
        "material_type": "GI",  "area": 8.5,   "thickness": 0.8,
        "material_cost": 190.0,  "unit_cost": 22.5,
        "label": "GI – small area / very thin",
    },
    {
        "material_type": "PI",  "area": 120.0, "thickness": 50.0,
        "material_cost": 29000.0,"unit_cost": 250.0,
        "label": "PI – very large / very thick",
    },
]


print("\n" + "=" * 60)
print("  INVENTORY WASTE — MODEL PREDICTIONS")
print("=" * 60)

run_knn_tests(SIMPLE_CASES)
run_lr_tests(SIMPLE_CASES)
run_rf_tests(EXTENDED_CASES)
run_svm_tests(EXTENDED_CASES)
run_ensemble_tests(EXTENDED_CASES)
# Waste Percentage Prediction for Metal Manufacturing 
### Solving Operational Challenges using AI
> Requirement for CSCI323 Modern Artificial Intelligence

## Installation
### 1. Clone the GitHub repository
```bash
git clone https://github.com/ofimikfra/CSCI323-Project.git
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

## Usage
> All models have been pre-trained and saved in `models/`.
```bash
python main.py
```
Runs predictions with simple and extended test cases for each model, as well as ensemble predictions (majority and soft vote).

## Reproducibility guide
1. Data files should be placed in `data\`.
2. Run each notebook in `notebooks\`:
    - `knn_model.ipynb`
    - `Logistic_Regression.ipynb`
    - `random_forest_model.ipynb`
    - `SVMClassificationModel.ipynb`
3. Run predictions.
```bash
python main.py
```
> All models were trained with `random_state=42` to ensure deterministic results. 

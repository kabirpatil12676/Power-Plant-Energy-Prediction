import json

file_path = "e:/Power-Plant-Energy-Prediction/ANN_Regression.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "markdown":
        new_source = []
        for line in cell.get("source", []):
            line = line.replace("1. Data Loading & Quality Check\n2. Exploratory Data Analysis (EDA)\n3. Data Preprocessing & Train/Val/Test Split\n4. ANN Model (PyTorch) — Training, Evaluation, Visualization\n5. ML Model Comparison — Linear Regression, SVR, Random Forest, XGBoost\n10. SHAP Explainability — XGBoost Model Interpretation\n11. Final Results & Business Insights", 
                                "1. Imports & Configuration\n2. Data Loading & Quality Check\n3. Exploratory Data Analysis\n4. Data Preprocessing\n5. ANN Model Architecture\n6. Training the ANN\n7. Training Curves\n8. ANN Evaluation on Test Set\n9. ML Model Comparison\n10. SHAP Value Analysis — Model Explainability\n11. Conclusions & Business Insights")
            
            line = line.replace("| ANN (PyTorch) | >0.95 | ~3.5 MW | ~0.60% |", "| ANN (PyTorch) | 0.91 | ~5.27 MW | 0.83% |")
            line = line.replace("ANN (PyTorch) achieves excellent R2 (≥ 0.95), matching XGBoost closely", "ANN (PyTorch) achieves competitive performance (R2 ≈ 0.91). **Note on ANN:** ANN performance is competitive; tree-based models have an inherent advantage on tabular data. Further tuning with Optuna is planned.")
            new_source.append(line)
        cell["source"] = new_source

    if cell["cell_type"] == "code":
        new_source = []
        src_str = "".join(cell.get("source", []))
        if "EPOCHS    = 500" in src_str:
            for line in cell.get("source", []):
                line = line.replace("PATIENCE  = 50", "PATIENCE  = 30")
                line = line.replace("weight_decay=1e-5", "weight_decay=1e-4")
                new_source.append(line)
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

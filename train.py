"""
=============================================================================
 Power Plant Energy Output Prediction -- Training Pipeline
=============================================================================
 Author  : Data Science Portfolio Project
 Dataset : UCI Combined Cycle Power Plant Dataset
 Task    : Regression -- Predict net hourly electrical energy output (PE)
 Models  : ANN (PyTorch), Linear Regression, SVR, Random Forest, XGBoost
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import shap

warnings.filterwarnings("ignore")

# --- Reproducibility ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Configuration -----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "powerplant_data.csv"
MODEL_DIR = "models"
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 30  # Early stopping patience

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"{'='*60}")
print(f"  Power Plant Energy Prediction -- Training Pipeline")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")


# =============================================================================
# 1. DATA LOADING & INSPECTION
# =============================================================================
print("[DATA] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Duplicates: {df.duplicated().sum()}")

# Remove duplicates if any
initial_rows = len(df)
df = df.drop_duplicates()
if len(df) < initial_rows:
    print(f"   Removed {initial_rows - len(df)} duplicate rows. New shape: {df.shape}")

print(f"\n   Dataset Statistics:")
print(df.describe().round(2).to_string())
print()


# =============================================================================
# 2. DATA SPLITTING
# =============================================================================
print("[SPLIT] Splitting data (70% train / 15% val / 15% test)...")
X = df.drop("PE", axis=1)
y = df["PE"]

FEATURE_NAMES = list(X.columns)
FEATURE_RANGES = {}
for col in FEATURE_NAMES:
    FEATURE_RANGES[col] = {
        "min": float(X[col].min()),
        "max": float(X[col].max()),
        "mean": float(X[col].mean()),
        "std": float(X[col].std()),
    }

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED
)
# Second split: 50% of temp = 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Val:   {X_val.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples")
print()


# =============================================================================
# 3. SCALING
# =============================================================================
print("[SCALE] Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"   Scaler saved -> {scaler_path}")
print()


# =============================================================================
# 4. PYTORCH DATA PREPARATION
# =============================================================================
print("[PREP]  Preparing PyTorch tensors & DataLoaders...")
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f"   Batches -- Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
print()


# =============================================================================
# 5. ANN MODEL DEFINITION
# =============================================================================
class PowerPlantANN(nn.Module):
    """
    Deep Artificial Neural Network for Power Plant Energy Prediction.
    """

    def __init__(self, input_dim=4):
        super(PowerPlantANN, self).__init__()
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            # Hidden Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            # Hidden Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            # Hidden Layer 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            # Output Layer
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# 6. TRAINING
# =============================================================================
print("[MODEL] Building ANN Model...")
model = PowerPlantANN(input_dim=X_train.shape[1]).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Architecture: 4 -> 256 -> 128 -> 64 -> 32 -> 1")
print(f"   Total Parameters: {total_params:,}")
print(f"   Trainable Parameters: {trainable_params:,}")
print()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Fixed L2 weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.3, patience=15
)

print(f"[TRAIN] Training for up to {EPOCHS} epochs (Early Stopping patience={PATIENCE})...")
print(f"   Optimizer: Adam (lr={LEARNING_RATE}, weight_decay=1e-4)")
print(f"   Scheduler: ReduceLROnPlateau (factor=0.3, patience=15)")
print(f"{'-'*60}")

train_losses = []
val_losses = []
lr_history = []
best_val_loss = float("inf")
early_stop_counter = 0
best_epoch = 0

for epoch in range(EPOCHS):
    # -- Training Phase --
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # -- Validation Phase --
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    lr_history.append(current_lr)

    # Step the scheduler
    scheduler.step(epoch_val_loss)

    # Print progress every 25 epochs
    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(
            f"   Epoch {epoch+1:>3}/{EPOCHS} | "
            f"Train Loss: {epoch_train_loss:>10.4f} | "
            f"Val Loss: {epoch_val_loss:>10.4f} | "
            f"LR: {current_lr:.6f}"
        )

    # -- Early Stopping & Checkpointing --
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_ann_model.pt"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"\n   [STOP] Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
            break

print(f"{'-'*60}")
print(f"   [OK] Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")
print()


# =============================================================================
# 7. ANN EVALUATION
# =============================================================================
print("[EVAL]  Evaluating ANN on Test Set...")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_ann_model.pt"), weights_only=True))
model.eval()

with torch.no_grad():
    X_test_device = X_test_t.to(DEVICE)
    test_preds = model(X_test_device).cpu().numpy().flatten()

y_test_np = y_test.values

ann_mae = mean_absolute_error(y_test_np, test_preds)
ann_mse = mean_squared_error(y_test_np, test_preds)
ann_rmse = np.sqrt(ann_mse)
ann_r2 = r2_score(y_test_np, test_preds)
ann_mape = np.mean(np.abs((y_test_np - test_preds) / y_test_np)) * 100

print(f"   MAE:  {ann_mae:.4f}")
print(f"   MSE:  {ann_mse:.4f}")
print(f"   RMSE: {ann_rmse:.4f}")
print(f"   R2:   {ann_r2:.4f}")
print(f"   MAPE: {ann_mape:.2f}%")
print()


# =============================================================================
# 8. ML MODEL COMPARISON
# =============================================================================
print("[COMPARE] Training comparison ML models...\n")

results = {
    "ANN (PyTorch)": {
        "MAE": round(ann_mae, 4),
        "MSE": round(ann_mse, 4),
        "RMSE": round(ann_rmse, 4),
        "R2": round(ann_r2, 4),
        "MAPE": round(ann_mape, 2),
    }
}

# -- Linear Regression --
print("   -> Linear Regression...")
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
results["Linear Regression"] = {
    "MAE": round(mean_absolute_error(y_test, y_pred_lr), 4),
    "MSE": round(mean_squared_error(y_test, y_pred_lr), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 4),
    "R2": round(r2_score(y_test, y_pred_lr), 4),
    "MAPE": round(np.mean(np.abs((y_test_np - y_pred_lr) / y_test_np)) * 100, 2),
}
print(f"      R2: {results['Linear Regression']['R2']}")

# -- SVR --
print("   -> Support Vector Regression...")
model_svr = SVR(kernel="rbf", C=100, epsilon=0.1)
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)
results["SVR (RBF)"] = {
    "MAE": round(mean_absolute_error(y_test, y_pred_svr), 4),
    "MSE": round(mean_squared_error(y_test, y_pred_svr), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_svr)), 4),
    "R2": round(r2_score(y_test, y_pred_svr), 4),
    "MAPE": round(np.mean(np.abs((y_test_np - y_pred_svr) / y_test_np)) * 100, 2),
}
print(f"      R2: {results['SVR (RBF)']['R2']}")

# -- Random Forest --
print("   -> Random Forest Regressor...")
model_rf = RandomForestRegressor(
    n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
results["Random Forest"] = {
    "MAE": round(mean_absolute_error(y_test, y_pred_rf), 4),
    "MSE": round(mean_squared_error(y_test, y_pred_rf), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 4),
    "R2": round(r2_score(y_test, y_pred_rf), 4),
    "MAPE": round(np.mean(np.abs((y_test_np - y_pred_rf) / y_test_np)) * 100, 2),
}
print(f"      R2: {results['Random Forest']['R2']}")

# -- XGBoost --
print("   -> XGBoost Regressor...")
model_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    random_state=SEED,
    verbosity=0,
)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
results["XGBoost"] = {
    "MAE": round(mean_absolute_error(y_test, y_pred_xgb), 4),
    "MSE": round(mean_squared_error(y_test, y_pred_xgb), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_xgb)), 4),
    "R2": round(r2_score(y_test, y_pred_xgb), 4),
    "MAPE": round(np.mean(np.abs((y_test_np - y_pred_xgb) / y_test_np)) * 100, 2),
}
print(f"      R2: {results['XGBoost']['R2']}")

# -- SHAP Feature Importance (XGBoost) --
print("   -> Computing SHAP Values for XGBoost...")
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = dict(zip(FEATURE_NAMES, mean_abs_shap.round(4).tolist()))

# -- Feature Importance (from Random Forest) --
feature_importance = dict(
    zip(FEATURE_NAMES, model_rf.feature_importances_.round(4).tolist())
)

print(f"\n{'-'*60}")
print(f"{'Model':<22} {'MAE':>8} {'MSE':>10} {'RMSE':>8} {'R2':>8} {'MAPE':>8}")
print(f"{'-'*60}")
for name, metrics in results.items():
    print(
        f"{name:<22} {metrics['MAE']:>8.4f} {metrics['MSE']:>10.4f} "
        f"{metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f} {metrics['MAPE']:>7.2f}%"
    )
print(f"{'-'*60}")

best_model_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n   [BEST] Best Model: {best_model_name} (R2 = {results[best_model_name]['R2']})")


# =============================================================================
# 9. SAVE METADATA
# =============================================================================
metadata = {
    "dataset": {
        "total_samples": len(df),
        "features": FEATURE_NAMES,
        "target": "PE",
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    },
    "feature_ranges": FEATURE_RANGES,
    "ann_config": {
        "architecture": "4 -> 256 -> BN -> LReLU -> D(0.15) -> 128 -> BN -> LReLU -> D(0.15) -> 64 -> BN -> LReLU -> D(0.15) -> 32 -> BN -> LReLU -> D(0.15) -> 1",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "best_epoch": best_epoch,
        "total_params": total_params,
    },
    "model_results": results,
    "feature_importance": feature_importance,
    "shap_importance": shap_importance,
    "training_history": {
        "train_losses": [round(l, 4) for l in train_losses],
        "val_losses": [round(l, 4) for l in val_losses],
        "lr_history": lr_history,
    },
    "best_model": best_model_name,
}

metadata_path = os.path.join(MODEL_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n[SAVE] Artifacts saved to '{MODEL_DIR}/':")
print(f"   +-- best_ann_model.pt")
print(f"   +-- scaler.pkl")
print(f"   +-- metadata.json")
print(f"\n{'='*60}")
print(f"  Training Pipeline Complete!")
print(f"{'='*60}")

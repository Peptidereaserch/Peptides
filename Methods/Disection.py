# multimodal_mlp_lstm_tree_model.py

import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================
CSV_PATH = "C:/Users/Waheg/Peptides/Peptide/Data/ML_Ready/mhc_class_I.csv"

TARGET_COL = "is_cancer"
DROP_COLS = ["id", "Tissue"]

SEED = 42
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SEED
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=DROP_COLS)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("Data shape:", X.shape)
print("\nClass balance:")
print(y.value_counts())
print(y.value_counts(normalize=True))


# ============================================================
# SPLIT DATA
# ============================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)


# ============================================================
# SCALE ORIGINAL FEATURES
# ============================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# TREE-BASED FEATURE GENERATION
# These models preprocess the data by creating extra probability features.
# ============================================================

tree_models = {
    "rf": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    ),

    "extra": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    ),

    "hgb": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        random_state=SEED
    )
}


def make_tree_features(X_train_raw, y_train_raw, X_val_raw, X_test_raw):
    train_meta = []
    val_meta = []
    test_meta = []

    for name, model in tree_models.items():
        print(f"\nCreating tree features from: {name}")

        # Out-of-fold predictions for train to reduce leakage
        oof_pred = np.zeros(len(X_train_raw))

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        for train_idx, holdout_idx in skf.split(X_train_raw, y_train_raw):
            X_fold_train = X_train_raw.iloc[train_idx]
            y_fold_train = y_train_raw.iloc[train_idx]

            X_fold_holdout = X_train_raw.iloc[holdout_idx]

            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)

            oof_pred[holdout_idx] = fold_model.predict_proba(X_fold_holdout)[:, 1]

        # Fit final model on full training data for val/test features
        final_model = model.__class__(**model.get_params())
        final_model.fit(X_train_raw, y_train_raw)

        val_pred = final_model.predict_proba(X_val_raw)[:, 1]
        test_pred = final_model.predict_proba(X_test_raw)[:, 1]

        train_meta.append(oof_pred.reshape(-1, 1))
        val_meta.append(val_pred.reshape(-1, 1))
        test_meta.append(test_pred.reshape(-1, 1))

    train_meta = np.hstack(train_meta)
    val_meta = np.hstack(val_meta)
    test_meta = np.hstack(test_meta)

    return train_meta, val_meta, test_meta


X_train_tree, X_val_tree, X_test_tree = make_tree_features(
    X_train,
    y_train,
    X_val,
    X_test
)


# ============================================================
# FINAL MULTIMODAL FEATURES
# Original scaled features + tree model features
# ============================================================
X_train_final = np.hstack([X_train_scaled, X_train_tree])
X_val_final = np.hstack([X_val_scaled, X_val_tree])
X_test_final = np.hstack([X_test_scaled, X_test_tree])

print("\nOriginal feature count:", X_train_scaled.shape[1])
print("Tree-added feature count:", X_train_tree.shape[1])
print("Final feature count:", X_train_final.shape[1])


# ============================================================
# TORCH TENSORS
# ============================================================
X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_final, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_tensor, y_val_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MULTIMODAL MODEL
# Branch 1: MLP learns tabular features
# Branch 2: LSTM learns feature-order interactions
# Then both branches merge
# ============================================================
class MultiModalMLPLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.mlp_branch = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
            bidirectional=True
        )

        self.merge = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        # MLP branch
        mlp_out = self.mlp_branch(x)

        # LSTM branch
        # Converts: [batch, features] -> [batch, features, 1]
        x_seq = x.unsqueeze(-1)

        lstm_out, _ = self.lstm(x_seq)

        # Take final LSTM output
        lstm_out = lstm_out[:, -1, :]

        # Combine both learning paths
        combined = torch.cat([mlp_out, lstm_out], dim=1)

        return self.merge(combined)


model = MultiModalMLPLSTM(input_size=X_train_final.shape[1]).to(DEVICE)

print("\nUsing device:", DEVICE)
print(model)


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================
num_zeros = (y_train_tensor == 0).sum().item()
num_ones = (y_train_tensor == 1).sum().item()

pos_weight = torch.tensor([num_zeros / num_ones], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=7
)


# ============================================================
# EVALUATION HELPERS
# ============================================================
def get_predictions(model, loader):
    model.eval()

    probs = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            logits = model(xb)
            batch_probs = torch.sigmoid(logits).cpu().numpy().ravel()

            probs.extend(batch_probs)
            labels.extend(yb.numpy().ravel())

    return np.array(probs), np.array(labels)


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_acc = 0

    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


# ============================================================
# TRAIN LOOP
# ============================================================
best_val_acc = 0
best_threshold = 0.5
best_model_state = None
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        # Noise injection for robustness
        noise = torch.randn_like(xb) * 0.01
        xb = xb + noise

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_prob, val_true = get_predictions(model, val_loader)

    threshold, val_acc = find_best_threshold(val_true, val_prob)
    val_pred = (val_prob >= threshold).astype(int)
    val_f1 = f1_score(val_true, val_pred)

    scheduler.step(val_acc)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Threshold: {threshold:.2f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_threshold = threshold
        best_model_state = model.state_dict()
        epochs_no_improve = 0

        torch.save(best_model_state, "best_multimodal_peptide_model.pt")

        print("New best model saved.")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("\nEarly stopping.")
        break


# ============================================================
# FINAL TEST
# ============================================================
model.load_state_dict(torch.load("best_multimodal_peptide_model.pt"))

test_prob, test_true = get_predictions(model, test_loader)
test_pred = (test_prob >= best_threshold).astype(int)

print("\n================ FINAL TEST RESULTS ================")
print("Best Validation Accuracy:", best_val_acc)
print("Best Threshold:", best_threshold)

print("\nTest Accuracy:", accuracy_score(test_true, test_pred))

print("\nClassification Report:")
print(classification_report(test_true, test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(test_true, test_pred))
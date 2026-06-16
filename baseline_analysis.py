import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = "NewDatasetWithValues.csv"
PLOT_DIR = "plots"


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({"—": np.nan, "�": np.nan})
    df["Peptide Sequence"] = df["Peptide Sequence"].astype(str).str.strip()
    df["Peptide Modifications"] = df["Peptide Modifications"].astype(str).str.strip()
    df["Best HLA Allele"] = df["Best HLA Allele"].astype(str).str.strip()
    df["MHC Class"] = df["MHC Class"].astype(str).str.strip()

    df["has_modification"] = df["Peptide Modifications"].apply(
        lambda x: 0
        if x in ["—", "nan", "None", "None", "", "nan"] or pd.isna(x)
        else 1
    )

    df["hla_group"] = df["Best HLA Allele"].fillna("UNKNOWN")
    top_hla = df["hla_group"].value_counts().nlargest(20).index
    df["hla_group"] = df["hla_group"].where(df["hla_group"].isin(top_hla), "OTHER")
    df["sequence_length"] = df["Peptide Sequence"].str.len()
    df["oxidation_flag"] = pd.to_numeric(df["oxidation_flag"], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "length",
        "molecular_weight",
        "charge_pH_7",
        "hydrophobicity_GRAVY",
        "isoelectric_point",
        "oxidation_flag",
        "has_modification",
        "sequence_length",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["feature_error", "Affinity % Rank", "Uniprot IDs"], errors="ignore")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [
        "length",
        "molecular_weight",
        "charge_pH_7",
        "hydrophobicity_GRAVY",
        "isoelectric_point",
        "oxidation_flag",
        "has_modification",
        "sequence_length",
    ]
    numeric_df = df[numeric_features].copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    categorical_df = pd.get_dummies(
        df[["MHC Class", "hla_group"]].astype(str),
        prefix=["MHC", "HLA"],
        drop_first=True,
    )

    X = pd.concat([numeric_df, categorical_df], axis=1)
    return X


def label_target(df: pd.DataFrame, target_column: str):
    y = df[target_column].astype(str).copy()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def plot_category_counts(df: pd.DataFrame, column: str) -> None:
    counts = df[column].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.head(40).plot.bar(ax=ax)
    ax.set_title(f"Top 40 counts for {column}")
    ax.set_ylabel("Count")
    ax.set_xlabel(column)
    plt.xticks(rotation=70, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"category_counts_{column}.png"))
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, numeric_columns):
    fig, axes = plt.subplots(len(numeric_columns) // 2 + len(numeric_columns) % 2, 2, figsize=(14, 16))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_columns):
        ax.hist(df[col].dropna(), bins=40, color="#4c72b0", edgecolor="black", alpha=0.8)
        ax.set_title(col)
    for extra in axes[len(numeric_columns) :]:
        extra.axis("off")
    fig.suptitle("Numeric Feature Distributions", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(os.path.join(PLOT_DIR, "numeric_distributions.png"))
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns):
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=90)
    ax.set_yticklabels(numeric_columns)
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
    plt.close(fig)


def plot_confusion_heatmap(cm, labels, filename: str, normalize: bool = True):
    labels = np.array(labels)
    if normalize:
        cm = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(filename.replace("_", " ").replace(".png", ""))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close(fig)


def plot_regression_scatter(y_true, y_pred, title, filename: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close(fig)


def plot_pca_clustering(X_scaled, labels, label_names, filename: str, title: str):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7, s=15)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close(fig)


def evaluate_classification(X, y, label_encoder, target_name: str):
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    if stratify is None:
        print(f"Warning: target '{target_name}' contains rare classes with only one sample. Using non-stratified train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=stratify, random_state=42
    )

    results = []

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    results.append(
        ("DummyClassifier", y_dummy, accuracy_score(y_test, y_dummy), balanced_accuracy_score(y_test, y_dummy))
    )

    lr = LogisticRegression(
        solver="saga",
        max_iter=600,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
    )
    lr.fit(X_train, y_train)
    y_lr = lr.predict(X_test)
    results.append(("LogisticRegression", y_lr, accuracy_score(y_test, y_lr), balanced_accuracy_score(y_test, y_lr)))

    ridge = RidgeClassifier()
    ridge.fit(X_train, y_train)
    y_ridge = ridge.predict(X_test)
    results.append(("RidgeClassifier", y_ridge, accuracy_score(y_test, y_ridge), balanced_accuracy_score(y_test, y_ridge)))

    print(f"\n=== Classification results for {target_name} ===")
    for name, y_pred, acc, bal_acc in results:
        print(f"{name}: accuracy={acc:.4f}, balanced_accuracy={bal_acc:.4f}")

    best_name, best_pred, _, _ = results[1]
    print("\nClassification report for LogisticRegression:\n")
    present_labels = np.unique(np.concatenate([y_test, best_pred]))
    present_names = label_encoder.inverse_transform(present_labels)
    print(
        classification_report(
            y_test,
            best_pred,
            labels=present_labels,
            target_names=present_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, best_pred, labels=present_labels)
    plot_confusion_heatmap(
        cm,
        label_encoder.classes_,
        f"confusion_matrix_{target_name}.png",
        normalize=True,
    )
    print(f"Saved confusion matrix heatmap at {os.path.join(PLOT_DIR, f'confusion_matrix_{target_name}.png')}")

    return X_test, y_test, best_pred


def evaluate_regression(df: pd.DataFrame, X: pd.DataFrame):
    y = df["hydrophobicity_GRAVY"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    dummy_rmse = mean_squared_error(y_test, y_dummy, squared=False)
    dummy_r2 = r2_score(y_test, y_dummy)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_ridge = ridge.predict(X_test)
    ridge_rmse = mean_squared_error(y_test, y_ridge, squared=False)
    ridge_r2 = r2_score(y_test, y_ridge)

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_linear = linear.predict(X_test)
    linear_rmse = mean_squared_error(y_test, y_linear, squared=False)
    linear_r2 = r2_score(y_test, y_linear)

    print("\n=== Regression results for hydrophobicity_GRAVY ===")
    print(f"DummyRegressor: RMSE={dummy_rmse:.4f}, R2={dummy_r2:.4f}")
    print(f"Ridge: RMSE={ridge_rmse:.4f}, R2={ridge_r2:.4f}")
    print(f"LinearRegression: RMSE={linear_rmse:.4f}, R2={linear_r2:.4f}")

    plot_regression_scatter(y_test, y_ridge, "Ridge Regression: Actual vs Predicted hydrophobicity_GRAVY", "regression_ridge_scatter.png")
    plot_regression_scatter(y_test, y_linear, "Linear Regression: Actual vs Predicted hydrophobicity_GRAVY", "regression_linear_scatter.png")

    return {
        "dummy": (dummy_rmse, dummy_r2),
        "ridge": (ridge_rmse, ridge_r2),
        "linear": (linear_rmse, linear_r2),
    }


def cluster_analysis(X: pd.DataFrame, df: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    cluster_range = [2, 3, 4, 5, 6, 8, 10]
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cluster_range, inertia, marker="o")
    ax.set_title("KMeans Elbow Plot")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "kmeans_elbow.png"))
    plt.close(fig)

    chosen_k = 5
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["kmeans_cluster"] = cluster_labels

    plot_pca_clustering(X_scaled, cluster_labels, np.unique(cluster_labels), "kmeans_pca_clusters.png", "KMeans PCA clusters")

    print("\n=== KMeans cluster composition ===")
    print(pd.crosstab(df["kmeans_cluster"], df["Disease"]).apply(lambda row: row / row.sum(), axis=1).round(3))
    return cluster_labels


def main():
    ensure_plot_dir()
    df = load_data(DATA_PATH)
    df = clean_data(df)

    numeric_columns = [
        "length",
        "molecular_weight",
        "charge_pH_7",
        "hydrophobicity_GRAVY",
        "isoelectric_point",
        "oxidation_flag",
        "has_modification",
        "sequence_length",
    ]

    print("Dataset shape:", df.shape)
    print("Numeric columns:", numeric_columns)

    plot_category_counts(df, "Disease")
    plot_category_counts(df, "Tissue")
    plot_category_counts(df, "MHC Class")
    plot_category_counts(df, "hla_group")
    plot_numeric_distributions(df, numeric_columns)
    plot_correlation_heatmap(df, numeric_columns)

    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    y_disease, disease_encoder = label_target(df, "Disease")
    evaluate_classification(X_scaled, y_disease, disease_encoder, "Disease")

    y_tissue, tissue_encoder = label_target(df, "Tissue")
    evaluate_classification(X_scaled, y_tissue, tissue_encoder, "Tissue")

    evaluate_regression(df, X_scaled)
    cluster_analysis(X_scaled, df)

    print(f"\nPlots were saved in the {PLOT_DIR} directory.")


if __name__ == "__main__":
    main()

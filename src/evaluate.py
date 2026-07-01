import os
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from src.preprocess import get_processed_data
from src.config import (
    BEST_MODEL_PATH,
    IMAGES_DIR
)
from src.utils import create_directory, print_section


def evaluate_model():

    print_section("Model Evaluation")

    create_directory(IMAGES_DIR)

    X_train, X_test, y_train, y_test, feature_names = get_processed_data()

    model = joblib.load(BEST_MODEL_PATH)

    predictions = model.predict(X_test)

    print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision : {precision_score(y_test, predictions):.4f}")
    print(f"Recall    : {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score  : {f1_score(y_test, predictions):.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

    # =====================================================
    # Confusion Matrix
    # =====================================================

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            IMAGES_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()

    # =====================================================
    # ROC Curve
    # =====================================================

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                IMAGES_DIR,
                "roc_curve.png"
            )
        )

        plt.close()

    # =====================================================
    # Feature Importance
    # =====================================================

    if hasattr(model, "feature_importances_"):

        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        )

        importance = importance.sort_values(
            ascending=False
        ).head(10)

        plt.figure(figsize=(9, 5))

        importance.plot(kind="bar")

        plt.title("Top 10 Feature Importance")

        plt.ylabel("Importance")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                IMAGES_DIR,
                "feature_importance.png"
            )
        )

        plt.close()

    print("\nEvaluation Complete.")

    print(f"\nGraphs saved inside '{IMAGES_DIR}'")


if __name__ == "__main__":

    evaluate_model()
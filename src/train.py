import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from src.preprocess import get_processed_data
from src.config import (
    BEST_MODEL_PATH,
    MODEL_DIR,
    RANDOM_STATE
)
from src.utils import create_directory, print_section


def train_models():

    print_section("Training Models")

    create_directory(MODEL_DIR)

    X_train, X_test, y_train, y_test, feature_names = get_processed_data()

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE
        )

    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    print("\nModel Performance\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"{name:<28} : {accuracy:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_name = name

    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n" + "=" * 60)
    print("Best Model")
    print("=" * 60)

    print(f"Model    : {best_name}")
    print(f"Accuracy : {best_accuracy:.4f}")

    print("\nSaved to")

    print(BEST_MODEL_PATH)


if __name__ == "__main__":

    train_models()
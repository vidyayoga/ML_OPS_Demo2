from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_processing import (
    load_data,
    prepare_data,
    create_preprocessor,
    split_data
)
from ml_model import get_models
# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
mlflow.set_experiment("Traditional_ML_Experiments")
# ---------------------------------------------------------
# Train and evaluate models
# ---------------------------------------------------------
def train_models():
    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------
    df = load_data()
    # -----------------------------------------------------
    # 2. Prepare data
    # -----------------------------------------------------
    X, y = prepare_data(df)
    # -----------------------------------------------------
    # 3. Create preprocessing object
    # -----------------------------------------------------
    preprocessor = create_preprocessor(X)
    # -----------------------------------------------------
    # 4. Train-test split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)
    # -----------------------------------------------------
    # 5. Get 3 models
    # -----------------------------------------------------
    models = get_models()
    print("\nModels to train:")
    for model_name in models:
        print(f"- {model_name}")
    # -----------------------------------------------------
    # 6. Train each model
    # -----------------------------------------------------
    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)
        # -------------------------------------------------
        # Create ML pipeline
        # -------------------------------------------------
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ]
        )
        # -------------------------------------------------
        # Start MLflow run
        # -------------------------------------------------
        with mlflow.start_run(run_name=model_name):
            # -------------------------------------------------
            # Log general parameters
            # -------------------------------------------------
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            # -------------------------------------------------
            # Log model-specific parameters
            # -------------------------------------------------
            if model_name == "Logistic Regression":
                mlflow.log_param("C", model.C)
                mlflow.log_param("max_iter", model.max_iter)
            elif model_name == "Random Forest":
                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators
                )
                mlflow.log_param(
                    "max_depth",
                    model.max_depth
                )
            elif model_name == "Gradient Boosting":
                mlflow.log_param(
                    "n_estimators",
                    model.n_estimators
                )
                mlflow.log_param(
                    "learning_rate",
                    model.learning_rate
                )
                mlflow.log_param(
                    "max_depth",
                    model.max_depth
                )
            # -------------------------------------------------
            # Train
            # -------------------------------------------------
            pipeline.fit(X_train, y_train)
            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------
            y_pred = pipeline.predict(X_test)
            # -------------------------------------------------
            # Calculate metrics
            # -------------------------------------------------
            accuracy = accuracy_score(
                y_test,
                y_pred
            )
            precision = precision_score(
                y_test,
                y_pred
            )
            recall = recall_score(
                y_test,
                y_pred
            )
            f1 = f1_score(
                y_test,
                y_pred
            )
            # -------------------------------------------------
            # Print results
            # -------------------------------------------------
            print("\nModel Performance:")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")
            # -------------------------------------------------
            # Log metrics to MLflow
            # -------------------------------------------------
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            # -------------------------------------------------
            # Confusion Matrix
            # -------------------------------------------------
            cm = confusion_matrix(
                y_test,
                y_pred
            )
            print("\nConfusion Matrix:")
            print(cm)
            # Create confusion matrix plot
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["No Churn", "Churn"]
            )
            disp.plot()
            plt.title(
                f"Confusion Matrix - {model_name}"
            )
            # Create safe filename
            safe_model_name = model_name.lower().replace(
                " ",
                "_"
            )
            confusion_matrix_path = (
                ARTIFACT_DIR /
                f"{safe_model_name}_confusion_matrix.png"
            )
            plt.savefig(
                confusion_matrix_path,
                bbox_inches="tight"
            )
            plt.close()
            # Log confusion matrix
            mlflow.log_artifact(
                str(confusion_matrix_path)
            )
            # -------------------------------------------------
            # Classification report
            # -------------------------------------------------
            report = classification_report(
                y_test,
                y_pred
            )
            print("\nClassification Report:")
            print(report)
            report_path = (
                ARTIFACT_DIR /
                f"{safe_model_name}_classification_report.txt"
            )
            with open(
                report_path,
                "w"
            ) as file:
                file.write(report)
            # Log classification report
            mlflow.log_artifact(
                str(report_path)
            )
            # -------------------------------------------------
            # Log trained model
            # -------------------------------------------------
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle"
            )
            print(
                f"\nMLflow run completed for {model_name}"
            )
# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    train_models()
    print("\n" + "=" * 60)
    print("All 3 models trained successfully!")
    print("=" * 60)
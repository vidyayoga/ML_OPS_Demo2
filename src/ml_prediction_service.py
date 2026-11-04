

from pathlib import Path


import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)




def load_best_model():
    runs = mlflow.search_runs(
        experiment_names=["Traditional_ML_Experiments"],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.f1_score DESC"],
        max_results=100,
    )


    if runs.empty or "metrics.f1_score" not in runs.columns:
        raise RuntimeError(
            "No completed ML runs found. "
            "Run src/ml_experiments.py first."
        )


    runs = runs.dropna(subset=["metrics.f1_score"])


    if runs.empty:
        raise RuntimeError(
            "No completed runs contain an F1 score."
        )


    best_run = runs.iloc[0]
    run_id = best_run["run_id"]


    model_uri = f"runs:/{run_id}/model"
    pipeline = mlflow.sklearn.load_model(model_uri)


    print("Best model loaded successfully")
    print("Model run ID:", run_id)
    print("F1 score:", best_run["metrics.f1_score"])
    print("Accuracy:", best_run.get("metrics.accuracy"))


    return pipeline, best_run




if __name__ == "__main__":
    pipeline, best_run = load_best_model()
    print("\nPipeline:")
    print(pipeline)
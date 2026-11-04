
from pathlib import Path
import sys


import pandas as pd
import streamlit as st
import mlflow




# ---------------------------------------------------------
# Project setup
# ---------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from data_processing import load_data, prepare_data
from ml_prediction_service import load_best_model
from llm_ops_mlflow import generate_recommendation




# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "LLMOps_Experiment_Demo"


mlflow.set_tracking_uri(TRACKING_URI)




# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------


st.set_page_config(
    page_title="Telecom Churn AI Assistant",
    page_icon="📊",
    layout="wide",
)


st.title("Telecom Customer Churn – ML + LLM")


st.write(
    "Predict customer churn using a trained ML model and generate "
    "retention recommendations using a local LLM. Successful LLM "
    "generations are logged to MLflow."
)


with st.sidebar:
    st.header("Configuration")


    prompt_version = st.selectbox(
        "Prompt version",
        options=["v1", "v2"],
        index=0,
    )


    st.caption(f"MLflow experiment: {EXPERIMENT_NAME}")
    st.caption(f"Tracking URI: {TRACKING_URI}")




# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------


@st.cache_data
def get_dataset():
    dataframe = load_data()
    features, target = prepare_data(dataframe)
    return dataframe, features, target




@st.cache_resource
def get_pipeline():
    return load_best_model()




try:
    with st.spinner("Loading dataset and trained ML model..."):
        df, X, y = get_dataset()
        pipeline, best_run = get_pipeline()


except Exception as exc:
    st.error("Could not load the dataset or trained ML model.")
    st.exception(exc)
    st.stop()




if X.empty:
    st.error("The prepared feature dataset is empty.")
    st.stop()




# ---------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------


st.subheader("Dataset Overview")


col1, col2, col3 = st.columns(3)


col1.metric("Dataset Rows", f"{len(df):,}")
col2.metric("Feature Columns", f"{X.shape[1]:,}")
col3.metric(
    "Selected Model Run",
    str(best_run.get("run_id", "Unknown"))[:12],
)


with st.expander("Preview Customer Data"):
    st.dataframe(df.head(10), use_container_width=True)




# ---------------------------------------------------------
# Select customer
# ---------------------------------------------------------


customer_index = st.number_input(
    "Customer Row Index",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1,
    help="Select a row from the prepared feature dataset.",
)


customer = X.iloc[[int(customer_index)]].copy()


with st.expander("View Selected Customer Features"):
    st.dataframe(customer, use_container_width=True)




# ---------------------------------------------------------
# Predict churn and generate recommendation
# ---------------------------------------------------------


if st.button(
    "Predict Churn and Generate Recommendation",
    type="primary",
):
    try:
        # Step 1: ML prediction
        with st.spinner("Predicting customer churn..."):
            prediction = int(pipeline.predict(customer)[0])


            probabilities = pipeline.predict_proba(customer)[0]
            classes = list(pipeline.classes_)


            if 1 not in classes:
                raise ValueError(
                    "The trained model does not contain class 1 for churn."
                )


            churn_probability = float(
                probabilities[classes.index(1)]
            )


        st.subheader("Churn Prediction")


        pred_col1, pred_col2 = st.columns(2)


        pred_col1.metric(
            "Prediction",
            "Churn" if prediction == 1 else "No Churn",
        )


        pred_col2.metric(
            "Churn Probability",
            f"{churn_probability:.2%}",
        )


        st.progress(churn_probability)


        # Step 2: LLM recommendation
        with st.spinner(
            f"Generating recommendations using prompt "
            f"{prompt_version.upper()}... The first run may take "
            "a few minutes on CPU."
        ):
            recommendation = generate_recommendation(
                customer=customer,
                prediction=prediction,
                churn_probability=churn_probability,
                prompt_version=prompt_version,
            )


        st.subheader("LLM Retention Recommendations")
        st.markdown(recommendation)


        st.success(
            "Recommendation generated. Check the LLMOps_Experiment "
            "in MLflow for the logged run and artifacts."
        )


    except Exception as exc:
        st.error(
            "The prediction or recommendation generation failed."
        )
        st.exception(exc)


        st.info(
            "Check that the prompt file exists, the LLM can load, "
            "and MLflow is using the correct database."
        )




# ---------------------------------------------------------
# View MLflow runs
# ---------------------------------------------------------


with st.expander("Check MLflow Runs"):


    if st.button("Refresh MLflow Runs"):
        try:
            experiment = mlflow.get_experiment_by_name(
                EXPERIMENT_NAME
            )


            if experiment is None:
                st.warning(
                    f"Experiment '{EXPERIMENT_NAME}' was not found."
                )


            else:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=100,
                )


                st.metric("Runs Found", len(runs))


                if not runs.empty:
                    display_columns = [
                        col
                        for col in [
                            "run_id",
                            "tags.prompt_version",
                            "params.prompt_version",
                            "metrics.input_tokens",
                            "metrics.output_tokens",
                            "metrics.latency_seconds",
                            "metrics.churn_probability",
                            "status",
                        ]
                        if col in runs.columns
                    ]


                    st.dataframe(
                        runs[display_columns]
                        if display_columns
                        else runs,
                        use_container_width=True,
                    )


                else:
                    st.info(
                        "No runs found in this experiment yet."
                    )


        except Exception as exc:
            st.error("Could not retrieve MLflow runs.")
            st.exception(exc)
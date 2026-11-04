from pathlib import Path
import time


import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM




# ---------------------------------------------------------
# Project paths and MLflow configuration
# ---------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EXPERIMENT_NAME = "LLMOps_Experiment_Demo"
MAX_NEW_TOKENS = 180


# IMPORTANT:
# Keep the path unescaped. Do not replace the space in "Gabrilla Sabadini"
# with "%20". The SQLite database is expected at PROJECT_ROOT / "mlflow.db".
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"


mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


_tokenizer = None
_model = None




# ---------------------------------------------------------
# Load LLM
# ---------------------------------------------------------


def load_llm():
    """Load the Hugging Face model once and reuse it."""
    global _tokenizer, _model


    if _tokenizer is None or _model is None:
        print(f"Loading language model: {MODEL_NAME}")


        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            # dtype replaces the deprecated torch_dtype argument
            # in current Transformers versions.
            dtype=torch.float32,
        )
        _model.eval()


        print("LLM loaded successfully.")
        print("Model:", MODEL_NAME)
        print("Device: CPU")


    return _tokenizer, _model




# ---------------------------------------------------------
# Prepare customer context
# ---------------------------------------------------------


def prepare_customer_context(customer, prediction, churn_probability):
    """Convert one customer's features and prediction into text."""
    customer_details = customer.iloc[0].to_dict()


    customer_details_text = "\n".join(
        f"- {key}: {value}" for key, value in customer_details.items()
    )


    return (
        "CUSTOMER CHURN ANALYSIS\n\n"
        f"Predicted churn: {'Yes' if prediction == 1 else 'No'}\n"
        f"Predicted churn probability: {churn_probability:.2%}\n\n"
        "CUSTOMER FEATURES\n"
        f"{customer_details_text}\n\n"
        "TASK\n"
        "Act as a telecom customer retention analyst. "
        "Provide exactly two practical, concise retention "
        "recommendations based on the supplied information. "
        "Explain the proposed action and why it may help. "
        "Do not claim that any feature caused churn. "
        "Do not invent customer information. "
        "If churn risk is low, recommend appropriate customer "
        "engagement rather than assuming the customer will leave. "
        "Number the recommendations 1 and 2."
    )




# ---------------------------------------------------------
# Generate recommendation and log an MLflow run
# ---------------------------------------------------------


def generate_recommendation(
    customer,
    prediction,
    churn_probability,
    prompt_version="v1",
):
    """
    Generate a recommendation and log its parameters, metrics,
    prompt, and output to the LLMOps_Experiment in MLflow.


    customer: one-row pandas DataFrame of customer features
    prediction: 0 = No Churn, 1 = Churn
    churn_probability: probability of Churn = 1
    prompt_version: v1 or v2
    """
    if customer is None or customer.empty:
        raise ValueError("Customer data cannot be empty.")


    if prediction not in (0, 1):
        raise ValueError("Prediction must be 0 or 1.")


    if not 0.0 <= churn_probability <= 1.0:
        raise ValueError("Churn probability must be between 0 and 1.")


    prompt_version = prompt_version.lower().strip()
    prompt_file = PROMPT_DIR / f"prompt_{prompt_version}.txt"


    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")


    prompt_template = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_template:
        raise ValueError(f"The prompt file is empty: {prompt_file}")


    context = prepare_customer_context(
        customer, prediction, churn_probability
    )


    if "{context}" in prompt_template:
        prompt = prompt_template.replace("{context}", context)
    else:
        prompt = f"{prompt_template}\n\n{context}"


    tokenizer, model = load_llm()


    messages = [
        {
            "role": "system",
            "content": (
                "You are a telecom customer retention analyst. "
                "Follow the user's instructions and use only the "
                "provided customer evidence."
            ),
        },
        {"role": "user", "content": prompt},
    ]


    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_token_count = int(inputs["input_ids"].shape[1])


    print("\n--- LLM GENERATION ---")
    print("Model:", MODEL_NAME)
    print("Prompt version:", prompt_version)
    print("Input tokens:", input_token_count)


    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_seconds = time.perf_counter() - start_time


    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    recommendation = tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    ).strip()
    output_token_count = int(generated_tokens.shape[0])


    print("Output tokens:", output_token_count)
    print(f"Latency: {latency_seconds:.3f} seconds")
    print("Recommendation:", repr(recommendation))


    if not recommendation:
        raise RuntimeError("The LLM returned an empty response.")


    # Log every successful generation as a separate MLflow run.
    # If another MLflow run is already active, create this run as a child.
    run_name = f"LLM_{prompt_version.upper()}{time.strftime('%Y%m%d%H%M%S')}"
    is_nested = mlflow.active_run() is not None


    with mlflow.start_run(run_name=run_name, nested=is_nested) as run:
        mlflow.set_tags(
            {
                "component": "llm_recommendation",
                "prompt_version": prompt_version,
                "llm_model": MODEL_NAME,
                "task": "telecom_churn_retention",
            }
        )


        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "prompt_version": prompt_version,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "prediction": "Churn" if prediction == 1 else "No Churn",
            }
        )


        mlflow.log_metrics(
            {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "latency_seconds": round(latency_seconds, 4),
                "churn_probability": float(churn_probability),
            }
        )


        # Save the exact prompt and response as run artifacts.
        mlflow.log_text(formatted_prompt, "artifacts/formatted_prompt.txt")
        mlflow.log_text(recommendation, "artifacts/recommendation.txt")
        mlflow.log_text(
            prompt_template, f"artifacts/prompt_{prompt_version}.txt"
        )


        print("\n--- MLFLOW LOGGING ---")
        print("Experiment:", EXPERIMENT_NAME)
        print("Tracking URI:", TRACKING_URI)
        print("Run ID:", run.info.run_id)
        print("Logged parameters, metrics, prompt, and recommendation.")


    return recommendation




# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------


if __name__ == "__main__":
    from data_processing import load_data, prepare_data
    from ml_prediction_service import load_best_model


    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", EXPERIMENT_NAME)
    print("MLflow database exists:", MLFLOW_DB.exists())


    # Load the trained MLOps pipeline.
    pipeline, best_run = load_best_model()


    # Load a real customer record.
    df = load_data()
    X, _ = prepare_data(df)
    customer = X.iloc[[0]].copy()


    # Predict churn.
    prediction = int(pipeline.predict(customer)[0])
    probabilities = pipeline.predict_proba(customer)[0]
    churn_probability = float(
        probabilities[list(pipeline.classes_).index(1)]
    )


    print("\n--- MLOPS PREDICTION ---")
    print("Model run ID:", best_run["run_id"])
    print("Prediction:", "Churn" if prediction == 1 else "No Churn")
    print(f"Churn probability: {churn_probability:.2%}")


    # Run V1 by default. Change to "v2" to log a V2 run.
    recommendation = generate_recommendation(
        customer=customer,
        prediction=prediction,
        churn_probability=churn_probability,
        prompt_version="v1",
    )


    print("\n--- LLM RECOMMENDATION ---")
    print(recommendation)
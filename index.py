import time
import json
import numpy as np
import pandas as pd
from typing import Mapping
from chassisml import ChassisModel
from chassis.builder import DockerBuilder, BuildOptions
import joblib

# ----------------------------------------------------------------------
# 1. Load your trained fraud detection model (previously saved with joblib or pickle)
# ----------------------------------------------------------------------
model = joblib.load("my_fraud_model.pkl")
# e.g., if you used joblib.dump(model, "my_fraud_model.pkl") in train_and_save_model.py


# ----------------------------------------------------------------------
# 2. Define the prediction function for ChassisML
# ----------------------------------------------------------------------
def predict(input_bytes: Mapping[str, bytes]) -> dict[str, bytes]:
    """
    ChassisML 'process_fn' that:
      1. Reads JSON input from input_bytes['input']
      2. Converts it to a DataFrame matching your training columns
      3. Runs the RandomForest model inference
      4. Returns predictions (class & probability) as JSON
    """

    # 1. Parse JSON input from the "input" key
    #    The user sends the data as a JSON-encoded array of rows
    input_json_str = input_bytes["input"].decode("utf-8")  # convert bytes -> str
    data_rows = json.loads(input_json_str)  # parse JSON -> Python list

    # 2. Convert the list into a DataFrame
    #    IMPORTANT: match these columns exactly to those used at training time.
    columns = [
        "Income",
        "Security_code",
        "Months_to_expiry",
        "Profession_ENGINEER",
        "Profession_LAWYER",
    ]
    X_new = pd.DataFrame(data_rows, columns=columns)

    # 3. Model inference
    y_pred = model.predict(X_new)  # predicted class
    y_prob = model.predict_proba(X_new)[:, 1]  # probability of Fraud (class = 1)

    # 4. Structure output for ChassisML
    results = []
    for c, p in zip(y_pred, y_prob):
        results.append({"predicted_class": int(c), "fraud_probability": float(p)})

    # Return a dict of {filename: file_contents}, here JSON with key "results.json"
    return {"results.json": json.dumps(results).encode("utf-8")}


# ----------------------------------------------------------------------
# 3. Create a ChassisModel object and specify dependencies & metadata
# ----------------------------------------------------------------------
chassis_model = ChassisModel(process_fn=predict)

# Add Python dependencies needed by your inference code
chassis_model.add_requirements(
    ["scikit-learn", "numpy", "pandas", "joblib", "imbalanced-learn"]
)

# Metadata is optional, but helps describe your model
chassis_model.metadata.model_name = "Fraud Detection Model"
chassis_model.metadata.model_version = "1.0.0"
chassis_model.metadata.add_input(
    key="input",
    accepted_media_types=["application/json"],
    max_size="10M",
    description="JSON array of rows, each row matching the trained feature columns",
)
chassis_model.metadata.add_output(
    key="results.json",
    media_type="application/json",
    max_size="1M",
    description="Array of predicted classes and fraud probabilities",
)

# ----------------------------------------------------------------------
# 4. Quick Local Test
# ----------------------------------------------------------------------
# Create a sample input that matches your training columns (2 example rows here):
sample_input = [[5000, 35, 4, 1, 0], [7000, 50, 3, 0, 1]]  # row 1  # row 2
# Convert the sample input to JSON bytes for chassis_model.test()
input_payload = {"input": json.dumps(sample_input).encode("utf-8")}

# Test the model locally (in-memory)
results = chassis_model.test(input_payload)
print("\nLocal Test Results:")
print(results)

# ----------------------------------------------------------------------
# 5. Build a Docker container with ChassisML
# ----------------------------------------------------------------------
options = BuildOptions(labels={"purpose": "fraud-detection"})
builder = DockerBuilder(chassis_model, options)

start_time = time.time()
res = builder.build_image(name="fraud-detection-model", tag="1.0.0", show_logs=True)
end_time = time.time()

print("\nDocker Build Output:")
print(res)
print(f"Container image built in {round((end_time - start_time)/60, 5)} minutes")

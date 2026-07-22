import os

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------------
# Paths — resolved relative to this file, not the process's working
# directory, so `python app.py` works the same no matter where it's run from.
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "human_activity_model.keras")

# 561 sensor features + "subject" = 562. Confirmed directly from the model
# file's own config: InputLayer batch_shape is [None, 562, 1].
EXPECTED_FEATURES = 562

# ----------------------------------------------------------------------------
# NOTE ON ACTIVITY LABELS
# The model's real config.json (inspected directly from the .keras file)
# confirms the final Dense layer has 7 units — matching the notebook's
# `to_categorical(y_train, num_classes=7)`. But the raw train.csv/test.csv
# were never provided, and the notebook never printed `encoder.classes_` or
# `sorted(train["Activity"].unique())`, so the exact 7 label strings and
# their order still can't be confirmed from what's available.
#
# The 6 labels below are the standard UCI HAR activity names; index 6 is a
# placeholder. Before trusting predictions, run this in the SAME
# environment/notebook that trained the model:
#
#     print(sorted(train["Activity"].unique()))
#
# scikit-learn's LabelEncoder assigns indices in that sorted (alphabetical)
# order, so CLASSES[i] must equal the i-th entry of that sorted list.
# ----------------------------------------------------------------------------
CLASSES = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS",
    "UNKNOWN_CLASS_7",  # placeholder — verify against encoder.classes_
]

app = Flask(__name__)

_model = None


def get_model():
    """Load the Keras model once and cache it in a module-level global."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Place "
                f"human_activity_model.keras in the same folder as app.py."
            )
        _model = load_model(MODEL_PATH)
    return _model


def preprocess(df: pd.DataFrame):
    """Mirrors the training notebook's preprocessing exactly:
      - drop only 'Activity' (the target column) — never 'subject', which
        the notebook keeps as a real input feature
        (`X_train = train.drop("Activity", axis=1)`)
      - coerce every column to numeric, exactly like
        `X_train.apply(pd.to_numeric, errors='coerce')`
      - fill any resulting NaNs (see docstring note below)
      - validate the 562-feature width the model's InputLayer expects
      - reshape to (n_samples, 562, 1), matching
        `X_train.reshape(X_train.shape[0], X_train.shape[1], 1)`
    """
    data = df.copy()

    if "Activity" in data.columns:
        data = data.drop(columns=["Activity"])

    data = data.apply(pd.to_numeric, errors="coerce")

    warning = None
    if data.isnull().values.any():
        # The notebook fits SimpleImputer(strategy="mean") on the training
        # set and reuses those exact means at inference time. That fitted
        # imputer wasn't saved/uploaded, so its per-column training means
        # aren't available here. As a fallback, missing values are filled
        # with this uploaded batch's own column means — NOT guaranteed
        # identical to the notebook's training-time imputation. For exact
        # parity, save the imputer with joblib during training and load it
        # here instead of this fallback.
        warning = (
            "Missing values were detected and filled using this file's own "
            "column means (not the original training-set means, which "
            "weren't provided). Predictions on affected rows may be less "
            "reliable."
        )
        data = data.fillna(data.mean(numeric_only=True))

    if data.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Column mismatch: model expects {EXPECTED_FEATURES} feature "
            f"columns (561 sensor features + 'subject'), but the uploaded "
            f"file has {data.shape[1]} after dropping 'Activity'. Check "
            f"that the CSV matches the training data's columns."
        )

    X = data.values.astype(np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, warning


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None
    warning = None
    preview = None

    if request.method == "POST":
        uploaded = request.files.get("file")

        if uploaded is None or uploaded.filename == "":
            error = "Please choose a CSV file first."
        elif not uploaded.filename.lower().endswith(".csv"):
            error = "Please upload a file with a .csv extension."
        else:
            try:
                df = pd.read_csv(uploaded)
                preview = df.head().to_html(classes="preview-table", index=False)

                X, warning = preprocess(df)
                

                model = get_model()
                probs = model.predict(X, verbose=0)
                pred_idx = np.argmax(probs, axis=1)

                # Warn (don't silently mislabel) if the model's real output
                # size ever stops matching CLASSES.
                if probs.shape[1] != len(CLASSES):
                    warning = (
                        (warning + " " if warning else "")
                        + f"Model has {probs.shape[1]} output classes but "
                        f"CLASSES has {len(CLASSES)} labels defined — "
                        f"verify against encoder.classes_ from training."
                    )

                predictions = []
                i = 0
                p= pred_idx[0]
                label = (
                        CLASSES[p] if p < len(CLASSES) else f"Class index {p} (unmapped)"
                    )
                confidence = float(probs[i][p])
                predictions.append(
                        {
                            "sample":1,
                            "label": label,
                            "confidence": f"{confidence:.2%}",
                        }
                    )
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        warning=warning,
        preview=preview,
    )


if __name__ == "__main__":
    # Load the model eagerly at boot, so a missing file or shape problem
    # surfaces immediately in the terminal instead of on the first upload.
    get_model()

    # debug=True must never run in production: it exposes Werkzeug's
    # interactive debugger (arbitrary code execution) to anyone who can
    # trigger an unhandled exception, and its reloader double-loads the
    # model in two processes. Controlled by FLASK_DEBUG so local dev can
    # still opt in with: FLASK_DEBUG=1 python app.py
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"

    # 0.0.0.0 (not the 127.0.0.1 default) so the app is reachable from
    # outside the container on Docker/Render/Railway/Heroku-style hosts.
    # PORT is read from the environment because most platforms assign it
    # dynamically rather than letting you hardcode 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
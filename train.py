
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. CONFIGURATION
# ==========================================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = Path("data/SMSSpamCollection")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MAX_WORDS = 5000
MAX_LENGTH = 60
EPOCHS = 15
BATCH_SIZE = 32

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading SMS dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

df = df.dropna()
df = df.drop_duplicates(subset="message")

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\nTotal messages:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

# ==========================================
# 3. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.125,
    random_state=42,
    stratify=y_train
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# ==========================================
# 4. TEXT VECTORIZATION
# ==========================================

print("\nPreparing text sequences...")

vectorizer = layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)

# Learn vocabulary only from training data
vectorizer.adapt(X_train.to_numpy())

vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)

with open(
    MODEL_DIR / "vocab.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(vocab, f)

X_train_seq = vectorizer(
    X_train.to_numpy()
).numpy()

X_val_seq = vectorizer(
    X_val.to_numpy()
).numpy()

X_test_seq = vectorizer(
    X_test.to_numpy()
).numpy()

# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

print("\nCalculating class weights...")

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train.to_numpy()
)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("Class weights:", class_weights)

# ==========================================
# 6. BUILD ANN MODEL
# ==========================================

ann = Sequential([
    layers.Input(shape=(MAX_LENGTH,)),
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=32,
        mask_zero=True
    ),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 7. BUILD LSTM MODEL
# ==========================================

lstm = Sequential([
    layers.Input(shape=(MAX_LENGTH,)),
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=32,
        mask_zero=True
    ),
    layers.LSTM(32),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

lstm.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 8. TRAIN AND EVALUATE BOTH MODELS
# ==========================================

models = {
    "ANN": ann,
    "LSTM": lstm
}

results = []
threshold_data = {}

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # ======================================
    # 9. FIND BEST THRESHOLD
    # ======================================

    print("\nFinding optimal threshold...")

    val_probabilities = model.predict(
        X_val_seq,
        verbose=0
    ).ravel()

    thresholds = np.arange(0.20, 0.81, 0.05)

    best_threshold = max(
        thresholds,
        key=lambda t: f1_score(
            y_val,
            (val_probabilities >= t).astype(int),
            zero_division=0
        )
    )

    best_threshold = float(best_threshold)

    print(
        "Best threshold:",
        round(best_threshold, 2)
    )

    threshold_data[name] = best_threshold

    # ======================================
    # 10. EVALUATE TEST DATA
    # ======================================

    probabilities = model.predict(
        X_test_seq,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test, predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nRESULTS FOR", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Threshold": round(best_threshold, 2)
    })

    # ======================================
    # 11. SAVE MODEL
    # ======================================

    model.save(
        str(MODEL_DIR / f"{name.lower()}.keras")
    )

    print(name, "model saved successfully!")

# ==========================================
# 12. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    MODEL_DIR / "results.csv",
    index=False
)

with open(
    MODEL_DIR / "thresholds.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(threshold_data, f, indent=4)

print("\n================================")
print("FINAL MODEL COMPARISON")
print("================================")

print(results_df.round(4).to_string(index=False))

# ==========================================
# 13. TEST CUSTOM MESSAGES
# ==========================================

print("\n================================")
print("TESTING SAMPLE SMS MESSAGES")
print("================================")

sample_messages = [
    "You have won a lottery!!",
    "Congratulations! Claim your cash prize now!",
    "URGENT! You have won $1000. Call now!",
    "Are you coming to college tomorrow?",
    "Please send me the assignment notes."
]

sample_sequences = vectorizer(
    np.array(sample_messages)
).numpy()

for name, model in models.items():

    print("\nMODEL:", name)

    sample_probabilities = model.predict(
        sample_sequences,
        verbose=0
    ).ravel()

    threshold = threshold_data[name]

    for message, probability in zip(
        sample_messages,
        sample_probabilities
    ):

        prediction = (
            "SPAM"
            if probability >= threshold
            else "HAM"
        )

        print("\nMessage:", message)
        print("Prediction:", prediction)
        print(
            "Spam probability:",
            f"{probability * 100:.2f}%"
        )

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("================================")

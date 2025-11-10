# DeepFlightAI
Predicting aircraft failures and accidents using deep learning with SHAP explainability.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1: Load dataset
file_path = r"C:\Users\BLESSY\Documents\DeepFlightAI\airline_accidents.csv"
df = pd.read_csv(file_path, low_memory=False)
print("Columns:", df.columns)
print("Dataset shape:", df.shape)
print(df.head())


#2: Create target variable and features
# Function to safely convert to numeric
def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)


# Convert all relevant columns to numeric, handling string values
numeric_cols = [
    'Total Fatal Injuries', 'Total Serious Injuries',
    'Total Minor Injuries', 'Total Uninjured',
    'Latitude', 'Longitude', 'Number of Engines'
]

print("Converting columns to numeric...")
for col in numeric_cols:
    if col in df.columns:
        original_type = df[col].dtype
        df[col] = safe_numeric(df[col])
        print(f"Converted {col}: {original_type} -> {df[col].dtype}")
    else:
        print(f"Column {col} not found in dataset")

#1. if there are any injuries/fatalities, 0 otherwise
df['Total_Casualties'] = df['Total Fatal Injuries'] + df['Total Serious Injuries'] + df['Total Minor Injuries']
df['Accident_Occurred'] = (df['Total_Casualties'] > 0).astype(int)

print(f"Accident distribution:\n{df['Accident_Occurred'].value_counts()}")

# Select only columns that exist in the dataset
available_cols = [col for col in numeric_cols if col in df.columns]
print(f"Available columns for features: {available_cols}")

if not available_cols:
    print("No suitable numeric columns found. Please check your dataset columns:")
    print(df.columns.tolist())
    print("\nFirst few rows of data:")
    print(df.head())
    exit()

# Prepare features
features = df[available_cols].values
target = df['Accident_Occurred'].values

# Remove rows where target is NaN or features contain NaN
valid_indices = ~(np.isnan(target) | np.isnan(features).any(axis=1))
features = features[valid_indices]
target = target[valid_indices].astype(int)

print(f"Final dataset shape: {features.shape}")
print(f"Target distribution: {np.bincount(target)}")

#3: Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#4:Create sequences for LSTM
SEQ_LEN = 30  # Reduced sequence length


def create_sequences(data, labels, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len + 1):
        X.append(data[i:i + seq_len])
        y.append(labels[i + seq_len - 1])  # Predict the last label in sequence
    return np.array(X), np.array(y)


# Only create sequences if we have enough data
if len(features_scaled) >= SEQ_LEN:
    X, y = create_sequences(features_scaled, target)
    print("Sequence shape:", X.shape)
    print("Labels shape:", y.shape)
    print(f"Sequence target distribution: {np.bincount(y)}")

    #5: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #6: Build simplified LSTM model
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    model.summary()

    #STEP 7: Train model with reduced epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=5,  # Reduced epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    #8: Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    #9: Evaluate model
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    #10: Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Training history
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Training accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['No Accident', 'Accident'],
                yticklabels=['No Accident', 'Accident'])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    # Prediction probabilities distribution
    axes[1, 1].hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    axes[1, 1].set_title('Prediction Probabilities Distribution')
    axes[1, 1].set_xlabel('Probability of Accident')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    #Feature importance analysis
    def analyze_feature_importance(model, X_test, feature_names):
        """Simple feature importance based on prediction variance"""
        baseline_pred = model.predict(X_test)
        importance_scores = []

        for i in range(X_test.shape[2]):
            # Create a copy and shuffle one feature
            X_modified = X_test.copy()
            np.random.shuffle(X_modified[:, :, i])
            modified_pred = model.predict(X_modified)

            # Calculate importance as variance in predictions
            importance = np.var(baseline_pred - modified_pred)
            importance_scores.append(importance)

        return np.array(importance_scores)


    # Calculate feature importance
    feature_names = available_cols  # Use only available columns

    importance_scores = analyze_feature_importance(model, X_test, feature_names)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance_scores)[::-1]
    plt.bar(range(len(importance_scores)), importance_scores[indices])
    plt.title('Feature Importance for Accident Prediction')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(range(len(importance_scores)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

    #12: Example predictions
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)

    # Show some example predictions
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        actual = y_test[idx]
        predicted_prob = y_pred_proba[idx][0]
        predicted = y_pred[idx][0]

        print(f"\nSample {idx}:")
        print(f"  Actual: {'Accident' if actual == 1 else 'No Accident'}")
        print(f"  Predicted: {'Accident' if predicted == 1 else 'No Accident'}")
        print(f"  Confidence: {predicted_prob:.3f}")
        print(f"  Correct: {'✓' if actual == predicted else '✗'}")

else:
    print(f"Not enough data to create sequences. Need at least {SEQ_LEN} rows, but got {len(features_scaled)}")

    # Alternative: Simple dense neural network
    print("Using simple dense neural network instead...")

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42, stratify=target
    )

    # Simple dense model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(features_scaled.shape[1],)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\n" + "=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
print("Reduced model complexity (fewer layers)")
print("Reduced training epochs (5 epochs with early stopping)")
print("Binary classification for accident prediction")
print("Added dropout for regularization")
print("Comprehensive evaluation metrics")
print("Feature importance analysis")
print("Fallback to dense network for small datasets")

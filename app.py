import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Data
print("Loading data...")
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# 2. Preprocessing & Cleaning
# Labeling: 1 for Fake, 0 for Real
fake_df['label'] = 1
true_df['label'] = 0

# IMPORTANT: Remove the "Publisher/Location" tag from True news 
# This prevents the model from just looking for the word "Reuters"
def remove_publisher(text):
    # Regex to remove "WASHINGTON (Reuters) - " pattern
    return re.sub(r'^.*? - ', '', text)

true_df['text'] = true_df['text'].apply(remove_publisher)

# Combine and Shuffle
df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Tokenization
max_features = 10000 
max_len = 300 # Limit each article to 300 words

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['text'].values)
X_seq = tokenizer.texts_to_sequences(df['text'].values)
X_pad = pad_sequences(X_seq, maxlen=max_len)
y = df['label'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# 4. Build RNN (Bidirectional LSTM)
model = Sequential([
    Embedding(max_features, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Training (Reduced Epochs)
print("\nStarting Training (2 Epochs)...")
model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.1)

# 6. Final Evaluation Metrics
print("\n--- Final Metrics ---")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))

# 7. Real-World Test Function
def predict_article(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    result = "FAKE" if pred > 0.5 else "REAL"
    print(f"\nAnalysis Result: {result} (Confidence: {pred if pred > 0.5 else 1-pred:.2%})")

# Example usage:
# predict_article("Your sample news text here...")
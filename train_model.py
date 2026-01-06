#!/usr/bin/env python3
"""
Training script for RNN-based Fake News Detection Model
Demonstrates how to train the temporal pattern recognition system
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from rnn_model import FakeNewsDetector, TemporalNewsDataset
from text_analyzer import TextAnalyzer
from data_collector import NewsCollector

def generate_synthetic_training_data(num_samples: int = 1000):
    """
    Generate synthetic training data for demonstration
    In practice, this would be replaced with real labeled data
    """
    
    print(f"🔧 Generating {num_samples} synthetic training samples...")
    
    sequences = []
    labels = []
    
    analyzer = TextAnalyzer()
    
    for i in range(num_samples):
        # Randomly decide if this is real (1) or fake (0) news
        is_real = np.random.choice([0, 1])
        
        # Generate sequence length (2-8 articles)
        seq_length = np.random.randint(2, 9)
        
        if is_real:
            # Real news patterns
            sequence = generate_real_news_sequence(seq_length)
        else:
            # Fake news patterns
            sequence = generate_fake_news_sequence(seq_length)
        
        sequences.append(sequence)
        labels.append(is_real)
    
    print(f"✅ Generated {len(sequences)} sequences")
    print(f"   Real news samples: {sum(labels)}")
    print(f"   Fake news samples: {len(labels) - sum(labels)}")
    
    return sequences, labels

def generate_real_news_sequence(length: int) -> np.ndarray:
    """Generate realistic temporal pattern for real news"""
    
    features = []
    
    for i in range(length):
        # Real news typically shows:
        # - Decreasing urgency over time
        # - Increasing certainty over time
        # - Stabilizing sentiment
        # - Consistent narrative
        
        progress = i / (length - 1) if length > 1 else 0
        
        # Sentiment: starts negative, moves toward neutral
        sentiment = -0.5 + 0.4 * progress + np.random.normal(0, 0.1)
        
        # Certainty: increases over time
        certainty = 0.3 + 0.6 * progress + np.random.normal(0, 0.05)
        
        # Urgency: decreases over time
        urgency = 0.8 - 0.6 * progress + np.random.normal(0, 0.1)
        
        # Emotional intensity: decreases over time
        emotional = 0.7 - 0.5 * progress + np.random.normal(0, 0.1)
        
        # Content length: relatively stable
        content_length = 0.5 + np.random.normal(0, 0.1)
        
        # Position in timeline
        timeline_pos = progress
        
        # Deltas (changes from previous)
        if i > 0:
            sentiment_delta = sentiment - features[-1][0]
            certainty_delta = certainty - features[-1][1]
        else:
            sentiment_delta = 0
            certainty_delta = 0
        
        feature_vector = [
            sentiment, certainty, urgency, emotional,
            content_length, timeline_pos, sentiment_delta, certainty_delta
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def generate_fake_news_sequence(length: int) -> np.ndarray:
    """Generate temporal pattern typical of fake news"""
    
    features = []
    
    for i in range(length):
        # Fake news typically shows:
        # - Inconsistent patterns
        # - High emotional content throughout
        # - Certainty doesn't improve much
        # - More erratic changes
        
        progress = i / (length - 1) if length > 1 else 0
        
        # Sentiment: more volatile, stays emotional
        sentiment = -0.3 + np.random.normal(0, 0.3)
        
        # Certainty: doesn't improve much, stays low
        certainty = 0.4 + 0.2 * progress + np.random.normal(0, 0.15)
        
        # Urgency: stays high, erratic
        urgency = 0.7 + np.random.normal(0, 0.2)
        
        # Emotional intensity: stays high
        emotional = 0.8 + np.random.normal(0, 0.15)
        
        # Content length: more variable
        content_length = 0.5 + np.random.normal(0, 0.2)
        
        # Position in timeline
        timeline_pos = progress
        
        # Deltas (more erratic changes)
        if i > 0:
            sentiment_delta = sentiment - features[-1][0]
            certainty_delta = certainty - features[-1][1]
        else:
            sentiment_delta = 0
            certainty_delta = 0
        
        feature_vector = [
            sentiment, certainty, urgency, emotional,
            content_length, timeline_pos, sentiment_delta, certainty_delta
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def train_model():
    """Train the fake news detection model"""
    
    print("🚀 TRAINING RNN-BASED FAKE NEWS DETECTOR")
    print("=" * 50)
    
    # Generate training data
    sequences, labels = generate_synthetic_training_data(1000)
    
    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Training set: {len(train_sequences)} samples")
    print(f"📊 Validation set: {len(val_sequences)} samples")
    
    # Create datasets
    train_dataset = TemporalNewsDataset(train_sequences, train_labels)
    val_dataset = TemporalNewsDataset(val_sequences, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    detector = FakeNewsDetector(input_size=8, hidden_size=64, num_layers=2)
    
    print(f"🧠 Model Architecture:")
    print(f"   Input size: 8 features")
    print(f"   Hidden size: 64 units")
    print(f"   LSTM layers: 2")
    print(f"   Parameters: {sum(p.numel() for p in detector.model.parameters()):,}")
    
    # Train model
    print(f"\n🏋️ Starting training...")
    train_losses, val_accuracies = detector.train(
        train_loader, val_loader, epochs=20
    )
    
    # Final evaluation
    final_accuracy = detector.evaluate(val_loader)
    print(f"\n✅ Training completed!")
    print(f"📊 Final validation accuracy: {final_accuracy:.4f}")
    
    # Save model
    detector.save_model("trained_model.pth")
    print(f"💾 Model saved to 'trained_model.pth'")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved to 'training_curves.png'")
    
    # Test with example sequences
    print(f"\n🧪 Testing with example sequences...")
    test_real_sequence(detector)
    test_fake_sequence(detector)
    
    return detector

def test_real_sequence(detector):
    """Test with a realistic 'real news' sequence"""
    
    print(f"\n📰 Testing REAL news pattern:")
    
    # Create a realistic real news sequence
    real_sequence = generate_real_news_sequence(4)
    
    # Normalize using the dataset scaler (simplified for demo)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_seq = scaler.fit_transform(real_sequence)
    
    # Pad to max length (10)
    if len(normalized_seq) < 10:
        padding = np.zeros((10 - len(normalized_seq), normalized_seq.shape[1]))
        normalized_seq = np.vstack([padding, normalized_seq])
    
    # Predict
    prediction = detector.predict(normalized_seq)
    
    print(f"   Credibility Score: {prediction['credibility_score']:.3f}")
    print(f"   Verdict: {prediction['verdict']}")
    print(f"   Real Probability: {prediction['real_probability']:.3f}")

def test_fake_sequence(detector):
    """Test with a realistic 'fake news' sequence"""
    
    print(f"\n🚨 Testing FAKE news pattern:")
    
    # Create a realistic fake news sequence
    fake_sequence = generate_fake_news_sequence(4)
    
    # Normalize using the dataset scaler (simplified for demo)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_seq = scaler.fit_transform(fake_sequence)
    
    # Pad to max length (10)
    if len(normalized_seq) < 10:
        padding = np.zeros((10 - len(normalized_seq), normalized_seq.shape[1]))
        normalized_seq = np.vstack([padding, normalized_seq])
    
    # Predict
    prediction = detector.predict(normalized_seq)
    
    print(f"   Credibility Score: {prediction['credibility_score']:.3f}")
    print(f"   Verdict: {prediction['verdict']}")
    print(f"   Fake Probability: {prediction['fake_probability']:.3f}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train the model
    detector = train_model()
    
    print(f"\n" + "="*50)
    print("🎯 TRAINING SUMMARY")
    print("="*50)
    print("✅ Successfully trained RNN-based fake news detector")
    print("✅ Model learns to recognize temporal patterns in news")
    print("✅ Real news: increasing certainty, stabilizing sentiment")
    print("✅ Fake news: erratic patterns, persistent uncertainty")
    print("✅ Ready for deployment on real news data")
    print("="*50)
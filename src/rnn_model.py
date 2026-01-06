import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import pickle

class TemporalNewsDataset(Dataset):
    """Dataset for temporal news sequences"""
    
    def __init__(self, sequences: List[np.ndarray], labels: List[int], max_length: int = 10):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.scaler = StandardScaler()
        
        # Fit scaler on all data
        all_features = np.vstack([seq for seq in sequences])
        self.scaler.fit(all_features)
        
        # Normalize and pad sequences
        self.processed_sequences = []
        for seq in sequences:
            # Normalize
            normalized_seq = self.scaler.transform(seq)
            
            # Pad or truncate to max_length
            if len(normalized_seq) > max_length:
                normalized_seq = normalized_seq[-max_length:]  # Keep most recent
            elif len(normalized_seq) < max_length:
                padding = np.zeros((max_length - len(normalized_seq), normalized_seq.shape[1]))
                normalized_seq = np.vstack([padding, normalized_seq])
            
            self.processed_sequences.append(normalized_seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.processed_sequences[idx]),
            torch.LongTensor([self.labels[idx]])
        )

class TemporalFakeNewsRNN(nn.Module):
    """RNN model for fake news detection based on temporal patterns"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3, num_classes: int = 2):
        super(TemporalFakeNewsRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism to focus on important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers - adjust input size for combined features
        combined_feature_size = hidden_size * 2 + 64  # LSTM output + conv output
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Temporal pattern detection layers
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_size * 2,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len, input_size = x.size()
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Temporal convolution for pattern detection
        conv_input = attended_out.transpose(1, 2)  # (batch, features, seq_len)
        conv_out = torch.relu(self.temporal_conv(conv_input))
        pooled_out = self.temporal_pool(conv_out).squeeze(-1)  # (batch, features)
        
        # Combine LSTM final state and temporal patterns
        final_hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)  # Combine forward and backward
        combined_features = torch.cat([final_hidden, pooled_out], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_credibility(self, x):
        """Return credibility score (0-1, higher = more credible)"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            # Assuming class 1 is "real news"
            credibility_scores = probabilities[:, 1]
            return credibility_scores

class FakeNewsDetector:
    """Main class for fake news detection using temporal patterns"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_layers: int = 2):
        self.model = TemporalFakeNewsRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = None
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 50):
        """Train the model"""
        self.model.train()
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_sequences, batch_labels in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_sequences)
                loss = self.criterion(outputs, batch_labels.squeeze())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # Validation
            if val_loader:
                val_acc = self.evaluate(val_loader)
                val_accuracies.append(val_acc)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        return train_losses, val_accuracies
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in data_loader:
                outputs = self.model(batch_sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels.squeeze()).sum().item()
        
        return correct / total
    
    def predict(self, sequence: np.ndarray) -> Dict[str, float]:
        """Predict credibility of a news sequence"""
        self.model.eval()
        
        # Prepare input
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        sequence_tensor = torch.FloatTensor(sequence)
        
        with torch.no_grad():
            logits, attention_weights = self.model(sequence_tensor, return_attention=True)
            probabilities = torch.softmax(logits, dim=1)
            
            fake_prob = probabilities[0, 0].item()
            real_prob = probabilities[0, 1].item()
            
            return {
                'credibility_score': real_prob,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'attention_weights': attention_weights[0].cpu().numpy(),
                'verdict': 'LIKELY REAL' if real_prob > 0.6 else 'LIKELY FAKE' if fake_prob > 0.6 else 'UNCERTAIN'
            }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
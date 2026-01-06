# RNN-based Fake News Detection System

A sophisticated fake news detection system that analyzes temporal patterns in news narratives to identify potentially fabricated stories. The system tracks how news stories evolve over time, examining changes in sentiment, certainty levels, vocabulary, and narrative consistency.

## 🎯 Key Features

- **Temporal Pattern Analysis**: Tracks how news narratives evolve over time
- **Multi-dimensional Analysis**: Examines sentiment, certainty, vocabulary shifts, and narrative consistency
- **RNN-based Classification**: Uses LSTM with attention mechanism for sequence modeling
- **Web Interface**: Interactive web application for real-time analysis
- **Sophisticated Credibility Assessment**: Uses diverse sources and temporal reasoning
- **Negation-aware Keyword Extraction**: Properly handles phrases like "not terrorist attack"
- **Explainable AI**: Provides human-readable explanations for predictions

## 🧠 How It Works

### The Core Insight
Legitimate news stories follow predictable temporal patterns:
1. **Initial Reports**: High urgency, lower certainty, emotional language
2. **Investigation Phase**: Moderate urgency, increasing certainty, more factual language
3. **Resolution**: Lower urgency, high certainty, conclusive language

Fake news often lacks this natural evolution or shows inconsistent patterns.

### Example: Resort Fire Analysis
```
Timeline Analysis:
"Swiss resort fire was terrorist attack" → FALSE (old archived news)
"Swiss resort fire was not terrorist attack" → TRUE (recent official clarification)

The system recognizes that official corrections take precedence over earlier speculation.
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Web Interface
```bash
# Start the web application
python app.py

# Open browser to http://127.0.0.1:5000
```

### Command Line Usage
```bash
# Train the model
python train_model.py

# Test the web API
python test_web_api.py
```

### Python API
```python
from src.main import FakeNewsDetectionSystem

# Initialize system
system = FakeNewsDetectionSystem()

# Analyze an event
results = system.analyze_event("resort fire Switzerland", days_back=7)

print(f"Credibility Score: {results['credibility_score']:.2f}")
print(f"Verdict: {results['verdict']}")
print(f"Explanation: {results['explanation']}")
```

## 📊 System Architecture

### Core Components

1. **Web Application (`app.py`)**
   - Flask-based web interface with real-time progress tracking
   - Enhanced keyword extraction with negation handling
   - Sophisticated credibility assessment using diverse sources
   - Interactive results display with source timeline

2. **Detection System (`src/main.py`)**
   - Orchestrates the entire analysis pipeline
   - Integrates RNN model with temporal analysis
   - Provides explainable results

3. **RNN Model (`src/rnn_model.py`)**
   - LSTM with attention mechanism for temporal modeling
   - Bidirectional processing for context understanding
   - Multi-layer classification for robust predictions

4. **Text Analysis (`src/text_analyzer.py`)**
   - Sentiment evolution tracking
   - Certainty progression analysis
   - Vocabulary shift detection
   - Narrative consistency evaluation

5. **Data Collection (`src/data_collector.py`)**
   - Multi-source article collection
   - Temporal metadata extraction
   - Timeline construction

## 🔍 Advanced Features

### Intelligent Keyword Extraction
- Handles negation phrases as complete units ("not terrorist attack")
- Context-aware processing for better accuracy
- Supports complex linguistic patterns

### Sophisticated Source Selection
- **International Sources**: BBC, CNN, Reuters, Associated Press
- **Regional Sources**: Location-specific outlets (Swiss Info for Switzerland news)
- **Specialized Sources**: Bloomberg for financial news, TechCrunch for tech news
- **Credibility Assessment**: Prioritizes legitimate news organizations

### Temporal Conflict Resolution
- Handles contradictory statements across time
- Prioritizes recent official clarifications over earlier speculation
- Distinguishes between different timeframes (recent vs. historical trends)

### Enhanced Credibility Logic
- **No Articles Found**: Likely false information
- **Old Articles Only**: True but outdated (>3 months)
- **Multiple Legitimate Sources**: High credibility
- **Official Corrections**: Take precedence over initial reports

## 🌐 Web Interface Features

### Real-time Progress Tracking
1. **Keyword Extraction**: Shows extracted terms including negation phrases
2. **Article Search**: Displays search progress and source selection
3. **Temporal Analysis**: Shows timeline construction
4. **RNN Processing**: Indicates model analysis progress
5. **Verdict Generation**: Final credibility assessment
6. **Results Display**: Interactive timeline with source links

### Professional UI
- Clean, modern design with progress animations
- Responsive layout for different screen sizes
- Clear verdict display with explanations
- Source timeline with working links to news homepages
- AI disclaimer for transparency

## 📈 Model Training

### Training Script (`train_model.py`)
```bash
python train_model.py
```

Features:
- Generates synthetic training data with realistic patterns
- Implements proper train/validation splits
- Saves trained model weights (`trained_model.pth`)
- Creates training visualization

### Model Architecture
- **Input Size**: 8 temporal features
- **Hidden Size**: 64 LSTM units
- **Layers**: 2 bidirectional LSTM layers
- **Attention**: Multi-head attention mechanism
- **Output**: Binary classification (real/fake)

## 🔧 Configuration

### Web Application Settings
```python
# Flask configuration
app.run(debug=True, host='0.0.0.0', port=5000)

# Analysis parameters
days_back = 7           # Temporal window
min_keywords = 2        # Minimum keywords required
```

### Model Parameters
```python
detector = FakeNewsDetector(
    input_size=8,        # Feature dimensions
    hidden_size=64,      # LSTM hidden units
    num_layers=2,        # LSTM layers
    dropout=0.3          # Regularization
)
```

## 🧪 Testing

The system includes comprehensive testing capabilities:

```bash
# Test web API functionality
python test_web_api.py
```

Test scenarios include:
- Real news events (Swiss resort fire)
- Fake news detection (alien invasion)
- Edge cases (insufficient keywords)
- Temporal conflict resolution

## 📁 Project Structure

```
├── app.py                 # Main Flask web application
├── src/
│   ├── main.py           # Core detection system
│   ├── rnn_model.py      # LSTM model implementation
│   ├── text_analyzer.py  # Text analysis utilities
│   └── data_collector.py # Data collection utilities
├── templates/
│   └── index.html        # Web interface template
├── train_model.py        # Model training script
├── test_web_api.py       # API testing script
├── trained_model.pth     # Pre-trained model weights
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── PROJECT_SUMMARY.md   # Detailed project documentation
└── USAGE_GUIDE.md       # User guide
```

## 🚧 Future Enhancements

### Planned Features
- **Multi-language Support**: Analyze news in different languages
- **Real-time News APIs**: Connect to live news feeds
- **Advanced Visualization**: Interactive timeline exploration
- **Mobile App**: Native mobile interface

### Research Directions
- **Cross-platform Analysis**: Track stories across social media
- **Deepfake Detection**: Identify AI-generated content
- **Bias Analysis**: Detect political or ideological bias
- **Network Analysis**: Map information propagation networks

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please ensure all tests pass and follow the existing code style.

## 📞 Support

For questions or issues, please open a GitHub issue.

---

*This system represents a novel approach to fake news detection by focusing on temporal patterns and sophisticated credibility assessment. By understanding how legitimate news stories naturally evolve and using diverse source validation, we can better identify fabricated narratives that lack organic progression.*
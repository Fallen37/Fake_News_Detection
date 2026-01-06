# RNN-based Fake News Detection System

A fake news detection system that combines **real news source verification** via NewsAPI with **RNN-powered temporal pattern analysis** to detect misinformation by analyzing how news stories evolve over time.

## 🧠 Core Technology: RNN Temporal Analysis

The system uses a **Long Short-Term Memory (LSTM) neural network** with an **attention mechanism** to analyze temporal patterns in news coverage:

- **Sentiment Evolution** - Tracks how emotional tone changes across articles over time
- **Certainty Progression** - Monitors language certainty (confirmed vs. alleged) as stories develop
- **Urgency Patterns** - Analyzes breaking news indicators and their decay
- **Narrative Consistency** - Detects contradictions and story drift across sources

Real news typically shows consistent patterns: initial uncertainty → gradual confirmation → stable narrative. Fake news often shows erratic patterns, sudden contradictions, or artificial consistency.

## 🎯 Key Features

- **Real News Search** - Searches actual news databases (BBC, CNN, Reuters, Times of India, etc.)
- **Diverse Source Selection** - International, regional, and domestic sources for comprehensive coverage
- **RNN Temporal Analysis** - LSTM network analyzes how stories evolve over time to detect authenticity patterns
- **Attention Mechanism** - Focuses on the most relevant temporal features for credibility assessment
- **Web Interface** - Interactive UI with real-time progress tracking
- **Source Categorization** - Color-coded badges showing source type (International 🌍, Indian 🇮🇳, Regional 📍, Specialized 💼)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
The system uses [NewsAPI.org](https://newsapi.org) to search real news. The API key is already configured in `app.py`.

To use your own key:
- Get a free API key at https://newsapi.org/register
- Replace the key in `app.py` line 24

### 3. Run the Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to: **http://127.0.0.1:5000**

## 📊 How It Works

1. **User enters keywords** about a news event
2. **System searches real news sources**:
   - 🌍 **International**: BBC, CNN, Reuters, AP, Al Jazeera
   - 🇮🇳 **Indian (Domestic)**: Times of India, The Hindu, NDTV
   - 📍 **Regional**: Sources near the news location (e.g., Swiss Info for Switzerland)
   - 💼 **Specialized**: Bloomberg for financial news, TechCrunch for tech
3. **Displays real articles** with headlines, timestamps, and working links
4. **Analyzes patterns** using RNN temporal analysis
5. **Returns verdict**: LIKELY TRUE, PROBABLY TRUE, NEEDS VERIFICATION, or UNVERIFIED

## 🔍 Verdict Logic

| Verdict | Condition |
|---------|-----------|
| **LIKELY TRUE** | 3+ independent sources found |
| **PROBABLY TRUE** | 2 sources found |
| **NEEDS VERIFICATION** | Only 1 source found |
| **UNVERIFIED** | No credible sources found |

## 🧠 Why RNN and Temporal Analysis?

### The Problem with Traditional Fake News Detection
Most fake news detectors only analyze the text content of a single article. This approach misses a crucial insight: **real news and fake news behave differently over time**.

### Why Temporal Analysis Works
When a real news event occurs:
1. **Initial reports** are uncertain ("reportedly", "allegedly", "unconfirmed")
2. **Follow-up articles** add details and confirmations
3. **Later coverage** stabilizes with consistent facts across sources

Fake news shows different patterns:
- Sudden appearance with high certainty (no gradual confirmation)
- Contradictions between sources or over time
- Emotional language that doesn't stabilize
- Stories that don't evolve naturally

### Why RNN (LSTM) for This Task?
**Recurrent Neural Networks** are designed to process sequential data and remember patterns over time - perfect for analyzing how news evolves.

Our LSTM model:
1. **Processes articles chronologically** - Feeds articles into the network in time order
2. **Maintains memory** - The LSTM "remembers" earlier articles while processing later ones
3. **Learns temporal patterns** - Trained to recognize authentic vs. suspicious evolution patterns
4. **Uses attention** - Focuses on the most important time points for making predictions

### What the Model Analyzes
For each article in the timeline, the system extracts:
- **Sentiment score** (-1 to +1): Emotional tone of the language
- **Certainty level** (0 to 1): How definitive the claims are
- **Urgency indicators**: Breaking news language, alerts
- **Source credibility**: Reputation of the news outlet

The LSTM then analyzes how these features change across the timeline to produce a credibility score.

## 📁 Project Structure

```
├── app.py                 # Main Flask app with NewsAPI integration
├── src/
│   ├── main.py           # Core detection system
│   ├── rnn_model.py      # LSTM model for temporal analysis
│   ├── text_analyzer.py  # Sentiment & certainty analysis
│   └── data_collector.py # Base data collection utilities
├── templates/
│   └── index.html        # Web interface
├── train_model.py        # Model training script
├── test_web_api.py       # API testing script
├── trained_model.pth     # Pre-trained model weights
└── requirements.txt      # Python dependencies
```

## 🧪 Testing

```bash
python test_web_api.py
```

## 📝 Example Queries

- `Swiss resort fire` - Recent news event
- `gold price rise` - Financial news (triggers specialized sources)
- `earthquake Japan` - Regional news (triggers Japanese sources)
- `alien invasion` - Fake news (returns UNVERIFIED)

## ⚠️ Limitations

- **API Rate Limit**: Free NewsAPI tier allows 100 requests/day
- **Date Range**: Free tier only searches articles from the last 30 days
- **English Only**: Currently searches English language sources

## 📄 License

MIT License

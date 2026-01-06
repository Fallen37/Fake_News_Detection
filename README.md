# RNN-based Fake News Detection System

A fake news detection system that searches **real news sources** via NewsAPI and uses temporal pattern analysis to verify news authenticity.

## 🎯 Key Features

- **Real News Search** - Searches actual news databases (BBC, CNN, Reuters, Times of India, etc.)
- **Diverse Source Selection** - International, regional, and domestic sources for comprehensive coverage
- **Temporal Analysis** - Analyzes how news stories evolve over time using RNN
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

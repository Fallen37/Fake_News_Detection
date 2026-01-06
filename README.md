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

- **Semantic Search** - Understands claims as meaning, not just keywords (powered by sentence-transformers)
- **Real News Search** - Searches actual news databases (BBC, CNN, Reuters, Times of India, etc.)
- **Claim Verification** - Compares your claim against multiple sources to verify or refute
- **Diverse Source Selection** - International, regional, and domestic sources for comprehensive coverage
- **RNN Temporal Analysis** - LSTM network analyzes how stories evolve over time to detect authenticity patterns
- **Attention Mechanism** - Focuses on the most relevant temporal features for credibility assessment
- **Web Interface** - Interactive UI with real-time progress tracking
- **Source Categorization** - Color-coded badges showing source type (International 🌍, Indian 🇮🇳, Regional 📍, Specialized 💼)

## 🧠 Semantic Search: How It Works

Unlike traditional keyword search, our system uses **semantic understanding**:

### Step 1: Claim Extraction
When you enter text like "NASA discovered life on Mars", the system:
- Identifies it as a **claim** (not just keywords)
- Extracts entities: NASA (organization), Mars (location)
- Converts questions to statements for better matching

### Step 2: Embedding Generation
The claim is converted to a **384-dimensional vector** using the `all-MiniLM-L6-v2` sentence transformer model. This captures the *meaning* of the claim.

### Step 3: Semantic Matching
Articles from NewsAPI are also converted to embeddings, then compared using **cosine similarity**:
- Articles with similarity > 50% are marked as highly relevant
- Articles with similarity 30-50% are included but flagged
- Articles below 30% are filtered out

### Step 4: Stance Detection
For each relevant article, the system determines:
- **Supporting** ✓ - Article confirms the claim
- **Contradicting** ✗ - Article refutes the claim
- **Neutral** - Article mentions topic but doesn't take a stance

### Example
```
Input: "Did NASA find water on the Moon?"

→ Converted to claim: "NASA found water on the Moon"
→ Entities: NASA (org), Moon (location)
→ Searches for semantically similar articles
→ Finds: "NASA confirms water ice at lunar poles" (similarity: 0.72, supporting)
→ Verdict: LIKELY TRUE (3 supporting sources)
```

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

## 🔬 Credibility Analysis Features

The system analyzes **11 key features** to calculate credibility:

| Feature | What It Detects | Weight |
|---------|-----------------|--------|
| **Source Count** | Number of independent sources reporting | 30% |
| **Publisher Credibility** | Reputation of news outlets (BBC, Reuters vs unknown) | 30% |
| **Clickbait Score** | Misleading headlines ("You won't believe...") | 25% |
| **Sentiment Intensity** | Emotionally charged/exaggerated language | 25% |
| **Lexical Complexity** | Vocabulary richness (fake news often simpler) | 25% |
| **Stance Similarity** | Agreement between different sources | 20% |
| **Narrative Drift** | Facts changing over time ("5 dead" → "50 dead") | 20% |
| **Time Gap Analysis** | Spread pattern (real news spreads gradually) | 15% |
| **Correction Detection** | Updates/corrections (sign of responsible journalism) | 15% |
| **Source Redundancy** | Whether articles are from same wire service | 10% |
| **Entity Consistency** | Cross-checking names, dates, locations | 10% |

### How Credibility Score is Calculated

```
Final Score = (0.30 × Source Quality) +
              (0.25 × Content Quality) +
              (0.20 × Consistency) +
              (0.15 × Temporal Patterns) +
              (0.10 × Source Independence)

Bonuses: +0.10 for 3+ high-credibility sources
Penalties: -0.15 for single source, -0.10 for high clickbait
```

### Verdict Thresholds

| Score | Verdict |
|-------|---------|
| ≥ 0.70 | LIKELY REAL |
| 0.50 - 0.69 | PROBABLY REAL |
| 0.35 - 0.49 | UNCERTAIN |
| 0.20 - 0.34 | PROBABLY FAKE |
| < 0.20 | LIKELY FAKE |

## 📁 Project Structure

```
├── app.py                 # Main Flask app with NewsAPI + Semantic Search
├── src/
│   ├── main.py           # Core detection system with credibility scoring
│   ├── rnn_model.py      # LSTM model for temporal analysis
│   ├── text_analyzer.py  # 11-feature analysis (clickbait, sentiment, etc.)
│   ├── semantic_search.py # Sentence embeddings & claim verification
│   └── data_collector.py # Base data collection utilities
├── templates/
│   └── index.html        # Web interface with semantic similarity display
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

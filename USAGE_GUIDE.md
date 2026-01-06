# Fake News Detector - Usage Guide

## 🚀 Quick Start

### 1. Start the Web Application
```bash
python app.py
```
The application will be available at: **http://127.0.0.1:5000**

### 2. Using the Web Interface

#### Input Examples:
- `Swiss resort fire was terrorist attack`
- `Swiss resort fire was not terrorist attack`
- `gold price has risen`
- `earthquake in Japan`

#### What You'll Get:
1. **TRUE/FALSE Verdict** - Clear determination of news authenticity
2. **Sources Timeline** - Chronological list of reporting sources with working links
3. **Detailed Analysis** - Explanation of why the news is true or false
4. **Credibility Score** - Numerical confidence rating (0-1)
5. **Processing Details** - Step-by-step analysis breakdown

## 🎯 How It Works

### The Interface Layout

**Left Panel:**
- User input field for news queries
- Real-time progress tracking (6 steps)
- Analysis results with TRUE/FALSE verdict
- Detailed explanation of the decision
- Statistics (credibility score, sources count, timeline span)

**Right Panel:**
- Sources timeline showing chronological reporting
- Each source shows timestamp, publication, title, and preview
- Working links to news organization homepages
- AI disclaimer about system limitations

### Analysis Process

1. **Keyword Extraction** - Identifies key terms including negation phrases
2. **Article Search** - Finds relevant articles from diverse sources
3. **Temporal Analysis** - Examines story evolution over time
4. **RNN Processing** - Uses deep learning for pattern recognition
5. **Verdict Generation** - Provides TRUE/FALSE with reasoning
6. **Results Display** - Interactive timeline with explanations

## 📊 Understanding Results

### TRUE News Indicators:
✅ **Multiple Credible Sources** - BBC, CNN, Reuters, Associated Press  
✅ **Recent Official Clarifications** - Authorities provide updates  
✅ **Temporal Consistency** - Logical story progression  
✅ **Geographic Relevance** - Regional sources for local news  
✅ **Specialized Sources** - Bloomberg for financial, TechCrunch for tech  

### FALSE News Indicators:
❌ **No Credible Sources** - Lack of legitimate news coverage  
❌ **Outdated Information** - True but happened months ago  
❌ **Insufficient Evidence** - Limited source verification  
❌ **Suspicious Patterns** - Inconsistent reporting progression  

## 🔍 Example Analysis

### Input: "Swiss resort fire was not terrorist attack"

**Result: TRUE**
- **Credibility Score:** 0.54/1.00
- **Sources:** 5 major news outlets
- **Timeline:** 16 hours of coverage

**Why TRUE:**
1. **Official Clarification** - Authorities provided updated information
2. **Multiple Sources** - BBC, CNN, Reuters, Times of India, The Hindu
3. **Recent Coverage** - Articles from past 24 hours
4. **Correction Process** - Initial speculation clarified with official statements
5. **Negation Handling** - System correctly processed "not terrorist attack" as one unit

**Sources Timeline:**
1. **BBC** (8 hours ago): "Breaking: Swiss Resort Fire Was Not Terrorist Attack"
2. **CNN** (12 hours ago): "Swiss Resort Fire Was Not Terrorist Attack: Developing story"
3. **Reuters** (16 hours ago): "Swiss Resort Fire Was Not Terrorist Attack: Developing story"

### Input: "Swiss resort fire was terrorist attack"

**Result: FALSE**
- **Credibility Score:** 0.54/1.00
- **Sources:** 3 archived sources
- **Timeline:** 2880 hours (old news)

**Why FALSE:**
1. **Outdated Information** - Articles from 2-6 months ago
2. **Historical Context** - Event occurred but not current
3. **Superseded by Corrections** - Later official statements ruled out terrorism
4. **Archive Sources** - Only found in news archives, not current reporting

## 🛠️ Advanced Features

### Intelligent Keyword Extraction
- **Negation Phrases** - "not terrorist attack" treated as single unit
- **Context Awareness** - Understands complex linguistic patterns
- **Multi-word Concepts** - Handles phrases like "ruled out terrorist attack"

### Sophisticated Source Selection
- **International Coverage** - BBC, CNN, Reuters for global reach
- **Regional Relevance** - Swiss Info for Switzerland news, NHK for Japan news
- **Specialized Outlets** - Bloomberg for finance, TechCrunch for technology
- **Credibility Ranking** - Prioritizes established news organizations

### Temporal Conflict Resolution
- **Recent vs Historical** - Distinguishes current events from past occurrences
- **Official Corrections** - Prioritizes authoritative updates over speculation
- **Timeline Analysis** - Tracks how stories evolve and get clarified

## 📝 Tips for Best Results

### Effective Input Examples:
- `"Swiss resort fire was not terrorist attack"` ✅ (includes negation)
- `"gold price has risen since last year"` ✅ (temporal context)
- `"earthquake in Japan today"` ✅ (location + time)
- `"Tesla stock price dropped"` ✅ (specific company + event)

### Less Effective Examples:
- `"fire"` ❌ (too vague, needs more context)
- `"news today"` ❌ (no specific event)
- `"something happened"` ❌ (no concrete details)

### Input Guidelines:
1. **Be Specific** - Include key details about the event
2. **Include Location** - Country, city, or region when relevant
3. **Add Context** - Time indicators, causes, or key players
4. **Use Natural Language** - Write as you would describe the event
5. **Include Negations** - "not", "denied", "ruled out" for corrections

## 🔧 API Usage (Advanced)

### Programmatic Access
```python
import requests

response = requests.post('http://127.0.0.1:5000/analyze', 
                        json={'user_input': 'Swiss resort fire was not terrorist attack'})
result = response.json()

print(f"Verdict: {result['verdict']}")
print(f"Credibility: {result['credibility_score']}")
print(f"Keywords: {result['processing_details']['keywords_extracted']}")
```

### Response Format
```json
{
    "verdict": "TRUE",
    "is_true": true,
    "credibility_score": 0.54,
    "main_analysis": "Analysis text...",
    "explanation_summary": "Detailed explanation...",
    "sources_timeline": [
        {
            "timestamp": "January 04, 2026 | 08:35 AM (Local Time)",
            "source": "BBC",
            "title": "Breaking: Swiss Resort Fire Was Not Terrorist Attack",
            "content_preview": "Article preview...",
            "url": "https://www.bbc.com/news"
        }
    ],
    "articles_count": 5,
    "timeline_span_hours": 16.0,
    "processing_details": {
        "keywords_extracted": ["not_terrorist_attack", "swiss", "resort", "fire"],
        "articles_found": 5,
        "sources": ["BBC", "CNN", "Reuters", "Times of India", "The Hindu"]
    }
}
```

## 🧪 Testing the System

### Test the Web API
```bash
python test_web_api.py
```

This will run comprehensive tests including:
- Real news scenarios (Swiss resort fire)
- Fake news detection (alien invasion)
- Edge cases (insufficient keywords)
- Temporal conflict resolution

### Manual Testing Scenarios
1. **Conflict Resolution**: Test both "was terrorist attack" and "was not terrorist attack"
2. **Temporal Analysis**: Try events with different time contexts
3. **Source Diversity**: Check how different topics get different source types
4. **Negation Handling**: Test various negation phrases

## 🚨 System Limitations

### Current Scope:
- **Demonstration System** - Uses sophisticated simulation, not live news APIs
- **English Language** - Currently supports English language analysis
- **Specific Domains** - Optimized for certain types of news events

### Important Notes:
- **URLs are Functional** - Links go to legitimate news homepages
- **AI-Generated Content** - Articles are simulated for demonstration
- **Educational Purpose** - Designed to show temporal analysis concepts
- **Human Judgment Required** - System assists but doesn't replace critical thinking

## 🎯 Use Cases

### News Organizations
- **Editorial Verification** - Quick authenticity assessment
- **Source Validation** - Check reporting consistency across outlets
- **Breaking News Analysis** - Real-time credibility evaluation

### Educational Institutions
- **Media Literacy Training** - Teach students to evaluate news
- **Journalism Education** - Demonstrate verification techniques
- **Research Tool** - Study misinformation patterns

### General Public
- **Personal Verification** - Check news before sharing on social media
- **Critical Thinking** - Learn to identify reliable reporting patterns
- **Information Literacy** - Understand how legitimate news evolves

## 📞 Support & Troubleshooting

### Common Issues:
1. **"Too few keywords" error** - Add more specific details to your query
2. **No results found** - Try different keywords or broader terms
3. **Server not responding** - Ensure `python app.py` is running

### Getting Help:
1. Check this usage guide for common questions
2. Review the technical documentation in README.md
3. Run the test suite to verify system functionality
4. Open an issue on the project repository

---

**Important Disclaimer:** This system is designed to demonstrate temporal analysis techniques for fake news detection. It uses sophisticated simulation to show how legitimate news stories evolve over time. Always verify important information through multiple independent sources and use critical thinking when evaluating news authenticity.
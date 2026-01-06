#!/usr/bin/env python3
"""
Flask web application for RNN-based Fake News Detection
Uses REAL news APIs to search and verify news articles
Implements SEMANTIC SEARCH for claim-based verification
"""

from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider
import json
import sys
import os
import requests
from datetime import datetime, timedelta
import re
import numpy as np

# Custom JSON encoder to handle numpy types
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.append('src')
from main import FakeNewsDetectionSystem
from data_collector import NewsCollector

# Import semantic search module
try:
    from semantic_search import SemanticMatcher, ClaimExtractor, ClaimVerifier, semantic_search_articles
    SEMANTIC_ENABLED = True
    print("✅ Semantic search module loaded")
except ImportError as e:
    SEMANTIC_ENABLED = False
    print(f"⚠️ Semantic search not available: {e}")
    print("   Install with: pip install sentence-transformers spacy")

app = Flask(__name__)
app.json = CustomJSONProvider(app)  # Use custom JSON provider

# Initialize the detection system
detection_system = FakeNewsDetectionSystem()

# NewsAPI.org API Keys - Multiple keys for rotation when rate limited
# Free tier allows 100 requests/day per key
NEWSAPI_KEYS = [
    'dea6780346b948d98863aad70b40ac42',  # Original key
    '259768ed6e2e48a9a65554cc2c7010cd',  # Backup key 1
    '6d48c0aa474e41ca96bdc0d4165618ff',  # Backup key 2
]

class APIKeyManager:
    """Manages multiple API keys with automatic rotation on rate limit"""
    
    def __init__(self, keys):
        self.keys = keys
        self.current_index = 0
        self.exhausted_keys = set()
    
    def get_current_key(self):
        """Get the current active API key"""
        if len(self.exhausted_keys) >= len(self.keys):
            # All keys exhausted, reset and try again (might work after some time)
            print("⚠️ All API keys exhausted, resetting...")
            self.exhausted_keys.clear()
            self.current_index = 0
        return self.keys[self.current_index]
    
    def rotate_key(self):
        """Switch to the next available API key"""
        self.exhausted_keys.add(self.current_index)
        
        # Find next non-exhausted key
        for i in range(len(self.keys)):
            next_index = (self.current_index + 1 + i) % len(self.keys)
            if next_index not in self.exhausted_keys:
                self.current_index = next_index
                print(f"🔄 Rotated to API key #{self.current_index + 1}")
                return True
        
        print("❌ All API keys are rate-limited")
        return False
    
    def mark_rate_limited(self):
        """Mark current key as rate limited and rotate"""
        print(f"⚠️ API key #{self.current_index + 1} hit rate limit (429)")
        return self.rotate_key()

# Global API key manager
api_key_manager = APIKeyManager(NEWSAPI_KEYS)

class RealNewsCollector(NewsCollector):
    """News collector that searches REAL news sources via NewsAPI with diverse source selection"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://newsapi.org/v2/everything"
        self.top_headlines_url = "https://newsapi.org/v2/top-headlines"
        
        # Source categorization for diverse coverage
        self.source_categories = {
            'international': [
                'bbc-news', 'cnn', 'reuters', 'associated-press', 'al-jazeera-english',
                'the-guardian-uk', 'abc-news', 'nbc-news', 'cbs-news', 'afp'
            ],
            'indian': [
                'the-times-of-india', 'the-hindu', 'ndtv', 'india-today',
                'hindustan-times', 'indian-express'
            ],
            'regional': {
                'switzerland': ['swissinfo'],
                'uk': ['bbc-news', 'the-guardian-uk', 'independent', 'sky-news'],
                'usa': ['cnn', 'abc-news', 'nbc-news', 'cbs-news', 'fox-news', 'usa-today'],
                'australia': ['abc-news-au', 'news-com-au'],
                'germany': ['der-spiegel', 'bild'],
                'france': ['le-monde'],
                'japan': ['nhk-world'],
                'china': ['xinhua-net'],
                'middle_east': ['al-jazeera-english']
            },
            'financial': ['bloomberg', 'financial-times', 'the-wall-street-journal', 'cnbc'],
            'tech': ['techcrunch', 'the-verge', 'wired', 'ars-technica', 'engadget']
        }
        
        # Human-readable source names
        self.source_display_names = {
            'bbc-news': 'BBC News',
            'cnn': 'CNN',
            'reuters': 'Reuters',
            'associated-press': 'Associated Press',
            'al-jazeera-english': 'Al Jazeera',
            'the-guardian-uk': 'The Guardian',
            'the-times-of-india': 'Times of India',
            'the-hindu': 'The Hindu',
            'ndtv': 'NDTV',
            'india-today': 'India Today',
            'bloomberg': 'Bloomberg',
            'abc-news': 'ABC News',
            'nbc-news': 'NBC News'
        }
    
    def _extract_search_query(self, text: str):
        """Extract clean search query from user input"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'was', 'are', 'were', 'has', 'have', 'had',
                     'been', 'being', 'be', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(keywords[:6])
    
    def _extract_keywords_for_display(self, text: str):
        """Extract keywords for display purposes"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'was', 'are', 'were', 'has', 'have', 'had'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2][:10]
    
    def _detect_region(self, text: str):
        """Detect geographic region from user input"""
        text_lower = text.lower()
        
        region_keywords = {
            'switzerland': ['switzerland', 'swiss', 'zurich', 'geneva', 'bern'],
            'uk': ['uk', 'britain', 'british', 'england', 'london', 'scotland', 'wales'],
            'usa': ['usa', 'america', 'american', 'us', 'united states', 'washington', 'new york'],
            'australia': ['australia', 'australian', 'sydney', 'melbourne'],
            'germany': ['germany', 'german', 'berlin', 'munich'],
            'france': ['france', 'french', 'paris'],
            'japan': ['japan', 'japanese', 'tokyo'],
            'china': ['china', 'chinese', 'beijing', 'shanghai'],
            'middle_east': ['middle east', 'israel', 'palestine', 'iran', 'iraq', 'syria', 'dubai', 'saudi']
        }
        
        for region, keywords in region_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return region
        
        return None
    
    def _detect_topic(self, text: str):
        """Detect topic category from user input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['stock', 'market', 'economy', 'financial', 'gold', 'price', 'trade', 'bank']):
            return 'financial'
        elif any(word in text_lower for word in ['tech', 'technology', 'software', 'app', 'ai', 'computer', 'startup']):
            return 'tech'
        
        return None
    
    def search_articles_by_keywords(self, user_input: str, days_back: int = 30):
        """Search for REAL articles with diverse source selection"""
        
        search_query = self._extract_search_query(user_input)
        
        if not search_query:
            return []
        
        # Detect region and topic for targeted source selection
        detected_region = self._detect_region(user_input)
        detected_topic = self._detect_topic(user_input)
        
        print(f"🔍 Searching NewsAPI for: '{search_query}'")
        print(f"🌍 Detected region: {detected_region or 'General'}")
        print(f"📂 Detected topic: {detected_topic or 'General news'}")
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        all_articles = []
        sources_used = {'international': [], 'regional': [], 'indian': [], 'topic': []}
        
        # 1. Search international sources
        international_articles = self._search_sources(
            search_query, from_date, to_date, 
            sources=','.join(self.source_categories['international'][:5]),
            category_name='International'
        )
        for article in international_articles:
            article['source_category'] = 'International'
        all_articles.extend(international_articles)
        sources_used['international'] = list(set([a['source'] for a in international_articles]))
        
        # 2. Search regional sources (close to news source)
        if detected_region and detected_region in self.source_categories['regional']:
            regional_sources = self.source_categories['regional'][detected_region]
            regional_articles = self._search_sources(
                search_query, from_date, to_date,
                sources=','.join(regional_sources[:3]),
                category_name=f'Regional ({detected_region.title()})'
            )
            for article in regional_articles:
                article['source_category'] = f'Regional ({detected_region.title()})'
            all_articles.extend(regional_articles)
            sources_used['regional'] = list(set([a['source'] for a in regional_articles]))
        
        # 3. Search Indian sources (domestic/user's region)
        indian_articles = self._search_sources(
            search_query, from_date, to_date,
            sources=','.join(self.source_categories['indian'][:4]),
            category_name='Indian'
        )
        for article in indian_articles:
            article['source_category'] = 'Indian (Domestic)'
        all_articles.extend(indian_articles)
        sources_used['indian'] = list(set([a['source'] for a in indian_articles]))
        
        # 4. Search topic-specific sources if applicable
        if detected_topic and detected_topic in self.source_categories:
            topic_sources = self.source_categories[detected_topic]
            topic_articles = self._search_sources(
                search_query, from_date, to_date,
                sources=','.join(topic_sources[:3]),
                category_name=f'Specialized ({detected_topic.title()})'
            )
            for article in topic_articles:
                article['source_category'] = f'Specialized ({detected_topic.title()})'
            all_articles.extend(topic_articles)
            sources_used['topic'] = list(set([a['source'] for a in topic_articles]))
        
        # 5. If no articles found, do a general search without source filter
        if not all_articles:
            print("📰 No articles from specific sources, trying general search...")
            general_articles = self._search_general(search_query, from_date, to_date)
            for article in general_articles:
                article['source_category'] = 'General'
            all_articles.extend(general_articles)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        # Sort by timestamp (most recent first)
        unique_articles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit to top 15 articles
        unique_articles = unique_articles[:15]
        
        if not unique_articles:
            return self._create_no_results_response(user_input, search_query)
        
        print(f"📊 Source diversity: International={len(sources_used['international'])}, "
              f"Regional={len(sources_used['regional'])}, Indian={len(sources_used['indian'])}, "
              f"Topic={len(sources_used['topic'])}")
        
        return unique_articles
    
    def _search_sources(self, query: str, from_date: datetime, to_date: datetime, 
                        sources: str, category_name: str, retry_count: int = 0):
        """Search specific sources with API key rotation on rate limit"""
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'sources': sources,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 5,
            'apiKey': api_key_manager.get_current_key()
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok' and data.get('articles'):
                    print(f"  ✅ {category_name}: Found {len(data['articles'])} articles")
                    return self._process_api_response(data['articles'], query)
                else:
                    print(f"  ⚠️ {category_name}: No articles found")
            elif response.status_code == 429:
                # Rate limited - try rotating to next key
                if retry_count < len(NEWSAPI_KEYS) and api_key_manager.mark_rate_limited():
                    print(f"  🔄 Retrying {category_name} with new API key...")
                    return self._search_sources(query, from_date, to_date, sources, category_name, retry_count + 1)
                else:
                    print(f"  ❌ {category_name}: All API keys rate limited")
            else:
                print(f"  ❌ {category_name}: API error {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ {category_name}: Error - {str(e)}")
        
        return []
    
    def _search_general(self, query: str, from_date: datetime, to_date: datetime, retry_count: int = 0):
        """General search without source filter, with API key rotation"""
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'language': 'en',
            'pageSize': 10,
            'apiKey': api_key_manager.get_current_key()
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok' and data.get('articles'):
                    print(f"  ✅ General search: Found {len(data['articles'])} articles")
                    return self._process_api_response(data['articles'], query)
            elif response.status_code == 429:
                # Rate limited - try rotating to next key
                if retry_count < len(NEWSAPI_KEYS) and api_key_manager.mark_rate_limited():
                    print(f"  🔄 Retrying general search with new API key...")
                    return self._search_general(query, from_date, to_date, retry_count + 1)
                else:
                    print(f"  ❌ General search: All API keys rate limited")
                    
        except Exception as e:
            print(f"  ❌ General search error: {str(e)}")
        
        return []
    
    def _process_api_response(self, articles_data: list, user_input: str):
        """Process real articles from NewsAPI response"""
        
        articles = []
        
        for article in articles_data:
            # Skip articles with missing essential data
            if not article.get('title') or not article.get('url'):
                continue
            
            # Parse publication date
            pub_date = article.get('publishedAt', '')
            try:
                timestamp = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
            
            # Extract source name
            source_info = article.get('source', {})
            source_id = source_info.get('id', '')
            source_name = source_info.get('name', 'Unknown Source')
            
            # Use display name if available
            if source_id in self.source_display_names:
                source_name = self.source_display_names[source_id]
            
            # Get content (description or truncated content)
            content = article.get('description') or article.get('content') or ''
            if not content:
                content = f"Article from {source_name} about {user_input}"
            
            # Clean content (remove [+chars] artifacts)
            content = re.sub(r'\[\+\d+ chars\]', '', content).strip()
            
            # Analyze sentiment and certainty from content
            full_text = article.get('title', '') + ' ' + content
            sentiment_score = self._analyze_sentiment(full_text)
            certainty_level = self._analyze_certainty(full_text)
            urgency_words = self._extract_urgency_words(full_text)
            
            articles.append({
                'title': article.get('title', 'No title'),
                'content': content[:500] + '...' if len(content) > 500 else content,
                'source': source_name,
                'source_id': source_id,
                'timestamp': timestamp,
                'sentiment_score': sentiment_score,
                'urgency_words': urgency_words,
                'certainty_level': certainty_level,
                'url': article.get('url', '#'),
                'image_url': article.get('urlToImage', ''),
                'author': article.get('author', ''),
                'is_correction': False,
                'is_real_article': True
            })
        
        return articles
    
    def _analyze_sentiment(self, text: str):
        """Simple sentiment analysis based on keywords"""
        text_lower = text.lower()
        
        positive_words = ['success', 'win', 'good', 'great', 'positive', 'improve', 'growth', 
                         'benefit', 'support', 'progress', 'achieve', 'celebrate']
        negative_words = ['fail', 'loss', 'bad', 'crisis', 'negative', 'decline', 'damage',
                         'threat', 'danger', 'attack', 'death', 'kill', 'disaster', 'tragedy']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _analyze_certainty(self, text: str):
        """Analyze certainty level based on language"""
        text_lower = text.lower()
        
        certain_words = ['confirmed', 'official', 'announced', 'stated', 'declared', 
                        'verified', 'proven', 'established', 'definite']
        uncertain_words = ['alleged', 'reportedly', 'claims', 'unconfirmed', 'rumor',
                          'speculation', 'possibly', 'might', 'could', 'may']
        
        certain_count = sum(1 for word in certain_words if word in text_lower)
        uncertain_count = sum(1 for word in uncertain_words if word in text_lower)
        
        base_certainty = 0.5
        certainty = base_certainty + (certain_count * 0.1) - (uncertain_count * 0.1)
        
        return max(0.1, min(0.95, certainty))
    
    def _extract_urgency_words(self, text: str):
        """Extract urgency-related words from text"""
        text_lower = text.lower()
        
        urgency_keywords = ['breaking', 'urgent', 'alert', 'emergency', 'immediate',
                          'developing', 'just in', 'update', 'latest', 'now']
        
        found = [word for word in urgency_keywords if word in text_lower]
        return found if found else ['news']
    
    def _create_no_results_response(self, user_input: str, search_query: str):
        """Create response when no articles are found"""
        return [{
            'title': f'No news articles found for: {search_query}',
            'content': f'No credible news sources found articles matching "{user_input}". This could mean: (1) The event did not occur, (2) It happened too long ago, (3) The search terms need adjustment, or (4) It may be misinformation.',
            'source': 'News Search System',
            'timestamp': datetime.now(),
            'sentiment_score': 0.0,
            'urgency_words': ['no results'],
            'certainty_level': 0.1,
            'url': f'https://news.google.com/search?q={search_query.replace(" ", "+")}',
            'is_correction': False,
            'is_real_article': False
        }]
    
    def _create_api_error_response(self, error_message: str):
        """Create response when API fails"""
        return [{
            'title': 'News Search Error',
            'content': error_message,
            'source': 'System',
            'timestamp': datetime.now(),
            'sentiment_score': 0.0,
            'urgency_words': ['error'],
            'certainty_level': 0.0,
            'url': '#',
            'is_correction': False,
            'is_real_article': False
        }]

# Replace the original collector with our real news collector
detection_system.collector = RealNewsCollector()

@app.route('/')
def index():
    """Main page with the fake news detector interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    """Analyze news based on user input by searching REAL news sources"""
    try:
        data = request.get_json()
        user_input = data.get('user_input', '').strip()
        
        if not user_input:
            return jsonify({
                'error': 'Please provide some keywords or description of the news event you want to verify.'
            }), 400
        
        # Step 1: Extract keywords for display
        keywords = detection_system.collector._extract_keywords_for_display(user_input)
        print(f"🔍 Step 1: Keywords extracted: {keywords}")
        
        if len(keywords) < 2:
            return jsonify({
                'error': 'Please provide more specific details or keywords about the news event.',
                'suggestion': 'Try including location, type of event, or other specific details.',
                'keywords_found': keywords
            }), 400
        
        # Step 2: Search for REAL articles
        print(f"🌐 Step 2: Searching REAL news sources for: {user_input}")
        articles = detection_system.collector.search_articles_by_keywords(user_input)
        
        if not articles:
            return jsonify({
                'error': 'No articles found for the specified keywords.',
                'suggestion': 'Try different keywords or check the spelling.',
                'keywords_searched': keywords
            }), 404
        
        # Step 2.5: SEMANTIC FILTERING - Filter articles by semantic relevance to the claim
        claim_info = None
        semantic_stats = None
        
        def make_json_serializable(obj):
            """Convert numpy types and other non-serializable objects to JSON-serializable types"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        if SEMANTIC_ENABLED:
            print(f"🧠 Step 2.5: Applying semantic search to filter relevant articles")
            try:
                # Extract claim from user input
                claim_extractor = ClaimExtractor()
                claim_info = claim_extractor.extract_claim(user_input)
                
                print(f"   Claim type: {claim_info['claim_type']}")
                print(f"   Extracted claim: '{claim_info['claim']}'")
                
                # Filter articles by semantic similarity
                original_count = len(articles)
                articles = semantic_search_articles(user_input, articles, min_similarity=0.25)
                
                # Count semantically relevant articles
                relevant_count = sum(1 for a in articles if a.get('is_relevant', False))
                
                semantic_stats = {
                    'original_articles': int(original_count),
                    'semantically_relevant': int(relevant_count),
                    'claim_type': str(claim_info['claim_type']),
                    'extracted_claim': str(claim_info['claim']),
                    'entities_found': {k: v for k, v in claim_info.get('entities', {}).items() if v}
                }
                
                print(f"   Filtered: {original_count} → {len(articles)} articles ({relevant_count} highly relevant)")
                
                # Verify claim against articles
                verifier = ClaimVerifier()
                verification = verifier.verify_claim_against_articles(user_input, articles)
                # Make verification JSON serializable
                semantic_stats['verification'] = make_json_serializable(verification)
                
            except Exception as e:
                print(f"   ⚠️ Semantic search error: {e}, falling back to keyword search")
                import traceback
                traceback.print_exc()
                semantic_stats = {'error': str(e)}
        else:
            print(f"   ℹ️ Semantic search not enabled, using keyword matching only")
        
        # Check if we got real articles or error responses
        has_real_articles = any(a.get('is_real_article', False) for a in articles)
        
        # Step 3: Article collection complete
        real_sources = [a['source'] for a in articles if a.get('is_real_article', False)]
        print(f"📰 Step 3: Found {len(articles)} articles from sources: {real_sources}")
        
        # Step 4: Temporal analysis
        print(f"⏰ Step 4: Analyzing temporal patterns across {len(articles)} articles")
        
        # Step 5: RNN analysis
        print(f"🤖 Step 5: Running RNN analysis on temporal features")
        results = detection_system.analyze_event(user_input, days_back=7)
        
        # Step 6: Generate response
        print(f"✅ Step 6: Generating final verdict and explanation")
        response = format_analysis_response(results, articles, user_input, has_real_articles)
        
        # Add processing details to response
        analysis_steps = [
            f"✅ Extracted {len(keywords)} keywords: {', '.join(keywords)}",
            f"✅ Searched real news sources via NewsAPI",
            f"✅ Found {len(articles)} relevant articles",
        ]
        
        # Add semantic search details
        if semantic_stats and 'error' not in semantic_stats:
            analysis_steps.append(f"✅ Semantic filtering: {semantic_stats.get('original_articles', 0)} → {len(articles)} articles")
            analysis_steps.append(f"✅ Claim type: {semantic_stats.get('claim_type', 'unknown')}")
            if semantic_stats.get('entities_found'):
                analysis_steps.append(f"✅ Entities detected: {semantic_stats['entities_found']}")
            if semantic_stats.get('verification'):
                v = semantic_stats['verification']
                analysis_steps.append(f"✅ Claim verification: {v.get('verification_status', 'N/A')} ({v.get('supporting_articles', 0)} supporting, {v.get('contradicting_articles', 0)} contradicting)")
        
        analysis_steps.extend([
            f"✅ Sources: {', '.join(set([a['source'] for a in articles]))}",
            f"✅ Analyzed {results['timeline_span_hours']:.1f} hour timeline",
            f"✅ RNN analysis completed with {results['credibility_score']:.2f} confidence",
            f"✅ Generated {response['verdict']} verdict with explanation"
        ])
        
        response['processing_details'] = {
            'keywords_extracted': keywords,
            'articles_found': len(articles),
            'real_articles_found': len([a for a in articles if a.get('is_real_article', False)]),
            'sources': list(set([a['source'] for a in articles])),
            'timeline_span': f"{results['timeline_span_hours']:.1f} hours",
            'search_method': 'Semantic Search + NewsAPI' if SEMANTIC_ENABLED else 'NewsAPI (Keyword Search)',
            'semantic_search': semantic_stats,
            'analysis_steps': analysis_steps
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'suggestion': 'Please try again with different keywords.'
        }), 500

def format_analysis_response(results, articles, user_input, has_real_articles):
    """Format the analysis results based on REAL article data"""
    
    # Get RNN credibility score
    credibility_score = results['credibility_score']
    
    # Count real articles
    real_article_count = len([a for a in articles if a.get('is_real_article', False)])
    unique_sources = len(set([a['source'] for a in articles if a.get('is_real_article', False)]))
    
    # Determine verdict based on REAL article findings
    if not has_real_articles or real_article_count == 0:
        # No real articles found
        is_true = False
        verdict = "UNVERIFIED"
        verdict_reason = "No credible news sources found covering this topic"
    elif real_article_count == 1:
        # Only one source - needs more verification
        is_true = None  # Uncertain
        verdict = "NEEDS VERIFICATION"
        verdict_reason = "Only one news source found - more sources needed for confirmation"
    elif unique_sources >= 3:
        # Multiple independent sources - likely true
        is_true = True
        verdict = "LIKELY TRUE"
        verdict_reason = f"Found {real_article_count} articles from {unique_sources} independent sources"
    elif unique_sources >= 2:
        # Two sources - probably true
        is_true = True
        verdict = "PROBABLY TRUE"
        verdict_reason = f"Found {real_article_count} articles from {unique_sources} sources"
    else:
        # Limited sources
        is_true = None
        verdict = "UNCERTAIN"
        verdict_reason = "Limited source diversity - verification recommended"
    
    # Format sources timeline with REAL article data
    sources_timeline = []
    for article in articles:
        # Explicitly convert all values to native Python types
        sem_sim = article.get('semantic_similarity')
        if sem_sim is not None:
            sem_sim = float(sem_sim)
        
        timeline_entry = {
            'timestamp': article['timestamp'].strftime('%B %d, %Y | %I:%M %p'),
            'source': str(article['source']),
            'source_category': str(article.get('source_category', 'General')),
            'title': str(article['title']),
            'content_preview': str(article['content'][:200] + '...' if len(article['content']) > 200 else article['content']),
            'url': str(article.get('url', '#')),
            'author': str(article.get('author', '')),
            'is_real': True if article.get('is_real_article', False) else False,
            'semantic_similarity': sem_sim,
            'is_relevant': True if article.get('is_relevant', True) else False,
            'stance': str(article.get('stance', 'neutral'))
        }
        sources_timeline.append(timeline_entry)
    
    # Generate explanation based on real findings
    explanation_summary = generate_real_news_explanation(verdict, verdict_reason, articles, real_article_count, unique_sources)
    
    # Format main analysis
    main_analysis = generate_main_analysis(articles, user_input, verdict, verdict_reason, real_article_count)
    
    return {
        'verdict': verdict,
        'is_true': is_true,
        'credibility_score': float(credibility_score),
        'user_input': str(user_input),
        'main_analysis': str(main_analysis),
        'explanation_summary': str(explanation_summary),
        'sources_timeline': sources_timeline,
        'articles_count': int(len(articles)),
        'real_articles_count': real_article_count,
        'unique_sources': unique_sources,
        'timeline_span_hours': float(results['timeline_span_hours']),
        'temporal_patterns': {
            'sentiment_evolution': [float(x) for x in results['temporal_patterns']['sentiment_evolution']],
            'certainty_evolution': [float(x) for x in results['temporal_patterns']['certainty_evolution']],
            'urgency_evolution': [float(x) for x in results['temporal_patterns']['urgency_evolution']],
            'emotional_evolution': [float(x) for x in results['temporal_patterns']['emotional_evolution']]
        },
        'narrative_consistency': {
            'consistency_score': float(results['narrative_consistency']['consistency_score']),
            'contradiction_count': int(results['narrative_consistency']['contradiction_count'])
        }
    }

def generate_real_news_explanation(verdict, reason, articles, real_count, source_count):
    """Generate explanation based on real news search results"""
    
    if verdict == "LIKELY TRUE" or verdict == "PROBABLY TRUE":
        return f"""
        <strong>Why this news is {verdict}:</strong><br><br>
        
        ✅ <strong>Reason:</strong> {reason}<br><br>
        
        ✅ <strong>Real News Coverage:</strong> Found {real_count} actual news articles<br><br>
        
        ✅ <strong>Source Diversity:</strong> {source_count} independent news organizations reported this<br><br>
        
        ✅ <strong>Verification Method:</strong> Searched real news databases via NewsAPI<br><br>
        
        📰 <strong>Note:</strong> Click the article links to read the original sources
        """
    
    elif verdict == "UNVERIFIED":
        return f"""
        <strong>Why this news is {verdict}:</strong><br><br>
        
        ❌ <strong>Reason:</strong> {reason}<br><br>
        
        ❌ <strong>No Coverage Found:</strong> Major news sources have not reported on this<br><br>
        
        ⚠️ <strong>Possible Explanations:</strong><br>
        • The event may not have occurred<br>
        • It could be misinformation or a rumor<br>
        • The search terms may need adjustment<br>
        • The event may be too old or too recent<br><br>
        
        🔍 <strong>Recommendation:</strong> Be cautious sharing this information
        """
    
    elif verdict == "NEEDS VERIFICATION":
        return f"""
        <strong>Why this news {verdict}:</strong><br><br>
        
        ⚠️ <strong>Reason:</strong> {reason}<br><br>
        
        ⚠️ <strong>Limited Coverage:</strong> Only found {real_count} article(s)<br><br>
        
        ℹ️ <strong>Recommendation:</strong> Wait for more sources to confirm before sharing<br><br>
        
        🔍 <strong>Suggestion:</strong> Check the linked article and look for additional sources
        """
    
    else:
        return f"""
        <strong>Verification Status: {verdict}</strong><br><br>
        
        ℹ️ <strong>Reason:</strong> {reason}<br><br>
        
        ℹ️ <strong>Articles Found:</strong> {real_count}<br><br>
        
        🔍 <strong>Recommendation:</strong> Review the sources and seek additional verification
        """

def generate_main_analysis(articles, user_input, verdict, reason, real_count):
    """Generate main analysis text based on real findings"""
    
    if real_count > 0:
        first_source = articles[0]['source'] if articles else 'Unknown'
        return f"""
        <strong>Search Query:</strong> {user_input}<br><br>
        
        <strong>Verdict: {verdict}</strong><br><br>
        
        <strong>Analysis:</strong> {reason}<br><br>
        
        <strong>Primary Source:</strong> {first_source}<br><br>
        
        <strong>Method:</strong> Real-time search of news databases including BBC, CNN, Reuters, 
        The Guardian, and thousands of other verified news sources via NewsAPI.
        """
    else:
        return f"""
        <strong>Search Query:</strong> {user_input}<br><br>
        
        <strong>Verdict: {verdict}</strong><br><br>
        
        <strong>Analysis:</strong> No credible news coverage found for this topic. 
        This suggests the information may be unverified, false, or the search terms 
        may need adjustment.<br><br>
        
        <strong>Recommendation:</strong> Do not share this information until it can be 
        verified through credible news sources.
        """

if __name__ == '__main__':
    # Check if API keys are configured
    if not NEWSAPI_KEYS or NEWSAPI_KEYS[0] == 'YOUR_API_KEY_HERE':
        print("⚠️  WARNING: NewsAPI keys not configured!")
        print("📝 To use real news search:")
        print("   1. Get a free API key at: https://newsapi.org/register")
        print("   2. Add your keys to the NEWSAPI_KEYS list in app.py")
        print("")
    else:
        print(f"✅ {len(NEWSAPI_KEYS)} NewsAPI keys configured for rotation")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

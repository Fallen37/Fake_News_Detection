import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
from typing import List, Dict, Optional
import re

class NewsCollector:
    """Collects news articles about specific events from multiple sources over time"""
    
    def __init__(self):
        self.sources = {
            'reuters': 'https://www.reuters.com/search/news?blob=',
            'bbc': 'https://www.bbc.com/search?q=',
            'ap': 'https://apnews.com/search?q='
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def collect_event_timeline(self, event_query: str, days_back: int = 30) -> List[Dict]:
        """
        Collect articles about an event over time
        
        Args:
            event_query: Search query for the event (e.g., "resort fire Switzerland")
            days_back: How many days back to search
            
        Returns:
            List of article dictionaries with temporal information
        """
        articles = []
        
        # Simulate collecting articles from different time periods
        # In a real implementation, you'd use news APIs or web scraping
        
        # Mock data structure for demonstration
        mock_articles = self._generate_mock_timeline(event_query, days_back)
        
        return mock_articles
    
    def _generate_mock_timeline(self, event_query: str, days_back: int) -> List[Dict]:
        """Generate mock article timeline for demonstration"""
        
        # Example: Resort fire in Switzerland timeline
        if "resort fire" in event_query.lower():
            return [
                {
                    'title': 'Fire breaks out at luxury Swiss resort, 50 evacuated',
                    'content': 'A fire erupted at the Grand Alpine Resort in Zermatt early this morning. Emergency services evacuated approximately 50 guests and staff. The cause is under investigation.',
                    'source': 'reuters',
                    'timestamp': datetime.now() - timedelta(days=3),
                    'sentiment_score': -0.3,
                    'urgency_words': ['fire', 'evacuated', 'emergency'],
                    'certainty_level': 0.7
                },
                {
                    'title': 'Swiss resort fire caused by faulty electrical system, officials confirm',
                    'content': 'Local fire department confirms the Grand Alpine Resort fire was caused by an electrical fault in the kitchen area. No injuries reported. Investigation reveals sparklers from a birthday celebration may have triggered smoke detectors.',
                    'source': 'bbc',
                    'timestamp': datetime.now() - timedelta(days=2),
                    'sentiment_score': -0.1,
                    'urgency_words': ['confirms', 'investigation'],
                    'certainty_level': 0.9
                },
                {
                    'title': 'Resort reopens after minor kitchen incident, sparklers blamed for false alarm',
                    'content': 'The Grand Alpine Resort has reopened after what officials now describe as a minor kitchen incident. Sparklers used in a birthday celebration triggered sensitive smoke detection systems, leading to precautionary evacuation.',
                    'source': 'ap',
                    'timestamp': datetime.now() - timedelta(days=1),
                    'sentiment_score': 0.2,
                    'urgency_words': ['reopens', 'minor'],
                    'certainty_level': 0.95
                }
            ]
        
        # Default mock for other events
        return [
            {
                'title': f'Breaking: {event_query} reported',
                'content': f'Initial reports suggest {event_query} has occurred. Details are still emerging.',
                'source': 'reuters',
                'timestamp': datetime.now() - timedelta(days=2),
                'sentiment_score': -0.2,
                'urgency_words': ['breaking', 'reported'],
                'certainty_level': 0.5
            }
        ]
    
    def extract_temporal_features(self, articles: List[Dict]) -> pd.DataFrame:
        """Extract temporal features from article timeline"""
        
        features = []
        
        for i, article in enumerate(articles):
            feature_dict = {
                'sequence_position': i,
                'timestamp': article['timestamp'],
                'sentiment_score': article['sentiment_score'],
                'certainty_level': article['certainty_level'],
                'urgency_word_count': len(article['urgency_words']),
                'title_length': len(article['title']),
                'content_length': len(article['content']),
                'source': article['source']
            }
            
            # Calculate temporal deltas
            if i > 0:
                time_delta = (article['timestamp'] - articles[i-1]['timestamp']).total_seconds() / 3600  # hours
                sentiment_delta = article['sentiment_score'] - articles[i-1]['sentiment_score']
                certainty_delta = article['certainty_level'] - articles[i-1]['certainty_level']
                
                feature_dict.update({
                    'time_delta_hours': time_delta,
                    'sentiment_delta': sentiment_delta,
                    'certainty_delta': certainty_delta
                })
            else:
                feature_dict.update({
                    'time_delta_hours': 0,
                    'sentiment_delta': 0,
                    'certainty_delta': 0
                })
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
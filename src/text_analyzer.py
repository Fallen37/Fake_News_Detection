import nltk
from textblob import TextBlob
import re
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextAnalyzer:
    """Analyzes text features that change over time in news narratives"""
    
    def __init__(self):
        self.urgency_words = {
            'high': ['breaking', 'urgent', 'emergency', 'crisis', 'disaster', 'tragedy'],
            'medium': ['reported', 'alleged', 'claimed', 'suspected', 'investigation'],
            'low': ['confirms', 'concludes', 'reveals', 'explains', 'clarifies']
        }
        
        self.certainty_indicators = {
            'high': ['confirmed', 'verified', 'official', 'authorities state', 'police confirm'],
            'medium': ['reports suggest', 'sources say', 'appears to', 'seems to'],
            'low': ['rumored', 'unconfirmed', 'alleged', 'claimed', 'speculation']
        }
        
        self.emotional_words = {
            'fear': ['terrified', 'panic', 'horror', 'devastating', 'shocking'],
            'anger': ['outraged', 'furious', 'condemn', 'blame', 'scandal'],
            'sadness': ['tragic', 'heartbreaking', 'mourning', 'grief', 'sorrow'],
            'neutral': ['reported', 'stated', 'announced', 'disclosed', 'revealed']
        }
    
    def analyze_sentiment_evolution(self, articles: List[Dict]) -> List[float]:
        """Track sentiment changes across article timeline"""
        sentiments = []
        
        for article in articles:
            blob = TextBlob(article['content'])
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
        
        return sentiments
    
    def analyze_certainty_evolution(self, articles: List[Dict]) -> List[float]:
        """Track certainty level changes across timeline"""
        certainty_scores = []
        
        for article in articles:
            text = (article['title'] + ' ' + article['content']).lower()
            
            high_count = sum(1 for phrase in self.certainty_indicators['high'] if phrase in text)
            medium_count = sum(1 for phrase in self.certainty_indicators['medium'] if phrase in text)
            low_count = sum(1 for phrase in self.certainty_indicators['low'] if phrase in text)
            
            # Calculate weighted certainty score
            total_indicators = high_count + medium_count + low_count
            if total_indicators == 0:
                certainty_score = 0.5  # neutral
            else:
                certainty_score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.1) / total_indicators
            
            certainty_scores.append(certainty_score)
        
        return certainty_scores
    
    def analyze_vocabulary_shift(self, articles: List[Dict]) -> Dict[str, List[float]]:
        """Analyze how vocabulary usage shifts over time"""
        
        urgency_scores = []
        emotional_scores = []
        
        for article in articles:
            text = (article['title'] + ' ' + article['content']).lower()
            words = nltk.word_tokenize(text)
            
            # Calculate urgency score
            urgency_count = 0
            for level, word_list in self.urgency_words.items():
                multiplier = {'high': 1.0, 'medium': 0.6, 'low': 0.3}[level]
                urgency_count += sum(multiplier for word in word_list if word in text)
            
            urgency_scores.append(urgency_count / len(words) if words else 0)
            
            # Calculate emotional intensity
            emotional_count = 0
            for emotion, word_list in self.emotional_words.items():
                multiplier = 0.3 if emotion == 'neutral' else 1.0
                emotional_count += sum(multiplier for word in word_list if word in text)
            
            emotional_scores.append(emotional_count / len(words) if words else 0)
        
        return {
            'urgency_evolution': urgency_scores,
            'emotional_evolution': emotional_scores
        }
    
    def detect_narrative_consistency(self, articles: List[Dict]) -> Dict[str, float]:
        """Detect inconsistencies in narrative across timeline"""
        
        if len(articles) < 2:
            return {'consistency_score': 1.0, 'contradiction_count': 0}
        
        # Extract key facts from each article
        facts_timeline = []
        for article in articles:
            facts = self._extract_key_facts(article['content'])
            facts_timeline.append(facts)
        
        # Check for contradictions
        contradictions = 0
        total_comparisons = 0
        
        for i in range(1, len(facts_timeline)):
            for fact_type in facts_timeline[i]:
                if fact_type in facts_timeline[i-1]:
                    total_comparisons += 1
                    if facts_timeline[i][fact_type] != facts_timeline[i-1][fact_type]:
                        # Allow for refinement (e.g., "50 people" -> "approximately 50 people")
                        if not self._is_refinement(facts_timeline[i-1][fact_type], facts_timeline[i][fact_type]):
                            contradictions += 1
        
        consistency_score = 1.0 - (contradictions / max(total_comparisons, 1))
        
        return {
            'consistency_score': consistency_score,
            'contradiction_count': contradictions
        }
    
    def _extract_key_facts(self, text: str) -> Dict[str, str]:
        """Extract key factual claims from text"""
        facts = {}
        
        # Extract numbers (casualties, quantities, etc.)
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            facts['numbers'] = numbers
        
        # Extract locations
        location_patterns = [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', r'\bin [A-Z][a-z]+\b']
        locations = []
        for pattern in location_patterns:
            locations.extend(re.findall(pattern, text))
        if locations:
            facts['locations'] = locations
        
        # Extract causes (simple pattern matching)
        cause_patterns = [r'caused by ([^.]+)', r'due to ([^.]+)', r'because of ([^.]+)']
        causes = []
        for pattern in cause_patterns:
            causes.extend(re.findall(pattern, text, re.IGNORECASE))
        if causes:
            facts['causes'] = causes
        
        return facts
    
    def _is_refinement(self, old_fact: str, new_fact: str) -> bool:
        """Check if new fact is a refinement rather than contradiction of old fact"""
        
        # Simple heuristic: if new fact contains old fact, it's likely a refinement
        if isinstance(old_fact, list) and isinstance(new_fact, list):
            return any(old in new for old in old_fact for new in new_fact)
        
        return str(old_fact).lower() in str(new_fact).lower()
    
    def calculate_temporal_features(self, articles: List[Dict]) -> np.ndarray:
        """Calculate comprehensive temporal features for RNN input"""
        
        sentiments = self.analyze_sentiment_evolution(articles)
        certainties = self.analyze_certainty_evolution(articles)
        vocab_shifts = self.analyze_vocabulary_shift(articles)
        consistency = self.detect_narrative_consistency(articles)
        
        features = []
        
        for i, article in enumerate(articles):
            feature_vector = [
                sentiments[i],
                certainties[i],
                vocab_shifts['urgency_evolution'][i],
                vocab_shifts['emotional_evolution'][i],
                len(article['content']) / 1000,  # normalized content length
                i / len(articles),  # position in timeline
            ]
            
            # Add deltas if not first article
            if i > 0:
                feature_vector.extend([
                    sentiments[i] - sentiments[i-1],  # sentiment delta
                    certainties[i] - certainties[i-1],  # certainty delta
                ])
            else:
                feature_vector.extend([0, 0])
            
            features.append(feature_vector)
        
        return np.array(features)
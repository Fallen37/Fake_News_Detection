#!/usr/bin/env python3
"""
Advanced Text Analyzer for Fake News Detection
Implements comprehensive feature extraction for credibility scoring
"""

import nltk
from textblob import TextBlob
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import Counter
import math

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextAnalyzer:
    """Advanced text analyzer with comprehensive fake news detection features"""
    
    def __init__(self):
        # Urgency indicators
        self.urgency_words = {
            'high': ['breaking', 'urgent', 'emergency', 'crisis', 'disaster', 'tragedy', 'alert'],
            'medium': ['reported', 'alleged', 'claimed', 'suspected', 'investigation', 'developing'],
            'low': ['confirms', 'concludes', 'reveals', 'explains', 'clarifies', 'update']
        }
        
        # Certainty indicators
        self.certainty_indicators = {
            'high': ['confirmed', 'verified', 'official', 'authorities state', 'police confirm', 
                    'government says', 'according to officials', 'statement released'],
            'medium': ['reports suggest', 'sources say', 'appears to', 'seems to', 'likely'],
            'low': ['rumored', 'unconfirmed', 'alleged', 'claimed', 'speculation', 'possibly', 
                   'might', 'could be', 'reportedly']
        }
        
        # Emotional words for sentiment intensity
        self.emotional_words = {
            'fear': ['terrified', 'panic', 'horror', 'devastating', 'shocking', 'alarming', 'frightening'],
            'anger': ['outraged', 'furious', 'condemn', 'blame', 'scandal', 'disgrace', 'shameful'],
            'sadness': ['tragic', 'heartbreaking', 'mourning', 'grief', 'sorrow', 'devastating'],
            'sensational': ['incredible', 'unbelievable', 'amazing', 'stunning', 'explosive', 'bombshell'],
            'neutral': ['reported', 'stated', 'announced', 'disclosed', 'revealed', 'said', 'noted']
        }
        
        # Clickbait patterns
        self.clickbait_patterns = [
            r"you won't believe",
            r"what happened next",
            r"shocking",
            r"mind-blowing",
            r"this is why",
            r"here's what",
            r"the truth about",
            r"exposed",
            r"secret",
            r"\d+ reasons",
            r"\d+ things",
            r"everyone is talking about",
            r"goes viral",
            r"breaks the internet",
            r"you need to see",
            r"will shock you",
            r"can't stop",
            r"jaw-dropping"
        ]
        
        # Publisher credibility database (simplified - in production, use external API)
        self.publisher_credibility = {
            # Tier 1: Highly credible (0.9-1.0)
            'reuters': 0.95, 'associated press': 0.95, 'ap': 0.95, 'bbc': 0.92, 'bbc news': 0.92,
            'the guardian': 0.88, 'the new york times': 0.88, 'washington post': 0.87,
            'the economist': 0.90, 'npr': 0.88, 'pbs': 0.88,
            
            # Tier 2: Generally credible (0.7-0.89)
            'cnn': 0.78, 'abc news': 0.80, 'cbs news': 0.80, 'nbc news': 0.79,
            'times of india': 0.75, 'the hindu': 0.82, 'ndtv': 0.78, 'india today': 0.75,
            'al jazeera': 0.80, 'france 24': 0.82, 'dw': 0.82,
            'bloomberg': 0.85, 'financial times': 0.88, 'wall street journal': 0.85,
            
            # Tier 3: Mixed credibility (0.5-0.69)
            'fox news': 0.55, 'daily mail': 0.50, 'new york post': 0.55,
            'huffpost': 0.60, 'buzzfeed news': 0.62,
            
            # Wire services (high credibility as original sources)
            'afp': 0.93, 'efe': 0.85, 'ansa': 0.85
        }
        
        # Wire services for source redundancy check
        self.wire_services = {'reuters', 'associated press', 'ap', 'afp', 'efe', 'ansa', 'xinhua', 'itar-tass'}
    
    # ==================== FEATURE 1: Sentiment Intensity ====================
    def analyze_sentiment_intensity(self, text: str) -> Dict[str, float]:
        """
        Analyze emotional intensity of text.
        Fake news often uses emotionally charged or exaggerated language.
        """
        text_lower = text.lower()
        words = nltk.word_tokenize(text_lower)
        word_count = len(words) if words else 1
        
        # Count emotional words by category
        emotion_counts = {}
        total_emotional = 0
        
        for emotion, word_list in self.emotional_words.items():
            count = sum(1 for word in word_list if word in text_lower)
            emotion_counts[emotion] = count
            if emotion != 'neutral':
                total_emotional += count
        
        # Calculate intensity score (0-1, higher = more emotional/sensational)
        intensity_score = min(1.0, total_emotional / (word_count * 0.05))  # Expect ~5% emotional words max
        
        # Check for ALL CAPS words (shouting)
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])
        caps_ratio = caps_words / word_count
        
        # Check for excessive punctuation (!!!, ???)
        excessive_punct = len(re.findall(r'[!?]{2,}', text))
        
        # Combined intensity
        combined_intensity = (
            0.5 * intensity_score +
            0.25 * min(1.0, caps_ratio * 10) +
            0.25 * min(1.0, excessive_punct * 0.2)
        )
        
        return {
            'intensity_score': combined_intensity,
            'emotion_breakdown': emotion_counts,
            'caps_ratio': caps_ratio,
            'excessive_punctuation': excessive_punct,
            'is_sensational': combined_intensity > 0.4
        }
    
    # ==================== FEATURE 2: Lexical Complexity ====================
    def analyze_lexical_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze lexical complexity of text.
        Real news tends to have balanced complexity; fake news often uses simpler, repetitive phrases.
        """
        words = nltk.word_tokenize(text.lower())
        sentences = nltk.sent_tokenize(text)
        
        if not words or not sentences:
            return {'complexity_score': 0.5, 'is_suspicious': False}
        
        # 1. Average word length
        avg_word_length = np.mean([len(w) for w in words if w.isalpha()])
        
        # 2. Vocabulary richness (Type-Token Ratio)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        
        # 3. Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # 4. Repetition score (how often words repeat)
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(10)
        repetition_score = sum(freq for _, freq in most_common_freq) / len(words) if words else 0
        
        # 5. Flesch Reading Ease approximation
        syllable_count = sum(self._count_syllables(w) for w in words if w.isalpha())
        if len(words) > 0 and len(sentences) > 0:
            flesch = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
            flesch = max(0, min(100, flesch))
        else:
            flesch = 50
        
        # Complexity score (0-1, where 0.4-0.7 is "normal" news complexity)
        # Too simple (< 0.3) or too complex (> 0.8) can be suspicious
        complexity_score = (
            0.25 * min(1.0, avg_word_length / 8) +  # Longer words = more complex
            0.25 * ttr +  # Higher TTR = more varied vocabulary
            0.25 * min(1.0, avg_sentence_length / 30) +  # Longer sentences = more complex
            0.25 * (1 - repetition_score)  # Less repetition = more complex
        )
        
        # Flag if complexity is unusually low (potential fake news indicator)
        is_suspicious = complexity_score < 0.3 or ttr < 0.4
        
        return {
            'complexity_score': complexity_score,
            'avg_word_length': avg_word_length,
            'vocabulary_richness': ttr,
            'avg_sentence_length': avg_sentence_length,
            'repetition_score': repetition_score,
            'flesch_reading_ease': flesch,
            'is_suspicious': is_suspicious
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count"""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e'):
            count -= 1
        return max(1, count)
    
    # ==================== FEATURE 3: Clickbait Score ====================
    def calculate_clickbait_score(self, title: str, content: str = "") -> Dict[str, float]:
        """
        Detect clickbait patterns in headlines.
        """
        title_lower = title.lower()
        
        # Count clickbait pattern matches
        pattern_matches = 0
        matched_patterns = []
        
        for pattern in self.clickbait_patterns:
            if re.search(pattern, title_lower):
                pattern_matches += 1
                matched_patterns.append(pattern)
        
        # Check for question headlines (often clickbait)
        is_question = title.strip().endswith('?')
        
        # Check for numbers at start (listicles)
        starts_with_number = bool(re.match(r'^\d+', title.strip()))
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in title if c.isupper()) / len(title) if title else 0
        
        # Check title-content mismatch (if content provided)
        content_mismatch = 0
        if content:
            title_words = set(title_lower.split())
            content_words = set(content.lower().split()[:100])  # First 100 words
            overlap = len(title_words & content_words) / len(title_words) if title_words else 0
            content_mismatch = 1 - overlap
        
        # Calculate clickbait score (0-1)
        clickbait_score = (
            0.4 * min(1.0, pattern_matches * 0.3) +
            0.15 * (1 if is_question else 0) +
            0.1 * (1 if starts_with_number else 0) +
            0.15 * min(1.0, caps_ratio * 2) +
            0.2 * content_mismatch
        )
        
        return {
            'clickbait_score': clickbait_score,
            'pattern_matches': pattern_matches,
            'matched_patterns': matched_patterns,
            'is_question': is_question,
            'is_clickbait': clickbait_score > 0.4
        }
    
    # ==================== FEATURE 4: Entity Extraction ====================
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities (people, locations, organizations, dates, numbers).
        Used for fact-entity consistency checking.
        """
        entities = {
            'people': [],
            'locations': [],
            'organizations': [],
            'dates': [],
            'numbers': [],
            'money': []
        }
        
        # Extract numbers with context
        number_patterns = [
            (r'(\d+)\s*(dead|killed|died|injured|wounded|casualties)', 'casualties'),
            (r'(\d+)\s*(people|persons|individuals)', 'people_count'),
            (r'\$[\d,]+(?:\.\d{2})?|\d+\s*(?:million|billion|trillion)', 'money'),
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'dates'),
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b', 'dates')
        ]
        
        for pattern, entity_type in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if entity_type == 'casualties':
                    entities['numbers'].extend([m[0] for m in matches])
                elif entity_type in entities:
                    entities[entity_type].extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
        
        # Extract capitalized phrases (potential names/places)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Common location indicators
        location_indicators = ['in', 'at', 'near', 'from', 'to']
        for i, phrase in enumerate(cap_phrases):
            # Check context for location hints
            phrase_pos = text.find(phrase)
            if phrase_pos > 0:
                before = text[max(0, phrase_pos-10):phrase_pos].lower()
                if any(ind in before for ind in location_indicators):
                    entities['locations'].append(phrase)
                elif phrase not in entities['locations']:
                    # Could be person or organization
                    entities['people'].append(phrase)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities

    # ==================== FEATURE 5: Stance Similarity ====================
    def analyze_stance_similarity(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Compare how different articles frame the same event.
        High agreement = more credible; complete disagreement = suspicious.
        """
        if len(articles) < 2:
            return {'stance_similarity': 0.5, 'agreement_ratio': 0.5, 'has_contradictions': False}
        
        # Extract key claims from each article
        article_stances = []
        for article in articles:
            stance = {
                'sentiment': TextBlob(article['content']).sentiment.polarity,
                'entities': self.extract_entities(article['content']),
                'numbers': re.findall(r'\b\d+\b', article['content'])[:10]
            }
            article_stances.append(stance)
        
        # Calculate sentiment agreement
        sentiments = [s['sentiment'] for s in article_stances]
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
        sentiment_agreement = 1 - min(1.0, sentiment_std * 2)
        
        # Calculate number consistency (do articles report similar figures?)
        all_numbers = []
        for stance in article_stances:
            all_numbers.extend(stance['numbers'])
        
        if all_numbers:
            number_freq = Counter(all_numbers)
            most_common = number_freq.most_common(1)[0][1] if number_freq else 0
            number_consistency = most_common / len(articles)
        else:
            number_consistency = 0.5
        
        # Check for contradictions in key numbers (e.g., death tolls)
        has_contradictions = False
        if len(set(all_numbers)) > len(articles) * 2:
            has_contradictions = True
        
        # Combined stance similarity
        stance_similarity = (
            0.6 * sentiment_agreement +
            0.4 * number_consistency
        )
        
        return {
            'stance_similarity': stance_similarity,
            'sentiment_agreement': sentiment_agreement,
            'number_consistency': number_consistency,
            'has_contradictions': has_contradictions
        }
    
    # ==================== FEATURE 6: Publisher Credibility ====================
    def get_publisher_credibility(self, source_name: str) -> float:
        """
        Get credibility score for a publisher.
        Returns score from 0-1 based on known publisher database.
        """
        source_lower = source_name.lower().strip()
        
        # Direct match
        if source_lower in self.publisher_credibility:
            return self.publisher_credibility[source_lower]
        
        # Partial match
        for known_source, score in self.publisher_credibility.items():
            if known_source in source_lower or source_lower in known_source:
                return score
        
        # Unknown source - return moderate score
        return 0.5
    
    def calculate_source_diversity_score(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Calculate diversity and credibility of sources.
        """
        if not articles:
            return {'diversity_score': 0, 'avg_credibility': 0, 'unique_sources': 0}
        
        sources = [a.get('source', 'Unknown') for a in articles]
        unique_sources = set(sources)
        
        # Get credibility for each source
        credibilities = [self.get_publisher_credibility(s) for s in unique_sources]
        avg_credibility = np.mean(credibilities) if credibilities else 0.5
        
        # Diversity score based on number of unique sources
        diversity_score = min(1.0, len(unique_sources) / 5)  # Max out at 5 sources
        
        # Bonus for having high-credibility sources
        high_cred_sources = sum(1 for c in credibilities if c >= 0.8)
        high_cred_bonus = min(0.2, high_cred_sources * 0.05)
        
        return {
            'diversity_score': diversity_score,
            'avg_credibility': avg_credibility,
            'unique_sources': len(unique_sources),
            'high_credibility_count': high_cred_sources,
            'combined_score': (diversity_score + avg_credibility + high_cred_bonus) / 2
        }
    
    # ==================== FEATURE 7: Source Redundancy ====================
    def check_source_redundancy(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Check if multiple articles actually originate from the same wire service.
        Many outlets republish AP/Reuters without original reporting.
        """
        if not articles:
            return {'redundancy_score': 0, 'wire_service_ratio': 0, 'original_sources': 0}
        
        wire_mentions = 0
        original_sources = set()
        
        for article in articles:
            content_lower = article.get('content', '').lower()
            source_lower = article.get('source', '').lower()
            
            # Check if article mentions wire service as source
            is_wire = False
            for wire in self.wire_services:
                if wire in content_lower or wire in source_lower:
                    wire_mentions += 1
                    is_wire = True
                    break
            
            if not is_wire:
                original_sources.add(article.get('source', 'Unknown'))
        
        wire_ratio = wire_mentions / len(articles) if articles else 0
        
        # High wire ratio means less independent verification
        redundancy_score = wire_ratio
        
        return {
            'redundancy_score': redundancy_score,
            'wire_service_ratio': wire_ratio,
            'original_sources': len(original_sources),
            'total_articles': len(articles),
            'is_mostly_wire': wire_ratio > 0.7
        }
    
    # ==================== FEATURE 8: Time Gap Analysis ====================
    def analyze_time_gaps(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Analyze time gaps between reports.
        Real stories spread gradually; fake ones spike quickly and fade.
        """
        if len(articles) < 2:
            return {'spread_pattern': 'insufficient_data', 'spike_score': 0.5, 'is_suspicious': False}
        
        # Sort by timestamp
        sorted_articles = sorted(articles, key=lambda x: x['timestamp'])
        
        # Calculate time gaps between consecutive articles
        gaps = []
        for i in range(1, len(sorted_articles)):
            gap = (sorted_articles[i]['timestamp'] - sorted_articles[i-1]['timestamp']).total_seconds() / 3600
            gaps.append(gap)
        
        if not gaps:
            return {'spread_pattern': 'single_burst', 'spike_score': 0.8, 'is_suspicious': True}
        
        avg_gap = np.mean(gaps)
        gap_std = np.std(gaps)
        
        # Check for spike pattern (many articles in short time, then nothing)
        first_half_count = sum(1 for g in gaps[:len(gaps)//2] if g < avg_gap)
        second_half_count = sum(1 for g in gaps[len(gaps)//2:] if g < avg_gap)
        
        # Spike score: high if articles clustered at beginning
        if len(gaps) > 2:
            early_clustering = first_half_count / (len(gaps)//2 + 1)
            late_clustering = second_half_count / (len(gaps) - len(gaps)//2 + 1)
            spike_score = early_clustering - late_clustering + 0.5
            spike_score = max(0, min(1, spike_score))
        else:
            spike_score = 0.5
        
        # Determine spread pattern
        if avg_gap < 1:  # Less than 1 hour average
            spread_pattern = 'rapid_burst'
            is_suspicious = True
        elif avg_gap > 24:  # More than 24 hours average
            spread_pattern = 'slow_spread'
            is_suspicious = False
        else:
            spread_pattern = 'normal_spread'
            is_suspicious = False
        
        return {
            'spread_pattern': spread_pattern,
            'avg_gap_hours': avg_gap,
            'gap_std': gap_std,
            'spike_score': spike_score,
            'is_suspicious': is_suspicious
        }
    
    # ==================== FEATURE 9: Correction/Update Detection ====================
    def detect_corrections(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Detect if articles contain corrections or updates.
        Real outlets publish updates/corrections; fake ones rarely do.
        """
        correction_keywords = [
            'correction', 'update', 'updated', 'editor\'s note', 'clarification',
            'we previously reported', 'earlier version', 'has been corrected',
            'this article was updated', 'breaking update', 'developing story'
        ]
        
        corrections_found = 0
        updates_found = 0
        
        for article in articles:
            content_lower = article.get('content', '').lower()
            title_lower = article.get('title', '').lower()
            combined = content_lower + ' ' + title_lower
            
            for keyword in correction_keywords:
                if keyword in combined:
                    if 'correction' in keyword or 'clarification' in keyword:
                        corrections_found += 1
                    else:
                        updates_found += 1
                    break
        
        # Having corrections/updates is actually a GOOD sign (indicates responsible journalism)
        correction_ratio = (corrections_found + updates_found) / len(articles) if articles else 0
        
        return {
            'corrections_found': corrections_found,
            'updates_found': updates_found,
            'correction_ratio': correction_ratio,
            'has_corrections': corrections_found > 0,
            'credibility_boost': min(0.15, correction_ratio * 0.3)  # Boost credibility if corrections exist
        }
    
    # ==================== FEATURE 10: Narrative Consistency (RNN) ====================
    def analyze_narrative_drift(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Analyze if facts drift over time (e.g., "5 dead" → "50 dead").
        Uses pattern matching to detect significant changes in reported facts.
        """
        if len(articles) < 2:
            return {'drift_score': 0, 'fact_changes': [], 'is_consistent': True}
        
        # Sort by timestamp
        sorted_articles = sorted(articles, key=lambda x: x['timestamp'])
        
        # Track key numbers across timeline
        number_timeline = []
        for article in sorted_articles:
            numbers = re.findall(r'\b(\d+)\s*(?:dead|killed|died|injured|wounded|casualties|people)', 
                               article['content'], re.IGNORECASE)
            number_timeline.append([int(n) for n in numbers] if numbers else [])
        
        # Detect significant changes
        fact_changes = []
        drift_score = 0
        
        for i in range(1, len(number_timeline)):
            if number_timeline[i] and number_timeline[i-1]:
                prev_max = max(number_timeline[i-1])
                curr_max = max(number_timeline[i])
                
                # Check for order-of-magnitude changes (suspicious)
                if prev_max > 0:
                    change_ratio = curr_max / prev_max
                    if change_ratio > 5 or change_ratio < 0.2:
                        fact_changes.append({
                            'from': prev_max,
                            'to': curr_max,
                            'ratio': change_ratio
                        })
                        drift_score += 0.3
        
        drift_score = min(1.0, drift_score)
        
        return {
            'drift_score': drift_score,
            'fact_changes': fact_changes,
            'is_consistent': drift_score < 0.3,
            'number_timeline': number_timeline
        }

    # ==================== LEGACY METHODS (for compatibility) ====================
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
            
            total_indicators = high_count + medium_count + low_count
            if total_indicators == 0:
                certainty_score = 0.5
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
            
            urgency_count = 0
            for level, word_list in self.urgency_words.items():
                multiplier = {'high': 1.0, 'medium': 0.6, 'low': 0.3}[level]
                urgency_count += sum(multiplier for word in word_list if word in text)
            
            urgency_scores.append(urgency_count / len(words) if words else 0)
            
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
        
        drift_analysis = self.analyze_narrative_drift(articles)
        
        return {
            'consistency_score': 1.0 - drift_analysis['drift_score'],
            'contradiction_count': len(drift_analysis['fact_changes'])
        }
    
    def calculate_temporal_features(self, articles: List[Dict]) -> np.ndarray:
        """Calculate comprehensive temporal features for RNN input"""
        sentiments = self.analyze_sentiment_evolution(articles)
        certainties = self.analyze_certainty_evolution(articles)
        vocab_shifts = self.analyze_vocabulary_shift(articles)
        
        features = []
        
        for i, article in enumerate(articles):
            feature_vector = [
                sentiments[i],
                certainties[i],
                vocab_shifts['urgency_evolution'][i],
                vocab_shifts['emotional_evolution'][i],
                len(article['content']) / 1000,
                i / len(articles),
            ]
            
            if i > 0:
                feature_vector.extend([
                    sentiments[i] - sentiments[i-1],
                    certainties[i] - certainties[i-1],
                ])
            else:
                feature_vector.extend([0, 0])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    def comprehensive_analysis(self, articles: List[Dict]) -> Dict:
        """
        Run all analysis features and return comprehensive results.
        """
        if not articles:
            return {'error': 'No articles to analyze'}
        
        results = {
            'article_count': len(articles),
            'features': {}
        }
        
        # Aggregate sentiment intensity across all articles
        intensity_scores = []
        clickbait_scores = []
        complexity_scores = []
        
        for article in articles:
            # Sentiment intensity
            intensity = self.analyze_sentiment_intensity(article['content'])
            intensity_scores.append(intensity['intensity_score'])
            
            # Clickbait
            clickbait = self.calculate_clickbait_score(article['title'], article['content'])
            clickbait_scores.append(clickbait['clickbait_score'])
            
            # Lexical complexity
            complexity = self.analyze_lexical_complexity(article['content'])
            complexity_scores.append(complexity['complexity_score'])
        
        results['features']['sentiment_intensity'] = {
            'avg_score': np.mean(intensity_scores),
            'max_score': max(intensity_scores),
            'is_sensational': np.mean(intensity_scores) > 0.4
        }
        
        results['features']['clickbait'] = {
            'avg_score': np.mean(clickbait_scores),
            'max_score': max(clickbait_scores),
            'is_clickbait': np.mean(clickbait_scores) > 0.4
        }
        
        results['features']['lexical_complexity'] = {
            'avg_score': np.mean(complexity_scores),
            'is_suspicious': np.mean(complexity_scores) < 0.3
        }
        
        # Source-level analysis
        results['features']['source_diversity'] = self.calculate_source_diversity_score(articles)
        results['features']['source_redundancy'] = self.check_source_redundancy(articles)
        results['features']['stance_similarity'] = self.analyze_stance_similarity(articles)
        
        # Temporal analysis
        results['features']['time_gaps'] = self.analyze_time_gaps(articles)
        results['features']['corrections'] = self.detect_corrections(articles)
        results['features']['narrative_drift'] = self.analyze_narrative_drift(articles)
        
        return results

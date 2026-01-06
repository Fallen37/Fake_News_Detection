#!/usr/bin/env python3
"""
Semantic Search Module for Fake News Detection
Uses sentence embeddings to find semantically similar news articles
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import re

# Lazy loading for heavy models
_sentence_model = None
_spacy_nlp = None


def get_sentence_model():
    """Lazy load sentence transformer model"""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("🔄 Loading sentence embedding model (first time may take a minute)...")
            # Using a lightweight but effective model
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence embedding model loaded")
        except ImportError:
            print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")
            return None
    return _sentence_model


def get_spacy_nlp():
    """Lazy load spaCy model - returns None if not available"""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded")
            except OSError:
                print("⚠️ spaCy model not found. Entity extraction will use fallback.")
                return None
        except ImportError:
            print("⚠️ spaCy not installed. Entity extraction will use fallback.")
            return None
        except Exception as e:
            print(f"⚠️ spaCy error: {e}. Entity extraction will use fallback.")
            return None
    return _spacy_nlp


class ClaimExtractor:
    """Extracts and normalizes claims from user input"""
    
    def __init__(self):
        self.nlp = None
    
    def _ensure_nlp(self):
        if self.nlp is None:
            self.nlp = get_spacy_nlp()
        return self.nlp is not None
    
    def _fallback_entity_extraction(self, text: str) -> Dict:
        """Simple regex-based entity extraction when spaCy is not available"""
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'events': []
        }
        
        # Extract capitalized phrases (potential names/places/orgs)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Common location indicators
        location_words = ['in', 'at', 'near', 'from', 'to']
        for phrase in cap_phrases:
            phrase_pos = text.find(phrase)
            if phrase_pos > 0:
                before = text[max(0, phrase_pos-10):phrase_pos].lower()
                if any(ind in before for ind in location_words):
                    entities['locations'].append(phrase)
                else:
                    # Could be person or organization
                    entities['people'].append(phrase)
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(matches)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_claim(self, user_input: str) -> Dict[str, any]:
        """
        Extract the main claim from user input.
        
        Returns:
            Dict with:
            - claim: The extracted/normalized claim
            - entities: Named entities found
            - is_question: Whether input is a question
            - claim_type: Type of claim (statement, question, keywords)
        """
        user_input = user_input.strip()
        
        # Detect input type
        is_question = user_input.endswith('?') or user_input.lower().startswith(('is ', 'are ', 'was ', 'were ', 'did ', 'does ', 'do ', 'has ', 'have ', 'will ', 'can ', 'could ', 'would ', 'should '))
        
        # Check if it's just keywords (no verbs, short)
        words = user_input.split()
        is_keywords = len(words) <= 5 and not any(w.lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'did', 'does', 'will', 'can', 'said', 'says', 'claims', 'reported'] for w in words)
        
        result = {
            'original_input': user_input,
            'claim': user_input,
            'entities': {},
            'is_question': is_question,
            'claim_type': 'keywords' if is_keywords else ('question' if is_question else 'statement'),
            'search_query': user_input
        }
        
        # Use spaCy for entity extraction if available, otherwise use fallback
        if self._ensure_nlp():
            doc = self.nlp(user_input)
            
            # Extract named entities
            entities = {
                'people': [],
                'organizations': [],
                'locations': [],
                'dates': [],
                'events': []
            }
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['people'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ('GPE', 'LOC'):
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'EVENT':
                    entities['events'].append(ent.text)
            
            result['entities'] = entities
            
            # If it's a question, convert to statement for better search
            if is_question:
                result['claim'] = self._question_to_statement(user_input, doc)
            
            # If it's keywords, try to form a search query
            if is_keywords:
                result['search_query'] = self._keywords_to_query(user_input, entities)
            else:
                # Extract key noun phrases for search
                noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                if noun_phrases:
                    result['search_query'] = ' '.join(noun_phrases[:5])
        else:
            # Use fallback entity extraction
            entities = self._fallback_entity_extraction(user_input)
            result['entities'] = entities
            
            # Simple question to statement conversion
            if is_question:
                result['claim'] = self._question_to_statement(user_input, None)
            
            if is_keywords:
                result['search_query'] = self._keywords_to_query(user_input, entities)
        
        return result
    
    def _question_to_statement(self, question: str, doc) -> str:
        """Convert a question to a statement for better semantic matching"""
        question = question.rstrip('?').strip()
        
        # Simple transformations
        transformations = [
            (r'^is\s+(.+)', r'\1 is'),
            (r'^are\s+(.+)', r'\1 are'),
            (r'^was\s+(.+)', r'\1 was'),
            (r'^were\s+(.+)', r'\1 were'),
            (r'^did\s+(.+)', r'\1'),
            (r'^does\s+(.+)', r'\1'),
            (r'^has\s+(.+)', r'\1 has'),
            (r'^have\s+(.+)', r'\1 have'),
            (r'^will\s+(.+)', r'\1 will'),
        ]
        
        for pattern, replacement in transformations:
            if re.match(pattern, question, re.IGNORECASE):
                return re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _keywords_to_query(self, keywords: str, entities: Dict) -> str:
        """Convert keywords to a better search query"""
        # If we have entities, prioritize them
        query_parts = []
        
        for entity_type in ['people', 'organizations', 'locations', 'events']:
            query_parts.extend(entities.get(entity_type, []))
        
        if query_parts:
            return ' '.join(query_parts) + ' ' + keywords
        
        return keywords


class SemanticMatcher:
    """Matches claims to articles using semantic similarity"""
    
    def __init__(self):
        self.model = None
        self.claim_extractor = ClaimExtractor()
    
    def _ensure_model(self):
        if self.model is None:
            self.model = get_sentence_model()
        return self.model is not None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Convert text to embedding vector"""
        if not self._ensure_model():
            return None
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Convert multiple texts to embedding vectors"""
        if not self._ensure_model():
            return None
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    
    def calculate_similarity(self, claim_embedding: np.ndarray, article_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between claim and articles"""
        # Reshape claim embedding if needed
        if len(claim_embedding.shape) == 1:
            claim_embedding = claim_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(claim_embedding, article_embeddings)[0]
        return similarities
    
    def rank_articles_by_relevance(self, claim: str, articles: List[Dict], 
                                    min_similarity: float = 0.3) -> List[Dict]:
        """
        Rank articles by semantic similarity to the claim.
        
        Args:
            claim: The user's claim/query
            articles: List of article dicts with 'title' and 'content'
            min_similarity: Minimum similarity threshold
            
        Returns:
            Articles sorted by relevance with similarity scores
        """
        if not articles:
            return []
        
        if not self._ensure_model():
            # Fallback: return articles as-is if model not available
            print("⚠️ Semantic model not available, using keyword matching")
            return articles
        
        # Extract claim
        claim_info = self.claim_extractor.extract_claim(claim)
        claim_text = claim_info['claim']
        
        print(f"🧠 Semantic search for claim: '{claim_text}'")
        print(f"   Claim type: {claim_info['claim_type']}")
        if claim_info['entities']:
            entities_found = {k: v for k, v in claim_info['entities'].items() if v}
            if entities_found:
                print(f"   Entities: {entities_found}")
        
        # Encode claim
        claim_embedding = self.encode_text(claim_text)
        
        # Encode articles (title + first part of content for efficiency)
        article_texts = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')[:500]  # First 500 chars
            article_texts.append(f"{title}. {content}")
        
        article_embeddings = self.encode_texts(article_texts)
        
        if article_embeddings is None:
            return articles
        
        # Calculate similarities
        similarities = self.calculate_similarity(claim_embedding, article_embeddings)
        
        # Add similarity scores to articles
        ranked_articles = []
        for i, article in enumerate(articles):
            article_copy = article.copy()
            article_copy['semantic_similarity'] = float(similarities[i])
            article_copy['is_relevant'] = similarities[i] >= min_similarity
            ranked_articles.append(article_copy)
        
        # Sort by similarity (highest first)
        ranked_articles.sort(key=lambda x: x['semantic_similarity'], reverse=True)
        
        # Log results
        relevant_count = sum(1 for a in ranked_articles if a['is_relevant'])
        print(f"📊 Semantic matching: {relevant_count}/{len(articles)} articles above {min_similarity} threshold")
        
        if ranked_articles:
            print(f"   Top match: '{ranked_articles[0]['title'][:60]}...' (similarity: {ranked_articles[0]['semantic_similarity']:.3f})")
        
        return ranked_articles
    
    def filter_relevant_articles(self, claim: str, articles: List[Dict],
                                  min_similarity: float = 0.3,
                                  max_articles: int = 15) -> List[Dict]:
        """
        Filter and return only relevant articles.
        
        Args:
            claim: The user's claim/query
            articles: List of articles to filter
            min_similarity: Minimum similarity threshold
            max_articles: Maximum number of articles to return
            
        Returns:
            Filtered list of relevant articles
        """
        ranked = self.rank_articles_by_relevance(claim, articles, min_similarity)
        
        # Filter by relevance
        relevant = [a for a in ranked if a.get('is_relevant', False)]
        
        # If too few relevant articles, include some below threshold
        if len(relevant) < 3 and len(ranked) > len(relevant):
            # Add next best articles even if below threshold
            for article in ranked:
                if article not in relevant:
                    relevant.append(article)
                if len(relevant) >= 3:
                    break
        
        return relevant[:max_articles]


class ClaimVerifier:
    """Verifies claims by comparing against multiple sources"""
    
    def __init__(self):
        self.matcher = SemanticMatcher()
    
    def verify_claim_against_articles(self, claim: str, articles: List[Dict]) -> Dict:
        """
        Verify a claim by checking semantic similarity and stance across articles.
        
        Returns:
            Dict with verification results
        """
        if not articles:
            return {
                'verification_status': 'NO_SOURCES',
                'confidence': 0.0,
                'supporting_articles': 0,
                'contradicting_articles': 0,
                'explanation': 'No articles found to verify this claim.'
            }
        
        # Get claim info
        claim_info = self.matcher.claim_extractor.extract_claim(claim)
        
        # Rank articles by relevance
        ranked_articles = self.matcher.rank_articles_by_relevance(claim, articles, min_similarity=0.25)
        
        # Analyze stance of relevant articles
        supporting = 0
        contradicting = 0
        neutral = 0
        
        for article in ranked_articles:
            if not article.get('is_relevant', False):
                continue
            
            # Simple stance detection based on content
            stance = self._detect_stance(claim_info['claim'], article)
            article['stance'] = stance
            
            if stance == 'supporting':
                supporting += 1
            elif stance == 'contradicting':
                contradicting += 1
            else:
                neutral += 1
        
        # Calculate verification confidence
        total_relevant = supporting + contradicting + neutral
        
        if total_relevant == 0:
            return {
                'verification_status': 'UNVERIFIABLE',
                'confidence': 0.0,
                'supporting_articles': 0,
                'contradicting_articles': 0,
                'neutral_articles': 0,
                'explanation': 'No semantically relevant articles found.'
            }
        
        # Confidence based on support ratio and source count
        support_ratio = supporting / total_relevant if total_relevant > 0 else 0
        source_confidence = min(1.0, total_relevant / 5)  # Max confidence at 5+ sources
        
        confidence = (0.6 * support_ratio + 0.4 * source_confidence)
        
        # Determine verification status
        if supporting >= 3 and contradicting == 0:
            status = 'VERIFIED'
        elif supporting >= 2 and contradicting <= 1:
            status = 'LIKELY_TRUE'
        elif contradicting >= 2 and supporting <= 1:
            status = 'LIKELY_FALSE'
        elif contradicting > supporting:
            status = 'DISPUTED'
        elif supporting > 0:
            status = 'PARTIALLY_VERIFIED'
        else:
            status = 'UNVERIFIED'
        
        return {
            'verification_status': status,
            'confidence': confidence,
            'supporting_articles': supporting,
            'contradicting_articles': contradicting,
            'neutral_articles': neutral,
            'total_relevant': total_relevant,
            'claim_info': claim_info,
            'explanation': self._generate_verification_explanation(status, supporting, contradicting, neutral)
        }
    
    def _detect_stance(self, claim: str, article: Dict) -> str:
        """
        Detect whether an article supports, contradicts, or is neutral to the claim.
        Uses semantic similarity and keyword analysis.
        """
        content = article.get('content', '').lower()
        title = article.get('title', '').lower()
        combined = title + ' ' + content
        
        # Check for contradiction indicators
        contradiction_phrases = [
            'false', 'fake', 'hoax', 'debunked', 'not true', 'incorrect',
            'misleading', 'no evidence', 'denied', 'refuted', 'disproven',
            'fact check', 'myth', 'rumor', 'unsubstantiated'
        ]
        
        # Check for support indicators
        support_phrases = [
            'confirmed', 'verified', 'true', 'correct', 'accurate',
            'evidence shows', 'according to', 'officials say', 'reported',
            'announced', 'revealed', 'discovered', 'found'
        ]
        
        contradiction_count = sum(1 for phrase in contradiction_phrases if phrase in combined)
        support_count = sum(1 for phrase in support_phrases if phrase in combined)
        
        # High similarity + support phrases = supporting
        similarity = article.get('semantic_similarity', 0.5)
        
        if contradiction_count > support_count and contradiction_count >= 2:
            return 'contradicting'
        elif similarity > 0.5 and support_count > contradiction_count:
            return 'supporting'
        elif similarity > 0.4:
            return 'supporting'  # High similarity implies relevance/support
        else:
            return 'neutral'
    
    def _generate_verification_explanation(self, status: str, supporting: int, 
                                           contradicting: int, neutral: int) -> str:
        """Generate human-readable explanation of verification result"""
        
        explanations = {
            'VERIFIED': f"✅ Claim is VERIFIED. {supporting} independent sources confirm this information.",
            'LIKELY_TRUE': f"✅ Claim is LIKELY TRUE. {supporting} sources support it with {contradicting} contradicting.",
            'LIKELY_FALSE': f"❌ Claim is LIKELY FALSE. {contradicting} sources contradict it.",
            'DISPUTED': f"⚠️ Claim is DISPUTED. Sources disagree: {supporting} support, {contradicting} contradict.",
            'PARTIALLY_VERIFIED': f"⚠️ Claim is PARTIALLY VERIFIED. Some support found but needs more sources.",
            'UNVERIFIED': f"❓ Claim is UNVERIFIED. No clear support or contradiction found.",
            'UNVERIFIABLE': f"❓ Cannot verify. No relevant sources found.",
            'NO_SOURCES': f"❓ Cannot verify. No news sources available."
        }
        
        return explanations.get(status, "Unable to determine verification status.")


# Convenience function for integration
def semantic_search_articles(claim: str, articles: List[Dict], 
                             min_similarity: float = 0.3) -> List[Dict]:
    """
    Convenience function to perform semantic search on articles.
    
    Args:
        claim: User's claim or query
        articles: List of articles from news API
        min_similarity: Minimum similarity threshold
        
    Returns:
        Filtered and ranked articles by semantic relevance
    """
    matcher = SemanticMatcher()
    return matcher.filter_relevant_articles(claim, articles, min_similarity)


def verify_claim(claim: str, articles: List[Dict]) -> Dict:
    """
    Convenience function to verify a claim against articles.
    
    Args:
        claim: The claim to verify
        articles: List of articles to check against
        
    Returns:
        Verification results
    """
    verifier = ClaimVerifier()
    return verifier.verify_claim_against_articles(claim, articles)

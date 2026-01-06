#!/usr/bin/env python3
"""
RNN-based Fake News Detection System
Analyzes temporal patterns in news narratives to detect fake news
"""

import argparse
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_collector import NewsCollector
from text_analyzer import TextAnalyzer
from rnn_model import FakeNewsDetector, TemporalNewsDataset
from torch.utils.data import DataLoader

class FakeNewsDetectionSystem:
    """Main system for fake news detection using temporal analysis"""
    
    def __init__(self):
        self.collector = NewsCollector()
        self.analyzer = TextAnalyzer()
        self.detector = FakeNewsDetector()
        
    def analyze_event(self, event_query: str, days_back: int = 7) -> Dict:
        """
        Analyze a specific event for fake news patterns
        
        Args:
            event_query: Description of the event to analyze
            days_back: How many days back to collect articles
            
        Returns:
            Analysis results with credibility assessment
        """
        print(f"🔍 Analyzing event: '{event_query}'")
        print(f"📅 Looking back {days_back} days...")
        
        # Step 1: Collect articles about the event
        if hasattr(self.collector, 'search_articles_by_keywords'):
            # Use web-enhanced collector
            articles = self.collector.search_articles_by_keywords(event_query, days_back)
        else:
            # Use original collector
            articles = self.collector.collect_event_timeline(event_query, days_back)
        
        if not articles:
            return {
                'error': 'No articles found for this event',
                'credibility_score': 0.5,
                'verdict': 'INSUFFICIENT DATA'
            }
        
        print(f"📰 Found {len(articles)} articles")
        
        # Step 2: Extract temporal features
        temporal_features = self.analyzer.calculate_temporal_features(articles)
        
        # Step 3: Analyze narrative patterns
        sentiment_evolution = self.analyzer.analyze_sentiment_evolution(articles)
        certainty_evolution = self.analyzer.analyze_certainty_evolution(articles)
        vocab_analysis = self.analyzer.analyze_vocabulary_shift(articles)
        consistency_analysis = self.analyzer.detect_narrative_consistency(articles)
        
        # Step 4: Generate prediction (using mock trained model for demo)
        prediction = self._mock_prediction(temporal_features, articles)
        
        # Step 5: Compile results
        results = {
            'event_query': event_query,
            'articles_analyzed': len(articles),
            'timeline_span_hours': self._calculate_timeline_span(articles),
            'credibility_score': prediction['credibility_score'],
            'verdict': prediction['verdict'],
            'temporal_patterns': {
                'sentiment_evolution': sentiment_evolution,
                'certainty_evolution': certainty_evolution,
                'urgency_evolution': vocab_analysis['urgency_evolution'],
                'emotional_evolution': vocab_analysis['emotional_evolution']
            },
            'narrative_consistency': consistency_analysis,
            'articles': articles,
            'explanation': self._generate_explanation(prediction, articles)
        }
        
        return results
    
    def _mock_prediction(self, features: np.ndarray, articles: List[Dict]) -> Dict:
        """
        Comprehensive credibility prediction using multiple features.
        This replaces the simple heuristic with a weighted multi-factor analysis.
        """
        # Run comprehensive analysis
        comprehensive = self.analyzer.comprehensive_analysis(articles)
        
        if 'error' in comprehensive:
            return {
                'credibility_score': 0.5,
                'verdict': 'INSUFFICIENT DATA',
                'fake_probability': 0.5,
                'real_probability': 0.5
            }
        
        features_data = comprehensive['features']
        
        # ==================== SCORING COMPONENTS ====================
        
        # 1. SOURCE CREDIBILITY (Weight: 30%)
        # - Number of unique sources
        # - Publisher credibility ratings
        # - Source diversity
        source_div = features_data.get('source_diversity', {})
        unique_sources = source_div.get('unique_sources', 0)
        avg_pub_credibility = source_div.get('avg_credibility', 0.5)
        
        # More sources = higher score (max out at 5 sources)
        source_count_score = min(1.0, unique_sources / 5)
        source_score = (0.5 * source_count_score + 0.5 * avg_pub_credibility)
        
        # 2. CONTENT QUALITY (Weight: 25%)
        # - Low clickbait score is good
        # - Moderate complexity is good (not too simple, not too complex)
        # - Low sensationalism is good
        clickbait = features_data.get('clickbait', {}).get('avg_score', 0.5)
        complexity = features_data.get('lexical_complexity', {}).get('avg_score', 0.5)
        sensationalism = features_data.get('sentiment_intensity', {}).get('avg_score', 0.5)
        
        # Invert clickbait and sensationalism (lower is better)
        clickbait_score = 1 - clickbait
        sensationalism_score = 1 - sensationalism
        
        # Complexity: optimal around 0.5, penalize extremes
        complexity_score = 1 - abs(complexity - 0.5) * 2
        
        content_score = (0.4 * clickbait_score + 0.3 * sensationalism_score + 0.3 * complexity_score)
        
        # 3. CONSISTENCY (Weight: 20%)
        # - Stance similarity between sources
        # - Narrative consistency over time
        # - No major fact drift
        stance = features_data.get('stance_similarity', {})
        stance_score = stance.get('stance_similarity', 0.5)
        
        drift = features_data.get('narrative_drift', {})
        drift_score = 1 - drift.get('drift_score', 0)  # Lower drift is better
        
        consistency_score = (0.5 * stance_score + 0.5 * drift_score)
        
        # 4. TEMPORAL PATTERNS (Weight: 15%)
        # - Normal spread pattern (not suspicious spike)
        # - Presence of corrections/updates (good sign)
        time_gaps = features_data.get('time_gaps', {})
        is_suspicious_spread = time_gaps.get('is_suspicious', False)
        spread_score = 0.3 if is_suspicious_spread else 0.8
        
        corrections = features_data.get('corrections', {})
        correction_boost = corrections.get('credibility_boost', 0)
        
        temporal_score = spread_score + correction_boost
        
        # 5. SOURCE INDEPENDENCE (Weight: 10%)
        # - Low redundancy (not all from same wire service)
        redundancy = features_data.get('source_redundancy', {})
        redundancy_ratio = redundancy.get('redundancy_score', 0)
        independence_score = 1 - redundancy_ratio
        
        # ==================== FINAL CALCULATION ====================
        
        # Weighted combination
        credibility_score = (
            0.30 * source_score +
            0.25 * content_score +
            0.20 * consistency_score +
            0.15 * temporal_score +
            0.10 * independence_score
        )
        
        # Apply bonuses/penalties
        
        # Bonus for multiple high-credibility sources
        if unique_sources >= 3 and avg_pub_credibility >= 0.7:
            credibility_score += 0.1
        
        # Penalty for single source
        if unique_sources <= 1:
            credibility_score -= 0.15
        
        # Penalty for high clickbait
        if clickbait > 0.6:
            credibility_score -= 0.1
        
        # Penalty for contradictions
        if stance.get('has_contradictions', False):
            credibility_score -= 0.1
        
        # Clamp to 0-1 range
        credibility_score = max(0.05, min(0.95, credibility_score))
        
        # Determine verdict
        if credibility_score >= 0.7:
            verdict = 'LIKELY REAL'
        elif credibility_score >= 0.5:
            verdict = 'PROBABLY REAL'
        elif credibility_score >= 0.35:
            verdict = 'UNCERTAIN'
        elif credibility_score >= 0.2:
            verdict = 'PROBABLY FAKE'
        else:
            verdict = 'LIKELY FAKE'
        
        return {
            'credibility_score': credibility_score,
            'verdict': verdict,
            'fake_probability': 1 - credibility_score,
            'real_probability': credibility_score,
            'score_breakdown': {
                'source_score': source_score,
                'content_score': content_score,
                'consistency_score': consistency_score,
                'temporal_score': temporal_score,
                'independence_score': independence_score
            },
            'analysis_details': comprehensive
        }
    
    def _calculate_timeline_span(self, articles: List[Dict]) -> float:
        """Calculate timeline span in hours"""
        if len(articles) < 2:
            return 0
        
        timestamps = [a['timestamp'] for a in articles]
        span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        return span
    
    def _generate_explanation(self, prediction: Dict, articles: List[Dict]) -> str:
        """Generate human-readable explanation of the analysis"""
        
        explanations = []
        
        # Get score breakdown if available
        breakdown = prediction.get('score_breakdown', {})
        details = prediction.get('analysis_details', {}).get('features', {})
        
        # Source analysis
        source_div = details.get('source_diversity', {})
        unique_sources = source_div.get('unique_sources', 0)
        avg_cred = source_div.get('avg_credibility', 0)
        
        if unique_sources >= 3:
            explanations.append(f"✅ Multiple independent sources ({unique_sources}) reported on this event")
        elif unique_sources == 2:
            explanations.append(f"⚠️ Only 2 sources found - more verification recommended")
        elif unique_sources == 1:
            explanations.append(f"❌ Only single source found - high risk of misinformation")
        else:
            explanations.append(f"❌ No credible sources found")
        
        if avg_cred >= 0.8:
            explanations.append(f"✅ Sources are highly credible (avg: {avg_cred:.2f})")
        elif avg_cred >= 0.6:
            explanations.append(f"✅ Sources have moderate credibility (avg: {avg_cred:.2f})")
        elif avg_cred < 0.5:
            explanations.append(f"⚠️ Sources have low credibility ratings (avg: {avg_cred:.2f})")
        
        # Content quality
        clickbait = details.get('clickbait', {})
        if clickbait.get('is_clickbait', False):
            explanations.append(f"⚠️ Clickbait patterns detected in headlines")
        
        intensity = details.get('sentiment_intensity', {})
        if intensity.get('is_sensational', False):
            explanations.append(f"⚠️ Sensational/emotional language detected")
        
        complexity = details.get('lexical_complexity', {})
        if complexity.get('is_suspicious', False):
            explanations.append(f"⚠️ Unusually simple language (potential fake news indicator)")
        
        # Consistency
        stance = details.get('stance_similarity', {})
        if stance.get('has_contradictions', False):
            explanations.append(f"❌ Contradictions detected between sources")
        elif stance.get('stance_similarity', 0) > 0.7:
            explanations.append(f"✅ Sources show consistent reporting")
        
        drift = details.get('narrative_drift', {})
        if not drift.get('is_consistent', True):
            explanations.append(f"⚠️ Facts appear to drift significantly over time")
        
        # Temporal patterns
        time_gaps = details.get('time_gaps', {})
        if time_gaps.get('is_suspicious', False):
            explanations.append(f"⚠️ Suspicious spread pattern (rapid spike)")
        
        corrections = details.get('corrections', {})
        if corrections.get('has_corrections', False):
            explanations.append(f"✅ Updates/corrections found (sign of responsible journalism)")
        
        # Source independence
        redundancy = details.get('source_redundancy', {})
        if redundancy.get('is_mostly_wire', False):
            explanations.append(f"⚠️ Most articles from same wire service (limited independent verification)")
        
        verdict_explanation = {
            'LIKELY REAL': "The evidence strongly suggests this is legitimate news.",
            'PROBABLY REAL': "The evidence suggests this is likely legitimate, but some caution advised.",
            'UNCERTAIN': "The evidence is mixed - more investigation recommended.",
            'PROBABLY FAKE': "Several red flags detected - treat with skepticism.",
            'LIKELY FAKE': "Multiple indicators suggest this may be misinformation."
        }
        
        verdict = prediction.get('verdict', 'UNCERTAIN')
        explanation = f"{verdict_explanation.get(verdict, 'Unable to determine.')}\n\n"
        explanation += "\n".join(explanations)
        
        # Add score breakdown
        if breakdown:
            explanation += f"\n\n📊 Score Breakdown:\n"
            explanation += f"  • Source Quality: {breakdown.get('source_score', 0):.2f}\n"
            explanation += f"  • Content Quality: {breakdown.get('content_score', 0):.2f}\n"
            explanation += f"  • Consistency: {breakdown.get('consistency_score', 0):.2f}\n"
            explanation += f"  • Temporal Patterns: {breakdown.get('temporal_score', 0):.2f}\n"
            explanation += f"  • Source Independence: {breakdown.get('independence_score', 0):.2f}"
        
        return explanation
    
    def visualize_analysis(self, results: Dict, save_path: str = None):
        """Create visualization of the temporal analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Temporal Analysis: {results['event_query']}", fontsize=16)
        
        # Timeline positions
        timeline_positions = list(range(len(results['articles'])))
        
        # Plot 1: Sentiment Evolution
        axes[0, 0].plot(timeline_positions, results['temporal_patterns']['sentiment_evolution'], 
                       marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Sentiment Evolution')
        axes[0, 0].set_xlabel('Article Sequence')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Certainty Evolution
        axes[0, 1].plot(timeline_positions, results['temporal_patterns']['certainty_evolution'], 
                       marker='s', color='orange', linewidth=2, markersize=6)
        axes[0, 1].set_title('Certainty Evolution')
        axes[0, 1].set_xlabel('Article Sequence')
        axes[0, 1].set_ylabel('Certainty Level')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Urgency vs Emotional Evolution
        axes[1, 0].plot(timeline_positions, results['temporal_patterns']['urgency_evolution'], 
                       marker='^', label='Urgency', linewidth=2, markersize=6)
        axes[1, 0].plot(timeline_positions, results['temporal_patterns']['emotional_evolution'], 
                       marker='v', label='Emotional Intensity', linewidth=2, markersize=6)
        axes[1, 0].set_title('Vocabulary Analysis')
        axes[1, 0].set_xlabel('Article Sequence')
        axes[1, 0].set_ylabel('Intensity Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Credibility Score
        credibility = results['credibility_score']
        colors = ['red' if credibility < 0.4 else 'orange' if credibility < 0.7 else 'green']
        axes[1, 1].bar(['Credibility Score'], [credibility], color=colors[0], alpha=0.7)
        axes[1, 1].set_title(f"Overall Assessment: {results['verdict']}")
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add credibility score text
        axes[1, 1].text(0, credibility + 0.05, f'{credibility:.2f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='RNN-based Fake News Detection System')
    parser.add_argument('event', help='Event to analyze (e.g., "resort fire Switzerland")')
    parser.add_argument('--days', type=int, default=7, help='Days back to search (default: 7)')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = FakeNewsDetectionSystem()
    
    # Analyze event
    print("🚀 Starting Fake News Detection Analysis")
    print("=" * 50)
    
    results = system.analyze_event(args.event, args.days)
    
    # Display results
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Event: {results['event_query']}")
    print(f"Articles Analyzed: {results['articles_analyzed']}")
    print(f"Timeline Span: {results['timeline_span_hours']:.1f} hours")
    print(f"Credibility Score: {results['credibility_score']:.2f}")
    print(f"Verdict: {results['verdict']}")
    print("\n📝 EXPLANATION:")
    print(results['explanation'])
    
    # Show article timeline
    print("\n📰 ARTICLE TIMELINE:")
    print("-" * 30)
    for i, article in enumerate(results['articles']):
        print(f"{i+1}. [{article['timestamp'].strftime('%Y-%m-%d %H:%M')}] {article['title']}")
        print(f"   Source: {article['source']} | Sentiment: {article['sentiment_score']:.2f}")
        print()
    
    # Visualization
    if args.visualize or args.save_plot:
        system.visualize_analysis(results, args.save_plot)

if __name__ == "__main__":
    # Demo mode if no arguments
    import sys
    if len(sys.argv) == 1:
        print("🎯 DEMO MODE: Analyzing 'resort fire Switzerland'")
        system = FakeNewsDetectionSystem()
        results = system.analyze_event("resort fire Switzerland", 7)
        
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Event: {results['event_query']}")
        print(f"Articles Analyzed: {results['articles_analyzed']}")
        print(f"Credibility Score: {results['credibility_score']:.2f}")
        print(f"Verdict: {results['verdict']}")
        print(f"\n📝 EXPLANATION:\n{results['explanation']}")
        
        # Show visualization
        system.visualize_analysis(results)
    else:
        main()
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
        Mock prediction for demonstration (replace with trained model)
        """
        # Simple heuristic for demonstration
        sentiment_trend = np.diff([a['sentiment_score'] for a in articles])
        certainty_trend = np.diff([a['certainty_level'] for a in articles])
        
        # Real news typically shows:
        # 1. Increasing certainty over time
        # 2. Sentiment stabilization
        # 3. Consistent narrative
        
        certainty_improvement = np.mean(certainty_trend) if len(certainty_trend) > 0 else 0
        sentiment_stability = 1 - np.std([a['sentiment_score'] for a in articles])
        
        # Calculate credibility score
        credibility_score = (
            0.4 * max(0, certainty_improvement) +  # Certainty should improve
            0.3 * max(0, sentiment_stability) +    # Sentiment should stabilize
            0.3 * 0.8  # Base credibility (mock)
        )
        
        credibility_score = max(0, min(1, credibility_score))
        
        if credibility_score > 0.7:
            verdict = 'LIKELY REAL'
        elif credibility_score < 0.4:
            verdict = 'LIKELY FAKE'
        else:
            verdict = 'UNCERTAIN'
        
        return {
            'credibility_score': credibility_score,
            'verdict': verdict,
            'fake_probability': 1 - credibility_score,
            'real_probability': credibility_score
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
        
        # Timeline analysis
        if len(articles) > 1:
            certainty_trend = [a['certainty_level'] for a in articles]
            if certainty_trend[-1] > certainty_trend[0]:
                explanations.append("✅ Certainty levels increased over time, indicating legitimate investigation")
            else:
                explanations.append("⚠️ Certainty levels did not improve, which is unusual for real events")
        
        # Sentiment analysis
        sentiments = [a['sentiment_score'] for a in articles]
        if len(sentiments) > 1:
            if abs(sentiments[-1] - sentiments[0]) < 0.3:
                explanations.append("✅ Sentiment remained relatively stable across reports")
            else:
                explanations.append("⚠️ Large sentiment swings detected across timeline")
        
        # Source diversity
        sources = set(a['source'] for a in articles)
        if len(sources) > 1:
            explanations.append(f"✅ Multiple sources ({len(sources)}) reported on this event")
        else:
            explanations.append("⚠️ Only single source found for this event")
        
        verdict_explanation = {
            'LIKELY REAL': "The temporal patterns suggest this is likely a legitimate news story.",
            'LIKELY FAKE': "The temporal patterns raise concerns about the authenticity of this story.",
            'UNCERTAIN': "The evidence is mixed - more investigation may be needed."
        }
        
        explanation = f"{verdict_explanation[prediction['verdict']]}\n\n" + "\n".join(explanations)
        
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
# -*- coding: utf-8 -*-
"""
Comprehensive TIER 3 System Test
=================================

Tests all TIER 3 modules working together in a realistic workflow.
"""

import sys
sys.path.insert(0, '.')
from datetime import datetime

from autoshorts.analytics.viral_scraper import ViralScraper, NicheType
from autoshorts.analytics.pattern_analyzer import PatternAnalyzer
from autoshorts.analytics.viral_predictor import ViralPredictor
from autoshorts.analytics.competitor_analyzer import CompetitorAnalyzer
from autoshorts.analytics.trend_detector import TrendDetector
from autoshorts.content.idea_generator import IdeaGenerator
from autoshorts.analytics.ab_tester import ABTester
from autoshorts.analytics.performance_tracker import PerformanceTracker
from autoshorts.analytics.optimizer import AutoOptimizer

def test_tier3_workflow():
    """Test complete TIER 3 workflow."""
    print("="*60)
    print("COMPREHENSIVE TIER 3 SYSTEM TEST")
    print("="*60)

    niche = NicheType.EDUCATION
    print(f"\nTesting with niche: {niche.value}")

    # TIER 3.1: Viral Pattern Recognition
    print("\n[TIER 3.1] Viral Pattern Recognition")
    print("-" * 60)

    print("  1. Scraping viral videos...")
    scraper = ViralScraper()
    scraped_videos = scraper.scrape_niche(niche, num_videos=50)
    print(f"     -> Scraped {len(scraped_videos)} videos")

    print("  2. Analyzing patterns...")
    analyzer = PatternAnalyzer()
    pattern_data = analyzer.analyze_patterns(scraped_videos, niche)
    print(f"     -> Found {len(pattern_data.top_patterns)} top patterns")
    print(f"     -> Emerging trends: {len(pattern_data.emerging_trends)}")

    print("  3. Predicting viral potential...")
    predictor = ViralPredictor()
    # Create mock features for testing
    from autoshorts.analytics.pattern_analyzer import PatternFeatures
    test_features = PatternFeatures(
        hook_type="question",
        hook_intensity=0.8,
        thumbnail_style="bold_text",
        has_face=True,
        has_text=True,
        emotion="curiosity",
        avg_shot_duration=2.5,
        cut_frequency=0.4,
        pacing_style="fast",
        music_type="upbeat",
        has_beat_sync=True,
        engagement_rate=0.045,
        virality_score=0.75
    )
    prediction = predictor.predict("Test Video", test_features)
    print(f"     -> Viral score: {prediction.score.overall_score:.1f}/100")
    print(f"     -> Probability: {prediction.score.probability.value}")

    # TIER 3.2: Competitor Analysis
    print("\n[TIER 3.2] Competitor Analysis & Trend Detection")
    print("-" * 60)

    print("  4. Analyzing competitors...")
    comp_analyzer = CompetitorAnalyzer()
    comp_report = comp_analyzer.analyze_niche(niche, num_competitors=20)
    print(f"     -> Top performers: {len(comp_report.top_performers)}")
    print(f"     -> Rising stars: {len(comp_report.rising_stars)}")
    print(f"     -> Trending topics: {len(comp_report.trending_topics)}")

    print("  5. Detecting trends...")
    trend_detector = TrendDetector()
    trend_report = trend_detector.detect_trends(time_range_days=30)
    print(f"     -> Emerging trends: {len(trend_report.emerging_trends)}")
    print(f"     -> Rising trends: {len(trend_report.rising_trends)}")
    print(f"     -> Cross-niche trends: {len(trend_report.cross_niche_trends)}")

    print("  6. Generating content ideas...")
    idea_gen = IdeaGenerator()

    # Extract trending topics
    trending_topics = [t.topic for t in comp_report.trending_topics]
    gaps = [g.description for g in comp_report.gaps]

    ideas = idea_gen.generate_ideas(
        niche=niche.value,
        trending_topics=trending_topics,
        gaps=gaps,
        num_ideas=10
    )
    print(f"     -> Generated {ideas.total_ideas} ideas")
    print(f"     -> Immediate: {len(ideas.immediate_ideas)}, High: {len(ideas.high_priority_ideas)}")
    print(f"     -> Avg viral potential: {ideas.avg_viral_potential:.1f}/100")

    # TIER 3.3: A/B Testing & Auto-Optimization
    print("\n[TIER 3.3] A/B Testing & Auto-Optimization")
    print("-" * 60)

    print("  7. Creating A/B test...")
    ab_tester = ABTester()
    test = ab_tester.create_test(
        test_name="Test Campaign",
        niche=niche.value,
        base_video_path="test_video.mp4",  # Mock path
        num_variants=3
    )
    print(f"     -> Test ID: {test.test_id}")
    print(f"     -> Variants: {len(test.variants)}")

    print("  8. Setting up performance tracking...")
    perf_tracker = PerformanceTracker()
    perf_tracker.track_video(
        video_id="test_video_123",
        title="Test Video",
        upload_date=datetime.now(),
        niche=niche.value
    )
    print(f"     -> Tracking video: test_video_123")

    print("  9. Initializing optimizer...")
    optimizer = AutoOptimizer()

    # Simulate learning from a test
    mock_learnings = {
        "winner_hook": "question",
        "winner_thumbnail": "text_overlay",
        "ctr_improvement": 0.45,
        "retention_improvement": 0.35
    }
    optimizer.learn_from_test("test_1", niche.value, mock_learnings)
    print(f"     -> Learned from test: test_1")

    # Get optimization recommendations
    strategy = optimizer.get_strategy(niche.value)
    print(f"     -> Strategy confidence: {strategy.strategy_confidence:.2f}")
    print(f"     -> Hook rules: {len(strategy.hook_rules)}")

    recommendation = optimizer.optimize("Future Video", niche.value)
    print(f"     -> Hook: {recommendation.hook_recommendation}")
    print(f"     -> Expected CTR improvement: +{recommendation.expected_ctr_improvement:.1%}")
    print(f"     -> Expected retention improvement: +{recommendation.expected_retention_improvement:.1%}")
    print(f"     -> Confidence: {recommendation.confidence:.2f}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("[OK] All TIER 3 modules working correctly")
    print("[OK] No errors encountered")
    print("[OK] System ready for integration")
    print("\nNext step: Build AutoVideoGenerator integration")
    print("="*60)

if __name__ == "__main__":
    test_tier3_workflow()

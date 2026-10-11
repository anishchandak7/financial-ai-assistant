import unittest
import os
import sys
from pathlib import Path

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.tools.technical_indicators import (
    get_complete_technical_profile,
    fetch_ticker_history,
    calculate_trend_indicators,
    calculate_momentum_indicators,
    calculate_volatility_indicators,
    calculate_volume_indicators,
    calculate_pivot_points,
    compute_technical_consensus
)
from agents.tools.stock_analyzer_tools import (
    calculate_technical_indicators,
    get_indicator_recommendations
)
from agents.specialists import StockAnalyzerAgent, MarketAnalysisAgent

class TestStockAnalyzer(unittest.TestCase):

    def test_technical_indicators_engine(self):
        """Verify that the indicator engine returns complete numerical outputs for AAPL."""
        profile = get_complete_technical_profile("AAPL", period="6mo")
        self.assertNotIn("error", profile, f"Error in profile: {profile.get('error')}")
        self.assertEqual(profile["symbol"], "AAPL")
        self.assertGreater(profile["current_price"], 0)
        self.assertGreaterEqual(profile["candles_analyzed"], 20)

        # Trend checks
        trend = profile["trend_indicators"]
        self.assertNotEqual(trend["SMA_20"], "N/A")
        self.assertNotEqual(trend["SMA_50"], "N/A")
        self.assertIsInstance(trend["above_sma20"], bool)

        # Momentum checks
        momentum = profile["momentum_indicators"]
        self.assertIsInstance(momentum["RSI_14"], (int, float))
        self.assertTrue(0 <= momentum["RSI_14"] <= 100)
        self.assertNotEqual(momentum["MACD_line"], "N/A")
        self.assertNotEqual(momentum["MACD_signal"], "N/A")

        # Volatility checks
        vol = profile["volatility_indicators"]
        self.assertNotEqual(vol["BB_Upper"], "N/A")
        self.assertNotEqual(vol["BB_Lower"], "N/A")
        self.assertNotEqual(vol["ATR_14"], "N/A")

        # Volume checks
        volume = profile["volume_indicators"]
        self.assertNotEqual(volume["Latest_Volume"], "N/A")
        self.assertNotEqual(volume["OBV_Trend"], "N/A")

        # Pivot checks
        pivots = profile["pivot_points"]
        self.assertNotEqual(pivots["Pivot_Point_P"], "N/A")
        self.assertNotEqual(pivots["Support_S1"], "N/A")
        self.assertNotEqual(pivots["Resistance_R1"], "N/A")

        # Consensus checks
        consensus = profile["consensus"]
        self.assertIn(
            consensus["consensus_rating"],
            [
                "STRONGLY BULLISH", "MODERATELY BULLISH", "NEUTRAL / BALANCED",
                "MODERATELY BEARISH", "STRONGLY BEARISH"
            ]
        )

    def test_indicator_category_filters(self):
        """Verify category filtering in calculate_technical_indicators tool."""
        momentum_rep = calculate_technical_indicators.invoke({"symbol": "NVDA", "indicator_category": "momentum"})
        self.assertIn("RSI (14-Period)", momentum_rep)
        self.assertIn("MACD", momentum_rep)

        trend_rep = calculate_technical_indicators.invoke({"symbol": "NVDA", "indicator_category": "trend"})
        self.assertIn("Simple Moving Averages", trend_rep)
        self.assertIn("20-Day SMA", trend_rep)

        pivots_rep = calculate_technical_indicators.invoke({"symbol": "NVDA", "indicator_category": "support_resistance"})
        self.assertIn("Floor Pivot Point (P)", pivots_rep)
        self.assertIn("Resistance Levels", pivots_rep)

    def test_indicator_recommendations(self):
        """Verify smart recommendations based on user intent."""
        rec_momentum = get_indicator_recommendations.invoke({"symbol": "TSLA", "query_intent": "is it overbought or peak"})
        self.assertIn("MOMENTUM", rec_momentum)

        rec_trend = get_indicator_recommendations.invoke({"symbol": "TSLA", "query_intent": "what is the long term trend"})
        self.assertIn("TREND", rec_trend)

        rec_pivots = get_indicator_recommendations.invoke({"symbol": "TSLA", "query_intent": "key support and resistance levels"})
        self.assertIn("SUPPORT_RESISTANCE", rec_pivots)

    def test_stock_analyzer_subagent_run(self):
        """Verify StockAnalyzerAgent executes end-to-end on a momentum query."""
        subagent = StockAnalyzerAgent()
        result = subagent.run(input_query="Is AAPL currently overbought? Check RSI and key momentum indicators.", ticker="AAPL")
        self.assertIsNotNone(result)
        self.assertIn("response", result)
        self.assertGreater(len(result["response"]), 50)
        self.assertTrue("AAPL" in result["response"] or "Apple" in result["response"])

    def test_market_analysis_agent_delegation(self):
        """Verify MarketAnalysisAgent spawns the sub-agent and provides a synthesized response."""
        parent_agent = MarketAnalysisAgent()
        result = parent_agent.run(input_query="Provide a technical analysis of NVDA and check if it is overbought.")
        self.assertIsNotNone(result)
        self.assertIn("response", result)
        self.assertGreater(len(result["response"]), 100)
        self.assertTrue("NVDA" in result["response"] or "Nvidia" in result["response"])

if __name__ == "__main__":
    unittest.main()

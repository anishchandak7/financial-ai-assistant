import json
from typing import Dict, Any
from langchain_core.tools import tool
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
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

INDICATOR_SUITE_MAPPING = {
    "momentum": {
        "description": "Evaluates price velocity, speed, and overbought/oversold extremes.",
        "primary_indicators": ["RSI (14-period)", "Stochastic Oscillator (%K, %D)", "MACD Histogram", "Rate of Change (ROC)"],
        "best_for": ["Overbought/oversold conditions", "Exhaustion tops/bottoms", "Mean reversion", "Momentum divergences"]
    },
    "trend": {
        "description": "Evaluates overall directional trajectory and structural strength.",
        "primary_indicators": ["SMA (20, 50, 200)", "EMA (12, 26, 50)", "Golden Cross / Death Cross", "Moving Average Alignment"],
        "best_for": ["Identifying bull vs bear markets", "Trend following", "Pullbacks to moving averages", "Long-term health"]
    },
    "volatility": {
        "description": "Measures price dispersion, compression, and expected range of movement.",
        "primary_indicators": ["Bollinger Bands (%B, Bandwidth)", "Bollinger Squeeze Alert", "ATR (Average True Range 14)", "Period Range Position"],
        "best_for": ["Breakout setups", "Volatility expansion/contraction", "Setting stop losses", "Target risk sizing"]
    },
    "volume": {
        "description": "Quantifies buying vs selling conviction and institutional footprint.",
        "primary_indicators": ["Volume SMA (20-period)", "Volume Surge Ratio", "On-Balance Volume (OBV) Accumulation/Distribution"],
        "best_for": ["Confirming trend breakouts", "Detecting institutional accumulation", "Spotting false rallies"]
    },
    "support_resistance": {
        "description": "Pinpoints actionable structural price inflection and entry/exit zones.",
        "primary_indicators": ["Floor Pivot Points (P, S1, S2, R1, R2)", "20-day Swing High", "20-day Swing Low", "Dynamic SMA Supports"],
        "best_for": ["Target prices", "Stop loss placement", "Key support bounces", "Resistance breakout triggers"]
    }
}

@tool
def get_indicator_recommendations(symbol: str, query_intent: str) -> str:
    """Provides smart technical indicator selection advice and mapping for a given stock and user query intent.
    
    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA').
        query_intent: What the user is trying to analyze. Examples:
            - 'overbought_or_oversold' (or momentum/reversal)
            - 'trend_direction' (or bullish/bearish market trend)
            - 'volatility_or_breakout' (or squeeze/risk)
            - 'volume_confirmation' (or institutional accumulation)
            - 'key_levels_or_targets' (or support/resistance/entries)
            - 'comprehensive_evaluation' (overall technical health)
            
    Returns:
        Structured recommendation detailing which specific indicators are best suited and why.
    """
    clean_sym = symbol.strip().upper()
    intent = query_intent.strip().lower()
    logger.info(msg=f"Recommending technical indicators for {clean_sym} with intent '{query_intent}'")

    selected_category = "all"
    if any(k in intent for k in ["overbought", "oversold", "momentum", "reversal", "rsi", "stochastic", "peak"]):
        selected_category = "momentum"
    elif any(k in intent for k in ["trend", "direction", "bullish", "bearish", "moving average", "golden cross", "death cross", "sma"]):
        selected_category = "trend"
    elif any(k in intent for k in ["volatility", "squeeze", "breakout", "bollinger", "atr", "range"]):
        selected_category = "volatility"
    elif any(k in intent for k in ["volume", "accumulation", "distribution", "obv", "surge", "conviction"]):
        selected_category = "volume"
    elif any(k in intent for k in ["support", "resistance", "pivot", "target", "entry", "exit", "stop loss"]):
        selected_category = "support_resistance"

    if selected_category in INDICATOR_SUITE_MAPPING:
        suite = INDICATOR_SUITE_MAPPING[selected_category]
        return (
            f"### Indicator Selection Recommendation for {clean_sym}:\n"
            f"- **Primary Category Selected**: `{selected_category.upper()}`\n"
            f"- **Rationale**: {suite['description']}\n"
            f"- **Recommended Indicators**: {', '.join(suite['primary_indicators'])}\n"
            f"- **Target Use-Cases**: {', '.join(suite['best_for'])}\n\n"
            f"Recommendation: Call `calculate_technical_indicators` with `symbol='{clean_sym}'` and `indicator_category='{selected_category}'` (or 'all' for full context)."
        )
    else:
        return (
            f"### Comprehensive Indicator Selection Recommendation for {clean_sym}:\n"
            f"- **Category Selected**: `ALL (4-Pillar Comprehensive Suite)`\n"
            f"- **Rationale**: The user query requests a full or multi-faceted evaluation. "
            f"Evaluating Trend, Momentum, Volatility, and Volume yields the most reliable holistic technical consensus.\n"
            f"- **Recommended Tool Call**: `calculate_technical_indicators(symbol='{clean_sym}', indicator_category='all')`"
        )

@tool
def calculate_technical_indicators(
    symbol: str,
    indicator_category: str = "all",
    period: str = "6mo"
) -> str:
    """Calculates quantitative technical analysis indicators derived from yfinance historical data.
    
    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA', 'MSFT').
        indicator_category: Category of indicators to retrieve:
            - 'momentum': RSI (14), MACD (12,26,9), Stochastic Oscillator, ROC.
            - 'trend': SMA (20, 50, 200), EMA (12, 26, 50), Golden/Death Crosses, price alignment.
            - 'volatility': Bollinger Bands, %B, Bandwidth Squeeze alert, ATR (14), 52-week range position.
            - 'volume': 20-day Volume SMA, Volume Surge ratio, On-Balance Volume (OBV) trend.
            - 'support_resistance': Classical Pivot Points (P, S1, S2, R1, R2) and 20-day Swing High/Low.
            - 'all': Complete multi-pillar profile with overall technical consensus rating.
        period: Historical lookback period for calculation (e.g. '3mo', '6mo', '1y'). Default is '6mo'.
        
    Returns:
        Structured string containing verified quantitative indicator values, interpretations, and signals.
    """
    clean_sym = symbol.strip().upper()
    cat = indicator_category.strip().lower()
    logger.info(msg=f"Calculating technical indicators for {clean_sym} | Category: {cat} | Period: {period}")

    profile = get_complete_technical_profile(clean_sym, period=period)
    if "error" in profile:
        return f"Error analyzing technical indicators for {clean_sym}: {profile['error']}"

    curr_p = profile["current_price"]
    chg = profile["day_change"]
    pct = profile["day_pct_change"]
    header = f"### Technical Indicator Report for {clean_sym}\n- **Current Price**: ${curr_p:,.2f} ({chg:+.2f}, {pct}) | Data Period: {profile['data_period']} ({profile['candles_analyzed']} bars)\n"

    # Category: Momentum
    if cat == "momentum":
        m = profile["momentum_indicators"]
        return header + (
            f"#### Momentum & Oscillator Indicators:\n"
            f"- **RSI (14-Period)**: {m['RSI_14']} | Status: **{m['RSI_status']}** (Slope: {m['RSI_slope']})\n"
            f"- **MACD (12, 26, 9)**:\n"
            f"  - MACD Line: {m['MACD_line']} | Signal Line: {m['MACD_signal']} | Histogram: {m['MACD_histogram']}\n"
            f"  - Interpretation: **{m['MACD_interpretation']}**\n"
            f"- **Stochastic Oscillator (%K, %D)**: %K={m['Stochastic_K']}, %D={m['Stochastic_D']} | Status: **{m['Stochastic_status']}**\n"
            f"- **Rate of Change (Momentum)**: 10-day: {m['ROC_10d_pct']} | 14-day: {m['ROC_14d_pct']}"
        )

    # Category: Trend
    elif cat == "trend":
        t = profile["trend_indicators"]
        return header + (
            f"#### Trend & Moving Average Indicators:\n"
            f"- **Simple Moving Averages (SMA)**:\n"
            f"  - 20-Day SMA: ${t['SMA_20']} (Price is {'Above' if t['above_sma20'] else 'Below'})\n"
            f"  - 50-Day SMA: ${t['SMA_50']} (Price is {'Above' if t['above_sma50'] else 'Below'})\n"
            f"  - 200-Day SMA: ${t['SMA_200']} (Price is {'Above' if t['above_sma200'] else 'Below'})\n"
            f"- **Exponential Moving Averages (EMA)**: 12-EMA: ${t['EMA_12']} | 26-EMA: ${t['EMA_26']} | 50-EMA: ${t['EMA_50']}\n"
            f"- **Crossover Signal**: **{t['crossover_status']}**\n"
            f"- **Short-Term Trend**: **{t['short_term_trend']}**"
        )

    # Category: Volatility
    elif cat == "volatility":
        v = profile["volatility_indicators"]
        return header + (
            f"#### Volatility & Range Indicators:\n"
            f"- **Bollinger Bands (20-day, 2 std)**:\n"
            f"  - Upper Band: ${v['BB_Upper']} | Middle Band: ${v['BB_Middle']} | Lower Band: ${v['BB_Lower']}\n"
            f"  - %B (Position): {v['BB_Percent_B']} | Bandwidth: {v['BB_Bandwidth_pct']}\n"
            f"  - Signal: **{v['BB_interpretation']}**\n"
            f"- **Average True Range (ATR-14)**: ${v['ATR_14']} ({v['ATR_pct_of_price']} of price)\n"
            f"- **Period High / Low**: ${v['Period_High']} / ${v['Period_Low']} (Position: {v['Range_Position_pct']})"
        )

    # Category: Volume
    elif cat == "volume":
        vol = profile["volume_indicators"]
        return header + (
            f"#### Volume & Money Flow Indicators:\n"
            f"- **Latest Volume**: {vol['Latest_Volume']} shares\n"
            f"- **20-Day Volume SMA**: {vol['Volume_SMA_20']} shares\n"
            f"- **Volume Surge Ratio**: {vol['Volume_Surge_Ratio']} ({vol['Volume_Interpretation']})\n"
            f"- **On-Balance Volume (OBV)**: **{vol['OBV_Trend']}**"
        )

    # Category: Support / Resistance
    elif cat in ["support_resistance", "pivots", "levels"]:
        p = profile["pivot_points"]
        return header + (
            f"#### Support, Resistance & Key Price Levels:\n"
            f"- **Floor Pivot Point (P)**: ${p['Pivot_Point_P']}\n"
            f"- **Resistance Levels**: R1: ${p['Resistance_R1']} | R2: ${p['Resistance_R2']} | R3: ${p['Resistance_R3']}\n"
            f"- **Support Levels**: S1: ${p['Support_S1']} | S2: ${p['Support_S2']} | S3: ${p['Support_S3']}\n"
            f"- **Recent 20-Day Swing High**: ${p['Recent_20d_Swing_High']}\n"
            f"- **Recent 20-Day Swing Low**: ${p['Recent_20d_Swing_Low']}"
        )

    # Default / Comprehensive
    else:
        t = profile["trend_indicators"]
        m = profile["momentum_indicators"]
        v = profile["volatility_indicators"]
        vol = profile["volume_indicators"]
        p = profile["pivot_points"]
        c = profile["consensus"]

        bullish_bullets = "\n".join([f"  + {s}" for s in c["bullish_signals"]]) if c["bullish_signals"] else "  None"
        bearish_bullets = "\n".join([f"  - {s}" for s in c["bearish_signals"]]) if c["bearish_signals"] else "  None"

        return (
            f"{header}\n"
            f"### 1. Overall Technical Consensus: **{c['consensus_rating']}**\n"
            f"- **Score**: Net {c['net_score']:+d} (Bullish signals: {c['bullish_count']}, Bearish: {c['bearish_count']}, Neutral: {c['neutral_count']})\n"
            f"- **Bullish Signals**:\n{bullish_bullets}\n"
            f"- **Bearish Signals**:\n{bearish_bullets}\n\n"
            f"### 2. Trend & Moving Averages:\n"
            f"- **MAs**: 20-SMA: ${t['SMA_20']} | 50-SMA: ${t['SMA_50']} | 200-SMA: ${t['SMA_200']}\n"
            f"- **Trend & Crossover**: {t['short_term_trend']} | {t['crossover_status']}\n\n"
            f"### 3. Momentum & Oscillators:\n"
            f"- **RSI (14)**: {m['RSI_14']} ({m['RSI_status']})\n"
            f"- **MACD**: {m['MACD_interpretation']} (Hist: {m['MACD_histogram']})\n"
            f"- **Stochastic**: %K={m['Stochastic_K']}, %D={m['Stochastic_D']} ({m['Stochastic_status']})\n\n"
            f"### 4. Volatility & Bollinger Bands:\n"
            f"- **Bollinger Bands**: Upper: ${v['BB_Upper']} | Mid: ${v['BB_Middle']} | Lower: ${v['BB_Lower']} | %B: {v['BB_Percent_B']}\n"
            f"- **ATR (14)**: ${v['ATR_14']} ({v['ATR_pct_of_price']}) | Signal: {v['BB_interpretation']}\n\n"
            f"### 5. Volume & Money Flow:\n"
            f"- **Volume Surge**: {vol['Volume_Surge_Ratio']} ({vol['Volume_Interpretation']})\n"
            f"- **OBV Trend**: {vol['OBV_Trend']}\n\n"
            f"### 6. Actionable Key Levels:\n"
            f"- **Pivot (P)**: ${p['Pivot_Point_P']} | **Resistance (R1/R2)**: ${p['Resistance_R1']} / ${p['Resistance_R2']}\n"
            f"- **Support (S1/S2)**: ${p['Support_S1']} / ${p['Support_S2']}\n"
            f"- **20-Day Range**: Low ${p['Recent_20d_Swing_Low']} - High ${p['Recent_20d_Swing_High']}"
        )

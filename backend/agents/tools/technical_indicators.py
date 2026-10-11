import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

def fetch_ticker_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch cleaned historical OHLCV data using yfinance."""
    clean_sym = symbol.strip().upper()
    ticker = yf.Ticker(clean_sym)
    df = ticker.history(period=period, interval=interval)
    if df.empty or len(df) < 15:
        # Fallback to longer period if short period gave insufficient data
        if period in ["1mo", "3mo"]:
            df = ticker.history(period="1y", interval=interval)
    return df

def calculate_trend_indicators(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Calculate Moving Averages (SMA, EMA), crossovers, and trend alignment."""
    closes = df["Close"]
    results = {}
    
    # Simple Moving Averages
    sma20 = float(closes.rolling(window=20).mean().iloc[-1]) if len(closes) >= 20 else None
    sma50 = float(closes.rolling(window=50).mean().iloc[-1]) if len(closes) >= 50 else None
    sma200 = float(closes.rolling(window=200).mean().iloc[-1]) if len(closes) >= 200 else None
    
    # Exponential Moving Averages
    ema12 = float(closes.ewm(span=12, adjust=False).mean().iloc[-1]) if len(closes) >= 12 else None
    ema26 = float(closes.ewm(span=26, adjust=False).mean().iloc[-1]) if len(closes) >= 26 else None
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1]) if len(closes) >= 50 else None

    # Trend Direction relative to MAs
    above_sma20 = current_price > sma20 if sma20 else None
    above_sma50 = current_price > sma50 if sma50 else None
    above_sma200 = current_price > sma200 if sma200 else None

    # Golden Cross / Death Cross detection (SMA 50 vs SMA 200)
    crossover_signal = "N/A (Insufficient Data)"
    if len(closes) >= 200:
        sma50_series = closes.rolling(window=50).mean()
        sma200_series = closes.rolling(window=200).mean()
        curr_50 = sma50_series.iloc[-1]
        curr_200 = sma200_series.iloc[-1]
        prev_50 = sma50_series.iloc[-5] if len(sma50_series) >= 5 else curr_50
        prev_200 = sma200_series.iloc[-5] if len(sma200_series) >= 5 else curr_200
        
        if curr_50 > curr_200 and prev_50 <= prev_200:
            crossover_signal = "Recent Golden Cross (Strong Bullish)"
        elif curr_50 < curr_200 and prev_50 >= prev_200:
            crossover_signal = "Recent Death Cross (Strong Bearish)"
        elif curr_50 > curr_200:
            crossover_signal = "Bullish Alignment (SMA 50 > SMA 200)"
        else:
            crossover_signal = "Bearish Alignment (SMA 50 < SMA 200)"

    # Short term trend
    short_term_trend = "Neutral"
    if sma20 and current_price:
        dist_20 = ((current_price - sma20) / sma20) * 100
        if dist_20 > 2.0:
            short_term_trend = f"Bullish (+{dist_20:.2f}% above 20-SMA)"
        elif dist_20 < -2.0:
            short_term_trend = f"Bearish ({dist_20:.2f}% below 20-SMA)"
        else:
            short_term_trend = f"Neutral ({dist_20:+.2f}% near 20-SMA)"

    results = {
        "SMA_20": round(sma20, 2) if sma20 else "N/A",
        "SMA_50": round(sma50, 2) if sma50 else "N/A",
        "SMA_200": round(sma200, 2) if sma200 else "N/A",
        "EMA_12": round(ema12, 2) if ema12 else "N/A",
        "EMA_26": round(ema26, 2) if ema26 else "N/A",
        "EMA_50": round(ema50, 2) if ema50 else "N/A",
        "above_sma20": above_sma20,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        "crossover_status": crossover_signal,
        "short_term_trend": short_term_trend
    }
    return results

def calculate_momentum_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate RSI, MACD, Stochastic Oscillator, and Rate of Change."""
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    results = {}

    # 1. RSI (14-period Wilder's smoothed)
    if len(closes) >= 15:
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.iloc[-1])
        prev_rsi = float(rsi_series.iloc[-5]) if len(rsi_series) >= 5 else current_rsi
        
        rsi_status = "Neutral"
        if current_rsi >= 70:
            rsi_status = f"Overbought (RSI {current_rsi:.1f} >= 70 - Risk of pullback)"
        elif current_rsi <= 30:
            rsi_status = f"Oversold (RSI {current_rsi:.1f} <= 30 - Potential bounce)"
        elif current_rsi > 55:
            rsi_status = f"Mildly Bullish (RSI {current_rsi:.1f} > 50)"
        elif current_rsi < 45:
            rsi_status = f"Mildly Bearish (RSI {current_rsi:.1f} < 50)"
        else:
            rsi_status = f"Neutral / Indecisive ({current_rsi:.1f})"
            
        rsi_slope = "Rising" if current_rsi > prev_rsi else "Falling"
    else:
        current_rsi = None
        rsi_status = "Insufficient data"
        rsi_slope = "N/A"

    # 2. MACD (12-EMA, 26-EMA, 9-EMA Signal)
    if len(closes) >= 30:
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        curr_macd = float(macd_line.iloc[-1])
        curr_sig = float(signal_line.iloc[-1])
        curr_hist = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2]) if len(histogram) >= 2 else curr_hist

        if curr_macd > curr_sig and prev_hist <= 0 and curr_hist > 0:
            macd_signal = "Fresh Bullish Crossover (Strong Buy Momentum)"
        elif curr_macd < curr_sig and prev_hist >= 0 and curr_hist < 0:
            macd_signal = "Fresh Bearish Crossover (Strong Sell Momentum)"
        elif curr_macd > curr_sig:
            macd_signal = f"Bullish (MACD Line > Signal Line, Histogram {curr_hist:+.2f})"
        else:
            macd_signal = f"Bearish (MACD Line < Signal Line, Histogram {curr_hist:+.2f})"
    else:
        curr_macd, curr_sig, curr_hist = None, None, None
        macd_signal = "Insufficient data"

    # 3. Stochastic Oscillator (%K 14, %D 3)
    if len(df) >= 17:
        low14 = lows.rolling(window=14).min()
        high14 = highs.rolling(window=14).max()
        k_percent = 100 * ((closes - low14) / (high14 - low14).replace(0, np.nan))
        d_percent = k_percent.rolling(window=3).mean()
        
        curr_k = float(k_percent.iloc[-1])
        curr_d = float(d_percent.iloc[-1])
        
        if curr_k >= 80:
            stoch_status = f"Overbought (%K: {curr_k:.1f}, %D: {curr_d:.1f})"
        elif curr_k <= 20:
            stoch_status = f"Oversold (%K: {curr_k:.1f}, %D: {curr_d:.1f})"
        elif curr_k > curr_d:
            stoch_status = f"Bullish Momentum (%K: {curr_k:.1f} > %D: {curr_d:.1f})"
        else:
            stoch_status = f"Bearish Momentum (%K: {curr_k:.1f} < %D: {curr_d:.1f})"
    else:
        curr_k, curr_d = None, None
        stoch_status = "Insufficient data"

    # 4. Rate of Change (ROC 10 & 14)
    roc10 = float(((closes.iloc[-1] - closes.iloc[-11]) / closes.iloc[-11]) * 100) if len(closes) >= 11 else None
    roc14 = float(((closes.iloc[-1] - closes.iloc[-15]) / closes.iloc[-15]) * 100) if len(closes) >= 15 else None

    results = {
        "RSI_14": round(current_rsi, 2) if current_rsi is not None else "N/A",
        "RSI_status": rsi_status,
        "RSI_slope": rsi_slope,
        "MACD_line": round(curr_macd, 2) if curr_macd is not None else "N/A",
        "MACD_signal": round(curr_sig, 2) if curr_sig is not None else "N/A",
        "MACD_histogram": round(curr_hist, 2) if curr_hist is not None else "N/A",
        "MACD_interpretation": macd_signal,
        "Stochastic_K": round(curr_k, 2) if curr_k is not None else "N/A",
        "Stochastic_D": round(curr_d, 2) if curr_d is not None else "N/A",
        "Stochastic_status": stoch_status,
        "ROC_10d_pct": f"{roc10:+.2f}%" if roc10 is not None else "N/A",
        "ROC_14d_pct": f"{roc14:+.2f}%" if roc14 is not None else "N/A"
    }
    return results

def calculate_volatility_indicators(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Calculate Bollinger Bands, Average True Range (ATR), and 52-week position."""
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    results = {}

    # 1. Bollinger Bands (20-day SMA, 2 standard deviations)
    if len(closes) >= 20:
        sma20 = closes.rolling(window=20).mean()
        std20 = closes.rolling(window=20).std()
        upper_bb = float((sma20 + (2 * std20)).iloc[-1])
        lower_bb = float((sma20 - (2 * std20)).iloc[-1])
        middle_bb = float(sma20.iloc[-1])
        
        # %B (Position of price relative to bands)
        percent_b = ((current_price - lower_bb) / (upper_bb - lower_bb)) if (upper_bb != lower_bb) else 0.5
        # Bandwidth ((Upper - Lower) / Middle) * 100
        bandwidth = ((upper_bb - lower_bb) / middle_bb) * 100 if middle_bb else 0.0

        # Bollinger squeeze detection
        bandwidth_series = ((sma20 + (2 * std20)) - (sma20 - (2 * std20))) / sma20 * 100
        min_bw_60 = float(bandwidth_series.tail(60).min()) if len(bandwidth_series) >= 60 else bandwidth
        is_squeeze = bandwidth <= (min_bw_60 * 1.15)

        if current_price >= upper_bb:
            bb_signal = f"Price at/above Upper Band (${upper_bb:,.2f}) - Overextended / Strong Breakout"
        elif current_price <= lower_bb:
            bb_signal = f"Price at/below Lower Band (${lower_bb:,.2f}) - Oversold / Downside Breakdown"
        elif is_squeeze:
            bb_signal = f"Bollinger Squeeze Alert (Bandwidth {bandwidth:.2f}%) - Expect Imminent Volatility Expansion"
        else:
            bb_signal = f"Normal Band Range (%B: {percent_b:.2f}, Bandwidth: {bandwidth:.2f}%)"
    else:
        upper_bb, lower_bb, middle_bb, percent_b, bandwidth = None, None, None, None, None
        bb_signal = "Insufficient data"

    # 2. Average True Range (ATR 14)
    if len(df) >= 15:
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = float(true_range.rolling(window=14).mean().iloc[-1])
        atr_pct = (atr14 / current_price * 100) if current_price else 0.0
    else:
        atr14, atr_pct = None, None

    # 3. 52-week or Period Range Position
    period_high = float(highs.max())
    period_low = float(lows.min())
    range_pos = ((current_price - period_low) / (period_high - period_low) * 100) if period_high != period_low else 50.0

    results = {
        "BB_Upper": round(upper_bb, 2) if upper_bb else "N/A",
        "BB_Middle": round(middle_bb, 2) if middle_bb else "N/A",
        "BB_Lower": round(lower_bb, 2) if lower_bb else "N/A",
        "BB_Percent_B": round(percent_b, 2) if percent_b is not None else "N/A",
        "BB_Bandwidth_pct": f"{bandwidth:.2f}%" if bandwidth is not None else "N/A",
        "BB_interpretation": bb_signal,
        "ATR_14": round(atr14, 2) if atr14 else "N/A",
        "ATR_pct_of_price": f"{atr_pct:.2f}%" if atr_pct else "N/A",
        "Period_High": round(period_high, 2),
        "Period_Low": round(period_low, 2),
        "Range_Position_pct": f"{range_pos:.1f}% (0% = Period Low, 100% = Period High)"
    }
    return results

def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume SMA, Volume Surge ratio, and On-Balance Volume (OBV)."""
    volumes = df["Volume"]
    closes = df["Close"]
    results = {}

    if len(volumes) >= 20:
        vol_sma20 = float(volumes.rolling(window=20).mean().iloc[-1])
        latest_vol = float(volumes.iloc[-1])
        vol_surge_ratio = latest_vol / vol_sma20 if vol_sma20 else 1.0
        
        if vol_surge_ratio >= 2.0:
            surge_desc = f"Heavy Volume Surge ({vol_surge_ratio:.2f}x of 20-day avg)"
        elif vol_surge_ratio >= 1.2:
            surge_desc = f"Above Average Volume ({vol_surge_ratio:.2f}x of 20-day avg)"
        elif vol_surge_ratio <= 0.6:
            surge_desc = f"Low Volume Session ({vol_surge_ratio:.2f}x of 20-day avg)"
        else:
            surge_desc = f"Normal Volume ({vol_surge_ratio:.2f}x of 20-day avg)"

        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(closes)):
            if closes.iloc[i] > closes.iloc[i-1]:
                obv.append(obv[-1] + volumes.iloc[i])
            elif closes.iloc[i] < closes.iloc[i-1]:
                obv.append(obv[-1] - volumes.iloc[i])
            else:
                obv.append(obv[-1])
        obv_series = pd.Series(obv, index=closes.index)
        obv_ema = obv_series.ewm(span=20, adjust=False).mean()
        
        curr_obv = float(obv_series.iloc[-1])
        curr_obv_ema = float(obv_ema.iloc[-1])
        obv_trend = "Accumulation (OBV > 20-EMA)" if curr_obv > curr_obv_ema else "Distribution (OBV < 20-EMA)"
    else:
        vol_sma20, latest_vol, vol_surge_ratio = None, None, None
        surge_desc, obv_trend = "Insufficient data", "Insufficient data"

    results = {
        "Latest_Volume": f"{latest_vol:,.0f}" if latest_vol else "N/A",
        "Volume_SMA_20": f"{vol_sma20:,.0f}" if vol_sma20 else "N/A",
        "Volume_Surge_Ratio": f"{vol_surge_ratio:.2f}x" if vol_surge_ratio else "N/A",
        "Volume_Interpretation": surge_desc,
        "OBV_Trend": obv_trend
    }
    return results

def calculate_pivot_points(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Calculate Classical Floor Pivot Points and recent Swing High/Low levels."""
    # Use the prior day / prior session data
    if len(df) >= 2:
        prev_row = df.iloc[-2]
        high = float(prev_row["High"])
        low = float(prev_row["Low"])
        close = float(prev_row["Close"])
        
        pivot = (high + low + close) / 3.0
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
    else:
        pivot, r1, s1, r2, s2, r3, s3 = None, None, None, None, None, None, None

    # Recent 20-day Swing High / Low
    recent_20 = df.tail(20)
    swing_high = float(recent_20["High"].max()) if not recent_20.empty else current_price
    swing_low = float(recent_20["Low"].min()) if not recent_20.empty else current_price

    results = {
        "Pivot_Point_P": round(pivot, 2) if pivot else "N/A",
        "Resistance_R1": round(r1, 2) if r1 else "N/A",
        "Resistance_R2": round(r2, 2) if r2 else "N/A",
        "Resistance_R3": round(r3, 2) if r3 else "N/A",
        "Support_S1": round(s1, 2) if s1 else "N/A",
        "Support_S2": round(s2, 2) if s2 else "N/A",
        "Support_S3": round(s3, 2) if s3 else "N/A",
        "Recent_20d_Swing_High": round(swing_high, 2),
        "Recent_20d_Swing_Low": round(swing_low, 2)
    }
    return results

def compute_technical_consensus(trend: Dict, momentum: Dict, volatility: Dict, volume: Dict) -> Dict[str, Any]:
    """Aggregates signals from all pillars into a composite quantitative rating."""
    bullish_signals = []
    bearish_signals = []
    neutral_signals = []

    # Trend signals
    if trend.get("above_sma20") is True:
        bullish_signals.append("Price > 20-day SMA")
    elif trend.get("above_sma20") is False:
        bearish_signals.append("Price < 20-day SMA")

    if trend.get("above_sma50") is True:
        bullish_signals.append("Price > 50-day SMA")
    elif trend.get("above_sma50") is False:
        bearish_signals.append("Price < 50-day SMA")

    if trend.get("above_sma200") is True:
        bullish_signals.append("Price > 200-day SMA (Long-term Bull Market)")
    elif trend.get("above_sma200") is False:
        bearish_signals.append("Price < 200-day SMA (Long-term Bear Market)")

    # Momentum signals
    rsi = momentum.get("RSI_14")
    if isinstance(rsi, (int, float)):
        if rsi >= 70:
            bearish_signals.append(f"RSI Overbought ({rsi:.1f})")
        elif rsi <= 30:
            bullish_signals.append(f"RSI Oversold ({rsi:.1f} - Potential Rebound)")
        elif rsi > 52:
            bullish_signals.append(f"RSI Bullish Momentum ({rsi:.1f})")
        elif rsi < 48:
            bearish_signals.append(f"RSI Bearish Momentum ({rsi:.1f})")
        else:
            neutral_signals.append(f"RSI Neutral ({rsi:.1f})")

    macd_hist = momentum.get("MACD_histogram")
    if isinstance(macd_hist, (int, float)):
        if macd_hist > 0:
            bullish_signals.append("MACD Histogram Positive")
        else:
            bearish_signals.append("MACD Histogram Negative")

    # Volume signals
    obv_tr = volume.get("OBV_Trend", "")
    if "Accumulation" in obv_tr:
        bullish_signals.append("OBV Accumulation Pattern")
    elif "Distribution" in obv_tr:
        bearish_signals.append("OBV Distribution Pattern")

    # Volatility signals
    bb_interp = volatility.get("BB_interpretation", "")
    if "Squeeze" in bb_interp:
        neutral_signals.append("Bollinger Bands Volatility Squeeze (Breakout Setup)")

    total_bull = len(bullish_signals)
    total_bear = len(bearish_signals)
    net_score = total_bull - total_bear

    if net_score >= 3:
        overall_rating = "STRONGLY BULLISH"
    elif net_score in [1, 2]:
        overall_rating = "MODERATELY BULLISH"
    elif net_score == 0:
        overall_rating = "NEUTRAL / BALANCED"
    elif net_score in [-1, -2]:
        overall_rating = "MODERATELY BEARISH"
    else:
        overall_rating = "STRONGLY BEARISH"

    return {
        "consensus_rating": overall_rating,
        "net_score": net_score,
        "bullish_count": total_bull,
        "bearish_count": total_bear,
        "neutral_count": len(neutral_signals),
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "neutral_signals": neutral_signals
    }

def get_complete_technical_profile(symbol: str, period: str = "6mo") -> Dict[str, Any]:
    """Retrieves full technical analysis profile for a stock ticker symbol."""
    clean_sym = symbol.strip().upper()
    df = fetch_ticker_history(clean_sym, period=period)
    
    if df.empty:
        return {"error": f"Unable to fetch historical price data for ticker '{clean_sym}'."}
        
    current_price = float(df["Close"].iloc[-1])
    day_change = float(df["Close"].iloc[-1] - df["Close"].iloc[-2]) if len(df) >= 2 else 0.0
    day_pct = (day_change / df["Close"].iloc[-2] * 100) if len(df) >= 2 else 0.0

    trend = calculate_trend_indicators(df, current_price)
    momentum = calculate_momentum_indicators(df)
    volatility = calculate_volatility_indicators(df, current_price)
    volume = calculate_volume_indicators(df)
    pivots = calculate_pivot_points(df, current_price)
    consensus = compute_technical_consensus(trend, momentum, volatility, volume)

    return {
        "symbol": clean_sym,
        "current_price": round(current_price, 2),
        "day_change": round(day_change, 2),
        "day_pct_change": f"{day_pct:+.2f}%",
        "data_period": period,
        "candles_analyzed": len(df),
        "trend_indicators": trend,
        "momentum_indicators": momentum,
        "volatility_indicators": volatility,
        "volume_indicators": volume,
        "pivot_points": pivots,
        "consensus": consensus
    }

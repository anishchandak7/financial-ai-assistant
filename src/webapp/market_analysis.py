import streamlit as st
from service.market_data_service import MarketDataService

market_data_service = MarketDataService()
    
def render_market_analysis():

    markets = market_data_service.markets
    
    st.subheader("Market Details")
    
    if not markets:
        st.info("No market data available. Please check your configuration.")
        return

    for market in markets:
        result = market_data_service.get_market_summary(market_name=market)
        
        if not result:
            st.warning(f"Could not fetch data for market: {market}")
            continue
            
        status, summary = result
        
        # 1. Safely parse market status (Handles status=None for non-US markets)
        if isinstance(status, dict):
            market_name = status.get('name', market.upper())
            market_state = status.get('status', 'UNKNOWN').upper()
            status_message = status.get('message', 'No status message provided.')
        else:
            # Fallback if Yahoo doesn't support market status for this region
            market_name = f"{market.upper()} Market"
            market_state = "N/A"
            status_message = "Market status endpoint not supported by Yahoo for this region."
        
        st.markdown(f"### {market_name}")
        
        # Display color-coded status message
        if market_state == 'OPEN':
            st.success(f"**{market_state}** | {status_message}")
        elif market_state == 'N/A':
            st.info(f"**STATUS:** {status_message}")
        else:
            st.warning(f"**{market_state}** | {status_message}")
            
        # 2. Safely parse summary metrics
        if isinstance(summary, dict) and summary:
            for ticker, data in summary.items():
                if not isinstance(data, dict):
                    continue
                    
                name = data.get('shortName', ticker)
                price = data.get('regularMarketPrice', 0.0)
                change = data.get('regularMarketChange', 0.0)
                change_percent = data.get('regularMarketChangePercent', 0.0)
                
                delta_string = f"{change:.2f} ({change_percent:.2f}%)"
                
                st.metric(
                    label=name,
                    value=f"{price:,.2f}",
                    delta=delta_string
                )
        else:
            st.caption("No financial summary metrics available.")
            
        st.divider()

    # News section
    st.subheader("Latest News")
    st.info("**Market Update**\n\nEquities open higher today as tech sector rallies.")
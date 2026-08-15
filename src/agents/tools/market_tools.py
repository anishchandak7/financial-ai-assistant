import yfinance as yf
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.documents.base import Document
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage

from rag.vector_store import vector_store
from rag.embeddings import EmbeddingLayer
from core.config import settings
from core.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

# Global registry to capture citations across tool executions during an agent run
_CURRENT_AGENT_CITATIONS: List[Dict[str, Any]] = []

def reset_agent_citations():
    """Clears citation cache before each agent execution."""
    global _CURRENT_AGENT_CITATIONS
    _CURRENT_AGENT_CITATIONS = []

def get_agent_citations() -> List[Dict[str, Any]]:
    """Returns captured citations from document lookups."""
    global _CURRENT_AGENT_CITATIONS
    return list(_CURRENT_AGENT_CITATIONS)

@tool
def fetch_realtime_market_data(
    symbol_or_query: str = "US",
    data_type: str = "quote",
    period: str = "1mo"
) -> str:
    """Fetch live and historical stock market data, market indices summaries, financial metrics, or company news.
    
    Args:
        symbol_or_query: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA', 'MSFT') or 'US'/'market' for broader index overview.
        data_type: The type of data to fetch:
            - 'quote': Real-time stock quote, latest price, day change, % change, 52-week range, market cap, P/E ratio.
            - 'market_summary': Overview of major US market indices (S&P 500, Dow 30, Nasdaq, Russell 2000, VIX).
            - 'history': Historical price performance and trend analysis over a specified period (e.g. '5d', '1mo', '6mo', '1y').
            - 'news': Top recent market/company news headlines and summaries.
            - 'financials': Key financial performance indicators (Revenue, Gross Profit, Operating Income, Margins).
        period: Time period for historical data (default '1mo'). Options: '5d', '1mo', '3mo', '6mo', '1y', 'ytd'.
    
    Returns:
        A structured string containing the requested financial market data.
    """
    try:
        clean_input = symbol_or_query.strip().upper() if symbol_or_query else "US"
        logger.info(msg=f"Executing fetch_realtime_market_data for: {clean_input} | Type: {data_type} | Period: {period}")
        
        # 1. Market Summary (Broad Indices)
        if data_type == "market_summary" or clean_input in ["US", "MARKET", "INDICES", "OVERVIEW"]:
            index_tickers = {
                "^GSPC": "S&P 500",
                "^DJI": "Dow Jones Industrial Average",
                "^IXIC": "Nasdaq Composite",
                "^RUT": "Russell 2000",
                "^VIX": "CBOE Volatility Index"
            }
            results = ["### US Major Market Indices Overview:"]
            
            for ticker_sym, name in index_tickers.items():
                try:
                    t = yf.Ticker(ticker_sym)
                    fast = t.fast_info
                    price = fast.last_price or 0.0
                    prev = fast.previous_close or price
                    change = price - prev
                    pct_change = (change / prev * 100) if prev else 0.0
                    sign = "+" if change >= 0 else ""
                    results.append(
                        f"- **{name} ({ticker_sym})**: {price:,.2f} USD | "
                        f"Change: {sign}{change:,.2f} ({sign}{pct_change:.2f}%) | "
                        f"Day Range: {fast.day_low:,.2f} - {fast.day_high:,.2f}"
                    )
                except Exception as e:
                    results.append(f"- **{name} ({ticker_sym})**: Unavailable ({str(e)})")
            
            return "\n".join(results)

        ticker = yf.Ticker(clean_input)

        # 2. Company / Stock Quote
        if data_type == "quote":
            fast = ticker.fast_info
            info = ticker.info or {}
            
            price = fast.last_price or info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
            prev = fast.previous_close or info.get("previousClose") or price
            change = price - prev
            pct_change = (change / prev * 100) if prev else 0.0
            sign = "+" if change >= 0 else ""

            company_name = info.get("longName") or info.get("shortName") or clean_input
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            mkt_cap = fast.market_cap or info.get("marketCap") or 0
            pe_ratio = info.get("trailingPE") or info.get("forwardPE") or "N/A"
            fifty_two_high = fast.year_high or info.get("fiftyTwoWeekHigh") or 0.0
            fifty_two_low = fast.year_low or info.get("fiftyTwoWeekLow") or 0.0
            
            formatted_quote = (
                f"### Stock Snapshot: {company_name} ({clean_input})\n"
                f"- **Sector / Industry**: {sector} | {industry}\n"
                f"- **Current Price**: ${price:,.2f} {fast.currency or 'USD'}\n"
                f"- **Today's Change**: {sign}${change:,.2f} ({sign}{pct_change:.2f}%)\n"
                f"- **Day Range**: ${fast.day_low:,.2f} - ${fast.day_high:,.2f}\n"
                f"- **52-Week Range**: ${fifty_two_low:,.2f} - ${fifty_two_high:,.2f}\n"
                f"- **Market Cap**: ${mkt_cap:,.0f}\n"
                f"- **P/E Ratio**: {pe_ratio}\n"
                f"- **50-Day Moving Avg**: ${fast.fifty_day_average:,.2f}\n"
                f"- **200-Day Moving Avg**: ${fast.two_hundred_day_average:,.2f}"
            )
            return formatted_quote

        # 3. Historical Price Trends
        elif data_type == "history":
            hist = ticker.history(period=period, interval="1d")
            if hist.empty:
                return f"No historical data available for {clean_input} with period '{period}'."
            
            start_price = float(hist["Close"].iloc[0])
            end_price = float(hist["Close"].iloc[-1])
            total_change = end_price - start_price
            total_pct = (total_change / start_price * 100) if start_price else 0.0
            period_high = float(hist["High"].max())
            period_low = float(hist["Low"].min())
            avg_volume = float(hist["Volume"].mean())
            sign = "+" if total_change >= 0 else ""

            recent_closes = [f"{date.strftime('%Y-%m-%d')}: ${close:,.2f}" for date, close in hist["Close"].tail(5).items()]

            return (
                f"### Historical Performance for {clean_input} (Period: {period})\n"
                f"- **Start Price**: ${start_price:,.2f}\n"
                f"- **End / Latest Price**: ${end_price:,.2f}\n"
                f"- **Total Period Return**: {sign}${total_change:,.2f} ({sign}{total_pct:.2f}%)\n"
                f"- **Period High / Low**: ${period_high:,.2f} / ${period_low:,.2f}\n"
                f"- **Average Daily Volume**: {avg_volume:,.0f}\n"
                f"- **Recent 5-Day Closes**:\n  " + "\n  ".join(recent_closes)
            )

        # 4. Company News
        elif data_type == "news":
            news_items = ticker.news or []
            if not news_items:
                return f"No recent news articles found for {clean_input}."
            
            news_list = [f"### Latest News for {clean_input}:"]
            for idx, item in enumerate(news_items[:5], 1):
                title = item.get("title", "No Title")
                publisher = item.get("publisher", "Unknown Publisher")
                link = item.get("link", "#")
                news_list.append(f"{idx}. **{title}** (Source: {publisher})\n   Link: {link}")
            return "\n".join(news_list)

        # 5. Financial Indicators
        elif data_type == "financials":
            info = ticker.info or {}
            revenue = info.get("totalRevenue", 0)
            gross_profit = info.get("grossProfits", 0)
            op_margins = info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else "N/A"
            profit_margins = info.get("profitMargins", 0) * 100 if info.get("profitMargins") else "N/A"
            free_cashflow = info.get("freeCashflow", 0)
            rev_growth = info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else "N/A"

            return (
                f"### Financial Fundamentals for {clean_input}\n"
                f"- **Total Revenue (TTM)**: ${revenue:,.0f}\n"
                f"- **Revenue Growth (YoY)**: {rev_growth}%\n"
                f"- **Gross Profit**: ${gross_profit:,.0f}\n"
                f"- **Operating Margin**: {op_margins}%\n"
                f"- **Net Profit Margin**: {profit_margins}%\n"
                f"- **Free Cash Flow**: ${free_cashflow:,.0f}"
            )

        else:
            return f"Unknown data_type '{data_type}'. Use 'quote', 'market_summary', 'history', 'news', or 'financials'."

    except Exception as e:
        logger.error(msg=f"Error in fetch_realtime_market_data for '{symbol_or_query}': {e}", exc_info=True)
        return f"Error fetching market data for '{symbol_or_query}': {str(e)}"

@tool
def document_lookup(query: str) -> str:
    """Search and retrieve information strictly from the MarketAnalysis vector collection using Multi-Query transformation.
    
    Args:
        query: The market analysis question, topic, company name, or industry report to search for.
        
    Returns:
        Relevant document excerpts, research notes, and citations extracted from the MarketAnalysis collection.
    """
    global _CURRENT_AGENT_CITATIONS
    try:
        logger.info(msg=f"Executing document_lookup on MarketAnalysis collection for query: {query}")
        
        # 1. Multi-Query Transformation: Generate multiple query perspectives via LLM
        multi_query_prompt = f"""You are an AI assistant helping search a financial research database.
                                The user is asking a question related to market analysis, equities, economic reports, or industry trends.
                                Generate 3 distinct search query variations (different perspectives, synonyms, or related technical queries) to maximize vector retrieval recall.
                                Output exactly 3 queries, one per line. Do NOT include numbers, bullets, quotation marks, or preamble.

                                User Query: {query}"""

        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_CONFIGS[0].get("model_name", "gemini-3.6-flash"),
            api_key=settings.GEMINI_KEY
        )
        
        mq_response = llm.invoke(input=[HumanMessage(content=multi_query_prompt)])
        raw_content = mq_response.content
        if isinstance(raw_content, str):
            raw_text = raw_content
        elif isinstance(raw_content, list):
            text_blocks = []
            for block in raw_content:
                if isinstance(block, dict) and "text" in block:
                    text_blocks.append(block["text"])
                elif isinstance(block, str):
                    text_blocks.append(block)
            raw_text = "\n".join(text_blocks)
        else:
            raw_text = str(raw_content)

        generated_queries = [q.strip() for q in raw_text.strip().split("\n") if q.strip() and not q.startswith("[{")]
        
        # Combine original query + generated variations
        all_queries = [query] + generated_queries[:3]
        logger.info(msg=f"Multi-Query variations generated ({len(all_queries)}): {all_queries}")

        # 2. Query the MarketAnalysis Chroma collection
        collection_name = settings.COLLECTION_NAMES[2]  # MarketAnalysis
        embedding_layer = EmbeddingLayer()
        retriever = Chroma(
            collection_name=collection_name,
            client=vector_store.client,
            embedding_function=embedding_layer.get_embedding_function()
        ).as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # 3. Retrieve and Deduplicate Documents across all queries
        unique_docs: Dict[str, Document] = {}
        for q in all_queries:
            try:
                retrieved = retriever.invoke(input=q)
                for doc in retrieved:
                    doc_id = doc.metadata.get("chunk_id") or hash(doc.page_content)
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
            except Exception as search_err:
                logger.warning(msg=f"Sub-query search failed for '{q}': {search_err}")

        docs_list = list(unique_docs.values())
        logger.info(msg=f"Total unique documents retrieved from {collection_name}: {len(docs_list)}")

        if not docs_list:
            return f"No documents found in the {collection_name} collection for query: '{query}'."

        # 4. Format Document Excerpts & Record Citations
        formatted_results = [f"### {collection_name} Documents Retrieved ({len(docs_list)} matches):"]
        
        for idx, doc in enumerate(docs_list):
            meta = doc.metadata or {}
            file_name = meta.get("file_name", f"{collection_name}_Doc_{idx}")
            page = meta.get("page", "N/A")
            excerpt = doc.page_content.strip()
            
            # Record citation for UI rendering
            citation_item = {
                "source_index": idx,
                "quote": excerpt[:250] + "..." if len(excerpt) > 250 else excerpt,
                "file_name": file_name,
                "metadata": meta
            }
            _CURRENT_AGENT_CITATIONS.append(citation_item)
            
            formatted_results.append(
                f"\n**[Source {idx}] File: {file_name} (Page: {page})**\n"
                f"Content:\n{excerpt}\n---"
            )

        return "\n".join(formatted_results)


    except Exception as e:
        logger.error(msg=f"Error in document_lookup: {e}", exc_info=True)
        return f"Error searching MarketAnalysis collection: {str(e)}"

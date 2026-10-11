import React, { useRef, useState, useEffect } from 'react';

// Helper to clean up ticker names for normal users
function getCleanTickerName(rawName) {
  if (!rawName) return 'Asset';
  let name = rawName;
  if (name.includes('Crude Oil')) return 'Crude Oil';
  if (name.includes('Copper')) return 'Copper';
  if (name.includes('Gold')) return 'Gold';
  if (name.includes('Silver')) return 'Silver';
  if (name.includes('S&P 500') || name.includes('SNP') || name.includes('GSPC')) return 'S&P 500';
  if (name.includes('Nasdaq') || name.includes('IXIC')) return 'Nasdaq';
  if (name.includes('Dow') || name.includes('DJI')) return 'Dow Jones';
  if (name.includes('Russell')) return 'Russell 2000';
  if (name.includes('Bitcoin')) return 'Bitcoin';
  if (name.includes('Ethereum')) return 'Ethereum';
  if (name.includes('Solana')) return 'Solana';
  return name.length > 18 ? name.substring(0, 16) + '...' : name;
}

export default function MarketTicker({ markets, onRefresh, isRefreshing }) {
  if (!markets || markets.length === 0) return null;

  const scrollContainerRef = useRef(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const allTickers = [];
  markets.forEach(m => {
    (m.tickers || []).forEach(t => {
      allTickers.push({
        ...t,
        cleanName: getCleanTickerName(t.name)
      });
    });
  });

  const usMarket = markets.find(m => m.market?.toLowerCase() === 'us') || markets[0];
  const isOpen = usMarket?.state === 'OPEN';

  const checkScroll = () => {
    const el = scrollContainerRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 10);
    setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 10);
  };

  useEffect(() => {
    checkScroll();
    window.addEventListener('resize', checkScroll);
    return () => window.removeEventListener('resize', checkScroll);
  }, [allTickers.length]);

  const scroll = (direction) => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const scrollAmount = direction === 'left' ? -320 : 320;
    el.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    setTimeout(checkScroll, 300);
  };

  const handleWheel = (e) => {
    const el = scrollContainerRef.current;
    if (!el) return;
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      el.scrollLeft += e.deltaY;
      checkScroll();
    }
  };

  return (
    <section className="bg-[#1e1f20] border-b border-[#3c4043]/40 px-3 sm:px-8 py-2 flex items-center justify-between text-xs gap-2 select-none relative">
      {/* Market Status Pill */}
      <div className="flex items-center space-x-2 shrink-0">
        <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-medium flex items-center gap-1.5 ${
          isOpen 
            ? 'bg-[#0d652d]/30 text-[#81c995] border border-[#81c995]/30' 
            : 'bg-[#3c4043]/30 text-[#c4c7c5] border border-[#3c4043]'
        }`}>
          <span className={`h-1.5 w-1.5 rounded-full ${isOpen ? 'bg-[#81c995]' : 'bg-[#9aa0a6]'}`}></span>
          {isOpen ? 'Markets Open' : 'Markets Closed'}
        </span>
      </div>

      {/* Left Chevron */}
      <button
        onClick={() => scroll('left')}
        disabled={!canScrollLeft}
        title="Scroll left"
        className={`h-7 w-7 rounded-full bg-[#282a2c] hover:bg-[#333538] text-[#c4c7c5] hover:text-white flex items-center justify-center border border-[#3c4043] transition shrink-0 z-10 ${
          !canScrollLeft ? 'opacity-20 cursor-not-allowed' : 'opacity-100'
        }`}
      >
        <span className="material-symbols-outlined text-base">chevron_left</span>
      </button>

      {/* Scrollable Clean Tickers */}
      <div
        ref={scrollContainerRef}
        onScroll={checkScroll}
        onWheel={handleWheel}
        className="flex-1 flex items-center space-x-2 overflow-x-auto py-0.5 scroll-smooth no-scrollbar"
      >
        {allTickers.map((t, idx) => {
          const isPositive = t.change >= 0;
          return (
            <div 
              key={`${t.symbol}_${idx}`}
              className="bg-[#282a2c] border border-[#3c4043]/80 rounded-full px-3.5 py-1 flex items-center space-x-2 shrink-0 hover:bg-[#333538] transition"
            >
              <span className="font-medium text-[#e3e3e3] text-xs whitespace-nowrap">{t.cleanName}</span>
              <span className="font-mono text-white font-medium text-xs whitespace-nowrap">
                ${typeof t.price === 'number' ? t.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : t.price}
              </span>
              <span className={`font-mono text-[11px] flex items-center font-medium whitespace-nowrap ${
                isPositive ? 'text-[#81c995]' : 'text-[#f28b82]'
              }`}>
                <span className="material-symbols-outlined text-[13px]">
                  {isPositive ? 'arrow_drop_up' : 'arrow_drop_down'}
                </span>
                {isPositive ? '+' : ''}{typeof t.change === 'number' ? t.change.toFixed(2) : t.change} ({isPositive ? '+' : ''}{typeof t.change_percent === 'number' ? t.change_percent.toFixed(2) : t.change_percent}%)
              </span>
            </div>
          );
        })}
      </div>

      {/* Right Chevron */}
      <button
        onClick={() => scroll('right')}
        disabled={!canScrollRight}
        title="Scroll right"
        className={`h-7 w-7 rounded-full bg-[#282a2c] hover:bg-[#333538] text-[#c4c7c5] hover:text-white flex items-center justify-center border border-[#3c4043] transition shrink-0 z-10 ${
          !canScrollRight ? 'opacity-20 cursor-not-allowed' : 'opacity-100'
        }`}
      >
        <span className="material-symbols-outlined text-base">chevron_right</span>
      </button>

      {/* Refresh Button */}
      <button
        onClick={onRefresh}
        title="Refresh market data"
        disabled={isRefreshing}
        className="h-7 w-7 rounded-full bg-[#282a2c] hover:bg-[#333538] text-[#c4c7c5] hover:text-white flex items-center justify-center border border-[#3c4043] transition shrink-0"
      >
        <span className={`material-symbols-outlined text-sm ${isRefreshing ? 'animate-spin text-[#8ab4f8]' : ''}`}>
          refresh
        </span>
      </button>
    </section>
  );
}

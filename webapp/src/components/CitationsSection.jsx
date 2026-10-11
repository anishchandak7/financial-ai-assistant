import React, { useState } from 'react';

export default function CitationsSection({ citations }) {
  const [isOpen, setIsOpen] = useState(false);
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-3.5 border border-[#3c4043] rounded-2xl overflow-hidden bg-[#1e1f20]/90">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2.5 text-left bg-[#282a2c] hover:bg-[#333538] transition flex items-center justify-between text-xs font-medium text-[#8ab4f8]"
      >
        <span className="flex items-center gap-2">
          <span className="material-symbols-outlined text-base">library_books</span>
          <span>Verified Sources & Citations ({citations.length})</span>
        </span>
        <span className="material-symbols-outlined text-sm transition-transform duration-200" style={{ transform: isOpen ? 'rotate(180deg)' : 'none' }}>
          expand_more
        </span>
      </button>

      {isOpen && (
        <div className="p-3.5 space-y-3 divide-y divide-[#3c4043]/60 text-xs">
          {citations.map((cit, idx) => (
            <div key={idx} className={idx > 0 ? "pt-3" : ""}>
              <div className="flex items-center justify-between font-medium text-[#e3e3e3] mb-1.5">
                <span className="flex items-center gap-1.5 text-[#8ab4f8]">
                  <span className="material-symbols-outlined text-sm">description</span>
                  <span className="font-semibold">[{cit.source_index !== undefined ? cit.source_index : idx}]</span>
                  <span className="truncate max-w-[280px] sm:max-w-md">{cit.file_name || 'MarketAnalysis Document'}</span>
                </span>
                {cit.metadata?.page && (
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#1e1f20] text-[#c4c7c5] font-mono border border-[#3c4043]">
                    Page {cit.metadata.page}
                  </span>
                )}
              </div>
              <blockquote className="border-l-2 border-[#8ab4f8] pl-3 py-1 text-[#c4c7c5] italic bg-[#8ab4f8]/5 rounded-r-lg text-[12px] leading-relaxed">
                "{cit.quote}"
              </blockquote>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

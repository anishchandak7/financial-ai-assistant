import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import { sendChatMessage } from '../services/api';

export default function ChatInterface({ activeModel, onShowToast, onOpenUpload }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your **Financial AI Assistant**.\n\nAsk me anything about stock analysis, market trends, company earnings, or financial concepts.'
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (e) => {
    e?.preventDefault();
    const query = inputQuery.trim();
    if (!query || isLoading) return;

    const updatedMessages = [...messages, { role: 'user', content: query }];
    setMessages(updatedMessages);
    setInputQuery('');
    setIsLoading(true);

    try {
      const data = await sendChatMessage(query, activeModel);
      if (data.success) {
        setMessages([...updatedMessages, { role: 'assistant', content: data.result }]);
      } else {
        setMessages([...updatedMessages, {
          role: 'assistant',
          content: `⚠️ **Unable to generate response**: ${data.detail || 'An unexpected error occurred. Please try again.'}`
        }]);
      }
    } catch (err) {
      setMessages([...updatedMessages, {
        role: 'assistant',
        content: `⚠️ **Connection Error**: Unable to reach the server (${err.message}). Please ensure the application backend is running.`
      }]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const promptChips = [
    {
      title: "Analyze NVIDIA (NVDA)",
      icon: "trending_up",
      prompt: "Provide a comprehensive stock analysis for NVIDIA (NVDA) including valuation, outlook, and recent performance."
    },
    {
      title: "Today's Market Overview",
      icon: "show_chart",
      prompt: "How are major US stock indices performing today, and what are the main market drivers?"
    },
    {
      title: "Roth IRA vs 401(k)",
      icon: "savings",
      prompt: "What are the key differences and tax advantages between a Roth IRA and a Traditional 401(k)?"
    },
    {
      title: "Company Research",
      icon: "find_in_page",
      prompt: "What are the latest semiconductor industry trends and future growth drivers?"
    }
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-[#131314]">
      {/* Chat Messages Canvas */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 max-w-4xl w-full mx-auto">
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} onCopyToast={onShowToast} />
        ))}

        {/* Clean Loading Indicator */}
        {isLoading && (
          <div className="flex gap-3.5 my-6 justify-start animate-m3-fade-in">
            <div className="h-8 w-8 rounded-full bg-[#1b3d6e] flex items-center justify-center text-[#8ab4f8] shrink-0 mt-0.5 shadow-sm">
              <span className="material-symbols-outlined text-lg animate-spin">sync</span>
            </div>
            <div className="bg-[#1e1f20] border border-[#3c4043]/50 rounded-3xl rounded-tl-none px-5 py-4 shadow-md flex items-center space-x-3 text-xs text-[#c4c7c5]">
              <span className="h-2 w-2 rounded-full bg-[#8ab4f8] animate-ping"></span>
              <span>Thinking and gathering market information...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Floating Google Gemini Prompt Bar */}
      <footer className="border-t border-[#3c4043]/30 bg-[#1e1f20]/95 backdrop-blur-xl p-4 sticky bottom-0 z-20">
        <div className="max-w-4xl mx-auto space-y-3">
          {/* Suggestion Chips (Shown on fresh conversation) */}
          {messages.length <= 1 && (
            <div className="flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar">
              {promptChips.map((chip, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInputQuery(chip.prompt);
                    inputRef.current?.focus();
                  }}
                  className="px-4 py-1.5 rounded-full bg-[#282a2c] hover:bg-[#333538] border border-[#3c4043] transition flex items-center space-x-2 shrink-0 text-xs font-medium text-[#e3e3e3] hover:text-[#8ab4f8]"
                >
                  <span className="material-symbols-outlined text-sm text-[#8ab4f8]">{chip.icon}</span>
                  <span>{chip.title}</span>
                </button>
              ))}
            </div>
          )}

          {/* Clean Input Form */}
          <form onSubmit={handleSendMessage} className="relative flex items-center">
            <button
              type="button"
              onClick={onOpenUpload}
              title="Upload document"
              className="absolute left-3 p-1.5 rounded-full hover:bg-[#333538] text-[#c4c7c5] hover:text-white transition flex items-center justify-center z-10"
            >
              <span className="material-symbols-outlined text-xl">attach_file</span>
            </button>

            <input
              ref={inputRef}
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask anything about stocks, markets, or financial topics..."
              disabled={isLoading}
              className="w-full bg-[#282a2c] border border-[#3c4043] rounded-full py-3.5 pl-12 pr-14 text-sm text-[#e3e3e3] placeholder-[#8e918f] focus:outline-none focus:border-[#8ab4f8] transition disabled:opacity-50 font-normal"
            />

            <button
              type="submit"
              disabled={!inputQuery.trim() || isLoading}
              className="absolute right-2.5 h-9 w-9 rounded-full bg-[#8ab4f8] hover:bg-[#a8c7fa] text-[#002d6c] disabled:opacity-30 disabled:cursor-not-allowed transition flex items-center justify-center shadow-md"
            >
              <span className="material-symbols-outlined text-lg">arrow_upward</span>
            </button>
          </form>

          {/* Simple Clean Subtext */}
          <div className="text-center text-[11px] text-[#8e918f]">
            AI can make mistakes. Verify important financial data.
          </div>
        </div>
      </footer>
    </div>
  );
}

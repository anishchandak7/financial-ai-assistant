import React, { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';
import CitationsSection from './CitationsSection';

export default function MessageBubble({ message, onCopyToast }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const content = message.content;

  let agentOutputs = {};
  let citations = [];
  let fullTextToCopy = '';

  if (!isUser && typeof content === 'object' && content !== null) {
    agentOutputs = content.agent_output || {};
    citations = content.citations || [];

    // Collect citations from agent_output if top level is empty
    if (citations.length === 0) {
      Object.values(agentOutputs).forEach((out) => {
        if (out && typeof out === 'object' && out.citations) {
          citations.push(...out.citations);
        }
      });
    }

    const parts = [];
    Object.entries(agentOutputs).forEach(([_, data]) => {
      const text = typeof data === 'object' && data !== null ? data.response : String(data);
      parts.push(text);
    });
    fullTextToCopy = parts.join('\n\n') || content?.response || '';
  } else if (typeof content === 'string') {
    fullTextToCopy = content;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(fullTextToCopy);
    setCopied(true);
    if (onCopyToast) onCopyToast("Copied to clipboard", 'success');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex gap-3.5 my-6 group ${isUser ? 'justify-end' : 'justify-start'} animate-m3-fade-in`}>
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="h-8 w-8 rounded-full bg-[#1b3d6e] flex items-center justify-center text-[#8ab4f8] shrink-0 mt-0.5 shadow-sm">
          <span className="material-symbols-outlined text-lg">auto_awesome</span>
        </div>
      )}

      {/* Message Bubble Card */}
      <div
        className={`max-w-3xl rounded-3xl p-5 relative transition-all duration-200 ${
          isUser
            ? 'bg-[#282a2c] border border-[#3c4043] text-[#e3e3e3] rounded-tr-none shadow-sm'
            : 'bg-[#1e1f20] border border-[#3c4043]/50 text-[#e3e3e3] rounded-tl-none shadow-md'
        }`}
      >
        {isUser ? (
          <p className="text-sm font-normal whitespace-pre-wrap leading-relaxed">{content}</p>
        ) : (
          <div>
            {/* Content Body */}
            {typeof content === 'string' ? (
              <MarkdownRenderer content={content} />
            ) : Object.keys(agentOutputs).length > 0 ? (
              <div className="space-y-4">
                {Object.entries(agentOutputs).map(([agentName, data], i) => {
                  const text = typeof data === 'object' && data !== null ? data.response : String(data);
                  return (
                    <div key={agentName} className={i > 0 ? 'pt-3 border-t border-[#3c4043]/40' : ''}>
                      <MarkdownRenderer content={text} />
                    </div>
                  );
                })}
              </div>
            ) : (
              <MarkdownRenderer content={content?.response || ''} />
            )}

            {/* Citations Section (Only when real citations exist) */}
            <CitationsSection citations={citations} />

            {/* Clean Floating Copy Action Button at bottom right */}
            <div className="mt-2.5 pt-1.5 flex items-center justify-end">
              <button
                onClick={handleCopy}
                title="Copy response"
                className="opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity h-7 px-2.5 rounded-full bg-[#282a2c] hover:bg-[#333538] text-[#c4c7c5] hover:text-white text-xs flex items-center gap-1 border border-[#3c4043]/70"
              >
                <span className="material-symbols-outlined text-xs">{copied ? 'check' : 'content_copy'}</span>
                <span className="text-[11px]">{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="h-8 w-8 rounded-full bg-[#3c4043] flex items-center justify-center text-[#e3e3e3] shrink-0 mt-0.5 shadow-sm">
          <span className="material-symbols-outlined text-lg">person</span>
        </div>
      )}
    </div>
  );
}

import React from 'react';

export default function Header({
  models,
  activeModel,
  onSelectModel,
  onOpenUpload,
  onClearChat
}) {
  return (
    <header className="border-b border-[#3c4043]/40 bg-[#1e1f20]/95 backdrop-blur-md sticky top-0 z-30 px-4 sm:px-8 py-3 flex items-center justify-between">
      {/* Clean Brand Logo & Title */}
      <div className="flex items-center space-x-3">
        <div className="h-9 w-9 rounded-full bg-[#1b3d6e] flex items-center justify-center text-[#8ab4f8] shadow-sm">
          <span className="material-symbols-outlined text-xl">auto_awesome</span>
        </div>
        <span className="text-base font-medium text-white tracking-tight">Financial AI</span>
      </div>

      {/* User-focused Actions */}
      <div className="flex items-center space-x-2.5">
        {/* Upload Document Button */}
        <button
          onClick={onOpenUpload}
          className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full bg-[#282a2c] hover:bg-[#333538] text-[#e3e3e3] border border-[#3c4043] transition text-xs font-medium"
        >
          <span className="material-symbols-outlined text-base text-[#8ab4f8]">upload_file</span>
          <span>Upload Document</span>
        </button>

        {/* New Chat Button */}
        <button
          onClick={onClearChat}
          title="New conversation"
          className="flex items-center space-x-1 px-3 py-1.5 rounded-full bg-[#282a2c] hover:bg-[#333538] text-[#c4c7c5] hover:text-white transition text-xs font-medium border border-[#3c4043]"
        >
          <span className="material-symbols-outlined text-base">add</span>
          <span className="hidden sm:inline">New Chat</span>
        </button>

        {/* Simplified Model Selector */}
        <div className="relative">
          <select
            value={activeModel || ''}
            onChange={(e) => onSelectModel(e.target.value)}
            className="bg-[#282a2c] text-[#e3e3e3] border border-[#3c4043] text-xs rounded-full px-3.5 py-1.5 pr-7 appearance-none focus:outline-none focus:border-[#8ab4f8] cursor-pointer font-medium hover:bg-[#333538] transition"
          >
            {models.map((m) => {
              const displayName = m.display_name || m;
              const cleanName = displayName.replace(/Gemini3\.6Flash/i, 'Gemini Flash').replace(/Gemini1\.5Pro/i, 'Gemini Pro');
              return (
                <option key={displayName} value={displayName}>
                  {cleanName}
                </option>
              );
            })}
          </select>
          <span className="material-symbols-outlined absolute right-2 top-2 text-sm text-[#c4c7c5] pointer-events-none">
            expand_more
          </span>
        </div>
      </div>
    </header>
  );
}

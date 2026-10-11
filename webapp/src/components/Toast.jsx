import React from 'react';

export default function Toast({ toast, onClose }) {
  if (!toast) return null;

  const isSuccess = toast.type === 'success';
  const isError = toast.type === 'error';

  return (
    <div className="fixed bottom-24 right-6 z-50 animate-m3-fade-in">
      <div className={`px-4 py-2.5 rounded-full shadow-2xl border flex items-center space-x-2.5 text-xs font-medium backdrop-blur-md ${
        isSuccess
          ? 'bg-[#0d652d] text-white border-[#81c995]/40 shadow-lg'
          : isError
          ? 'bg-[#d93025] text-white border-[#f28b82]/40 shadow-lg'
          : 'bg-[#282a2c] text-[#e3e3e3] border-[#3c4043] shadow-lg'
      }`}>
        <span className="material-symbols-outlined text-base">
          {isSuccess ? 'check_circle' : isError ? 'error' : 'info'}
        </span>
        <span>{toast.message}</span>
        {onClose && (
          <button onClick={onClose} className="p-0.5 hover:opacity-75 text-white/80">
            <span className="material-symbols-outlined text-sm">close</span>
          </button>
        )}
      </div>
    </div>
  );
}

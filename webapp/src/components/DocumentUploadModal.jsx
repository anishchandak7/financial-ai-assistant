import React, { useState } from 'react';
import { uploadDocument } from '../services/api';

export default function DocumentUploadModal({ isOpen, onClose, collections, onUploadSuccess }) {
  const [selectedCollection, setSelectedCollection] = useState(collections[0] || 'Finance');
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  if (!isOpen) return null;

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const data = await uploadDocument(file, selectedCollection);
      if (data.success) {
        setUploadStatus({ type: 'success', message: data.message });
        setFile(null);
        if (onUploadSuccess) onUploadSuccess(data);
      } else {
        setUploadStatus({ type: 'error', message: data.detail || 'Upload failed' });
      }
    } catch (err) {
      setUploadStatus({ type: 'error', message: err.message });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-m3-fade-in">
      <div className="bg-[#282a2c] border border-[#3c4043] rounded-3xl max-w-lg w-full p-6 shadow-2xl relative">
        {/* M3 Dialog Header */}
        <div className="flex items-center justify-between pb-3 border-b border-[#3c4043]/60">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-full bg-[#1b3d6e] flex items-center justify-center text-[#8ab4f8]">
              <span className="material-symbols-outlined text-lg">drive_folder_upload</span>
            </div>
            <div>
              <h3 className="text-base font-medium text-white">Knowledge Vault Ingestion</h3>
              <p className="text-[11px] text-[#c4c7c5]">Upload documents to ChromaDB collections</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 rounded-full hover:bg-[#333538] text-[#c4c7c5] hover:text-white flex items-center justify-center transition"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleUpload} className="mt-5 space-y-4">
          <div>
            <label className="block text-xs font-medium text-[#e3e3e3] mb-1.5">Target Vector Collection</label>
            <select
              value={selectedCollection}
              onChange={(e) => setSelectedCollection(e.target.value)}
              className="w-full bg-[#1e1f20] border border-[#3c4043] text-[#e3e3e3] text-xs rounded-2xl p-3 focus:outline-none focus:border-[#8ab4f8] transition"
            >
              {collections.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            <p className="text-[11px] text-[#8e918f] mt-1">
              Select target collection (e.g. <b>MarketAnalysis</b> for equity research, <b>Finance</b> for general reports).
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-[#e3e3e3] mb-1.5">Select Financial Report (.pdf, .txt, .csv)</label>
            <input
              type="file"
              accept=".pdf,.txt,.csv"
              onChange={(e) => setFile(e.target.files[0])}
              className="w-full text-xs text-[#c4c7c5] file:mr-3 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-medium file:bg-[#1b3d6e] file:text-[#8ab4f8] hover:file:bg-[#284a7e] cursor-pointer border border-[#3c4043] rounded-2xl p-3 bg-[#1e1f20]"
            />
          </div>

          {uploadStatus && (
            <div className={`p-3 rounded-2xl text-xs font-mono flex items-start gap-2 ${
              uploadStatus.type === 'success'
                ? 'bg-[#0d652d]/30 text-[#81c995] border border-[#81c995]/40'
                : 'bg-[#d93025]/20 text-[#f28b82] border border-[#f28b82]/40'
            }`}>
              <span className="material-symbols-outlined text-sm mt-0.5">
                {uploadStatus.type === 'success' ? 'check_circle' : 'error'}
              </span>
              <span>{uploadStatus.message}</span>
            </div>
          )}

          {/* Dialog Action Buttons */}
          <div className="flex items-center justify-end gap-2 pt-3 border-t border-[#3c4043]/60">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-medium text-[#c4c7c5] hover:text-white rounded-full hover:bg-[#333538] transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!file || isUploading}
              className="px-5 py-2 text-xs font-medium rounded-full bg-[#8ab4f8] hover:bg-[#a8c7fa] text-[#002d6c] disabled:opacity-30 disabled:cursor-not-allowed transition flex items-center gap-1.5 shadow-md"
            >
              {isUploading ? (
                <>
                  <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
                  Ingesting Chunks...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-sm">upload</span>
                  Upload & Ingest
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

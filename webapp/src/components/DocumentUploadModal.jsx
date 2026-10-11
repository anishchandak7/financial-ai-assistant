import React, { useState, useEffect } from 'react';
import { uploadDocument, fetchDocuments, deleteDocument, fetchCollections } from '../services/api';

export default function DocumentUploadModal({ isOpen, onClose, collections, onUploadSuccess }) {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'manage'
  const [selectedCollection, setSelectedCollection] = useState(collections[0] || 'Finance');
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Manage tab states
  const [documents, setDocuments] = useState([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [filterCollection, setFilterCollection] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const [manageStatus, setManageStatus] = useState(null);

  useEffect(() => {
    if (isOpen && activeTab === 'manage') {
      loadDocuments();
    }
  }, [isOpen, activeTab, filterCollection]);

  const loadDocuments = async () => {
    setIsLoadingDocs(true);
    setManageStatus(null);
    try {
      const data = await fetchDocuments(filterCollection || null);
      setDocuments(data.documents || []);
    } catch (err) {
      setManageStatus({ type: 'error', message: err.message });
    } finally {
      setIsLoadingDocs(false);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const data = await uploadDocument(file, selectedCollection);
      if (data.success) {
        setUploadStatus({
          type: 'success',
          message: `${data.message} (${data.chunk_count} chunks indexed)`
        });
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

  const handleDelete = async (fileId, fileName) => {
    if (!window.confirm(`Delete "${fileName}" and all associated vector chunks from ChromaDB?`)) {
      return;
    }

    setDeletingId(fileId);
    try {
      const res = await deleteDocument(fileId);
      setManageStatus({
        type: 'success',
        message: `Deleted "${fileName}" (${res.deleted_chunks} chunks removed).`
      });
      setDocuments((prev) => prev.filter((doc) => doc.file_id !== fileId));
    } catch (err) {
      setManageStatus({ type: 'error', message: err.message });
    } finally {
      setDeletingId(null);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 animate-m3-fade-in">
      <div className="bg-[#282a2c] border border-[#3c4043] rounded-3xl max-w-2xl w-full p-6 shadow-2xl relative flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-[#3c4043]/60">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-full bg-[#1b3d6e] flex items-center justify-center text-[#8ab4f8]">
              <span className="material-symbols-outlined text-lg">
                {activeTab === 'upload' ? 'drive_folder_upload' : 'library_books'}
              </span>
            </div>
            <div>
              <h3 className="text-base font-medium text-white">ChromaDB Knowledge Vault</h3>
              <p className="text-[11px] text-[#c4c7c5]">
                {activeTab === 'upload'
                  ? 'Ingest financial research, filings (10-K, 10-Q), and tabular data'
                  : 'Manage indexed collections, metadata, and vector chunks'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 rounded-full hover:bg-[#333538] text-[#c4c7c5] hover:text-white flex items-center justify-center transition"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Tab Toggle Navigation */}
        <div className="flex gap-2 mt-4 pb-2 border-b border-[#3c4043]/40">
          <button
            type="button"
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-1.5 rounded-full text-xs font-medium transition flex items-center gap-1.5 ${
              activeTab === 'upload'
                ? 'bg-[#1b3d6e] text-[#8ab4f8] border border-[#8ab4f8]/30'
                : 'text-[#c4c7c5] hover:bg-[#333538]'
            }`}
          >
            <span className="material-symbols-outlined text-sm">upload_file</span>
            Ingest Document
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('manage')}
            className={`px-4 py-1.5 rounded-full text-xs font-medium transition flex items-center gap-1.5 ${
              activeTab === 'manage'
                ? 'bg-[#1b3d6e] text-[#8ab4f8] border border-[#8ab4f8]/30'
                : 'text-[#c4c7c5] hover:bg-[#333538]'
            }`}
          >
            <span className="material-symbols-outlined text-sm">folder_managed</span>
            Manage Vault Chunks
          </button>
        </div>

        {/* TAB 1: Ingest Form */}
        {activeTab === 'upload' && (
          <form onSubmit={handleUpload} className="mt-4 space-y-4 overflow-y-auto pr-1">
            <div>
              <label className="block text-xs font-medium text-[#e3e3e3] mb-1.5">
                Target Vector Collection
              </label>
              <select
                value={selectedCollection}
                onChange={(e) => setSelectedCollection(e.target.value)}
                className="w-full bg-[#1e1f20] border border-[#3c4043] text-[#e3e3e3] text-xs rounded-2xl p-3 focus:outline-none focus:border-[#8ab4f8] transition"
              >
                {collections.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-[#8e918f] mt-1">
                Select target domain (e.g. <b>MarketAnalysis</b> for 10-K/10-Q research, <b>Finance</b> for general financial QA, <b>Tax</b> for IRC rules).
              </p>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#e3e3e3] mb-1.5">
                Financial Report File (.pdf, .txt, .csv)
              </label>
              <input
                type="file"
                accept=".pdf,.txt,.csv"
                onChange={(e) => setFile(e.target.files[0])}
                className="w-full text-xs text-[#c4c7c5] file:mr-3 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-medium file:bg-[#1b3d6e] file:text-[#8ab4f8] hover:file:bg-[#284a7e] cursor-pointer border border-[#3c4043] rounded-2xl p-3 bg-[#1e1f20]"
              />
              <p className="text-[10px] text-[#8e918f] mt-1">
                Supported formats: PDF (PyPDFium2 high-fidelity text extraction), Plain Text/Markdown, and CSV tables. Metadata (ticker, year, quarter, doc_type) is automatically extracted and indexed.
              </p>
            </div>

            {uploadStatus && (
              <div
                className={`p-3 rounded-2xl text-xs font-mono flex items-start gap-2 ${
                  uploadStatus.type === 'success'
                    ? 'bg-[#0d652d]/30 text-[#81c995] border border-[#81c995]/40'
                    : 'bg-[#d93025]/20 text-[#f28b82] border border-[#f28b82]/40'
                }`}
              >
                <span className="material-symbols-outlined text-sm mt-0.5">
                  {uploadStatus.type === 'success' ? 'check_circle' : 'error'}
                </span>
                <span>{uploadStatus.message}</span>
              </div>
            )}

            <div className="flex items-center justify-end gap-2 pt-3 border-t border-[#3c4043]/60">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-xs font-medium text-[#c4c7c5] hover:text-white rounded-full hover:bg-[#333538] transition"
              >
                Close
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
                    Ingest & Index
                  </>
                )}
              </button>
            </div>
          </form>
        )}

        {/* TAB 2: Manage Vault Documents */}
        {activeTab === 'manage' && (
          <div className="mt-4 flex-1 flex flex-col overflow-hidden">
            {/* Filter bar */}
            <div className="flex items-center justify-between gap-3 mb-3">
              <div className="flex items-center gap-2 flex-1">
                <span className="text-[11px] text-[#8e918f]">Filter:</span>
                <select
                  value={filterCollection}
                  onChange={(e) => setFilterCollection(e.target.value)}
                  className="bg-[#1e1f20] border border-[#3c4043] text-[#e3e3e3] text-xs rounded-xl px-2.5 py-1.5 focus:outline-none focus:border-[#8ab4f8]"
                >
                  <option value="">All Collections</option>
                  {collections.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              <button
                onClick={loadDocuments}
                className="p-1.5 rounded-full hover:bg-[#333538] text-[#c4c7c5] hover:text-white transition"
                title="Refresh Documents"
              >
                <span className="material-symbols-outlined text-sm">refresh</span>
              </button>
            </div>

            {manageStatus && (
              <div
                className={`p-2.5 mb-3 rounded-xl text-xs font-mono flex items-center gap-2 ${
                  manageStatus.type === 'success'
                    ? 'bg-[#0d652d]/30 text-[#81c995] border border-[#81c995]/40'
                    : 'bg-[#d93025]/20 text-[#f28b82] border border-[#f28b82]/40'
                }`}
              >
                <span className="material-symbols-outlined text-sm">
                  {manageStatus.type === 'success' ? 'check_circle' : 'error'}
                </span>
                <span>{manageStatus.message}</span>
              </div>
            )}

            {/* Document list */}
            <div className="flex-1 overflow-y-auto space-y-2 pr-1">
              {isLoadingDocs ? (
                <div className="flex flex-col items-center justify-center py-12 text-[#8e918f] gap-2">
                  <span className="material-symbols-outlined text-2xl animate-spin">progress_activity</span>
                  <span className="text-xs">Loading Knowledge Vault documents...</span>
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-10 text-[#8e918f] text-xs">
                  No documents found in selected collection(s). Use the Ingest tab to add documents.
                </div>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.file_id}
                    className="p-3 rounded-2xl bg-[#1e1f20] border border-[#3c4043]/60 flex items-center justify-between hover:border-[#8ab4f8]/40 transition group"
                  >
                    <div className="min-w-0 flex-1 pr-3">
                      <div className="flex items-center gap-2">
                        <span className="material-symbols-outlined text-base text-[#8ab4f8]">
                          description
                        </span>
                        <span className="text-xs font-medium text-white truncate" title={doc.file_name}>
                          {doc.file_name}
                        </span>
                        {doc.ticker && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-[#1b3d6e] text-[#8ab4f8]">
                            ${doc.ticker}
                          </span>
                        )}
                        <span className="px-2 py-0.5 rounded-full text-[10px] bg-[#333538] text-[#c4c7c5]">
                          {doc.collection}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-[10px] text-[#8e918f] mt-1 pl-6">
                        <span>Chunks: <b>{doc.chunk_count}</b></span>
                        {doc.doc_type && <span>Type: <b>{doc.doc_type}</b></span>}
                        {doc.fiscal_year && (
                          <span>Period: <b>{doc.fiscal_year}{doc.fiscal_quarter ? ` ${doc.fiscal_quarter}` : ''}</b></span>
                        )}
                        {doc.created_at && (
                          <span>Indexed: {new Date(doc.created_at).toLocaleDateString()}</span>
                        )}
                      </div>
                    </div>

                    <button
                      onClick={() => handleDelete(doc.file_id, doc.file_name)}
                      disabled={deletingId === doc.file_id}
                      className="p-2 rounded-full text-[#f28b82] hover:bg-[#d93025]/20 disabled:opacity-40 transition"
                      title="Delete document and purge chunks"
                    >
                      {deletingId === doc.file_id ? (
                        <span className="material-symbols-outlined text-base animate-spin">progress_activity</span>
                      ) : (
                        <span className="material-symbols-outlined text-base">delete</span>
                      )}
                    </button>
                  </div>
                ))
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-3 border-t border-[#3c4043]/60 mt-3">
              <span className="text-[11px] text-[#8e918f]">
                Total Documents: <b>{documents.length}</b>
              </span>
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-1.5 text-xs font-medium text-[#c4c7c5] hover:text-white rounded-full hover:bg-[#333538] transition"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

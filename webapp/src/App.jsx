import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MarketTicker from './components/MarketTicker';
import ChatInterface from './components/ChatInterface';
import DocumentUploadModal from './components/DocumentUploadModal';
import Toast from './components/Toast';
import {
  fetchHealth,
  fetchModels,
  setModel as apiSetModel,
  fetchCollections,
  fetchMarkets
} from './services/api';

export default function App() {
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState('');
  const [collections, setCollections] = useState(['Finance', 'MarketAnalysis']);
  const [markets, setMarkets] = useState([]);
  const [isRefreshingMarkets, setIsRefreshingMarkets] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(loadMarketData, 60000);
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      const [modelsData, collectionsData] = await Promise.all([
        fetchModels(),
        fetchCollections()
      ]);

      if (modelsData?.models) {
        setModels(modelsData.models);
        setActiveModel(modelsData.active_model || modelsData.display_names?.[0] || '');
      }

      if (collectionsData?.collections) {
        setCollections(collectionsData.collections);
      }
    } catch (err) {
      console.error('Error fetching initial configs:', err);
    }

    loadMarketData();
  };

  const loadMarketData = async () => {
    setIsRefreshingMarkets(true);
    try {
      const data = await fetchMarkets();
      if (data?.markets) {
        setMarkets(data.markets);
      }
    } catch (err) {
      console.error('Error fetching markets:', err);
    } finally {
      setIsRefreshingMarkets(false);
    }
  };

  const handleSelectModel = async (model) => {
    setActiveModel(model);
    try {
      await apiSetModel(model);
      showToast(`Model switched`, 'success');
    } catch (err) {
      showToast(`Failed to switch model: ${err.message}`, 'error');
    }
  };

  const handleClearChat = () => {
    window.location.reload();
  };

  return (
    <div className="flex flex-col h-screen bg-[#131314] text-[#e3e3e3] selection:bg-[#8ab4f8]/30 selection:text-[#c2e7ff]">
      {/* Clean Top Header */}
      <Header
        models={models}
        activeModel={activeModel}
        onSelectModel={handleSelectModel}
        onOpenUpload={() => setIsUploadOpen(true)}
        onClearChat={handleClearChat}
      />

      {/* Market Ticker Strip */}
      <MarketTicker
        markets={markets}
        onRefresh={loadMarketData}
        isRefreshing={isRefreshingMarkets}
      />

      {/* Conversational Workspace */}
      <ChatInterface
        activeModel={activeModel}
        onShowToast={showToast}
        onOpenUpload={() => setIsUploadOpen(true)}
      />

      {/* Document Upload Modal */}
      <DocumentUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        collections={collections}
        onUploadSuccess={(res) => showToast(`Document "${res.file_name}" uploaded successfully!`, 'success')}
      />

      {/* Snackbar Toast */}
      <Toast toast={toast} onClose={() => setToast(null)} />
    </div>
  );
}

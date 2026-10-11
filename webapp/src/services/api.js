// API Client for Financial AI Assistant

const rawApiUrl = import.meta.env.VITE_API_BASE_URL || '';
const API_BASE = rawApiUrl ? rawApiUrl.replace(/\/$/, '') : (window.location.port === '5173' ? 'http://127.0.0.1:8000' : '');

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed with status: ${res.status}`);
  return await res.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error(`Failed to fetch models with status: ${res.status}`);
  return await res.json();
}

export async function setModel(model) {
  const res = await fetch(`${API_BASE}/api/models/set`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to set model: ${res.status}`);
  }
  return await res.json();
}

export async function fetchCollections() {
  const res = await fetch(`${API_BASE}/api/collections`);
  if (!res.ok) throw new Error(`Failed to fetch collections: ${res.status}`);
  return await res.json();
}

export async function fetchMarkets() {
  const res = await fetch(`${API_BASE}/api/markets`);
  if (!res.ok) throw new Error(`Failed to fetch markets: ${res.status}`);
  return await res.json();
}

export async function sendChatMessage(prompt, model, sessionId = null) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model, session_id: sessionId })
  });
  return await res.json();
}

export async function resetChatSession(sessionId) {
  if (!sessionId) return null;
  const res = await fetch(`${API_BASE}/api/chat/session/reset?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'POST'
  });
  return await res.json();
}

export async function uploadDocument(file, collection) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('collection', collection);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData
  });
  return await res.json();
}

export async function fetchDocuments(collection = null) {
  const url = collection
    ? `${API_BASE}/api/documents?collection=${encodeURIComponent(collection)}`
    : `${API_BASE}/api/documents`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.status}`);
  return await res.json();
}

export async function deleteDocument(fileId) {
  const res = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(fileId)}`, {
    method: 'DELETE'
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to delete document: ${res.status}`);
  }
  return await res.json();
}

export async function clearCollection(collection) {
  const res = await fetch(`${API_BASE}/api/collections/${encodeURIComponent(collection)}/clear`, {
    method: 'DELETE'
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to clear collection: ${res.status}`);
  }
  return await res.json();
}

export async function fetchCollectionStats(collection) {
  const res = await fetch(`${API_BASE}/api/collections/${encodeURIComponent(collection)}/stats`);
  if (!res.ok) throw new Error(`Failed to fetch stats: ${res.status}`);
  return await res.json();
}

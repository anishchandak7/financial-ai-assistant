// API Client for Financial AI Assistant

const API_BASE = window.location.port === '5173' ? 'http://127.0.0.1:8000' : '';

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

export async function sendChatMessage(prompt, model) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model })
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

const API_BASE = "http://localhost:8000"; // your backend URL

export async function apiRequest(endpoint, options = {}) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!res.ok) {
    throw new Error("API request failed");
  }

  return res.json();
}
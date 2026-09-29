const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000";


async function request(endpoint, options = {}) {
  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    },
  );

  let data;

  try {
    data = await response.json();
  } catch {
    data = {
      detail: "Invalid server response.",
    };
  }

  if (!response.ok) {
    throw new Error(
      data.detail ||
      `Request failed with status ${response.status}`,
    );
  }

  return data;
}


export async function analyzeUrl(url) {
  return request(
    "/api/v1/analyze/url",
    {
      method: "POST",
      body: JSON.stringify({
        url,
      }),
    },
  );
}


export async function analyzeSms(text) {
  return request(
    "/api/v1/analyze/sms",
    {
      method: "POST",
      body: JSON.stringify({
        text,
      }),
    },
  );
}


export async function getHistory(limit = 50) {
  return request(
    `/api/v1/history?limit=${limit}`,
  );
}


export async function getHealth() {
  return request(
    "/api/v1/health",
  );
}


export async function getModelInfo() {
  return request(
    "/api/v1/model-info",
  );
}

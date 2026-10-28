// JSON API of the Prodigy server (prodigy/web/app.py). URLs are relative to the page, so the
// GUI also works behind a reverse proxy under a path prefix.

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

async function request(method, path, body, { signal } = {}) {
  const options = { method, headers: { Accept: "application/json" }, signal };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(new URL(path, document.baseURI), options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("The Prodigy server cannot be reached. Is `prodigy web` still running?", 0, null);
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    throw new ApiError(data?.error || `${response.status} ${response.statusText}`, response.status, data);
  }
  return data;
}

export const api = {
  meta: () => request("GET", "api/meta"),
  check: (source, signal) => request("POST", "api/check", { source }, { signal }),
  analyze: (payload) => request("POST", "api/analyze", payload),
  job: (id, logOffset = 0) => request("GET", `api/jobs/${encodeURIComponent(id)}?log_offset=${logOffset}`),
  cancel: (id) => request("DELETE", `api/jobs/${encodeURIComponent(id)}`),
  examples: () => request("GET", "api/examples"),
  example: (id) => request("GET", `api/examples/${id.split("/").map(encodeURIComponent).join("/")}`),
};

import type {
  DocumentRef,
  HealthResponse,
  QueryResponse,
  UploadResponse,
} from '../types';
import { parseSSE } from './sse';

const BASE = (import.meta.env.VITE_API_BASE as string) ?? '/api';

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`, { signal });
  return parseJson<HealthResponse>(res);
}

export async function postQuery(
  query: string,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  });
  return parseJson<QueryResponse>(res);
}

export async function uploadPdf(
  file: File,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    body: form,
    signal,
  });
  return parseJson<UploadResponse>(res);
}

export interface StreamHandlers {
  onDocs?: (docs: DocumentRef[]) => void;
  onToken?: (text: string) => void;
  onDone?: (info: { cached: boolean; elapsed_ms: number }) => void;
  onError?: (detail: string) => void;
}

export async function streamQuery(
  query: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  if (!res.body) {
    throw new Error('응답 본문이 비어있습니다');
  }

  for await (const ev of parseSSE(res.body)) {
    switch (ev.event) {
      case 'docs':
        handlers.onDocs?.(JSON.parse(ev.data) as DocumentRef[]);
        break;
      case 'token':
        handlers.onToken?.(JSON.parse(ev.data) as string);
        break;
      case 'done':
        handlers.onDone?.(JSON.parse(ev.data) as { cached: boolean; elapsed_ms: number });
        break;
      case 'error':
        handlers.onError?.((JSON.parse(ev.data) as { detail: string }).detail);
        break;
      default:
        break;
    }
  }
}

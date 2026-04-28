export interface DocumentRef {
  page_content: string;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  query: string;
  answer: string;
  documents: DocumentRef[];
  cached: boolean;
  elapsed_ms: number;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  doc_id: string | null;
  doc_name: string | null;
}

export interface UploadResponse {
  doc_id: string;
  doc_name: string;
  chunks_indexed: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  documents?: DocumentRef[];
  cached?: boolean;
  elapsedMs?: number;
  pending?: boolean;
  error?: string;
}

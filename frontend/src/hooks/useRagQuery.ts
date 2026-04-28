import { useCallback, useRef, useState } from 'react';
import { streamQuery } from '../lib/api';
import type { ChatMessage } from '../types';

function makeId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function useRagQuery() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const updateAssistant = useCallback(
    (id: string, patch: (m: ChatMessage) => ChatMessage) => {
      setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));
    },
    [],
  );

  const send = useCallback(
    async (query: string) => {
      if (!query.trim() || pending) return;

      const userMsg: ChatMessage = { id: makeId(), role: 'user', text: query };
      const assistantId = makeId();
      const placeholder: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        text: '',
        pending: true,
      };
      setMessages((prev) => [...prev, userMsg, placeholder]);
      setPending(true);

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        await streamQuery(
          query,
          {
            onDocs: (docs) =>
              updateAssistant(assistantId, (m) => ({ ...m, documents: docs })),
            onToken: (text) =>
              updateAssistant(assistantId, (m) => ({
                ...m,
                text: m.text + text,
              })),
            onDone: (info) =>
              updateAssistant(assistantId, (m) => ({
                ...m,
                pending: false,
                cached: info.cached,
                elapsedMs: info.elapsed_ms,
              })),
            onError: (detail) =>
              updateAssistant(assistantId, (m) => ({
                ...m,
                pending: false,
                error: detail,
              })),
          },
          ctrl.signal,
        );
      } catch (err) {
        const aborted = err instanceof DOMException && err.name === 'AbortError';
        const message = aborted
          ? '취소되었습니다.'
          : err instanceof Error
            ? err.message
            : '알 수 없는 오류';
        updateAssistant(assistantId, (m) => ({
          ...m,
          pending: false,
          error: m.error ?? message,
        }));
      } finally {
        abortRef.current = null;
        setPending(false);
      }
    },
    [pending, updateAssistant],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, pending, send, cancel, clear };
}

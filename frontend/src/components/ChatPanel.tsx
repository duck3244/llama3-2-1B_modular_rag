import { useEffect, useState } from 'react';
import { getHealth } from '../lib/api';
import { useRagQuery } from '../hooks/useRagQuery';
import type { ChatMessage, HealthResponse } from '../types';
import { DocumentPanel } from './DocumentPanel';
import { InputBox } from './InputBox';
import { MessageList } from './MessageList';

export function ChatPanel() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const { messages, pending, send, cancel, clear } = useRagQuery();

  const refreshHealth = async () => {
    try {
      setHealth(await getHealth());
    } catch {
      setHealth(null);
    }
  };

  useEffect(() => {
    void refreshHealth();
    const id = setInterval(refreshHealth, 5000);
    return () => clearInterval(id);
  }, []);

  const lastDocs =
    [...messages]
      .reverse()
      .find((m: ChatMessage) => m.role === 'assistant' && m.documents?.length)
      ?.documents ?? [];

  return (
    <div className="flex h-full">
      <DocumentPanel health={health} onUploaded={() => void refreshHealth()} />

      <main className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-100 bg-white px-4 py-3">
          <h1 className="text-base font-semibold text-ink-900">
            Llama 3.2 Modular RAG
          </h1>
          <button
            type="button"
            onClick={clear}
            className="text-xs text-ink-800/60 hover:text-ink-900"
          >
            대화 비우기
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-4">
          <MessageList messages={messages} />
        </div>

        {lastDocs.length > 0 && (
          <details className="border-t border-ink-100 bg-ink-50 px-4 py-2 text-xs">
            <summary className="cursor-pointer text-ink-800/70">
              참조 문서 {lastDocs.length}개
            </summary>
            <div className="mt-2 flex flex-col gap-2">
              {lastDocs.map((d, i) => (
                <div
                  key={i}
                  className="rounded border border-ink-100 bg-white px-3 py-2"
                >
                  <div className="text-[10px] text-ink-800/50">
                    {JSON.stringify(d.metadata)}
                  </div>
                  <div className="mt-1 whitespace-pre-wrap text-ink-800/80">
                    {d.page_content.slice(0, 400)}
                    {d.page_content.length > 400 ? '…' : ''}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}

        <InputBox
          disabled={!health?.ready}
          pending={pending}
          onSend={send}
          onCancel={cancel}
        />
      </main>
    </div>
  );
}

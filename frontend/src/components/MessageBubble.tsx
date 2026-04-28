import type { ChatMessage } from '../types';

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const wrapper = isUser
    ? 'flex justify-end'
    : 'flex justify-start';
  const bubble = isUser
    ? 'bg-ink-900 text-white'
    : 'bg-white text-ink-900 border border-ink-100';

  return (
    <div className={wrapper}>
      <div className={`${bubble} max-w-[80%] rounded-lg px-4 py-3 shadow-sm`}>
        {message.error ? (
          <span className="text-red-600">⚠ {message.error}</span>
        ) : message.pending && !message.text ? (
          <span className="inline-flex items-center gap-2 text-ink-800/70">
            <span className="h-2 w-2 animate-pulse rounded-full bg-ink-800/60" />
            생성 중…
          </span>
        ) : (
          <p className="whitespace-pre-wrap leading-relaxed">
            {message.text}
            {message.pending && (
              <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-ink-800/60 align-baseline" />
            )}
          </p>
        )}

        {!message.pending && !message.error && message.role === 'assistant' && (
          <div className="mt-2 text-xs text-ink-800/60">
            {message.cached ? '캐시 응답' : '신규 응답'}
            {typeof message.elapsedMs === 'number'
              ? ` · ${message.elapsedMs}ms`
              : ''}
          </div>
        )}
      </div>
    </div>
  );
}

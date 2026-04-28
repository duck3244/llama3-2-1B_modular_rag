// `fetch` 응답 본문에서 SSE 이벤트를 비동기로 파싱한다.

export interface SSEEvent {
  event: string;
  data: string;
}

export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEEvent, void, void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let event = 'message';
  const dataLines: string[] = [];

  const dispatch = (): SSEEvent | null => {
    if (dataLines.length === 0) {
      event = 'message';
      return null;
    }
    const data = dataLines.join('\n');
    const ev: SSEEvent = { event, data };
    dataLines.length = 0;
    event = 'message';
    return ev;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx: number;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;

        if (line === '') {
          const ev = dispatch();
          if (ev) yield ev;
          continue;
        }
        if (line.startsWith(':')) continue; // SSE 주석
        const colon = line.indexOf(':');
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? '' : line.slice(colon + 1);
        if (value.startsWith(' ')) value = value.slice(1);

        if (field === 'event') {
          event = value;
        } else if (field === 'data') {
          dataLines.push(value);
        }
        // id/retry 필드는 MVP에서 무시
      }
    }
    const tail = dispatch();
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

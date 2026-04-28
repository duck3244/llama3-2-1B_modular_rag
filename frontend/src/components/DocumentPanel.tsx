import { ChangeEvent, useRef, useState } from 'react';
import { uploadPdf } from '../lib/api';
import type { HealthResponse } from '../types';

interface Props {
  health: HealthResponse | null;
  onUploaded: () => void;
}

export function DocumentPanel({ health, onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      await uploadPdf(file);
      onUploaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : '업로드 실패');
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  return (
    <aside className="flex w-72 flex-col gap-3 border-r border-ink-100 bg-white p-4">
      <div>
        <h2 className="text-sm font-semibold text-ink-800">활성 문서</h2>
        <p className="mt-1 break-all text-xs text-ink-800/70">
          {health?.doc_name ?? '없음'}
        </p>
        {health?.doc_id && (
          <p className="mt-0.5 text-[10px] text-ink-800/40">
            {health.doc_id.slice(0, 16)}…
          </p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <label
          className={`cursor-pointer rounded-md border border-dashed border-ink-100 px-3 py-4 text-center text-xs ${
            uploading ? 'opacity-50' : 'hover:bg-ink-50'
          }`}
        >
          {uploading ? '업로드 중…' : 'PDF 선택'}
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            disabled={uploading}
            onChange={onChange}
          />
        </label>
        {error && <p className="text-xs text-red-600">⚠ {error}</p>}
      </div>

      <div className="mt-auto text-[11px] text-ink-800/50">
        <div>상태: {health?.ready ? '준비됨' : '대기'}</div>
      </div>
    </aside>
  );
}

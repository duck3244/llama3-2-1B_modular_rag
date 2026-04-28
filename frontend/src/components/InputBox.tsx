import { useState, KeyboardEvent } from 'react';

interface Props {
  disabled?: boolean;
  pending?: boolean;
  onSend: (text: string) => void;
  onCancel: () => void;
}

export function InputBox({ disabled, pending, onSend, onCancel }: Props) {
  const [value, setValue] = useState('');

  const submit = () => {
    if (!value.trim() || disabled || pending) return;
    onSend(value);
    setValue('');
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex items-end gap-2 border-t border-ink-100 bg-white p-3">
      <textarea
        className="min-h-[44px] flex-1 resize-none rounded-md border border-ink-100 bg-ink-50 px-3 py-2 text-sm leading-relaxed focus:border-ink-800 focus:outline-none disabled:opacity-60"
        placeholder={
          disabled ? '문서를 먼저 업로드하세요' : '질문을 입력하세요 (Enter 전송)'
        }
        rows={2}
        value={value}
        disabled={disabled || pending}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {pending ? (
        <button
          type="button"
          onClick={onCancel}
          className="h-10 rounded-md bg-red-500 px-4 text-sm font-medium text-white hover:bg-red-600"
        >
          취소
        </button>
      ) : (
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="h-10 rounded-md bg-ink-900 px-4 text-sm font-medium text-white hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          전송
        </button>
      )}
    </div>
  );
}

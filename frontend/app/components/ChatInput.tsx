import { useRef, type KeyboardEvent } from "react";
import { Button } from "~/components/ui/button";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export function ChatInput({ value, onChange, onSubmit, isLoading }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && value.trim()) onSubmit();
    }
  }

  function handleInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  return (
    <div className="flex items-end gap-3 rounded-2xl border border-white/10 bg-card/50 backdrop-blur-md px-4 py-3 shadow-lg">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        disabled={isLoading}
        onChange={(e) => onChange(e.target.value)}
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        placeholder="Ask about SEC filings… (Enter to send, Shift+Enter for newline)"
        className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none disabled:opacity-50"
        style={{ minHeight: "24px", maxHeight: "160px" }}
      />
      <Button
        size="sm"
        disabled={isLoading || !value.trim()}
        onClick={onSubmit}
        className="shrink-0 h-8 px-4"
      >
        {isLoading ? (
          <span className="flex items-center gap-1.5">
            <span className="size-3 rounded-full border-2 border-current border-t-transparent animate-spin" />
            Thinking
          </span>
        ) : (
          "Send"
        )}
      </Button>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { ChatMessage } from "~/components/ChatMessage";
import { ChatInput } from "~/components/ChatInput";
import { FilterBar } from "~/components/FilterBar";
import type { Filters, Message, Source } from "~/types";

const API_URL = "http://localhost:8000";

function uid() {
  return Math.random().toString(36).slice(2);
}

export function meta() {
  return [
    { title: "FinanceRAG — SEC Filing Assistant" },
    { name: "description", content: "AI-powered SEC filing research" },
  ];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({ company: "", document_type: "" });
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef(false);

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    userScrolledRef.current = distFromBottom > 80;
  }

  // Auto-scroll only when the user hasn't scrolled away.
  useEffect(() => {
    if (!userScrolledRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  async function handleSubmit() {
    const query = input.trim();
    if (!query || isLoading) return;
    userScrolledRef.current = false;

    const userMsg: Message = { id: uid(), role: "user", content: query };
    const assistantId = uid();
    const assistantMsg: Message = { id: assistantId, role: "assistant", content: "" };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const resp = await fetch(`${API_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          company: filters.company || undefined,
          document_type: filters.document_type || undefined,
        }),
      });

      if (!resp.ok || !resp.body) {
        throw new Error(`Server error: ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;

          let event: {
            type: string;
            content?: string;
            sources?: Source[];
            message?: string;
          };
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }

          if (event.type === "status" && event.message) {
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, status: event.message } : m)));
          } else if (event.type === "token" && event.content) {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + event.content } : m)),
            );
          } else if (event.type === "sources" && event.sources) {
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, sources: event.sources } : m)));
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: `Error: ${event.message}` } : m)),
            );
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `Failed to connect to the API. Is the backend running?\n\n${err}`,
              }
            : m,
        ),
      );
    } finally {
      setIsLoading(false);
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="dark h-dvh bg-gray-950 text-foreground flex flex-col overflow-hidden">
      {/* Ambient gradient blobs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden="true">
        <div className="absolute -top-40 -left-40 size-150 rounded-full bg-emerald-600/10 blur-[120px]" />
        <div className="absolute -bottom-40 -right-40 size-150 rounded-full bg-blue-600/10 blur-[120px]" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/5 backdrop-blur-sm bg-gray-950/60">
        <div className="mx-auto max-w-3xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="size-7 rounded-lg bg-emerald-600 flex items-center justify-center text-white text-xs font-bold">
              F
            </div>
            <span className="font-semibold text-sm tracking-tight">
              Finance<span className="text-emerald-400">RAG</span>
            </span>
          </div>
          {/* <FilterBar filters={filters} onChange={setFilters} disabled={isLoading} /> */}
        </div>
      </header>

      {/* Chat area */}
      <main className="relative z-10 flex-1 flex flex-col mx-auto w-full max-w-3xl px-4 overflow-hidden">
        {isEmpty ? (
          /* Hero state */
          <div className="flex-1 flex flex-col items-center justify-center gap-6 py-24 text-center">
            <div className="size-16 rounded-2xl bg-emerald-600/20 border border-emerald-500/30 flex items-center justify-center">
              <span className="text-3xl">📊</span>
            </div>
            <div className="space-y-2">
              <h1 className="text-2xl font-bold tracking-tight">Ask anything about SEC filings</h1>
              <p className="text-muted-foreground text-sm max-w-sm">
                Query 10-Ks, 10-Qs, and 8-Ks from top public companies. Answers are grounded in real filings with source
                citations.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 text-xs">
              {[
                "What was Apple's revenue in FY2023?",
                "Compare Microsoft and Google cloud margins",
                "NVIDIA gross profit trends 2022–2024",
              ].map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setInput(q)}
                  className="rounded-full border border-white/10 bg-card/40 px-3 py-1.5 text-muted-foreground hover:text-foreground hover:border-white/20 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Message list */
          <div ref={scrollRef} onScroll={handleScroll} className="flex-1 min-h-0 overflow-y-auto">
            <div className="flex flex-col gap-6 py-6">
              {messages.map((msg, i) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  isStreaming={isLoading && msg.role === "assistant" && i === messages.length - 1}
                />
              ))}
              <div ref={bottomRef} />
            </div>
          </div>
        )}

        {/* Input pinned to bottom */}
        <div className="sticky bottom-0 py-4 bg-linear-to-t from-gray-950 via-gray-950/90 to-transparent">
          <ChatInput value={input} onChange={setInput} onSubmit={handleSubmit} isLoading={isLoading} />
          <p className="text-center text-[11px] text-muted-foreground/50 mt-2">
            Responses are grounded in SEC filings. Not financial advice.
          </p>
        </div>
      </main>
    </div>
  );
}

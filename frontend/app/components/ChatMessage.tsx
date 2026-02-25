import ReactMarkdown from "react-markdown";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "~/components/ui/accordion";
import { Avatar, AvatarFallback } from "~/components/ui/avatar";
import type { Message } from "~/types";

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"} items-start`}>
      {/* Avatar */}
      <Avatar className="size-8 shrink-0 mt-1">
        <AvatarFallback
          className={isUser ? "bg-primary text-primary-foreground text-xs" : "bg-emerald-600 text-white text-xs"}
        >
          {isUser ? "You" : "AI"}
        </AvatarFallback>
      </Avatar>

      {/* Bubble */}
      <div className={`flex flex-col gap-2 max-w-[80%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-sm"
              : "bg-card/60 backdrop-blur-sm border border-white/10 text-card-foreground rounded-tl-sm"
          }`}
        >
          {isUser ? (
            <p>{message.content}</p>
          ) : message.content ? (
            <div className="prose prose-sm prose-invert max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
              {isStreaming && <span className="inline-block w-1.5 h-4 bg-primary animate-pulse ml-0.5 rounded-sm" />}
            </div>
          ) : (
            /* Pipeline is running but no tokens yet — show status label */
            <p className="text-muted-foreground/70 text-sm italic animate-pulse">{message.status ?? "Thinking…"}</p>
          )}
        </div>

        {/* Sources accordion */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="sources" className="border-white/10">
              <AccordionTrigger className="text-xs text-muted-foreground hover:text-foreground py-2">
                {message.sources.length} source
                {message.sources.length > 1 ? "s" : ""} cited
              </AccordionTrigger>
              <AccordionContent>
                <div className="flex flex-col gap-1.5 pt-1">
                  {message.sources.map((s) => (
                    <div
                      key={s.source_number}
                      className="flex items-center gap-2 text-xs text-muted-foreground bg-muted/30 rounded-lg px-3 py-2"
                    >
                      <span className="font-mono text-primary font-semibold">[{s.source_number}]</span>
                      <span className="font-medium text-foreground">{s.company}</span>
                      <span className="rounded border border-white/20 px-1 text-[10px] leading-4">
                        {s.document_type}
                      </span>
                      {s.page_number != null && <span>p.{s.page_number}</span>}
                      <span className="ml-auto font-mono">{s.score.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        )}
      </div>
    </div>
  );
}

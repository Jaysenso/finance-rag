import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "~/components/ui/select";
import type { Filters } from "~/types";

const COMPANIES = ["", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"];
const DOC_TYPES = ["", "10-K", "10-Q", "8-K"];

interface FilterBarProps {
  filters: Filters;
  onChange: (filters: Filters) => void;
  disabled?: boolean;
}

export function FilterBar({ filters, onChange, disabled }: FilterBarProps) {
  return (
    <div className="flex items-center gap-2">
      <Select
        disabled={disabled}
        value={filters.company}
        onValueChange={(v) => onChange({ ...filters, company: v === "all" ? "" : v })}
      >
        <SelectTrigger className="h-8 w-36 text-xs bg-card/40 border-white/10 backdrop-blur-sm">
          <SelectValue placeholder="Company" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All companies</SelectItem>
          {COMPANIES.filter(Boolean).map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select
        disabled={disabled}
        value={filters.document_type}
        onValueChange={(v) => onChange({ ...filters, document_type: v === "all" ? "" : v })}
      >
        <SelectTrigger className="h-8 w-36 text-xs bg-card/40 border-white/10 backdrop-blur-sm">
          <SelectValue placeholder="Doc type" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All types</SelectItem>
          {DOC_TYPES.filter(Boolean).map((d) => (
            <SelectItem key={d} value={d}>
              {d}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

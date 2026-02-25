export interface Source {
  source_number: number;
  company: string;
  document_type: string;
  filing_date: string;
  page_number: number | null;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  /** Latest pipeline status label shown while content is still empty. */
  status?: string;
}

export interface Filters {
  company: string;
  document_type: string;
}

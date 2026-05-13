from __future__ import annotations

import json
import re
from typing import Any

from openai import RateLimitError

from annual_report_extractor.config import get_settings
from annual_report_extractor.models import StatementExtraction
from annual_report_extractor.prompts import EXTRACTION_PROMPT
from annual_report_extractor.services.llm_service import LLMService


STATEMENT_TITLES = {
    "balance_sheet": "Balance Sheet",
    "profit_and_loss": "Statement of Profit and Loss",
    "cash_flow": "Cash Flow Statement",
}
SECTION_HEADERS = {
    "profit_and_loss": (
        "profit & loss statement",
        "profit and loss statement",
        "statement of profit and loss",
    ),
    "balance_sheet": (
        "balance sheet summary",
        "balance sheet",
        "financial position",
        "statement of financial position",
    ),
    "cash_flow": (
        "cash flow statement summary",
        "cash flow statement",
        "statement of cash flows",
        "cash flow summary",
    ),
}
MAX_LLM_CHARS = 12000

LINE_ITEM_PATTERN = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9 ,.&()'/%-]{2,}?)\s{2,}(?P<values>.*)$"
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:[$₹]?\(?-?\d[\d,]*(?:\.\d+)?\)?(?:\s*[A-Za-z%]+)?|-{1,2}|N/?A)(?![A-Za-z])",
    re.IGNORECASE,
)


class StatementExtractorService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def extract(
        self,
        statement_type: str,
        statement_text: str,
        company: dict[str, Any],
        page_numbers: list[int],
    ) -> dict[str, Any]:
        provider = self.settings.llm_provider.lower().strip() or "auto"
        if provider == "openai":
            return self._extract_with_llm(statement_type, statement_text, company, page_numbers)
        if provider == "heuristic":
            return self._extract_with_heuristics(statement_type, statement_text, page_numbers)
        if provider == "auto":
            try:
                return self._extract_with_llm(statement_type, statement_text, company, page_numbers)
            except (RateLimitError, ValueError):
                return self._extract_with_heuristics(statement_type, statement_text, page_numbers)
        raise ValueError(f"Unsupported LLM_PROVIDER: {self.settings.llm_provider}")

    def _extract_with_llm(
        self,
        statement_type: str,
        statement_text: str,
        company: dict[str, Any],
        page_numbers: list[int],
    ) -> dict[str, Any]:
        llm = LLMService()
        relevant_text = self._extract_relevant_section(statement_text, statement_type)
        prepared_text = self._prepare_text_for_llm(relevant_text)
        payload = llm.invoke_json(
            EXTRACTION_PROMPT,
            (
                f"Company metadata: {json.dumps(company)}\n"
                f"Statement type: {statement_type}\n"
                f"Statement text:\n{prepared_text}\n\n"
                "Return JSON with keys: statement_type, title, fiscal_year, prior_fiscal_year, page_numbers, line_items. "
                "Each line item must include line_item, current_fy_value, prior_fiscal_year, prior_fy_value, unit, notes. "
                "If only current-year values are present, return prior_fy_value as an empty string."
            ),
        )
        payload["statement_type"] = statement_type
        payload["page_numbers"] = page_numbers
        return StatementExtraction.model_validate(payload).model_dump()

    def _extract_with_heuristics(
        self,
        statement_type: str,
        statement_text: str,
        page_numbers: list[int],
    ) -> dict[str, Any]:
        relevant_text = self._extract_relevant_section(statement_text, statement_type)
        line_items = []
        unit = self._infer_unit(relevant_text or statement_text)
        fiscal_year, prior_fiscal_year = self._infer_years(statement_text)
        title = self._infer_title(relevant_text or statement_text, statement_type)
        lines = [" ".join(raw_line.split()).strip() for raw_line in (relevant_text or statement_text).splitlines()]
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if not line or line.startswith("[Page "):
                idx += 1
                continue
            if self._is_header_line(line, statement_type):
                idx += 1
                continue

            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if self._looks_like_label(line) and self._is_value_only_line(next_line):
                line_items.append(
                    {
                        "line_item": line,
                        "current_fy_value": next_line,
                        "prior_fy_value": "",
                        "unit": unit,
                        "notes": "Heuristic fallback extraction.",
                    }
                )
                idx += 2
                continue

            match = LINE_ITEM_PATTERN.match(line)
            candidate = match.group("label").strip() if match else line
            value_source = match.group("values") if match else line
            numbers = NUMBER_PATTERN.findall(value_source)

            if len(numbers) < 1:
                idx += 1
                continue
            if not re.search(r"[A-Za-z]", candidate):
                idx += 1
                continue
            normalized = candidate.lower()
            if self._is_header_line(normalized, statement_type):
                idx += 1
                continue

            current_value = numbers[-2].strip() if len(numbers) >= 2 else numbers[-1].strip()
            prior_value = numbers[-1].strip() if len(numbers) >= 2 else ""
            line_items.append(
                {
                    "line_item": re.sub(r"\s+", " ", candidate).strip(" .:-"),
                    "current_fy_value": current_value,
                    "prior_fy_value": prior_value,
                    "unit": unit,
                    "notes": "Heuristic fallback extraction.",
                }
            )
            idx += 1

        payload = {
            "statement_type": statement_type,
            "title": title,
            "fiscal_year": fiscal_year,
            "prior_fiscal_year": prior_fiscal_year,
            "page_numbers": page_numbers,
            "line_items": line_items,
        }
        return StatementExtraction.model_validate(payload).model_dump()

    def _infer_title(self, statement_text: str, statement_type: str) -> str:
        for raw_line in statement_text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line or line.startswith("[Page "):
                continue
            lowered = line.lower()
            if any(header in lowered for header in SECTION_HEADERS[statement_type]):
                return line[:200]
            if "consolidated" in lowered or any(word in lowered for word in ("balance", "profit", "cash flow", "financial position")):
                return line[:200]
        return STATEMENT_TITLES[statement_type]

    def _infer_years(self, statement_text: str) -> tuple[str, str]:
        fy_match = re.search(r"\bFY\s*(20\d{2})\s*[–-]\s*(\d{2,4})\b", statement_text, re.IGNORECASE)
        if fy_match:
            start_year = fy_match.group(1)
            end_year = fy_match.group(2)
            if len(end_year) == 2:
                end_year = start_year[:2] + end_year
            prior_start = str(int(start_year) - 1)
            prior_end = str(int(end_year) - 1)
            return f"FY{start_year}-{end_year[-2:]}", f"FY{prior_start}-{prior_end[-2:]}"

        date_match = re.search(r"financial year ended\s+[A-Za-z]+\s+\d{1,2},\s+(20\d{2})", statement_text, re.IGNORECASE)
        if date_match:
            current = date_match.group(1)
            return current, str(int(current) - 1)

        years = re.findall(r"\b20\d{2}\b", statement_text)
        unique_years = []
        for year in years:
            if year not in unique_years:
                unique_years.append(year)
        if len(unique_years) >= 2:
            return unique_years[0], unique_years[1]
        if len(unique_years) == 1:
            current = unique_years[0]
            return current, str(int(current) - 1)
        return "", ""

    def _infer_unit(self, statement_text: str) -> str:
        lowered = statement_text.lower()
        if "crore" in lowered or "cr." in lowered:
            return "crore"
        if "lakhs" in lowered or "lakh" in lowered:
            return "lakhs"
        if "billion" in lowered:
            return "billion"
        if "million" in lowered:
            return "million"
        if "thousand" in lowered:
            return "thousand"
        if "rs." in lowered or "inr" in lowered or "rupees" in lowered:
            return "rupees"
        if "$" in statement_text:
            return "USD"
        return ""

    def _prepare_text_for_llm(self, statement_text: str) -> str:
        selected_lines: list[str] = []
        seen_lines: set[str] = set()

        for raw_line in statement_text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            lowered = line.lower()
            keep_line = False

            if line.startswith("[Page "):
                keep_line = True
            elif NUMBER_PATTERN.search(line) and re.search(r"[A-Za-z]", line):
                keep_line = True
            elif any(
                keyword in lowered
                for keyword in (
                    "consolidated",
                    "balance sheet",
                    "statement of financial position",
                    "statement of profit and loss",
                    "profit & loss",
                    "cash flow",
                    "particulars",
                    "assets",
                    "equity",
                    "liabilities",
                    "expenses",
                    "income",
                )
            ):
                keep_line = True

            if not keep_line or line in seen_lines:
                continue

            selected_lines.append(line)
            seen_lines.add(line)

        compact_text = "\n".join(selected_lines)
        if len(compact_text) <= MAX_LLM_CHARS:
            return compact_text
        return compact_text[:MAX_LLM_CHARS]

    def _extract_relevant_section(self, statement_text: str, statement_type: str) -> str:
        lines = statement_text.splitlines()
        normalized_lines = [" ".join(line.split()).strip() for line in lines]
        headers = SECTION_HEADERS[statement_type]

        start_index = None
        for idx, line in enumerate(normalized_lines):
            lowered = line.lower()
            if any(header in lowered for header in headers):
                start_index = idx
                break

        if start_index is None:
            return statement_text

        end_index = len(lines)
        other_headers = []
        for key, values in SECTION_HEADERS.items():
            if key != statement_type:
                other_headers.extend(values)

        for idx in range(start_index + 1, len(normalized_lines)):
            lowered = normalized_lines[idx].lower()
            if any(header in lowered for header in other_headers):
                end_index = idx
                break
            if lowered in ("operational highlights", "conclusion"):
                end_index = idx
                break

        return "\n".join(lines[start_index:end_index]).strip()

    def _is_header_line(self, line: str, statement_type: str) -> bool:
        lowered = line.lower()
        if any(header in lowered for header in SECTION_HEADERS[statement_type]):
            return True
        return lowered in {
            "particulars",
            "amount",
            "assets",
            "cash flow category",
            "operational highlights",
            "conclusion",
        }

    def _is_value_only_line(self, line: str) -> bool:
        if not line:
            return False
        compact = line.replace(",", "").strip()
        return bool(re.fullmatch(r"-?[$₹]?\d+(?:\.\d+)?(?:\s+[A-Za-z%]+)?", compact))

    def _looks_like_label(self, line: str) -> bool:
        if not line or not re.search(r"[A-Za-z]", line):
            return False
        lowered = line.lower()
        if lowered.startswith("[page "):
            return False
        if self._is_value_only_line(line):
            return False
        if "fy20" in lowered:
            return False
        return True

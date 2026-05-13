from __future__ import annotations

from annual_report_extractor.models import GraphState
from annual_report_extractor.services.pdf_service import collect_pages
from annual_report_extractor.services.statement_extractor import StatementExtractorService


def extract_statements(state: GraphState) -> GraphState:
    extractor = StatementExtractorService()
    statements: dict[str, dict] = {}
    for statement_type in ("balance_sheet", "profit_and_loss", "cash_flow"):
        page_numbers = state.get("page_map", {}).get(statement_type, [])
        if not page_numbers:
            continue
        statement_text = collect_pages(state["pages"], page_numbers)
        statements[statement_type] = extractor.extract(
            statement_type=statement_type,
            statement_text=statement_text,
            company=state["company"],
            page_numbers=page_numbers,
        )
    return {"statements": statements}

"""
document_reader.py — Universal Document Reader
Reads: .txt .md .html .htm .pdf .docx .doc .epub .rtf .csv
Returns plain text. All heavy dependencies are optional with clear error messages.
"""
from pathlib import Path
from typing import Optional, Tuple


# ─── Format dispatcher ────────────────────────────────────────────────────────

def read_document(path: str) -> Tuple[Optional[str], str]:
    """
    Read any supported document and return (text, format_name).
    Returns (None, error_message) on failure.
    """
    p = Path(path)
    if not p.exists():
        return None, f"File not found: {path}"

    ext = p.suffix.lower()
    dispatch = {
        ".txt":  _read_txt,
        ".md":   _read_txt,
        ".csv":  _read_txt,
        ".html": _read_html,
        ".htm":  _read_html,
        ".pdf":  _read_pdf,
        ".docx": _read_docx,
        ".doc":  _read_docx,
        ".epub": _read_epub,
        ".rtf":  _read_rtf,
    }
    reader = dispatch.get(ext, _read_txt)
    try:
        text = reader(path)
        if text is None or not text.strip():
            return None, "File appears to be empty or unreadable."
        return text, ext.lstrip(".")
    except Exception as e:
        return None, f"Error reading {ext} file: {e}"


# ─── Individual readers ───────────────────────────────────────────────────────

def _read_txt(path: str) -> Optional[str]:
    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252", "cp850")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _read_html(path: str) -> Optional[str]:
    raw = _read_txt(path)
    if raw is None:
        return None
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "lxml")
        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except ImportError:
        # Fallback: strip tags with regex
        text = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", text).strip()


def _read_pdf(path: str) -> Optional[str]:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages) if pages else None
    except ImportError:
        raise RuntimeError(
            "pdfplumber not installed. Run: pip install pdfplumber --break-system-packages"
        )


def _read_docx(path: str) -> Optional[str]:
    try:
        from docx import Document
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text)
        return "\n".join(paragraphs) if paragraphs else None
    except ImportError:
        raise RuntimeError(
            "python-docx not installed. Run: pip install python-docx --break-system-packages"
        )


def _read_epub(path: str) -> Optional[str]:
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book  = epub.read_epub(path)
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "lxml")
            parts.append(soup.get_text(separator=" ", strip=True))
        return "\n".join(parts) if parts else None
    except ImportError:
        raise RuntimeError(
            "ebooklib/beautifulsoup4 not installed. "
            "Run: pip install ebooklib beautifulsoup4 --break-system-packages"
        )


def _read_rtf(path: str) -> Optional[str]:
    try:
        from striprtf.striprtf import rtf_to_text
        raw = _read_txt(path)
        if raw is None:
            return None
        return rtf_to_text(raw)
    except ImportError:
        # Fallback: crude RTF stripping
        raw = _read_txt(path)
        if raw is None:
            return None
        text = re.sub(r"\{[^{}]*\}", "", raw)
        text = re.sub(r"\\[a-z]+\d*\s?", "", text)
        text = re.sub(r"[{}\\]", "", text)
        return re.sub(r"\s+", " ", text).strip()


# ─── Size / preview helpers ───────────────────────────────────────────────────

def preview(text: str, chars: int = 300) -> str:
    """Return a short preview of extracted text."""
    clean = " ".join(text.split())
    return clean[:chars] + ("…" if len(clean) > chars else "")


def supported_extensions() -> str:
    return ".txt  .md  .html  .htm  .pdf  .docx  .doc  .epub  .rtf  .csv"

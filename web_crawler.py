"""
web_crawler.py — Web Content Extractor
Features:
  • Wikipedia REST API  — clean article text, no scraping noise
  • Wikipedia topic search — find and pull the best matching article
  • Generic URL scraper  — BeautifulSoup content extraction
  • Link follower        — crawl 1-hop out-links from a page
  • Smart content filter — removes boilerplate, nav, ads
  • Language detection   — follows ?lang= for Wikipedia
"""
import re
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote

# All imports are lazy to give clean error messages
try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False


# ─── Constants ────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "VoychinetResearchBot/3.0 "
        "(linguistic cipher analysis; educational)"
    )
}
WIKI_REST = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_SEARCH = "https://{lang}.wikipedia.org/w/api.php"
DEFAULT_TIMEOUT = 12

# Boilerplate tags to remove during scraping
JUNK_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "advertisement", "figure", "noscript", "iframe", "form",
    "button", "input", "select", "textarea", "svg",
]

# Wikipedia article sections that are noise for NLP training
WIKI_NOISE_SECTIONS = {
    "references", "see also", "further reading", "external links",
    "notes", "bibliography", "gallery", "footnotes",
    "referencias", "véase también", "notas", "bibliografía",
}


# ─── Guard ────────────────────────────────────────────────────────────────────

def _check_deps():
    if not _REQUESTS_OK:
        raise RuntimeError(
            "requests not installed. Run: pip install requests --break-system-packages"
        )
    if not _BS4_OK:
        raise RuntimeError(
            "beautifulsoup4 not installed. Run: pip install beautifulsoup4 --break-system-packages"
        )


# ─── Wikipedia REST ───────────────────────────────────────────────────────────

def fetch_wikipedia_article(
    topic: str,
    lang:    str = "en",
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Optional[str], str]:
    """
    Fetch full article text for a Wikipedia topic via the REST API + HTML endpoint.
    Falls back to summary if full article unavailable.
    Returns (text, source_url) or (None, error).
    """
    _check_deps()
    # URL-encode the topic
    slug = quote(topic.replace(" ", "_"), safe="")
    url  = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{slug}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 404:
            # Try search to find best match
            return _wiki_search_and_fetch(topic, lang, timeout)
        r.raise_for_status()
        text = _clean_wiki_html(r.text, lang)
        if text and len(text) > 100:
            return text, f"https://{lang}.wikipedia.org/wiki/{slug}"
        # Fallback to summary
        return _wiki_summary(slug, lang, timeout)
    except requests.RequestException as e:
        return None, f"Network error: {e}"


def _wiki_summary(slug: str, lang: str, timeout: int) -> Tuple[Optional[str], str]:
    url = WIKI_REST.format(lang=lang, title=slug)
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data.get("extract", "")
        src  = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        return (text if text else None), src
    except requests.RequestException as e:
        return None, f"Summary error: {e}"


def _wiki_search_and_fetch(
    query: str, lang: str, timeout: int
) -> Tuple[Optional[str], str]:
    """Use MediaWiki search API to find the best matching article, then fetch it."""
    params = {
        "action":  "opensearch",
        "search":  query,
        "limit":   5,
        "format":  "json",
    }
    url = WIKI_SEARCH.format(lang=lang)
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data  = r.json()
        titles = data[1] if len(data) > 1 else []
        if not titles:
            return None, f"No Wikipedia article found for: {query}"
        # Use first result
        best = titles[0]
        return fetch_wikipedia_article(best, lang=lang, timeout=timeout)
    except Exception as e:
        return None, f"Search error: {e}"


def _clean_wiki_html(html: str, lang: str) -> str:
    """Extract clean readable text from Wikipedia HTML, skip noise sections."""
    soup = BeautifulSoup(html, "lxml")

    # Remove junk elements
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    # Remove edit-section spans, reference numbers
    for tag in soup.find_all(class_=re.compile(r"mw-editsection|reference|noprint")):
        tag.decompose()

    paragraphs: List[str] = []
    skip = False

    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ("h1", "h2", "h3"):
            heading = el.get_text().lower().strip()
            skip    = any(noise in heading for noise in WIKI_NOISE_SECTIONS)
            continue
        if skip:
            continue
        text = el.get_text(separator=" ", strip=True)
        if len(text) > 30:          # skip stub lines
            paragraphs.append(text)

    return "\n".join(paragraphs)


# ─── Multi-topic Wikipedia search ─────────────────────────────────────────────

def search_and_fetch_related_topics(
    base_topic:  str,
    extra_terms: List[str],
    lang:        str = "en",
    max_articles: int = 5,
    timeout:     int = DEFAULT_TIMEOUT,
) -> List[Tuple[str, str, str]]:
    """
    Search Wikipedia for `base_topic` plus each term in `extra_terms`.
    Returns list of (topic_queried, text, url).
    Great for: fetch_related("Latin", ["medieval", "ecclesiastical", "vulgar"])
    """
    _check_deps()
    results = []
    queries = [base_topic] + [f"{base_topic} {t}" for t in extra_terms[:max_articles-1]]

    for q in queries:
        text, url = fetch_wikipedia_article(q, lang=lang, timeout=timeout)
        if text:
            results.append((q, text, url))
        time.sleep(0.3)   # polite rate limit

    return results


# ─── Generic URL scraper ──────────────────────────────────────────────────────

def scrape_url(
    url: str,
    timeout:       int  = DEFAULT_TIMEOUT,
    follow_links:  bool = False,
    max_links:     int  = 5,
) -> Tuple[Optional[str], str]:
    """
    Scrape text content from any URL.
    If follow_links=True, also fetches up to max_links in-domain links
    and appends their content (useful for single-domain wikis, archives, etc.)
    Returns (combined_text, source_url) or (None, error).
    """
    _check_deps()
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        return None, f"Network error: {e}"

    content_type = r.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return None, f"Unsupported content type: {content_type}"

    soup  = BeautifulSoup(r.text, "lxml")
    texts = [_extract_main_content(soup, url)]

    if follow_links:
        base    = urlparse(url)
        seen    = {url}
        links   = _collect_links(soup, base, seen, max_links)
        for link in links:
            try:
                lr = requests.get(link, headers=HEADERS, timeout=timeout)
                lr.raise_for_status()
                lsoup = BeautifulSoup(lr.text, "lxml")
                t = _extract_main_content(lsoup, link)
                if t and len(t) > 100:
                    texts.append(t)
                time.sleep(0.4)
            except Exception:
                continue

    combined = "\n\n".join(t for t in texts if t)
    return (combined if combined.strip() else None), url


def _extract_main_content(soup: "BeautifulSoup", base_url: str) -> str:
    """
    Heuristic main-content extractor.
    Tries <main>, <article>, <div id="content">, then falls back to all <p>.
    """
    for junk in soup(JUNK_TAGS):
        junk.decompose()

    # Priority selectors for main content
    for selector in ["main", "article", '[id="content"]', '[id="main"]',
                     '[class*="article"]', '[class*="content"]']:
        el = soup.select_one(selector)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return _normalise(text)

    # Fallback: collect all non-empty paragraphs
    paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return _normalise(" ".join(p for p in paras if len(p) > 40))


def _collect_links(
    soup: "BeautifulSoup",
    base: "ParseResult",
    seen: set,
    limit: int,
) -> List[str]:
    """Collect in-domain href links not yet visited."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(f"{base.scheme}://{base.netloc}", href)
        parsed = urlparse(full)
        if (parsed.netloc == base.netloc
                and full not in seen
                and not full.endswith((".jpg", ".png", ".pdf", ".svg", ".gif"))
                and "#" not in full):
            links.append(full)
            seen.add(full)
        if len(links) >= limit:
            break
    return links


def _normalise(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─── Convenience: detect Wikipedia URL ───────────────────────────────────────

def is_wikipedia_url(url: str) -> bool:
    return "wikipedia.org/wiki/" in url


def topic_from_wikipedia_url(url: str) -> Tuple[str, str]:
    """Extract (topic_slug, lang) from a wikipedia URL."""
    match = re.search(r"([a-z]{2})\.wikipedia\.org/wiki/(.+)", url)
    if match:
        lang  = match.group(1)
        topic = match.group(2).replace("_", " ")
        return topic, lang
    return url, "en"

"""Fetch and extract readable article text from a news URL.

Public API:
    fetch_article(url) -> Article(title, text, url)

Raises a ScraperError subclass when extraction is not possible.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
import time
from typing import List, Optional
from urllib.parse import urlparse, urljoin

import requests
import urllib3
from bs4 import BeautifulSoup, Tag

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_TIMEOUT = 15  # seconds
MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags that never contain article prose.
NOISE_TAGS = (
    "script",
    "style",
    "nav",
    "footer",
    "aside",
    "form",
    "noscript",
    "header",
    "iframe",
    "svg",
    "button",
    "figcaption",
)

MIN_PARAGRAPH_CHARS = 50  # shorter <p> tags are usually captions/menu items
MAX_LINK_DENSITY = 0.35  # a <p> that is mostly links is navigation
MIN_ARTICLE_CHARS = 300  # below this we assume extraction failed

BOILERPLATE_MARKERS = (
    "sign up",
    "subscribe",
    "newsletter",
    "cookie",
    "advertisement",
    "all rights reserved",
    "follow us",
    "share this",
    "read more",
    "hide caption",
    "getty images",
    "toggle caption",
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ScraperError(Exception):
    """Base error for this module."""


class InvalidURLError(ScraperError):
    """The URL is malformed or uses an unsupported scheme."""


class FetchError(ScraperError):
    """The page could not be downloaded (network, HTTP, or content type)."""


class ExtractionError(ScraperError):
    """The page downloaded, but no usable article text was found."""


@dataclass
class Article:
    """Result of a successful scrape."""

    title: str
    text: str
    url: str

    def __len__(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------
# Step 1: validate
# --------------------------------------------------------------------------


def validate_url(url: str) -> str:
    """Return a cleaned URL, or raise InvalidURLError."""
    if not isinstance(url, str) or not url.strip():
        raise InvalidURLError("URL is empty.")

    url = url.strip()
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise InvalidURLError("Invalid URL or port.") from exc

    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError("URL must start with http:// or https://")
    if not parsed.hostname or "." not in parsed.hostname:
        raise InvalidURLError("URL must contain a public domain or IPv4 address.")
    if parsed.username is not None or parsed.password is not None:
        raise InvalidURLError("URLs containing credentials are not supported.")
    if port not in (None, 80, 443) or any(ord(char) < 32 for char in url):
        raise InvalidURLError("Only standard HTTP/HTTPS article URLs are supported.")

    return url


# --------------------------------------------------------------------------
# Step 2: download
# --------------------------------------------------------------------------


def _public_ipv4(hostname: str, port: int) -> str:
    """Resolve once and return an approved address; never re-resolve on connect."""
    try:
        addresses = socket.getaddrinfo(
            hostname, port, socket.AF_INET, socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise FetchError("Could not resolve the article website.") from exc
    ips = list(dict.fromkeys(item[4][0] for item in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise InvalidURLError(
            "Only public websites are allowed; private server addresses are blocked."
        )
    return ips[0]


def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch bounded HTML from public IPv4 hosts, validating every redirect.

    Connect to the validated numeric IP to prevent DNS rebinding, while keeping
    the original Host header, TLS SNI and certificate hostname verification.
    """
    deadline = time.monotonic() + timeout * 2
    for _ in range(MAX_REDIRECTS + 1):
        url = validate_url(url)
        parsed = urlparse(url)
        hostname = parsed.hostname.encode("idna").decode("ascii")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        ip = _public_ipv4(hostname, port)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FetchError("Article download timed out.")
        options = {
            "host": ip,
            "port": port,
            "timeout": urllib3.Timeout(
                connect=min(timeout, remaining), read=min(timeout, remaining)
            ),
        }
        if parsed.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                **options,
                server_hostname=hostname,
                assert_hostname=hostname,
                cert_reqs="CERT_REQUIRED",
                ca_certs=requests.certs.where(),
            )
        else:
            pool = urllib3.HTTPConnectionPool(**options)
        headers = {
            **HEADERS,
            "Host": hostname + (f":{port}" if parsed.port else ""),
            "Accept-Encoding": "identity",
        }
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        response = None
        try:
            response = pool.urlopen(
                "GET",
                target,
                headers=headers,
                redirect=False,
                retries=False,
                preload_content=False,
                assert_same_host=False,
            )
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise FetchError("The website returned an invalid redirect.")
                url = urljoin(url, location)
                continue
            if response.status >= 400:
                raise FetchError(f"Server returned HTTP {response.status}.")
            if "html" not in response.headers.get("Content-Type", "").lower():
                raise FetchError("The URL did not return an HTML article.")
            if (
                response.headers.get("Content-Encoding", "identity").lower()
                != "identity"
            ):
                raise FetchError(
                    "The website requires an unsupported compressed response."
                )
            body = bytearray()
            for chunk in response.stream(65536, decode_content=False):
                body.extend(chunk)
                if len(body) > MAX_HTML_BYTES:
                    raise FetchError("Article page is too large (maximum 2 MB).")
                if time.monotonic() > deadline:
                    raise FetchError("Article download timed out.")
            encoding = (
                requests.utils.get_encoding_from_headers(response.headers) or "utf-8"
            )
            try:
                return body.decode(encoding, errors="replace")
            except LookupError:
                return body.decode("utf-8", errors="replace")
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            raise FetchError(
                "Could not securely download the article. Check the URL or try another website."
            ) from exc
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise FetchError("Too many redirects; the site may be blocking scrapers.")


# --------------------------------------------------------------------------
# Step 3: extract
# --------------------------------------------------------------------------


def _strip_noise(soup: BeautifulSoup) -> None:
    """Delete tags that never hold article prose (in place)."""
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()


def _extract_title(soup: BeautifulSoup) -> str:
    """Title from og:title, then <h1>, then <title>."""
    og = soup.find("meta", property="og:title")
    if og and og.get("content", "").strip():
        return og["content"].strip()

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    return "Untitled"


def _link_density(paragraph: Tag) -> float:
    """Fraction of a paragraph's characters that sit inside <a> tags."""
    total = len(paragraph.get_text(strip=True))
    if total == 0:
        return 1.0
    linked = sum(len(a.get_text(strip=True)) for a in paragraph.find_all("a"))
    return linked / total


def _is_content_paragraph(paragraph: Tag) -> bool:
    """Keep real prose, drop captions, menus and promo lines."""
    text = paragraph.get_text(" ", strip=True)

    if len(text) < MIN_PARAGRAPH_CHARS:
        return False
    if _link_density(paragraph) > MAX_LINK_DENSITY:
        return False

    lowered = text.lower()
    if any(marker in lowered for marker in BOILERPLATE_MARKERS) and len(text) < 200:
        return False

    return True


def _paragraphs_from(container: Tag) -> List[str]:
    """Collect qualifying paragraph texts from a container."""
    return [
        p.get_text(" ", strip=True)
        for p in container.find_all("p")
        if _is_content_paragraph(p)
    ]


def _best_container(soup: BeautifulSoup) -> Optional[Tag]:
    """Generic fallback: the element whose direct <p> children hold the most text.

    Works because article bodies group their paragraphs under one parent,
    while navigation and sidebars are scattered across many small ones.
    """
    scores: dict[int, tuple[int, Tag]] = {}

    for p in soup.find_all("p"):
        parent = p.parent
        if parent is None:
            continue
        if not _is_content_paragraph(p):
            continue
        key = id(parent)
        length, _ = scores.get(key, (0, parent))
        scores[key] = (length + len(p.get_text(strip=True)), parent)

    if not scores:
        return None
    return max(scores.values(), key=lambda item: item[0])[1]


def extract_article(html: str, url: str) -> Article:
    """Turn raw HTML into a title and body text, or raise ExtractionError."""
    soup = BeautifulSoup(html, "lxml")
    _strip_noise(soup)

    title = _extract_title(soup)

    # Preferred source, then progressively more generic fallbacks.
    paragraphs: List[str] = []
    for container in (soup.find("article"), soup.find("main")):
        if container is not None:
            paragraphs = _paragraphs_from(container)
            if paragraphs:
                break

    if not paragraphs:
        container = _best_container(soup)
        if container is not None:
            paragraphs = _paragraphs_from(container)

    text = "\n\n".join(paragraphs)

    if len(text) < MIN_ARTICLE_CHARS:
        raise ExtractionError(
            "Could not extract readable article text. The page may be "
            "paywalled, JavaScript-rendered, or not an article page."
        )

    return Article(title=title, text=text, url=url)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def fetch_article(url: str, timeout: int = DEFAULT_TIMEOUT) -> Article:
    """Scrape a news article.

    Args:
        url: Full http(s) URL of a news article.
        timeout: Request timeout in seconds.

    Returns:
        Article with .title, .text and .url.

    Raises:
        InvalidURLError, FetchError, ExtractionError (all ScraperError).
    """
    url = validate_url(url)
    html = fetch_html(url, timeout=timeout)
    return extract_article(html, url)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.scraper.news_scraper <article_url>")
        sys.exit(1)

    try:
        article = fetch_article(sys.argv[1])
    except ScraperError as error:
        print(f"[FAILED] {type(error).__name__}: {error}")
        sys.exit(1)

    print(f"TITLE : {article.title}")
    print(f"CHARS : {len(article)}")
    print("-" * 60)
    print(article.text[:800])

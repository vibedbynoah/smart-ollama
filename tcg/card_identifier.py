"""
Card Identifier — fast identification of trading cards AND sports cards.

Supports: Pokemon, Magic, Yu-Gi-Oh, Digimon, One Piece, Lorcana,
          Baseball, Football, Basketball, Soccer, Hockey (Topps, Panini,
          Upper Deck, Bowman, Donruss, Fleer, etc.)

Fast path (~50-80ms): CV + OCR only — no AI call.
Enhanced path (optional): adds Ollama AI interpretation.
"""

import re
import json
import logging
import time
import cv2
import numpy as np
import pytesseract
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# eBay price LRU cache
# ---------------------------------------------------------------------------

_EBAY_CACHE_MAX = 500
_EBAY_CACHE_TTL = 3600  # seconds

# OrderedDict used as an LRU cache: key=(card_name_lower, set_name_lower)
# value=(timestamp, listings)
_ebay_cache: OrderedDict = OrderedDict()


def _ebay_cache_get(card_name: str, set_name: str):
    """Return cached listings or None if missing/expired."""
    key = (card_name.lower(), set_name.lower())
    entry = _ebay_cache.get(key)
    if entry is None:
        return None
    ts, listings = entry
    if time.time() - ts > _EBAY_CACHE_TTL:
        del _ebay_cache[key]
        return None
    # Move to end (most-recently-used)
    _ebay_cache.move_to_end(key)
    return listings


def _ebay_cache_set(card_name: str, set_name: str, listings: list):
    """Store listings in the LRU cache, evicting oldest entry if at capacity."""
    key = (card_name.lower(), set_name.lower())
    if key in _ebay_cache:
        _ebay_cache.move_to_end(key)
    _ebay_cache[key] = (time.time(), listings)
    while len(_ebay_cache) > _EBAY_CACHE_MAX:
        _ebay_cache.popitem(last=False)


@dataclass
class CardIdentity:
    name: str = ""
    set_name: str = ""
    set_code: str = ""
    card_number: str = ""
    year: str = ""
    rarity: str = ""
    category: str = ""      # "TCG" or "Sports"
    game: str = ""           # Pokemon, Magic, Baseball, Football, etc.
    brand: str = ""          # Topps, Panini, Upper Deck, etc.
    player: str = ""         # For sports cards
    team: str = ""           # For sports cards
    language: str = "English"
    raw_ocr: str = ""
    confidence: float = 0.0
    ai_description: str = ""
    card_detected: bool = False
    in_case: bool = False
    case_type: str = ""      # "PSA slab", "BGS slab", "one-touch", "top loader", "penny sleeve", ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPORTS_BRANDS = [
    "topps", "panini", "upper deck", "bowman", "donruss", "fleer",
    "score", "prizm", "select", "mosaic", "optic", "chrome",
    "stadium club", "heritage", "gypsy queen", "allen & ginter",
    "national treasures", "immaculate", "spectra", "contenders",
]

_SPORTS_KEYWORDS = [
    "rookie", "rc", "baseball", "football", "basketball", "hockey",
    "soccer", "nfl", "nba", "mlb", "nhl", "mls", "fifa",
    "quarterback", "pitcher", "forward", "goalie", "center",
    "touchdown", "home run", "slam dunk", "hat trick",
    "pro bowl", "all-star", "mvp", "hall of fame",
]

_TCG_KEYWORDS = {
    "Pokemon": ["pokemon", "pokémon", "hp", "weakness", "resistance", "retreat", "trainer", "energy"],
    "Magic: The Gathering": ["magic", "wizards of the coast", "mana", "creature", "sorcery", "instant", "enchantment", "artifact", "planeswalker"],
    "Yu-Gi-Oh!": ["yu-gi-oh", "yugioh", "konami", "atk", "def", "trap card", "spell card", "monster"],
    "Digimon": ["digimon", "bandai", "digivolve"],
    "One Piece": ["one piece", "don!!", "counter"],
    "Lorcana": ["lorcana", "disney"],
    "Flesh and Blood": ["flesh and blood"],
    "Weiss Schwarz": ["weiss schwarz", "weiß"],
}

_NFL_TEAMS = [
    "cardinals", "falcons", "ravens", "bills", "panthers", "bears", "bengals",
    "browns", "cowboys", "broncos", "lions", "packers", "texans", "colts",
    "jaguars", "chiefs", "raiders", "chargers", "rams", "dolphins", "vikings",
    "patriots", "saints", "giants", "jets", "eagles", "steelers", "49ers",
    "seahawks", "buccaneers", "titans", "commanders",
]

_NBA_TEAMS = [
    "hawks", "celtics", "nets", "hornets", "bulls", "cavaliers", "mavericks",
    "nuggets", "pistons", "warriors", "rockets", "pacers", "clippers", "lakers",
    "grizzlies", "heat", "bucks", "timberwolves", "pelicans", "knicks", "thunder",
    "magic", "76ers", "suns", "blazers", "kings", "spurs", "raptors", "jazz", "wizards",
]

_MLB_TEAMS = [
    "diamondbacks", "braves", "orioles", "red sox", "cubs", "white sox", "reds",
    "guardians", "rockies", "tigers", "astros", "royals", "angels", "dodgers",
    "marlins", "brewers", "twins", "mets", "yankees", "athletics", "phillies",
    "pirates", "padres", "giants", "mariners", "cardinals", "rays", "rangers", "blue jays", "nationals",
]

_ALL_TEAMS = _NFL_TEAMS + _NBA_TEAMS + _MLB_TEAMS

# Thread pool for parallel OCR
_ocr_pool = ThreadPoolExecutor(max_workers=4)

# Grading service labels commonly found on slabs
_SLAB_KEYWORDS = ["psa", "bgs", "sgc", "cgc", "bvg", "gma", "ags", "hga",
                   "beckett", "gem mint", "gem-mt", "mint", "nm-mt", "authentic"]

_CASE_KEYWORDS = ["ultra pro", "one-touch", "magnetic", "top loader",
                   "card saver", "semi-rigid", "penny sleeve", "team bag"]


# ---------------------------------------------------------------------------
# Card presence detection
# ---------------------------------------------------------------------------

def detect_card_presence(img: np.ndarray) -> tuple[bool, np.ndarray | None]:
    """
    Detect if the image contains a card. Returns (found, contour).
    A card is a roughly 2.5x3.5 aspect ratio rectangle (0.63-0.78 ratio).
    """
    h, w = img.shape[:2]

    # Downscale for speed
    scale = 1.0
    if w > 800:
        scale = 800 / w
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sh, sw = small.shape[:2]
    min_area = sh * sw * 0.05  # Card must be at least 5% of image

    for cnt in contours[:15]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # Check aspect ratio
            rect = cv2.minAreaRect(cnt)
            box_w, box_h = rect[1]
            if box_w == 0 or box_h == 0:
                continue
            ratio = min(box_w, box_h) / max(box_w, box_h)
            # Card ratio is ~0.714 (2.5/3.5), allow wider range for cases/slabs
            if 0.45 < ratio < 0.85:
                return True, (approx.astype(np.float32) / scale).astype(np.int32)

        # Also accept rectangles detected via bounding rect
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw > 0 and rh > 0:
            ratio = min(rw, rh) / max(rw, rh)
            if 0.45 < ratio < 0.85 and area > min_area:
                return True, cnt

    return False, None


# ---------------------------------------------------------------------------
# Case / slab detection
# ---------------------------------------------------------------------------

def detect_case(img: np.ndarray) -> tuple[bool, str]:
    """
    Detect if a card is in a graded slab, case, or holder.
    Returns (in_case, case_type).

    Detection methods:
    - Look for a thick rectangular border around the card (slab/holder)
    - Look for label area above card (grading slab)
    - Color analysis for plastic/acrylic sheen
    - OCR the label area for grading company names
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method 1: Look for nested rectangles (outer = case, inner = card)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    rect_contours = []
    for cnt in contours_sorted[:20]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > h * w * 0.03:
            rect_contours.append(cnt)

    # If we have 2+ nested rectangles, likely a case
    nested = len(rect_contours) >= 2
    if nested:
        outer_area = cv2.contourArea(rect_contours[0])
        inner_area = cv2.contourArea(rect_contours[1])
        # The inner card should be 30-80% of the outer case
        ratio = inner_area / outer_area if outer_area > 0 else 0
        if not (0.25 < ratio < 0.85):
            nested = False

    # Method 2: Check top area for slab label (graded slabs have a label above the card)
    top_region = img[0:int(h * 0.25), 0:w]
    top_gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY) if len(top_region.shape) == 3 else top_region

    # Slab labels tend to be white/light with dark text
    white_pct = np.mean(top_gray > 200)

    # OCR the top region for grading company names
    label_text = ""
    try:
        processed = cv2.threshold(top_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        label_text = pytesseract.image_to_string(processed, config="--psm 6 --oem 3").strip().lower()
    except Exception:
        pass

    # Check for slab keywords
    slab_found = ""
    for kw in _SLAB_KEYWORDS:
        if kw in label_text:
            if kw in ("psa", "gem mint", "gem-mt", "mint", "nm-mt", "authentic"):
                slab_found = "PSA slab"
            elif kw in ("bgs", "beckett", "bvg"):
                slab_found = "BGS slab"
            elif kw == "sgc":
                slab_found = "SGC slab"
            elif kw == "cgc":
                slab_found = "CGC slab"
            elif kw in ("gma", "ags", "hga"):
                slab_found = f"{kw.upper()} slab"
            break

    if slab_found:
        return True, slab_found

    # Check for case keywords in full OCR
    for kw in _CASE_KEYWORDS:
        if kw in label_text:
            return True, kw.title()

    # Method 3: Detect plastic sheen / thick borders
    # Graded slabs have uniform color borders, look for that
    if nested and white_pct > 0.3:
        # Top region is mostly white/light + nested rectangles = likely a slab
        return True, "graded slab"

    # Method 4: Check for thick uniform borders (one-touch/top loader)
    border_samples = [
        gray[0:10, :],           # top edge
        gray[h-10:h, :],        # bottom edge
        gray[:, 0:10],          # left edge
        gray[:, w-10:w],        # right edge
    ]
    border_stds = [np.std(b) for b in border_samples]
    avg_border_std = np.mean(border_stds)
    # Very uniform borders = plastic case
    if avg_border_std < 15 and nested:
        return True, "card holder"

    return False, ""


# ---------------------------------------------------------------------------
# eBay search
# ---------------------------------------------------------------------------

def search_ebay_listings(card_name: str, year: str = "", set_name: str = "",
                         card_number: str = "", game: str = "") -> list:
    """Search eBay for listings of this card (LRU-cached, 10s timeout)."""
    import requests as req

    # Check cache first
    cached = _ebay_cache_get(card_name, set_name)
    if cached is not None:
        return cached

    # Build search query
    parts = []
    if card_name:
        parts.append(card_name)
    if year:
        parts.append(year)
    if set_name:
        parts.append(set_name)
    elif game and game not in card_name:
        parts.append(game)
    if card_number:
        parts.append(f"#{card_number}")

    query = " ".join(parts)
    if not query.strip():
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    listings = []
    for domain in ["www.ebay.com", "www.ebay.co.uk"]:
        try:
            from urllib.parse import quote_plus
            # LH_Complete=1 + LH_Sold=1 = sold listings only, _sop=13 = newest first
            url = f"https://{domain}/sch/i.html?_nkw={quote_plus(query)}&_sacat=0&LH_Complete=1&LH_Sold=1&_sop=13"
            r = req.get(url, headers=headers, timeout=10)
            if r.status_code != 200 or len(r.text) < 5000:
                continue

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "lxml")

            # Try new eBay layout
            for card_el in soup.select(".s-card")[:10]:
                title_el = card_el.select_one(".s-card__title span, .s-card__title")
                price_el = card_el.select_one(".s-card__price .s-price, .s-card__price")
                link_el = card_el.select_one("a.s-card__link, a")
                img_el = card_el.select_one("img")

                title = title_el.get_text(strip=True) if title_el else ""
                price = price_el.get_text(strip=True) if price_el else ""
                link = link_el.get("href", "") if link_el else ""
                img = img_el.get("src", "") if img_el else ""

                if title and "shop on ebay" not in title.lower():
                    listings.append({
                        "title": title,
                        "price": price,
                        "url": link.split("?")[0] if link else "",
                        "image": img,
                        "source": domain,
                    })

            # Fallback to legacy layout
            if not listings:
                for item in soup.select("li.s-item")[:10]:
                    title_el = item.select_one(".s-item__title span, .s-item__title")
                    price_el = item.select_one(".s-item__price")
                    link_el = item.select_one("a.s-item__link")
                    img_el = item.select_one("img")

                    title = title_el.get_text(strip=True) if title_el else ""
                    price = price_el.get_text(strip=True) if price_el else ""
                    link = link_el.get("href", "") if link_el else ""
                    img = img_el.get("src", "") if img_el else ""

                    if title and "shop on ebay" not in title.lower():
                        listings.append({
                            "title": title,
                            "price": price,
                            "url": link.split("?")[0] if link else "",
                            "image": img,
                            "source": domain,
                        })

            if listings:
                break
        except Exception as exc:
            logger.error("eBay request failed for domain %s: %s", domain, exc)
            continue

    # Deduplicate
    seen = set()
    unique = []
    for l in listings:
        key = l.get("title", "")
        if key not in seen:
            seen.add(key)
            unique.append(l)

    result = unique[:10]
    _ebay_cache_set(card_name, set_name, result)
    return result


# ---------------------------------------------------------------------------
# Beckett price guide
# ---------------------------------------------------------------------------

def get_beckett_prices(card_name: str, year: str = "", set_name: str = "",
                       game: str = "") -> dict:
    """Scrape Beckett for price guide values."""
    import requests as req
    from urllib.parse import quote_plus

    parts = []
    if card_name:
        parts.append(card_name)
    if year:
        parts.append(year)
    if set_name:
        parts.append(set_name)

    query = " ".join(parts)
    if not query.strip():
        return {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html",
    }

    prices = {"source": "beckett", "query": query}

    try:
        url = f"https://www.beckett.com/search?term={quote_plus(query)}&type=cards"
        r = req.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            prices["error"] = f"HTTP {r.status_code}"
            return prices

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "lxml")

        results = []
        for row in soup.select(".search-result-item, .item-row, tr.card-row, .listing-item")[:5]:
            title_el = row.select_one(".item-title, .card-title, a, h4, td:first-child")
            low_el = row.select_one(".price-low, .low, .beckett-price-low")
            high_el = row.select_one(".price-high, .high, .beckett-price-high")

            title = title_el.get_text(strip=True) if title_el else ""
            low = low_el.get_text(strip=True) if low_el else ""
            high = high_el.get_text(strip=True) if high_el else ""

            if title:
                results.append({"title": title, "low": low, "high": high})

        prices["results"] = results

        # Also try to find price data from page text
        text = soup.get_text(" ", strip=True)
        price_matches = re.findall(r'\$[\d,]+\.?\d*', text)
        if price_matches and not results:
            prices["prices_found"] = price_matches[:10]

    except Exception as e:
        logger.error("Beckett request failed: %s", e)
        prices["error"] = str(e)

    return prices


# ---------------------------------------------------------------------------
# Parallel price fetching
# ---------------------------------------------------------------------------

def fetch_all_prices(card_name: str, year: str = "", set_name: str = "",
                     card_number: str = "", game: str = "") -> dict:
    """
    Fetch eBay and Beckett prices concurrently using ThreadPoolExecutor.
    Returns {"ebay": [...], "beckett": {...}}.
    """
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_ebay = pool.submit(
            search_ebay_listings, card_name, year, set_name, card_number, game
        )
        fut_beckett = pool.submit(
            get_beckett_prices, card_name, year, set_name, game
        )
        results = {}
        for label, fut in (("ebay", fut_ebay), ("beckett", fut_beckett)):
            try:
                results[label] = fut.result()
            except Exception as exc:
                logger.error("Price fetch failed for %s: %s", label, exc)
                results[label] = [] if label == "ebay" else {}
    return results


# ---------------------------------------------------------------------------
# Image preprocessing — optimized for speed
# ---------------------------------------------------------------------------

def _fast_warp(img: np.ndarray) -> np.ndarray:
    """Fast card detection and warp. Skips expensive operations."""
    h, w = img.shape[:2]

    # Downscale for contour detection if large
    scale = 1.0
    if w > 800:
        scale = 800 / w
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        sh, sw = small.shape[:2]
        if len(approx) == 4 and cv2.contourArea(cnt) > sh * sw * 0.1:
            pts = (approx.reshape(4, 2).astype(np.float32) / scale)
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).flatten()
            ordered = np.array([
                pts[np.argmin(s)], pts[np.argmin(d)],
                pts[np.argmax(s)], pts[np.argmax(d)],
            ], dtype=np.float32)
            dst = np.array([[0, 0], [500, 0], [500, 700], [0, 700]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered, dst)
            return cv2.warpPerspective(img, M, (500, 700))

    return cv2.resize(img, (500, 700))


def _fast_ocr_prep(region: np.ndarray) -> np.ndarray:
    """Minimal preprocessing for fast OCR."""
    h, w = region.shape[:2]
    if w < 250:
        region = cv2.resize(region, (250, int(h * 250 / w)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# ---------------------------------------------------------------------------
# Fast parallel OCR
# ---------------------------------------------------------------------------

def _ocr_one(region: np.ndarray, psm: int = 7) -> str:
    """OCR a single region."""
    try:
        processed = _fast_ocr_prep(region)
        return pytesseract.image_to_string(processed, config=f"--psm {psm} --oem 3").strip()
    except Exception:
        return ""


def _fast_ocr(card: np.ndarray) -> dict:
    """Parallel OCR on 3 key regions for speed."""
    h, w = card.shape[:2]

    # Only 3 regions for speed: name strip, bottom strip, full card
    name_region = card[0:int(h * 0.12), 0:w]
    bottom_region = card[int(h * 0.86):h, 0:w]
    mid_region = card[int(h * 0.12):int(h * 0.22), 0:w]

    # Run OCR in parallel
    f_name = _ocr_pool.submit(_ocr_one, name_region, 7)
    f_bottom = _ocr_pool.submit(_ocr_one, bottom_region, 6)
    f_mid = _ocr_pool.submit(_ocr_one, mid_region, 7)
    f_full = _ocr_pool.submit(_ocr_one, card, 3)

    return {
        "name": f_name.result(timeout=5),
        "bottom": f_bottom.result(timeout=5),
        "mid": f_mid.result(timeout=5),
        "full": f_full.result(timeout=5),
    }


# ---------------------------------------------------------------------------
# Card type + game detection
# ---------------------------------------------------------------------------

def _detect_card_type(ocr: dict) -> tuple[str, str, str]:
    """Detect category (TCG/Sports), game/sport, and brand. Returns (category, game, brand)."""
    all_text = " ".join(ocr.values()).lower()

    # Check sports brands first
    brand = ""
    for b in _SPORTS_BRANDS:
        if b in all_text:
            brand = b.title()
            break

    # Check sports keywords
    is_sports = brand != "" or any(kw in all_text for kw in _SPORTS_KEYWORDS)

    if is_sports:
        # Detect sport
        sport = "Sports Card"
        if any(kw in all_text for kw in ["baseball", "mlb", "pitcher", "home run"]):
            sport = "Baseball"
        elif any(kw in all_text for kw in ["football", "nfl", "quarterback", "touchdown"]):
            sport = "Football"
        elif any(kw in all_text for kw in ["basketball", "nba", "slam dunk"]):
            sport = "Basketball"
        elif any(kw in all_text for kw in ["hockey", "nhl", "hat trick", "goalie"]):
            sport = "Hockey"
        elif any(kw in all_text for kw in ["soccer", "mls", "fifa"]):
            sport = "Soccer"
        return "Sports", sport, brand

    # Check TCG games
    for game, keywords in _TCG_KEYWORDS.items():
        if any(kw in all_text for kw in keywords):
            return "TCG", game, ""

    # If brand detected but no sport keywords, still likely sports
    if brand:
        return "Sports", "Sports Card", brand

    return "Unknown", "Unknown", ""


def _detect_team(all_text: str) -> str:
    """Try to detect a team name from OCR text."""
    lower = all_text.lower()
    for team in _ALL_TEAMS:
        if team in lower:
            return team.title()
    return ""


# ---------------------------------------------------------------------------
# Parse structured info
# ---------------------------------------------------------------------------

def _parse_info(ocr: dict, category: str, game: str) -> dict:
    """Parse card info from OCR text."""
    all_text = " ".join(ocr.values())
    bottom = ocr.get("bottom", "")
    info = {}

    # Card/player name — top of card
    name_raw = ocr.get("name", "") or ocr.get("mid", "")
    name = re.sub(r'[^a-zA-Z0-9\s\-\'\.\,éÉ]', '', name_raw.split("\n")[0]).strip()
    info["name"] = name

    # Year
    year_match = re.search(r'(19|20)\d{2}', bottom + " " + all_text)
    info["year"] = year_match.group(0) if year_match else ""

    # Card number
    for pattern in [r'(\d{1,4}\s*/\s*\d{1,4})', r'([A-Z]{2,5}[-]?\d{1,4})', r'#\s*(\d+)']:
        m = re.search(pattern, bottom)
        if m:
            info["card_number"] = m.group(1) if m.lastindex else m.group(0)
            break
    else:
        info["card_number"] = ""

    # Set code
    m = re.search(r'([A-Z]{2,5}\d{0,2})', bottom)
    if m and m.group(1) not in ("HP", "ATK", "DEF", "GX", "EX", "RC", "MVP"):
        info["set_code"] = m.group(1)
    else:
        info["set_code"] = ""

    # Rarity
    info["rarity"] = ""
    rarity_map = {
        "secret rare": "Secret Rare", "ultra rare": "Ultra Rare",
        "rare holo": "Rare Holo", "rare": "Rare", "uncommon": "Uncommon",
        "common": "Common", "mythic": "Mythic Rare", "rookie": "Rookie",
        "refractor": "Refractor", "prizm": "Prizm", "auto": "Autograph",
        "patch": "Patch", "numbered": "Numbered", "parallel": "Parallel",
        "chrome": "Chrome", "insert": "Insert",
    }
    lower_all = all_text.lower()
    for kw, label in rarity_map.items():
        if kw in lower_all:
            info["rarity"] = label
            break

    # Sports-specific
    info["team"] = _detect_team(all_text) if category == "Sports" else ""
    info["player"] = name if category == "Sports" else ""

    return info


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def identify_card(img_data: bytes, use_ai: bool = False) -> CardIdentity:
    """
    Fast card identification from raw image bytes.

    Args:
        img_data: Raw image bytes (JPEG/PNG)
        use_ai: If True, also call Ollama for enhanced identification (slower)

    Returns:
        CardIdentity with all parsed fields including card_detected and in_case
    """
    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # Step 0: Detect if image contains a card at all
    card_found, contour = detect_card_presence(img)

    if not card_found:
        return CardIdentity(card_detected=False, confidence=0.0)

    # Step 0.5: Detect if card is in a case/slab
    in_case, case_type = detect_case(img)

    # Step 1: Detect + warp card
    card = _fast_warp(img)

    # Step 2: Parallel OCR on key regions
    ocr = _fast_ocr(card)

    # Step 3: Detect type
    category, game, brand = _detect_card_type(ocr)

    # Step 4: Parse structured info
    parsed = _parse_info(ocr, category, game)

    # Step 5: Optional AI enhancement
    ai_result = {}
    if use_ai:
        ai_result = _ai_identify(ocr, category, game, parsed)

    # Step 6: Determine confidence score
    # AI overrides confidence when present and reliable
    if ai_result.get("confidence"):
        confidence = float(ai_result["confidence"])
    else:
        name_found = bool(parsed.get("name", "").strip())
        card_number_found = bool(parsed.get("card_number", "").strip())
        set_code_found = bool(parsed.get("set_code", "").strip())
        # Exact match: name + at least one of card_number/set_code found
        if name_found and (card_number_found or set_code_found):
            confidence = 0.95
        # Partial match: name found but no number/set anchor
        elif name_found:
            confidence = 0.70
        # Fallback: no reliable name extracted
        else:
            confidence = 0.40

    # Build result — AI overrides where available
    return CardIdentity(
        name=ai_result.get("name", parsed.get("name", "")),
        set_name=ai_result.get("set_name", ""),
        set_code=ai_result.get("set_code", parsed.get("set_code", "")),
        card_number=ai_result.get("card_number", parsed.get("card_number", "")),
        year=ai_result.get("year", parsed.get("year", "")),
        rarity=ai_result.get("rarity", parsed.get("rarity", "")),
        category=category,
        game=ai_result.get("game", game),
        brand=ai_result.get("brand", brand),
        player=ai_result.get("player", parsed.get("player", "")),
        team=ai_result.get("team", parsed.get("team", "")),
        language=ai_result.get("language", "English"),
        raw_ocr=ocr.get("full", "")[:1000],
        confidence=confidence,
        ai_description=ai_result.get("description", ""),
        card_detected=True,
        in_case=in_case,
        case_type=case_type,
    )


def _ai_identify(ocr: dict, category: str, game: str, parsed: dict) -> dict:
    """Optional AI-enhanced identification via Ollama."""
    import requests

    ocr_summary = "\n".join(f"  {k}: {v}" for k, v in ocr.items() if v.strip())
    card_type = f"{category} - {game}"

    prompt = f"""Identify this card from OCR text. Card type: {card_type}
Parsed: name={parsed.get('name','?')}, number={parsed.get('card_number','?')}, set={parsed.get('set_code','?')}, year={parsed.get('year','?')}

OCR:
{ocr_summary}

Respond ONLY with JSON:
{{"name":"card/player name","set_name":"full set name","set_code":"abbreviation","card_number":"number","year":"year","rarity":"rarity","game":"{game}","brand":"manufacturer","player":"player name if sports","team":"team if sports","language":"language","confidence":0.85,"description":"brief description"}}"""

    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": "tcg-grader",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }, timeout=10)
        ai_text = r.json().get("message", {}).get("content", "")
        m = re.search(r'\{[^{}]+\}', ai_text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as exc:
        logger.error("AI identify request failed: %s", exc)
    return {}

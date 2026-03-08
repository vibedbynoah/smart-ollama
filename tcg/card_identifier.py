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
import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict


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
        CardIdentity with all parsed fields
    """
    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

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
        confidence=ai_result.get("confidence", 0.6 if parsed.get("name") else 0.2),
        ai_description=ai_result.get("description", ""),
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
        }, timeout=30)
        ai_text = r.json().get("message", {}).get("content", "")
        m = re.search(r'\{[^{}]+\}', ai_text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}

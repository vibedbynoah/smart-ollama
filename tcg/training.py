"""
TCG Grading Model Trainer — Continuous self-improvement loop.

- Collects grading data + user feedback
- Learns correction rules from discrepancies
- Rebuilds Ollama Modelfile with updated system prompt
- Runs calibration cycles in the background
"""

import json
import os
import sqlite3
import time
import threading
import statistics
from dataclasses import asdict

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "smart_ollama.db")
MODELFILE_PATH = os.path.join(os.path.dirname(__file__), "Modelfile.tcg")
TRAINING_INTERVAL = 900  # 15 minutes


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_training_tables():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS tcg_grades (
            id TEXT PRIMARY KEY,
            image_hash TEXT,
            ai_overall REAL,
            ai_centering REAL,
            ai_corners REAL,
            ai_edges REAL,
            ai_surface REAL,
            cv_overall REAL,
            cv_centering REAL,
            cv_corners REAL,
            cv_edges REAL,
            cv_surface REAL,
            actual_overall REAL,
            actual_service TEXT,
            feedback TEXT,
            labeled_at REAL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tcg_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_type TEXT NOT NULL,
            description TEXT NOT NULL,
            adjustment REAL NOT NULL,
            confidence REAL NOT NULL,
            sample_size INTEGER NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tcg_training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entries_processed INTEGER,
            labeled_entries INTEGER,
            corrections_generated INTEGER,
            mae_before REAL,
            mae_after REAL,
            modelfile_updated INTEGER,
            created_at REAL NOT NULL
        );
    """)
    db.commit()
    db.close()


def store_grade(grade_id: str, image_hash: str, cv_result: dict, ai_result: dict):
    """Store a grading result for training."""
    db = get_db()
    db.execute("""
        INSERT OR REPLACE INTO tcg_grades
        (id, image_hash, ai_overall, ai_centering, ai_corners, ai_edges, ai_surface,
         cv_overall, cv_centering, cv_corners, cv_edges, cv_surface, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        grade_id, image_hash,
        ai_result.get("overall"), ai_result.get("centering"),
        ai_result.get("corners"), ai_result.get("edges"), ai_result.get("surface"),
        cv_result.get("overall"), cv_result.get("centering"),
        cv_result.get("corners"), cv_result.get("edges"), cv_result.get("surface"),
        time.time(),
    ))
    db.commit()
    db.close()


def submit_feedback(grade_id: str, actual_grade: float, service: str, feedback: str = ""):
    """User submits the actual PSA/BGS grade for training."""
    db = get_db()
    db.execute("""
        UPDATE tcg_grades SET actual_overall=?, actual_service=?, feedback=?, labeled_at=?
        WHERE id=?
    """, (actual_grade, service, feedback, time.time(), grade_id))
    db.commit()
    db.close()


def compute_corrections() -> list[dict]:
    """Analyze labeled data and generate correction rules."""
    db = get_db()
    labeled = db.execute("""
        SELECT * FROM tcg_grades WHERE actual_overall IS NOT NULL
    """).fetchall()
    db.close()

    if len(labeled) < 3:
        return []

    corrections = []

    # Overall bias: AI vs actual
    ai_diffs = [r["ai_overall"] - r["actual_overall"] for r in labeled if r["ai_overall"]]
    cv_diffs = [r["cv_overall"] - r["actual_overall"] for r in labeled if r["cv_overall"]]

    if ai_diffs:
        ai_bias = statistics.mean(ai_diffs)
        if abs(ai_bias) > 0.1:
            corrections.append({
                "rule_type": "ai_overall_bias",
                "description": f"AI overall grades are {'high' if ai_bias > 0 else 'low'} by {abs(ai_bias):.2f}",
                "adjustment": -ai_bias,
                "confidence": min(0.95, len(ai_diffs) / 20),
                "sample_size": len(ai_diffs),
            })

    if cv_diffs:
        cv_bias = statistics.mean(cv_diffs)
        if abs(cv_bias) > 0.1:
            corrections.append({
                "rule_type": "cv_overall_bias",
                "description": f"CV overall grades are {'high' if cv_bias > 0 else 'low'} by {abs(cv_bias):.2f}",
                "adjustment": -cv_bias,
                "confidence": min(0.95, len(cv_diffs) / 20),
                "sample_size": len(cv_diffs),
            })

    # Per-service corrections
    for service in ["PSA", "BGS", "SGC", "CGC"]:
        service_entries = [r for r in labeled if r["actual_service"] == service]
        if len(service_entries) >= 3:
            diffs = [r["ai_overall"] - r["actual_overall"] for r in service_entries if r["ai_overall"]]
            if diffs:
                bias = statistics.mean(diffs)
                if abs(bias) > 0.15:
                    corrections.append({
                        "rule_type": f"service_bias_{service.lower()}",
                        "description": f"When grading for {service}, AI is off by {bias:+.2f}",
                        "adjustment": -bias,
                        "confidence": min(0.9, len(diffs) / 10),
                        "sample_size": len(diffs),
                    })

    # Per-subgrade corrections
    for subgrade in ["centering", "corners", "edges", "surface"]:
        ai_key = f"ai_{subgrade}"
        actual_key = "actual_overall"  # We only have overall actuals usually
        entries = [r for r in labeled if r[ai_key] is not None]
        if len(entries) >= 5:
            # Compare subgrade distribution to overall
            sub_diffs = [r[ai_key] - r[actual_key] for r in entries]
            sub_bias = statistics.mean(sub_diffs)
            if abs(sub_bias) > 0.3:
                corrections.append({
                    "rule_type": f"subgrade_bias_{subgrade}",
                    "description": f"AI {subgrade} grades tend to be {sub_bias:+.2f} off from actual",
                    "adjustment": -sub_bias,
                    "confidence": min(0.85, len(entries) / 15),
                    "sample_size": len(entries),
                })

    return corrections


def build_correction_prompt(corrections: list[dict]) -> str:
    """Build correction rules as text for the system prompt."""
    if not corrections:
        return ""

    lines = ["\n## LEARNED CORRECTIONS (from verified training data):"]
    for c in corrections:
        conf_str = f"{c['confidence']:.0%}"
        lines.append(f"- {c['description']} (adjust by {c['adjustment']:+.2f}, confidence: {conf_str}, n={c['sample_size']})")

    lines.append("\nApply these corrections to your raw grades before reporting final scores.")
    return "\n".join(lines)


def build_modelfile(corrections: list[dict]):
    """Generate an Ollama Modelfile with TCG grading specialization."""
    correction_text = build_correction_prompt(corrections)

    system_prompt = f"""You are an expert TCG (Trading Card Game) card grader with deep knowledge of PSA, BGS, SGC, and CGC grading standards.

## YOUR ROLE:
You receive computer vision analysis of a trading card and must provide a final grading assessment.
The CV system gives you objective measurements. Your job is to interpret them, apply grading knowledge, and produce a final grade.

## GRADING SCALE (PSA 1-10):
- GEM MINT 10: Perfect card in every way. 50/50 centering or better (55/45 max).
- MINT 9: A superb condition card with only minor flaws. Centering 55/45 or better.
- NM-MT 8: Near Mint-Mint. Slight wax staining, minor printing imperfection allowed. 60/40 centering.
- NEAR MINT 7: Minor wear on corners. Slight roughness on edges. Centering 65/35.
- EX-MT 6: Visible wear on corners. Noticeable roughness on edges. 70/30 centering.
- EXCELLENT 5: Moderate rounding of corners. Noticeable surface wear. 75/25 centering.
- VG-EX 4: Obvious rounding. Surface scuffing. Moderate staining. 80/20.
- VERY GOOD 3: Heavy rounding. Surface scratching. Major staining.
- GOOD 2: Extreme wear throughout. Major creasing possible.
- POOR 1: Barely identifiable as a card. Major damage throughout.

## SUB-GRADE WEIGHTS (PSA-aligned):
- Centering: 15% (border symmetry L/R and T/B)
- Corners: 25% (sharpness of all 4 corners)
- Edges: 25% (straightness, whitening, nicks)
- Surface: 35% (scratches, print quality, whitening, gloss)

## RESPONSE FORMAT:
Always respond with a JSON object:
{{
  "overall": 8.0,
  "centering": 8.5,
  "corners": 7.5,
  "edges": 8.0,
  "surface": 8.0,
  "confidence": 0.85,
  "explanation": "Brief explanation of grade",
  "defects": ["list of notable defects"],
  "psa_equivalent": "NM-MT 8",
  "bgs_equivalent": "8.0",
  "recommended_service": "PSA"
}}
{correction_text}"""

    modelfile = f"""FROM qwen2.5:0.5b
SYSTEM \"\"\"{system_prompt}\"\"\"
PARAMETER temperature 0.1
PARAMETER num_predict 512
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""
    with open(MODELFILE_PATH, "w") as f:
        f.write(modelfile)

    return MODELFILE_PATH


def get_system_prompt(corrections: list[dict] = None) -> str:
    """Get the full TCG system prompt with corrections."""
    if corrections is None:
        corrections = []
    correction_text = build_correction_prompt(corrections)

    return f"""You are an expert TCG (Trading Card Game) card grader with deep knowledge of PSA, BGS, SGC, and CGC grading standards.

## YOUR ROLE:
You receive computer vision analysis of a trading card and must provide a final grading assessment.
The CV system gives you objective measurements. Your job is to interpret them, apply grading knowledge, and produce a final grade.

## GRADING SCALE (PSA 1-10):
- GEM MINT 10: Perfect card in every way. 50/50 centering or better (55/45 max).
- MINT 9: A superb condition card with only minor flaws. Centering 55/45 or better.
- NM-MT 8: Near Mint-Mint. Slight wax staining, minor printing imperfection allowed. 60/40 centering.
- NEAR MINT 7: Minor wear on corners. Slight roughness on edges. Centering 65/35.
- EX-MT 6: Visible wear on corners. Noticeable roughness on edges. 70/30 centering.
- EXCELLENT 5: Moderate rounding of corners. Noticeable surface wear. 75/25 centering.
- VG-EX 4: Obvious rounding. Surface scuffing. Moderate staining. 80/20.
- VERY GOOD 3: Heavy rounding. Surface scratching. Major staining.
- GOOD 2: Extreme wear throughout. Major creasing possible.
- POOR 1: Barely identifiable as a card. Major damage throughout.

## SUB-GRADE WEIGHTS (PSA-aligned):
- Centering: 15% (border symmetry L/R and T/B)
- Corners: 25% (sharpness of all 4 corners)
- Edges: 25% (straightness, whitening, nicks)
- Surface: 35% (scratches, print quality, whitening, gloss)

## RESPONSE FORMAT:
Always respond with a JSON object:
{{
  "overall": 8.0,
  "centering": 8.5,
  "corners": 7.5,
  "edges": 8.0,
  "surface": 8.0,
  "confidence": 0.85,
  "explanation": "Brief explanation of grade",
  "defects": ["list of notable defects"],
  "psa_equivalent": "NM-MT 8",
  "bgs_equivalent": "8.0",
  "recommended_service": "PSA"
}}
{correction_text}"""


def create_ollama_model(corrections: list[dict] = None):
    """Create/update the TCG grading model in Ollama."""
    import requests
    system_prompt = get_system_prompt(corrections or [])

    try:
        r = requests.post("http://localhost:11434/api/create", json={
            "name": "tcg-grader",
            "from": "qwen2.5:0.5b",
            "system": system_prompt,
            "parameters": {
                "temperature": 0.1,
                "num_predict": 512,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }, timeout=120)
        return r.status_code == 200
    except Exception as e:
        print(f"Failed to create Ollama model: {e}")
        return False


def run_training_cycle():
    """Run one training cycle: analyze data, compute corrections, update model."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM tcg_grades").fetchone()["c"]
    labeled = db.execute("SELECT COUNT(*) as c FROM tcg_grades WHERE actual_overall IS NOT NULL").fetchone()["c"]
    db.close()

    corrections = compute_corrections()

    # Store corrections
    db = get_db()
    db.execute("DELETE FROM tcg_corrections")  # Replace all
    for c in corrections:
        db.execute(
            "INSERT INTO tcg_corrections (rule_type, description, adjustment, confidence, sample_size, created_at) VALUES (?,?,?,?,?,?)",
            (c["rule_type"], c["description"], c["adjustment"], c["confidence"], c["sample_size"], time.time()),
        )
    db.commit()

    # Rebuild modelfile with corrections and update Ollama model
    build_modelfile(corrections)
    model_updated = create_ollama_model(corrections)

    # Log training run
    db.execute(
        "INSERT INTO tcg_training_runs (entries_processed, labeled_entries, corrections_generated, modelfile_updated, created_at) VALUES (?,?,?,?,?)",
        (total, labeled, len(corrections), 1 if model_updated else 0, time.time()),
    )
    db.commit()
    db.close()

    return {
        "entries_processed": total,
        "labeled_entries": labeled,
        "corrections": corrections,
        "model_updated": model_updated,
    }


def get_active_corrections() -> list[dict]:
    """Get current active correction rules."""
    db = get_db()
    rows = db.execute("SELECT * FROM tcg_corrections ORDER BY confidence DESC").fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_training_stats() -> dict:
    """Get training statistics."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM tcg_grades").fetchone()["c"]
    labeled = db.execute("SELECT COUNT(*) as c FROM tcg_grades WHERE actual_overall IS NOT NULL").fetchone()["c"]
    corrections = db.execute("SELECT COUNT(*) as c FROM tcg_corrections").fetchone()["c"]
    runs = db.execute("SELECT * FROM tcg_training_runs ORDER BY created_at DESC LIMIT 5").fetchall()
    db.close()

    return {
        "total_grades": total,
        "labeled_grades": labeled,
        "active_corrections": corrections,
        "recent_runs": [dict(r) for r in runs],
    }


# ---------------------------------------------------------------------------
# Background trainer thread
# ---------------------------------------------------------------------------

_trainer_thread = None
_trainer_running = False


def _trainer_loop():
    global _trainer_running
    while _trainer_running:
        try:
            result = run_training_cycle()
            print(f"[TCG Trainer] Cycle complete: {result['entries_processed']} entries, "
                  f"{result['labeled_entries']} labeled, {len(result['corrections'])} corrections")
        except Exception as e:
            print(f"[TCG Trainer] Error: {e}")
        time.sleep(TRAINING_INTERVAL)


def start_trainer():
    global _trainer_thread, _trainer_running
    if _trainer_running:
        return
    _trainer_running = True
    _trainer_thread = threading.Thread(target=_trainer_loop, daemon=True)
    _trainer_thread.start()
    print("[TCG Trainer] Background training started (every 15 min)")


def stop_trainer():
    global _trainer_running
    _trainer_running = False

"""
==============================================================================
  app.py
  Flask API Gateway – Smart Home Energy Management System
  JKUAT Final Year Project – Phase 12: Arduino Serial Integration

  POST /api/command        {"text": "Washa taa"}
  POST /api/voice          multipart audio → Whisper STT → mBERT → GPIO
  GET  /api/status         → current pin state + uptime
  GET  /api/logs           → 10 most recent command logs
  GET  /api/energy         → energy sessions + totals
  POST /api/energy/wattage {"wattage": 9.0}
  POST /api/energy/cost    {"cost_per_kwh": 23.0}
  GET  /api/tips           → dynamic energy saving tips
  GET  /                   → Web dashboard UI
==============================================================================
"""

# ── CRITICAL: env vars BEFORE any other imports ─────────────────────────────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import time
import tempfile
import serial
import serial.tools.list_ports
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CFG = {
    "model_dir"      : "./saved_relay_model",
    "db_path"        : "./smart_home.db",
    "relay_pin"      : 17,
    "max_length"     : 64,
    "host"           : "0.0.0.0",
    "port"           : 5000,
    # ── Label map ─────────────────────────────
    "id2label"       : {0: "TURN_ON", 1: "TURN_OFF",
                        2: "GET_STATUS", 3: "GET_ADVICE"},
    # ── Noise filters ─────────────────────────
    "min_confidence" : 0.75,
    "domain_keywords": {
        "light", "bulb", "switch", "turn", "on", "off",
        "status", "save", "energy", "power", "lamp",
        "taa", "washa", "zima", "stima", "hali", "nini",
        "ushauri", "nishati", "punguza", "umeme", "iwashwe", "izimwe",
    },
    # ── Arduino serial ────────────────────────
    "arduino_port"     : None,       # e.g. "COM3" — None = auto-detect
    "arduino_baud"     : 9600,
    "arduino_channel"  : 1,          # relay channel number sent to Arduino
    "arduino_timeout"  : 2,          # seconds to wait for Arduino response
    # ── Energy monitoring ─────────────────────
    "default_wattage"  : 5.0,
    # ── Whisper STT ───────────────────────────
    "whisper_model"    : "small",   # tiny / base / small / medium
    "whisper_device"   : "cpu",
    "whisper_compute"  : "int8",
    "whisper_min_words": 2,
}

# ─────────────────────────────────────────────
# RUNTIME STATE
# ─────────────────────────────────────────────
ENERGY_STATE = {
    "wattage"      : CFG["default_wattage"],
    "session_start": None,
    "cost_per_kwh" : 23.0,   # KSh — configurable from dashboard
}

# ── Arduino relay state (tracked in software) ─────────────────────────────
RELAY_STATE = {"value": 0}   # 0 = OFF, 1 = ON

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
START_TIME = time.time()

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db() -> None:
    with sqlite3.connect(CFG["db_path"]) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id               INTEGER  PRIMARY KEY AUTOINCREMENT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input       TEXT     NOT NULL,
                predicted_intent TEXT     NOT NULL,
                confidence       REAL     NOT NULL,
                language         TEXT     NOT NULL,
                relay_state      INTEGER  NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS energy_sessions (
                id            INTEGER  PRIMARY KEY AUTOINCREMENT,
                turned_on_at  DATETIME NOT NULL,
                turned_off_at DATETIME,
                duration_s    REAL,
                wattage_w     REAL     NOT NULL,
                wh_consumed   REAL
            )
        """)
        conn.commit()
    logger.info(f"  ✅  Database ready: {CFG['db_path']}")


@contextmanager
def get_db():
    conn = sqlite3.connect(CFG["db_path"], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def log_transaction(user_input, predicted_intent, confidence, language, relay_state):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO system_logs
                (user_input, predicted_intent, confidence, language, relay_state)
            VALUES (?, ?, ?, ?, ?)
        """, (user_input, predicted_intent, round(confidence, 4), language, relay_state))
    logger.info(f"  [DB] intent={predicted_intent} | conf={confidence*100:.1f}% | relay={relay_state}")


# ─────────────────────────────────────────────
# ENERGY HELPERS
# ─────────────────────────────────────────────
def energy_on():
    ENERGY_STATE["session_start"] = time.time()
    logger.info(f"  [ENERGY] Session started — {ENERGY_STATE['wattage']}W")


def energy_off():
    start = ENERGY_STATE.get("session_start")
    if start is None:
        return
    duration_s  = time.time() - start
    wattage_w   = ENERGY_STATE["wattage"]
    wh_consumed = (wattage_w * duration_s) / 3600.0
    turned_on_at  = datetime.utcfromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
    turned_off_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO energy_sessions
                (turned_on_at, turned_off_at, duration_s, wattage_w, wh_consumed)
            VALUES (?, ?, ?, ?, ?)
        """, (turned_on_at, turned_off_at,
              round(duration_s, 2), wattage_w, round(wh_consumed, 6)))
    ENERGY_STATE["session_start"] = None
    logger.info(f"  [ENERGY] Session closed — {duration_s:.1f}s | {wh_consumed*1000:.2f}mWh")


# ─────────────────────────────────────────────
# ARDUINO SERIAL CONNECTION
# ─────────────────────────────────────────────
def find_arduino_port() -> str | None:
    """Auto-detect Arduino Uno by USB vendor/product ID."""
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        # Arduino Uno USB identifiers
        if any(k in desc for k in ["arduino", "uno", "ch340", "ch341", "ftdi", "16u2"]):
            return port.device
    return None


def init_arduino() -> serial.Serial | None:
    """Open serial connection to Arduino. Returns None if not found."""
    port = CFG["arduino_port"] or find_arduino_port()
    if not port:
        logger.warning("  ⚠️  Arduino not found. Running in SOFTWARE SIMULATION mode.")
        logger.warning("     Set 'arduino_port' in CFG or connect the Arduino.")
        return None
    try:
        ser = serial.Serial(port, CFG["arduino_baud"],
                            timeout=CFG["arduino_timeout"])
        time.sleep(2)   # wait for Arduino to reset after serial connect
        logger.info(f"  ✅  Arduino connected on {port} @ {CFG['arduino_baud']} baud")
        return ser
    except serial.SerialException as e:
        logger.error(f"  ❌  Could not open {port}: {e}")
        logger.warning("     Falling back to SOFTWARE SIMULATION mode.")
        return None


logger.info("Connecting to Arduino …")
arduino: serial.Serial | None = init_arduino()
ARDUINO_CONNECTED = arduino is not None

# ─────────────────────────────────────────────
# mBERT MODEL
# ─────────────────────────────────────────────
model_path = CFG["model_dir"]
if not Path(model_path).exists():
    raise FileNotFoundError(f"[ERROR] Model directory '{model_path}' not found.")

logger.info(f"Loading tokenizer from {model_path} …")
tokenizer = AutoTokenizer.from_pretrained(model_path)

logger.info(f"Loading mBERT from {model_path} …")
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model   = AutoModelForSequenceClassification.from_pretrained(model_path)
bert_model.to(torch_device)
bert_model.eval()
logger.info(f"  ✅  mBERT loaded on {str(torch_device).upper()}")

# ─────────────────────────────────────────────
# WHISPER STT MODEL
# ─────────────────────────────────────────────
logger.info(f"Loading Whisper '{CFG['whisper_model']}' model …")
whisper = WhisperModel(
    CFG["whisper_model"],
    device       = CFG["whisper_device"],
    compute_type = CFG["whisper_compute"],
)
logger.info("  ✅  Whisper STT ready.")

# ─────────────────────────────────────────────
# ML MODELS — trained once at startup
# ─────────────────────────────────────────────
ML_READY  = False
iso_model = None
reg_model = None

try:
    _csv = Path("energy.csv")
    if _csv.exists():
        _df          = pd.read_csv(_csv).dropna()
        _df["power"] = _df["voltage"] * _df["current"]

        # Isolation Forest
        iso_model = IsolationForest(contamination=0.1, random_state=42)
        iso_model.fit(_df[["voltage", "current", "power"]])

        # Linear Regression
        _X = _df[["voltage", "current"]]
        _y = _df["power"]
        _Xtr, _Xte, _ytr, _yte = train_test_split(
            _X, _y, test_size=0.2, random_state=42)
        reg_model = LinearRegression()
        reg_model.fit(_Xtr, _ytr)

        ML_READY = True
        logger.info("  ✅  ML models (IsolationForest + LinearRegression) ready.")
    else:
        logger.warning("  ⚠️  energy.csv not found — ML endpoints disabled.")
except Exception as e:
    logger.error(f"  ❌  ML model training failed: {e}")
init_db()

# ─────────────────────────────────────────────
# RESPONSE TEMPLATES
# ─────────────────────────────────────────────
RESPONSES = {
    "TURN_ON"   : {"en": "The light has been turned ON.",
                   "sw": "Taa imewashwa."},
    "TURN_OFF"  : {"en": "The light has been turned OFF.",
                   "sw": "Taa imezimwa."},
    "GET_STATUS": {"en": "The light is currently {state}.",
                   "sw": "Taa iko {state} sasa hivi."},
    "GET_ADVICE": {"en": None,   # dynamically generated — see get_top_tip()
                   "sw": None},
}

# ─────────────────────────────────────────────
# TIPS ENGINE
# ─────────────────────────────────────────────
def generate_tips() -> list[dict]:
    """
    Analyse energy_sessions from the database and return a ranked list
    of dynamic, data-driven energy saving tips.
    Each tip has: title, body_en, body_sw, severity (info|warn|critical)
    """
    tips = []
    cost = ENERGY_STATE["cost_per_kwh"]

    with get_db() as conn:
        sessions = conn.execute("""
            SELECT duration_s, wh_consumed, wattage_w,
                   turned_on_at, turned_off_at
            FROM energy_sessions
            WHERE turned_off_at IS NOT NULL
            ORDER BY id DESC
        """).fetchall()

        daily = conn.execute("""
            SELECT DATE(turned_on_at) AS day,
                   SUM(duration_s)    AS total_s,
                   SUM(wh_consumed)   AS total_wh,
                   COUNT(*)           AS sessions
            FROM energy_sessions
            WHERE turned_off_at IS NOT NULL
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
        """).fetchall()

    if not sessions:
        return [{
            "title"   : "No usage data yet",
            "body_en" : "Start using the system to get personalised energy tips.",
            "body_sw" : "Anza kutumia mfumo kupata vidokezo vya nishati.",
            "severity": "info",
        }]

    durations   = [r["duration_s"]  for r in sessions]
    wh_list     = [r["wh_consumed"] for r in sessions]
    total_wh    = sum(wh_list)
    total_s     = sum(durations)
    avg_dur     = total_s / len(durations)
    max_dur     = max(durations)
    total_cost  = (total_wh / 1000) * cost
    n_sessions  = len(sessions)

    # ── Tip 1: Long sessions ──────────────────
    long_sessions = [d for d in durations if d > 3600]
    if long_sessions:
        hrs = max_dur / 3600
        sev = "critical" if hrs > 8 else "warn"
        tips.append({
            "title"   : f"⚠️ Long usage sessions detected",
            "body_en" : (f"Your longest session was {hrs:.1f} hours. "
                         f"{len(long_sessions)} session(s) exceeded 1 hour. "
                         f"Consider switching off the light when leaving the room."),
            "body_sw" : (f"Kipindi chako kirefu zaidi kilikuwa saa {hrs:.1f}. "
                         f"Vikao {len(long_sessions)} vilizidi saa moja. "
                         f"Zima taa unapoondoka chumbani."),
            "severity": sev,
        })

    # ── Tip 2: Total cost estimate ────────────
    if total_cost > 0:
        monthly_est = total_cost * (30 / max(len(set(r["turned_on_at"][:10] for r in sessions)), 1))
        tips.append({
            "title"   : f"💰 Estimated electricity cost",
            "body_en" : (f"Total usage so far: {total_wh*1000:.1f} mWh "
                         f"≈ KSh {total_cost:.4f}. "
                         f"Projected monthly cost at this rate: KSh {monthly_est:.2f}."),
            "body_sw" : (f"Matumizi yote hadi sasa: {total_wh*1000:.1f} mWh "
                         f"≈ KSh {total_cost:.4f}. "
                         f"Gharama inayokadiriwa kwa mwezi: KSh {monthly_est:.2f}."),
            "severity": "info",
        })

    # ── Tip 3: Session frequency ──────────────
    if n_sessions >= 3:
        tips.append({
            "title"   : f"📊 Usage pattern",
            "body_en" : (f"You have {n_sessions} recorded sessions with an average "
                         f"duration of {avg_dur/60:.1f} minutes. "
                         f"{'Your usage is consistent.' if avg_dur < 1800 else 'Consider reducing session duration.'}"),
            "body_sw" : (f"Una vikao {n_sessions} vilivyorekodiwa na wastani wa "
                         f"dakika {avg_dur/60:.1f}. "
                         f"{'Matumizi yako ni ya kawaida.' if avg_dur < 1800 else 'Fikiria kupunguza muda wa matumizi.'}"),
            "severity": "info" if avg_dur < 1800 else "warn",
        })

    # ── Tip 4: Daily usage trend ──────────────
    if daily and len(daily) >= 2:
        latest_wh = daily[0]["total_wh"] or 0
        prev_wh   = daily[1]["total_wh"] or 0
        if prev_wh > 0:
            change = ((latest_wh - prev_wh) / prev_wh) * 100
            if change > 20:
                tips.append({
                    "title"   : "📈 Usage increased",
                    "body_en" : (f"Today's usage is {change:.0f}% higher than yesterday. "
                                 f"Remember to switch off the light when not needed."),
                    "body_sw" : (f"Matumizi ya leo ni {change:.0f}% zaidi ya jana. "
                                 f"Kumbuka kuzima taa isipohitajika."),
                    "severity": "warn",
                })
            elif change < -20:
                tips.append({
                    "title"   : "📉 Great improvement!",
                    "body_en" : (f"Today's usage is {abs(change):.0f}% lower than yesterday. "
                                 f"Keep up the energy-saving habit!"),
                    "body_sw" : (f"Matumizi ya leo ni {abs(change):.0f}% chini ya jana. "
                                 f"Endelea na tabia ya kuokoa nishati!"),
                    "severity": "info",
                })

    # ── Tip 5: Short-cycle waste ──────────────
    short = [d for d in durations if d < 30]
    if len(short) > 2:
        tips.append({
            "title"   : "🔁 Frequent short switching",
            "body_en" : (f"{len(short)} sessions lasted under 30 seconds. "
                         f"Frequent switching can reduce bulb lifespan. "
                         f"Only switch on when you intend to use the light."),
            "body_sw" : (f"Vikao {len(short)} vilidumu chini ya sekunde 30. "
                         f"Kuwasha na kuzima mara kwa mara kunaweza kupunguza maisha ya taa. "
                         f"Washa taa tu unapohitaji."),
            "severity": "warn",
        })

    # ── Tip 6: General if nothing triggered ───
    if not tips:
        tips.append({
            "title"   : "✅ Usage looks efficient",
            "body_en" : (f"Total consumption: {total_wh*1000:.2f} mWh across "
                         f"{n_sessions} sessions. No major inefficiencies detected."),
            "body_sw" : (f"Jumla ya matumizi: {total_wh*1000:.2f} mWh katika "
                         f"vikao {n_sessions}. Hakuna upotevu mkubwa uliogunduliwa."),
            "severity": "info",
        })

    return tips


def get_top_tip() -> dict:
    """Return the single most important tip for GET_ADVICE responses."""
    tips = generate_tips()
    # Priority: critical > warn > info
    for sev in ("critical", "warn", "info"):
        for t in tips:
            if t["severity"] == sev:
                return {"en": t["body_en"], "sw": t["body_sw"]}
    return {
        "en": "Turn off lights when not in use to save energy.",
        "sw": "Zima taa unapotoka ili kuokoa nishati.",
    }


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────
def is_domain_relevant(text):
    return bool(set(text.lower().split()) & CFG["domain_keywords"])


def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=CFG["max_length"])
    inputs = {k: v.to(torch_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs      = torch.softmax(logits, dim=-1)[0]
    label_id   = int(torch.argmax(probs).item())
    confidence = float(probs[label_id].item())
    return CFG["id2label"][label_id], confidence


def detect_language(text):
    sw_kw = {"taa","washa","zima","hali","nini","je","nipe","ushauri",
             "nishati","ninawezaje","umeme","stima","imewashwa","imezimwa",
             "iwashwe","izimwe","punguza"}
    return "Swahili" if set(text.lower().split()) & sw_kw else "English"


def send_relay_command(state: str) -> bool:
    """
    Send relay command to Arduino over serial.
    Command format: "RELAY:<channel>:<ON|OFF>\n"
    Arduino should echo back "ACK:<state>\n"
    Returns True if acknowledged, False otherwise.
    """
    if not ARDUINO_CONNECTED or arduino is None:
        return False
    try:
        cmd = f"RELAY:{CFG['arduino_channel']}:{state}\n"
        arduino.write(cmd.encode())
        arduino.flush()
        response = arduino.readline().decode(errors="ignore").strip()
        logger.info(f"  [SERIAL] Sent: {cmd.strip()} | Arduino: {response}")
        return response.startswith("ACK")
    except serial.SerialException as e:
        logger.error(f"  [SERIAL] Write error: {e}")
        return False


def execute_action(intent):
    prev = RELAY_STATE["value"]

    if intent == "TURN_ON":
        ok = send_relay_command("ON")
        if ok or not ARDUINO_CONNECTED:
            RELAY_STATE["value"] = 1
    elif intent == "TURN_OFF":
        ok = send_relay_command("OFF")
        if ok or not ARDUINO_CONNECTED:
            RELAY_STATE["value"] = 0

    curr = RELAY_STATE["value"]

    if not prev and curr:
        energy_on()
    elif prev and not curr:
        energy_off()

    if prev != curr:
        mode = "Arduino" if ARDUINO_CONNECTED else "Simulation"
        logger.info(f"  [RELAY] [{mode}] Ch{CFG['arduino_channel']}: "
                    f"{'OFF → ON 💡' if curr else 'ON → OFF 🌑'}")

    return curr


def build_response_text(intent, pin_state):
    state_en = "ON" if pin_state else "OFF"
    state_sw = "ON" if pin_state else "IMEZIMWA"
    if intent == "GET_ADVICE":
        tip = get_top_tip()
        return tip["en"], tip["sw"]
    tmpl = RESPONSES[intent]
    return tmpl["en"].format(state=state_en), tmpl["sw"].format(state=state_sw)


def process_text_command(text):
    """Shared pipeline: text → filters → mBERT → GPIO → JSON dict."""
    language = detect_language(text)

    if not is_domain_relevant(text):
        logger.info(f"  [FILTER] Off-topic: '{text}'")
        return {
            "input": text, "intent": "UNKNOWN/NOISE", "confidence": 0.0,
            "pin_state": RELAY_STATE["value"],
            "response_en": "Input not recognised as a smart home command.",
            "response_sw": "Amri haikutambuliwa kama amri ya nyumba.",
            "filtered_by": "keyword_guard",
        }, 400

    intent, confidence = predict_intent(text)

    if confidence < CFG["min_confidence"]:
        log_transaction(text, "UNCERTAIN", confidence, language, RELAY_STATE["value"])
        return {
            "input": text, "intent": "UNCERTAIN",
            "confidence": round(confidence, 4),
            "pin_state": RELAY_STATE["value"],
            "response_en": f"Not sure ({confidence*100:.1f}%). Please repeat.",
            "response_sw": f"Sijaelewea vizuri ({confidence*100:.1f}%). Tafadhali rudia.",
            "filtered_by": "confidence_gate",
        }, 200

    pin_state = execute_action(intent)
    response_en, response_sw = build_response_text(intent, pin_state)
    log_transaction(text, intent, confidence, language, pin_state)
    logger.info(f"  [CMD] '{text}' → {intent} ({confidence*100:.1f}%) | pin={pin_state}")

    return {
        "input": text, "intent": intent,
        "confidence": round(confidence, 4),
        "pin_state": pin_state, "language": language,
        "response_en": response_en, "response_sw": response_sw,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }, 200


# ─────────────────────────────────────────────
# ROUTE: POST /api/command  (text)
# ─────────────────────────────────────────────
@app.route("/api/command", methods=["POST"])
def command():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "JSON body with 'text' key required.",
                        "example": {"text": "Washa taa"}}), 400
    text = str(payload["text"]).strip()
    if not text:
        return jsonify({"error": "'text' must not be empty."}), 400
    result, status_code = process_text_command(text)
    return jsonify(result), status_code


# ─────────────────────────────────────────────
# ROUTE: POST /api/voice  (browser audio)
# ─────────────────────────────────────────────
@app.route("/api/voice", methods=["POST"])
def voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Send as multipart 'audio' field."}), 400

    import subprocess

    FFMPEG = r"ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

    # ── Save raw browser audio (webm) ─────────
    audio_data = request.files["audio"].read()
    raw_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    raw_path = raw_tmp.name
    raw_tmp.write(audio_data)
    raw_tmp.close()

    wav_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = wav_tmp.name
    wav_tmp.close()

    transcript    = ""
    detected_lang = "unknown"

    try:
        # ── Convert to 16 kHz mono WAV ─────────
        proc = subprocess.run(
            [FFMPEG, "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="ignore")[-300:]
            logger.error(f"  [VOICE] ffmpeg failed: {err}")
            return jsonify({"error": "Audio conversion failed."}), 500

        wav_size = Path(wav_path).stat().st_size
        logger.info(f"  [VOICE] WAV ready: {wav_size} bytes — transcribing …")

        if wav_size == 0:
            return jsonify({
                "transcript": "", "intent": "UNKNOWN/NOISE", "confidence": 0.0,
                "pin_state": RELAY_STATE["value"],
                "response_en": "Empty audio. Please try again.",
                "response_sw": "Sauti tupu. Jaribu tena.",
            }), 200

        # ── Whisper transcription ──────────────
        segments, info = whisper.transcribe(wav_path, language=None, beam_size=1, best_of=1)
        transcript     = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang  = info.language
        logger.info(f"  [VOICE] Transcript: \'{transcript}\' (lang: {detected_lang})")

    except Exception as e:
        logger.error(f"  [VOICE] Error: {e}")
        return jsonify({"error": f"Voice processing failed: {str(e)}"}), 500

    finally:
        try: os.unlink(raw_path)
        except: pass
        try: os.unlink(wav_path)
        except: pass

    # ── Guard: empty or too short ──────────────
    if not transcript:
        return jsonify({
            "transcript": "", "intent": "UNKNOWN/NOISE", "confidence": 0.0,
            "pin_state": RELAY_STATE["value"],
            "response_en": "No speech detected. Please try again.",
            "response_sw": "Hakuna sauti. Jaribu tena.",
        }), 200

    if len(transcript.split()) < CFG["whisper_min_words"]:
        return jsonify({
            "transcript": transcript, "intent": "UNKNOWN/NOISE", "confidence": 0.0,
            "pin_state": RELAY_STATE["value"],
            "response_en": f"Too short: \'{transcript}\'. Say a full command.",
            "response_sw": f"Fupi sana: \'{transcript}\'. Sema amri kamili.",
        }), 200

    # ── Process through mBERT pipeline ────────
    result, status_code      = process_text_command(transcript)
    result["transcript"]     = transcript
    result["stt_language"]   = detected_lang
    return jsonify(result), status_code


# ─────────────────────────────────────────────
# ROUTE: GET /api/logs
# ─────────────────────────────────────────────
@app.route("/api/logs", methods=["GET"])
def logs():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, user_input, predicted_intent,
                   confidence, language, relay_state
            FROM system_logs ORDER BY id DESC LIMIT 10
        """).fetchall()
    entries = [{
        "id": r["id"], "timestamp": r["timestamp"],
        "user_input": r["user_input"],
        "predicted_intent": r["predicted_intent"],
        "confidence": round(r["confidence"] * 100, 1),
        "language": r["language"],
        "relay_state": r["relay_state"],
        "bulb": "ON" if r["relay_state"] else "OFF",
    } for r in rows]
    return jsonify({"total_returned": len(entries), "logs": entries}), 200


# ─────────────────────────────────────────────
# ROUTE: GET /api/energy
# ─────────────────────────────────────────────
@app.route("/api/energy", methods=["GET"])
def energy():
    with get_db() as conn:
        sessions = conn.execute("""
            SELECT id, turned_on_at, turned_off_at,
                   duration_s, wattage_w, wh_consumed
            FROM energy_sessions ORDER BY id DESC LIMIT 20
        """).fetchall()
        totals = conn.execute("""
            SELECT COUNT(*) AS total_sessions,
                   SUM(duration_s) AS total_duration_s,
                   SUM(wh_consumed) AS total_wh
            FROM energy_sessions WHERE turned_off_at IS NOT NULL
        """).fetchone()

    live_duration = 0.0
    live_wh       = 0.0
    if ENERGY_STATE["session_start"] is not None:
        live_duration = time.time() - ENERGY_STATE["session_start"]
        live_wh       = (ENERGY_STATE["wattage"] * live_duration) / 3600.0

    total_wh = (totals["total_wh"] or 0.0) + live_wh
    return jsonify({
        "wattage_w"       : ENERGY_STATE["wattage"],
        "relay_on"        : RELAY_STATE["value"] == 1,
        "live_duration_s" : round(live_duration, 1),
        "live_wh"         : round(live_wh, 6),
        "total_sessions"  : totals["total_sessions"] or 0,
        "total_duration_s": round((totals["total_duration_s"] or 0.0) + live_duration, 1),
        "total_wh"        : round(total_wh, 6),
        "total_kwh"       : round(total_wh / 1000, 8),
        "sessions"        : [{
            "id"           : r["id"],
            "turned_on_at" : r["turned_on_at"],
            "turned_off_at": r["turned_off_at"] or "ongoing",
            "duration_s"   : round(r["duration_s"] or 0, 2),
            "wattage_w"    : r["wattage_w"],
            "wh_consumed"  : round(r["wh_consumed"] or 0, 6),
        } for r in sessions],
    }), 200


# ─────────────────────────────────────────────
# ROUTE: POST /api/energy/wattage
# ─────────────────────────────────────────────
@app.route("/api/energy/wattage", methods=["POST"])
def set_wattage():
    payload = request.get_json(silent=True)
    if not payload or "wattage" not in payload:
        return jsonify({"error": "JSON body with 'wattage' key required."}), 400
    try:
        w = float(payload["wattage"])
        if not (0.1 <= w <= 10000):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "wattage must be a number between 0.1 and 10000."}), 400
    ENERGY_STATE["wattage"] = w
    logger.info(f"  [ENERGY] Wattage updated → {w}W")
    return jsonify({"wattage_w": w, "message": f"Bulb wattage set to {w}W."}), 200


# ─────────────────────────────────────────────
# ROUTE: GET /api/tips
# ─────────────────────────────────────────────
@app.route("/api/tips", methods=["GET"])
def tips():
    return jsonify({
        "cost_per_kwh": ENERGY_STATE["cost_per_kwh"],
        "tips"        : generate_tips(),
    }), 200

# ─────────────────────────────────────────────
# ROUTE: GET /api/ml/insights
# ─────────────────────────────────────────────
@app.route("/api/ml/insights", methods=["GET"])
def ml_insights():
    """
    Returns ML anomaly and prediction results
    formatted as insights cards for the dashboard.
    """
    if not ML_READY:
        return jsonify({"insights": [{
            "title"   : "⚙️ ML Insights Unavailable",
            "body_en" : "energy.csv not found. ML models could not be loaded.",
            "body_sw" : "Faili energy.csv halipatikani. Mifano ya ML haikupakiwa.",
            "severity": "info"
        }]}), 200

    try:
        _df                  = pd.read_csv("energy.csv").dropna()
        _df["power"]         = _df["voltage"] * _df["current"]
        _df["anomaly"]       = iso_model.predict(
                                   _df[["voltage", "current", "power"]])
        _df["anomaly_score"] = iso_model.decision_function(
                                   _df[["voltage", "current", "power"]])

        total        = len(_df)
        anomaly_df   = _df[_df["anomaly"] == -1]
        normal_df    = _df[_df["anomaly"] ==  1]
        n_anomalies  = len(anomaly_df)
        pct          = round((n_anomalies / total) * 100, 1)

        avg_v_normal  = round(normal_df["voltage"].mean(), 2)
        avg_i_normal  = round(normal_df["current"].mean(), 2)
        avg_v_anomaly = round(anomaly_df["voltage"].mean(), 2)
        avg_i_anomaly = round(anomaly_df["current"].mean(), 2)

        # Predict power for average normal reading
        predicted_power = round(float(
            reg_model.predict([[avg_v_normal, avg_i_normal]])[0]), 2)

        insights = []

        # ── Card 1: Anomaly summary ───────────
        sev = "critical" if pct > 15 else "warn" if pct > 5 else "info"
        insights.append({
            "title"   : f"🔍 Anomaly Detection Results",
            "body_en" : (f"{n_anomalies} out of {total} readings "
                         f"({pct}%) were flagged as anomalous. "
                         f"Normal operating range: "
                         f"{avg_v_normal}V / {avg_i_normal}A."),
            "body_sw" : (f"Matumizi {n_anomalies} kati ya {total} "
                         f"({pct}%) yaliashiriwa kama ya kawaida si ya kawaida. "
                         f"Masafa ya kawaida: "
                         f"{avg_v_normal}V / {avg_i_normal}A."),
            "severity": sev,
        })

        # ── Card 2: Anomalous readings detail ─
        if n_anomalies > 0:
            insights.append({
                "title"   : f"⚡ Abnormal Readings Profile",
                "body_en" : (f"Anomalous readings averaged "
                             f"{avg_v_anomaly}V and {avg_i_anomaly}A — "
                             f"compared to normal averages of "
                             f"{avg_v_normal}V and {avg_i_normal}A. "
                             f"Check wiring or connected devices."),
                "body_sw" : (f"Matumizi yasiyo ya kawaida yalipima wastani wa "
                             f"{avg_v_anomaly}V na {avg_i_anomaly}A — "
                             f"ikilinganishwa na wastani wa kawaida wa "
                             f"{avg_v_normal}V na {avg_i_normal}A. "
                             f"Angalia nyaya au vifaa vilivyounganishwa."),
                "severity": "warn",
            })

        # ── Card 3: Power prediction ──────────
        insights.append({
            "title"   : f"📈 Power Consumption Prediction",
            "body_en" : (f"Based on your normal voltage ({avg_v_normal}V) "
                         f"and current ({avg_i_normal}A), "
                         f"predicted power consumption is "
                         f"{predicted_power}W."),
            "body_sw" : (f"Kulingana na volti yako ya kawaida ({avg_v_normal}V) "
                         f"na mkondo ({avg_i_normal}A), "
                         f"matumizi ya nguvu yanayotabiriwa ni "
                         f"{predicted_power}W."),
            "severity": "info",
        })

        return jsonify({
            "total_rows"    : total,
            "anomaly_count" : n_anomalies,
            "anomaly_pct"   : pct,
            "insights"      : insights,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# ROUTE: POST /api/energy/cost
# ─────────────────────────────────────────────
@app.route("/api/energy/cost", methods=["POST"])
def set_cost():
    payload = request.get_json(silent=True)
    if not payload or "cost_per_kwh" not in payload:
        return jsonify({"error": "JSON body with 'cost_per_kwh' key required."}), 400
    try:
        c = float(payload["cost_per_kwh"])
        if not (0.01 <= c <= 100000):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "cost_per_kwh must be a positive number."}), 400
    ENERGY_STATE["cost_per_kwh"] = c
    logger.info(f"  [ENERGY] Cost updated → KSh {c}/kWh")
    return jsonify({"cost_per_kwh": c, "message": f"Cost set to KSh {c}/kWh."}), 200


# ─────────────────────────────────────────────
# ROUTE: POST /api/arduino/config
# ─────────────────────────────────────────────
@app.route("/api/arduino/config", methods=["POST"])
def arduino_config():
    """
    Update Arduino connection settings at runtime.
    Payload: {"port": "COM3", "baud": 9600, "channel": 1}
    Reconnects automatically after update.
    """
    global arduino, ARDUINO_CONNECTED
    payload = request.get_json(silent=True) or {}

    if "port"    in payload: CFG["arduino_port"]    = payload["port"]
    if "baud"    in payload: CFG["arduino_baud"]    = int(payload["baud"])
    if "channel" in payload: CFG["arduino_channel"] = int(payload["channel"])

    # Close existing connection
    if arduino and arduino.is_open:
        arduino.close()

    # Reconnect
    arduino = init_arduino()
    ARDUINO_CONNECTED = arduino is not None

    return jsonify({
        "arduino_connected": ARDUINO_CONNECTED,
        "port"    : CFG["arduino_port"],
        "baud"    : CFG["arduino_baud"],
        "channel" : CFG["arduino_channel"],
        "message" : "Connected" if ARDUINO_CONNECTED else "Not found — check port and cable",
    }), 200


# ─────────────────────────────────────────────
# ROUTE: GET /api/arduino/ports
# ─────────────────────────────────────────────
@app.route("/api/arduino/ports", methods=["GET"])
def list_ports():
    """List all available serial ports — useful for finding the Arduino COM port."""
    ports = [
        {
            "device"     : p.device,
            "description": p.description,
            "hwid"       : p.hwid,
        }
        for p in serial.tools.list_ports.comports()
    ]
    return jsonify({"ports": ports, "count": len(ports)}), 200


# ─────────────────────────────────────────────
# ROUTE: GET /api/status
# ─────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def status():
    with get_db() as conn:
        total_logs = conn.execute(
            "SELECT COUNT(*) FROM system_logs").fetchone()[0]
    return jsonify({
        "server"    : "Smart Home API Gateway",
        "model"     : "bert-base-multilingual-cased (fine-tuned)",
        "pin"       : f"Ch{CFG['arduino_channel']}",
        "pin_state" : RELAY_STATE["value"],
        "bulb"      : "ON" if RELAY_STATE["value"] else "OFF",
        "uptime_s"  : int(time.time() - START_TIME),
        "device"    : str(torch_device).upper(),
        "arduino_connected": ARDUINO_CONNECTED,
        "arduino_port"     : CFG["arduino_port"] or "auto-detect",
        "arduino_channel"  : CFG["arduino_channel"],
        "total_logs": total_logs,
        "wattage_w" : ENERGY_STATE["wattage"],
        "whisper"   : CFG["whisper_model"],
    }), 200

# ─────────────────────────────────────────────
# ROUTE: POST /api/ml/detect
# ─────────────────────────────────────────────
@app.route("/api/ml/detect", methods=["POST"])
def ml_detect():
    if not ML_READY:
        return jsonify({"error": "ML model not available. Check energy.csv."}), 503
    payload = request.get_json(silent=True)
    if not payload or "voltage" not in payload or "current" not in payload:
        return jsonify({"error": "Provide 'voltage' and 'current'."}), 400
    try:
        voltage = float(payload["voltage"])
        current = float(payload["current"])
        power   = voltage * current
        score   = float(iso_model.decision_function([[voltage, current, power]])[0])
        anomaly = int(iso_model.predict([[voltage, current, power]])[0])
        return jsonify({
            "voltage": voltage,
            "current": current,
            "power"  : round(power, 2),
            "anomaly": anomaly,
            "score"  : round(score, 4),
            "status" : "ANOMALY" if anomaly == -1 else "NORMAL",
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ROUTE: POST /api/ml/predict
# ─────────────────────────────────────────────
@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    if not ML_READY:
        return jsonify({"error": "ML model not available. Check energy.csv."}), 503
    payload = request.get_json(silent=True)
    if not payload or "voltage" not in payload or "current" not in payload:
        return jsonify({"error": "Provide 'voltage' and 'current'."}), 400
    try:
        voltage         = float(payload["voltage"])
        current         = float(payload["current"])
        predicted_power = float(reg_model.predict([[voltage, current]])[0])
        return jsonify({
            "voltage"        : voltage,
            "current"        : current,
            "predicted_power": round(predicted_power, 2),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ROUTE: GET /api/ml/scan
# ─────────────────────────────────────────────
@app.route("/api/ml/scan", methods=["GET"])
def ml_scan():
    if not ML_READY:
        return jsonify({"error": "ML model not available. Check energy.csv."}), 503
    try:
        _df                  = pd.read_csv("energy.csv").dropna()
        _df["power"]         = _df["voltage"] * _df["current"]
        _df["anomaly"]       = iso_model.predict(_df[["voltage", "current", "power"]])
        _df["anomaly_score"] = iso_model.decision_function(
                                   _df[["voltage", "current", "power"]])
        anomalies = _df[_df["anomaly"] == -1].to_dict(orient="records")
        return jsonify({
            "total_rows"   : len(_df),
            "anomaly_count": len(anomalies),
            "anomalies"    : anomalies,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ─────────────────────────────────────────────
# ROUTE: GET /  (Dashboard)
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Smart Home API Gateway  –  Arduino Mode")
    logger.info(f"  http://127.0.0.1:{CFG['port']}")
    logger.info("  Endpoints:")
    logger.info(f"    POST /api/command")
    logger.info(f"    POST /api/voice")
    logger.info(f"    GET  /api/energy")
    logger.info(f"    POST /api/energy/wattage")
    logger.info(f"    GET  /api/status")
    logger.info(f"    GET  /api/logs")
    logger.info(f"    GET  /api/tips")
    logger.info(f"    POST /api/energy/cost")
    logger.info(f"    GET  /api/arduino/ports")
    logger.info(f"    POST /api/arduino/config")
    logger.info(f"    GET  / (Dashboard)")
    logger.info(f"    POST /api/ml/detect")
    logger.info(f"    POST /api/ml/predict")
    logger.info(f"    GET  /api/ml/scan")
    logger.info("=" * 55)
    app.run(host=CFG["host"], port=CFG["port"], debug=False)
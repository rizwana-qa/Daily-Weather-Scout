
import os
import re
import sys
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

CITY = os.getenv("WEATHER_CITY", "Dubai")

CITIES = {
    "Lahore": (31.52, 74.32),
    "Karachi": (24.86, 67.01),
    "Islamabad": (33.69, 73.05),
    "Dubai": (25.20, 55.27),
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

GUIDELINES_PATH = Path(__file__).resolve().parent / "guidelines.md"

SHEET_URL = os.getenv("SHEET_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MAX_STEPS = 1

# External emails remain disabled until explicitly authorized.
AUTO_APPROVE_ALERTS = (
    os.getenv("AUTO_APPROVE_ALERTS", "false").lower() == "true"
)

# These two settings must both be enabled before an email
# can be requested through the existing Apps Script.
ENABLE_EMAIL_ALERTS = (
    os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
)


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def require_configuration():
    if CITY not in CITIES:
        raise ValueError(
            f"Unsupported city: {CITY}. Allowed cities: {list(CITIES)}"
        )

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    if not SHEET_URL:
        raise ValueError("SHEET_URL is not configured.")


def safe_number(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)) and math.isfinite(value):
        return value

    return None


def average(values):
    valid = [v for v in values if safe_number(v) is not None]

    if not valid:
        return None

    return round(sum(valid) / len(valid), 1)


def local_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def json_text(value):
    return json.dumps(value, ensure_ascii=False, default=str)


# ============================================================
# 3. WEATHER DATA
# ============================================================

def get_weather(city):
    if city not in CITIES:
        raise ValueError(f"City not allowed: {city}")

    lat, lon = CITIES[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "visibility"
        ),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "visibility"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "wind_speed_10m_max,"
            "wind_gusts_10m_max,"
            "precipitation_sum"
        ),
        "past_days": 5,
        "forecast_days": 2,
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    raw = response.json()

    if not all(k in raw for k in ("current", "hourly", "daily")):
        raise ValueError("Weather API returned incomplete data.")

    current = raw["current"]
    hourly = raw["hourly"]
    daily = raw["daily"]

    current_time = local_datetime(current.get("time"))

    if current_time is None:
        raise ValueError("Weather API returned an invalid current timestamp.")

    today = current_time.date()

    # Previous completed days, excluding today.
    previous_days = []

    for i, date_value in enumerate(daily.get("time", [])):
        date = datetime.fromisoformat(date_value).date()

        if date >= today:
            continue

        previous_days.append({
            "date": date_value,
            "high": daily["temperature_2m_max"][i],
            "low": daily["temperature_2m_min"][i],
            "wind": daily["wind_speed_10m_max"][i],
            "gust": daily["wind_gusts_10m_max"][i],
            "precipitation": daily["precipitation_sum"][i],
        })

    previous_days = previous_days[-5:]

    # Upcoming 24 hours, based on the API's local timestamps.
    hourly_forecast = []

    for i, time_value in enumerate(hourly.get("time", [])):
        forecast_time = local_datetime(time_value)

        if forecast_time is None:
            continue

        if not current_time <= forecast_time < current_time + timedelta(hours=24):
            continue

        hourly_forecast.append({
            "time": time_value,
            "temperature": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "apparent_temperature": hourly["apparent_temperature"][i],
            "precipitation": hourly["precipitation"][i],
            "wind": hourly["wind_speed_10m"][i],
            "gust": hourly["wind_gusts_10m"][i],
            "visibility": hourly["visibility"][i],
        })

    today_index = None

    for i, date_value in enumerate(daily.get("time", [])):
        if datetime.fromisoformat(date_value).date() == today:
            today_index = i
            break

    today_forecast = None

    if today_index is not None:
        today_forecast = {
            "high": daily["temperature_2m_max"][today_index],
            "low": daily["temperature_2m_min"][today_index],
            "wind": daily["wind_speed_10m_max"][today_index],
            "gust": daily["wind_gusts_10m_max"][today_index],
            "precipitation": daily["precipitation_sum"][today_index],
        }

    if not hourly_forecast and today_forecast is None:
        raise ValueError("No usable current or forecast weather data.")

    return {
        "city": city,
        "coordinates": {
            "latitude": lat,
            "longitude": lon,
        },
        "timezone": raw.get("timezone"),
        "source": "Open-Meteo",
        "source_type": "Weather API, not an official UAE warning source",
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "current": current,
        "today_forecast": today_forecast,
        "previous_days": previous_days,
        "hourly_forecast_next_24h": hourly_forecast,
        "history_summary": {
            "average_high": average([d["high"] for d in previous_days]),
            "average_low": average([d["low"] for d in previous_days]),
            "peak_wind": max(
                [d["wind"] for d in previous_days
                 if safe_number(d["wind"]) is not None],
                default=None,
            ),
        },
    }


# ============================================================
# 4. GROQ API
# ============================================================

def ask_groq(messages, temperature=0.1):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": "Bearer " + GROQ_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=90,
    )

    if response.status_code != 200:
        raise RuntimeError("Groq API error: " + response.text[:1000])

    result = response.json()

    return result["choices"][0]["message"]["content"]


def parse_json_response(text):
    text = (text or "").strip()

    text = text.removeprefix("```json")
    text = text.removeprefix("```")
    text = text.removesuffix("```").strip()

    return json.loads(text)


# ============================================================
# 5. GUIDELINES RETRIEVAL
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "for", "on", "at", "or", "when", "what",
    "do", "should", "with", "from", "by", "be",
}


def words(text):
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    ]


def load_guidelines():
    print("\nGUIDELINES CHECK")

    print("Expected path:", GUIDELINES_PATH)
    print("File exists:", GUIDELINES_PATH.is_file())

    if not GUIDELINES_PATH.is_file():
        raise FileNotFoundError(
            f"guidelines.md not found at {GUIDELINES_PATH}"
        )

    content = GUIDELINES_PATH.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError("guidelines.md is empty.")

    chunks = {}

    for block in re.split(r"\n(?=## )", content):
        block = block.strip()

        if block.startswith("## "):
            title, _, body = block[3:].partition("\n")

            chunks[title.strip()] = body.strip()

    if not chunks:
        raise ValueError(
            "No Markdown sections found. Expected headings beginning with ##"
        )

    policy_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    print("Sections loaded:", len(chunks))
    print("Policy SHA256:", policy_hash)

    return {
        "content": content,
        "sections": chunks,
        "version": "2.0",
        "hash": policy_hash,
        "path": str(GUIDELINES_PATH),
    }


def search_guidelines(query, k=5):
    policy = load_guidelines()
    chunks = policy["sections"]

    query_words = set(words(query))

    documents = {
        title: words(title + " " + body)
        for title, body in chunks.items()
    }

    number_of_documents = len(documents)
    scores = {}

    for title, document_words in documents.items():
        score = 0.0

        for word in query_words:
            frequency = document_words.count(word)

            if frequency:
                document_frequency = sum(
                    1
                    for other in documents.values()
                    if word in other
                )

                score += (
                    (1 + math.log(frequency))
                    * (math.log(
                        (number_of_documents + 1)
                        / document_frequency
                    ) + 1)
                )

        scores[title] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # Core policy sections are always included.
    mandatory_terms = [
        "purpose and governing principles",
        "evidence quality and confidence",
        "risk assessment dimensions",
        "risk calculation model",
        "alert decision framework",
        "official warning override",
        "data quality and assessment completeness",
        "human approval and notification governance",
        "final decision rule",
    ]

    selected = []

    for term in mandatory_terms:
        for title in chunks:
            if term in title.lower() and title not in selected:
                selected.append(title)

    for title, score in ranked:
        if score > 0 and title not in selected:
            selected.append(title)

        if len(selected) >= len(mandatory_terms) + k:
            break

    results = [
        {
            "section": title,
            "text": chunks[title],
        }
        for title in selected
    ]

    print("Retrieved policy sections:", [r["section"] for r in results])

    return results


# ============================================================
# 6. GOOGLE SHEETS HISTORY
# ============================================================

def tool_get_history(n=5):
    if not SHEET_URL:
        return {"error": "SHEET_URL is not configured."}

    try:
        response = requests.get(
            SHEET_URL,
            params={"action": "history", "n": min(max(int(n), 1), 10)},
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return {"error": "Unexpected history response format."}

        return data

    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}


# ============================================================
# 7. RISK CALCULATION
# ============================================================

def calculate_risk_score(
    likelihood,
    severity,
    exposure,
    duration,
    geographic_impact,
):
    dimensions = [
        likelihood,
        severity,
        exposure,
        duration,
        geographic_impact,
    ]

    if any(type(value) is not int or not 1 <= value <= 5
           for value in dimensions):
        return None, "UNASSESSED"

    score = (
        0.35 * likelihood * severity
        + 0.20 * exposure * severity
        + 0.15 * duration * severity
        + 0.30 * geographic_impact * severity
    )

    score = round(score, 2)

    if score <= 5:
        band = "LOW"
    elif score <= 10:
        band = "MODERATE"
    elif score <= 15:
        band = "HIGH"
    else:
        band = "CRITICAL"

    return score, band


# ============================================================
# 8. ASSESSMENT VALIDATION
# ============================================================

VALID_RECOMMENDATIONS = {
    "NO_ALERT_RECOMMENDED",
    "MONITOR",
    "ADVISORY_RECOMMENDED",
    "ESCALATE_FOR_REVIEW",
    "OFFICIAL_WARNING_PASS_THROUGH",
    "UNABLE_TO_ASSESS",
}


def validate_decision(decision, retrieved_sections):
    if not isinstance(decision, dict):
        raise ValueError("AI returned an invalid assessment.")

    required = [
        "hazard_type",
        "likelihood_score",
        "likelihood_reason",
        "severity_score",
        "severity_reason",
        "exposure_score",
        "exposure_reason",
        "duration_score",
        "duration_reason",
        "geographic_impact_score",
        "geographic_impact_reason",
        "alert_recommendation",
        "advice",
        "reasoning",
        "recommended_actions",
        "limitations",
        "missing_evidence",
        "sources",
    ]

    missing = [key for key in required if key not in decision]

    if missing:
        raise ValueError("Missing assessment fields: " + ", ".join(missing))

    score, band = calculate_risk_score(
        decision["likelihood_score"],
        decision["severity_score"],
        decision["exposure_score"],
        decision["duration_score"],
        decision["geographic_impact_score"],
    )

    if score is None:
        raise ValueError(
            "One or more risk dimensions are missing or outside 1-5."
        )

    if decision["alert_recommendation"] not in VALID_RECOMMENDATIONS:
        raise ValueError("Invalid alert recommendation.")

    if not isinstance(decision["sources"], list):
        raise ValueError("Assessment sources must be a list.")

    available_titles = {
        section["section"]
        for section in retrieved_sections
    }

    valid_sources = [
        source
        for source in decision["sources"]
        if isinstance(source, str) and source in available_titles
    ]

    if not valid_sources:
        raise ValueError(
            "Assessment did not identify valid retrieved policy sections."
        )

    for field in (
        "likelihood_reason",
        "severity_reason",
        "exposure_reason",
        "duration_reason",
        "geographic_impact_reason",
    ):
        if not isinstance(decision[field], str) or not decision[field].strip():
            raise ValueError(f"Missing rationale: {field}")

    if not isinstance(decision["advice"], str):
        raise ValueError("Invalid advice.")

    if not isinstance(decision["reasoning"], str):
        raise ValueError("Invalid reasoning.")

    return {
        **decision,
        "sources": valid_sources,
        "risk_score": score,
        "risk_band": band,
        "assessment_status": "COMPLETE",
    }


# ============================================================
# 9. WEATHER ASSESSMENT
# ============================================================

def assess_weather(weather, history, policy, retrieved_sections):
    context = {
        "city": CITY,
        "weather": weather,
        "recent_agent_history": history,
        "policy_name": "Dubai Weather Safety Guidelines",
        "policy_version": policy["version"],
        "policy_hash": policy["hash"],
        "retrieved_guideline_sections": retrieved_sections,
        "official_warning_status": (
            "NOT_CHECKED: No NCM or other official warning feed "
            "is currently integrated."
        ),
    }

    system_prompt = """
You are an evidence-based weather safety assessment agent.

You must follow the supplied weather safety policy.

Do not invent measurements, official warnings, regulatory requirements,
risk thresholds or geographic information.

Use only the supplied weather data and guideline sections.

IMPORTANT REQUIREMENTS:

1. Assess the actual weather conditions, not generic seasonal advice.
2. Identify the relevant hazard, if any.
3. Evaluate all five risk dimensions using the policy's 1-5 anchors.
4. Provide a separate evidence-based rationale for each dimension.
5. Do not treat missing evidence as a low score.
6. Do not claim that an official warning has been checked.
7. Official warning status is NOT_CHECKED unless authoritative evidence
   is explicitly supplied.
8. Do not describe Open-Meteo as an official UAE warning authority.
9. Distinguish current measurements, historical observations and forecasts.
10. Consider exposure and duration. Do not assume all Dubai residents
    have the same exposure.
11. Use recent history only as supporting context. A previous alert
    must not suppress a materially different hazard.
12. Do not recommend an alert merely because a numerical measurement
    is higher than usual.
13. Use the exact recommendation values defined in the policy.
14. Explain missing evidence and limitations.
15. Return JSON only.

The five risk dimensions must be integers from 1 to 5.
If a dimension cannot be assessed, do not invent a score.

Return exactly these keys:

hazard_type
likelihood_score
likelihood_reason
severity_score
severity_reason
exposure_score
exposure_reason
duration_score
duration_reason
geographic_impact_score
geographic_impact_reason
alert_recommendation
advice
reasoning
recommended_actions
limitations
missing_evidence
sources

The sources field must contain exact section titles from the supplied
retrieved guideline sections.
"""

    user_prompt = (
        "Assess the following weather situation using the supplied policy.\n\n"
        + json_text(context)
    )

    response = ask_groq([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    decision = parse_json_response(response)

    return validate_decision(decision, retrieved_sections)


# ============================================================
# 10. SAFE INCOMPLETE ASSESSMENT
# ============================================================

def incomplete_assessment(reason):
    return {
        "hazard_type": "UNKNOWN",
        "likelihood_score": None,
        "likelihood_reason": "Not assessed.",
        "severity_score": None,
        "severity_reason": "Not assessed.",
        "exposure_score": None,
        "exposure_reason": "Not assessed.",
        "duration_score": None,
        "duration_reason": "Not assessed.",
        "geographic_impact_score": None,
        "geographic_impact_reason": "Not assessed.",
        "risk_score": None,
        "risk_band": "UNASSESSED",
        "assessment_status": "INCOMPLETE",
        "alert_recommendation": "UNABLE_TO_ASSESS",
        "advice": "A reliable safety assessment could not be completed.",
        "reasoning": reason,
        "recommended_actions": [],
        "limitations": [reason],
        "missing_evidence": [],
        "sources": [],
    }


# ============================================================
# 11. MAIN AGENT
# ============================================================

def run_weather_agent():
    used = []

    try:
        require_configuration()

        # Policy retrieval is mandatory.
        policy = load_guidelines()
        used.append("load_guidelines")

        # Weather retrieval is mandatory.
        weather = get_weather(CITY)
        used.append("get_weather")

        # Retrieve recent history before making a decision.
        history = tool_get_history(5)
        used.append("get_history")

        # Retrieve relevant and mandatory policy sections.
        query = (
            "Dubai weather safety heat wind rain thunderstorms "
            "cold exposure duration likelihood severity geographic "
            "impact risk calculation evidence quality alert decision "
            "official warning and human approval"
        )

        retrieved_sections = search_guidelines(query)
        used.append("search_guidelines")

        if not retrieved_sections:
            raise ValueError("No policy sections were retrieved.")

        if isinstance(history, dict) and "error" in history:
            print("History unavailable:", history["error"])
            history = []

        decision = assess_weather(
            weather,
            history,
            policy,
            retrieved_sections,
        )

        # Record the fact that official warning verification
        # is not implemented in the current architecture.
        decision["official_warning_status"] = "NOT_CHECKED"
        decision["official_warning_authority"] = None

        # Open-Meteo is a single supporting weather data source.
        decision["evidence_quality"] = "E3"
        decision["evidence_confidence"] = "LOW"

        decision["policy_status"] = "AVAILABLE"
        decision["policy_version"] = policy["version"]
        decision["policy_hash"] = policy["hash"]
        decision["policy_path"] = policy["path"]

        decision["weather_source"] = weather["source"]
        decision["weather_retrieval_status"] = "AVAILABLE"
        decision["weather_data_timestamp"] = weather["retrieved_at"]

        decision["assessment_timestamp"] = datetime.now().astimezone().isoformat()
        decision["city"] = CITY
        decision["timezone"] = weather["timezone"]
        decision["tools_used"] = used

        decision["weather_summary"] = weather
        decision["history_checked"] = True

        # A completed internal assessment is not proof that
        # official warnings or regulatory instructions were checked.
        decision["limitations"] = list(decision["limitations"])
        decision["limitations"].append(
            "Official NCM and emergency authority warnings were not "
            "retrieved or verified by this implementation."
        )

        # External emails are disabled unless explicitly enabled.
        authorized = AUTO_APPROVE_ALERTS and ENABLE_EMAIL_ALERTS

        decision["approval_status"] = (
            "APPROVED" if authorized else "PENDING_REVIEW"
        )

        decision["send_alert"] = (
            authorized
            and decision["assessment_status"] == "COMPLETE"
            and decision["alert_recommendation"] in {
                "ADVISORY_RECOMMENDED",
                "OFFICIAL_WARNING_PASS_THROUGH",
            }
        )

        decision["alert_message"] = (
            decision["advice"][:300]
            if decision["send_alert"]
            else ""
        )

        decision["notification_status"] = (
            "PENDING" if decision["send_alert"] else "NOT_SENT"
        )

        return decision, used

    except Exception as exc:
        print("\nASSESSMENT ERROR:", str(exc))

        decision = incomplete_assessment(str(exc))

        decision["policy_status"] = (
            "UNAVAILABLE"
            if isinstance(exc, (FileNotFoundError, ValueError))
            and "guideline" in str(exc).lower()
            else "UNKNOWN"
        )

        decision["policy_version"] = None
        decision["policy_hash"] = None
        decision["policy_path"] = str(GUIDELINES_PATH)

        decision["official_warning_status"] = "NOT_CHECKED"
        decision["evidence_quality"] = "E0"
        decision["evidence_confidence"] = "NONE"

        decision["approval_status"] = "NOT_AUTHORIZED"
        decision["send_alert"] = False
        decision["alert_message"] = ""
        decision["notification_status"] = "NOT_SENT"

        decision["assessment_timestamp"] = datetime.now().astimezone().isoformat()
        decision["city"] = CITY
        decision["tools_used"] = used

        return decision, used


# ============================================================
# 12. GOOGLE SHEETS INTEGRATION
# ============================================================

def run_agent_job(dry=False):
    decision, used = run_weather_agent()

    print("\n========== WEATHER ASSESSMENT ==========")

    print("CITY:", decision.get("city"))
    print("POLICY:", decision.get("policy_status"))
    print("POLICY VERSION:", decision.get("policy_version"))
    print("ASSESSMENT:", decision.get("assessment_status"))
    print("HAZARD:", decision.get("hazard_type"))
    print("RISK SCORE:", decision.get("risk_score"))
    print("RISK BAND:", decision.get("risk_band"))
    print("RECOMMENDATION:", decision.get("alert_recommendation"))

    print("\nADVICE:", decision.get("advice"))
    print("REASONING:", decision.get("reasoning"))

    print("\nRISK DIMENSIONS:")

    for dimension in (
        "likelihood",
        "severity",
        "exposure",
        "duration",
        "geographic_impact",
    ):
        print(
            dimension.upper() + ":",
            decision.get(dimension + "_score"),
            "|",
            decision.get(dimension + "_reason"),
        )

    print("\nSOURCES:", decision.get("sources"))
    print("ALERT AUTHORIZED:", decision.get("send_alert"))
    print("NOTIFICATION:", decision.get("notification_status"))

    # Preserve existing Apps Script field names.
    payload = {
        "type": "agent",
        "city": CITY,
        "tools_used": ", ".join(used),

        "advice": decision.get("advice", ""),
        "reasoning": decision.get("reasoning", ""),

        "send_alert": decision.get("send_alert") is True,
        "alert_message": decision.get("alert_message", ""),

        "sources": ", ".join(decision.get("sources", [])),

        # New fields. Current Apps Script ignores these until updated.
        "policy_status": decision.get("policy_status"),
        "policy_version": decision.get("policy_version"),
        "policy_hash": decision.get("policy_hash"),
        "assessment_status": decision.get("assessment_status"),

        "hazard_type": decision.get("hazard_type"),
        "likelihood_score": decision.get("likelihood_score"),
        "likelihood_reason": decision.get("likelihood_reason"),

        "severity_score": decision.get("severity_score"),
        "severity_reason": decision.get("severity_reason"),

        "exposure_score": decision.get("exposure_score"),
        "exposure_reason": decision.get("exposure_reason"),

        "duration_score": decision.get("duration_score"),
        "duration_reason": decision.get("duration_reason"),

        "geographic_impact_score": decision.get("geographic_impact_score"),
        "geographic_impact_reason": decision.get("geographic_impact_reason"),

        "risk_score": decision.get("risk_score"),
        "risk_band": decision.get("risk_band"),

        "evidence_quality": decision.get("evidence_quality"),
        "evidence_confidence": decision.get("evidence_confidence"),

        "official_warning_status": decision.get("official_warning_status"),
        "alert_recommendation": decision.get("alert_recommendation"),

        "assessment_timestamp": decision.get("assessment_timestamp"),
        "approval_status": decision.get("approval_status"),
        "notification_status": decision.get("notification_status"),

        "limitations": json_text(decision.get("limitations", [])),
        "missing_evidence": json_text(decision.get("missing_evidence", [])),
    }

    if dry:
        print("\nDRY RUN: Nothing was sent to Google Sheets or email.")
        print(json_text(payload))
        return

    if not SHEET_URL:
        raise SystemExit("SHEET_URL is missing.")

    response = requests.post(
        SHEET_URL,
        json=payload,
        timeout=30,
    )

    print("\nSHEET RESPONSE:", response.status_code, response.text)

    response.raise_for_status()

    if response.text.strip() != "OK":
        raise SystemExit("Google Apps Script did not confirm success.")


# ============================================================
# 13. DETERMINISTIC TESTS
# ============================================================

def run_tests():
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed

        if condition:
            passed += 1
            print("PASS:", name)
        else:
            failed += 1
            print("FAIL:", name)

    # Risk scoring tests.
    score, band = calculate_risk_score(1, 1, 1, 1, 1)

    check(
        "Minimum valid risk score",
        score == 1.0 and band == "LOW",
    )

    score, band = calculate_risk_score(5, 5, 5, 5, 5)

    check(
        "Maximum valid risk score",
        score == 25.0 and band == "CRITICAL",
    )

    score, band = calculate_risk_score(3, 3, 3, 3, 3)

    check(
        "Moderate risk score calculation",
        score == 9.0 and band == "MODERATE",
    )

    score, band = calculate_risk_score(0, 3, 3, 3, 3)

    check(
        "Invalid risk dimension rejected",
        score is None and band == "UNASSESSED",
    )

    score, band = calculate_risk_score(6, 3, 3, 3, 3)

    check(
        "Out-of-range dimension rejected",
        score is None and band == "UNASSESSED",
    )

    # Guidelines retrieval tests.
    try:
        policy = load_guidelines()

        check(
            "Guidelines file exists and is readable",
            bool(policy["content"]),
        )

        check(
            "Guidelines sections loaded",
            len(policy["sections"]) > 0,
        )

        check(
            "Policy hash generated",
            len(policy["hash"]) == 64,
        )

        results = search_guidelines(
            "heat exposure risk assessment alert",
        )

        check(
            "Relevant guideline sections retrieved",
            len(results) > 0,
        )

    except Exception as exc:
        print("Guidelines test error:", exc)

        check("Guidelines retrieval", False)

    print("\n========== TEST RESULTS ==========")
    print("Passed:", passed)
    print("Failed:", failed)
    print("Total:", passed + failed)

    if failed:
        raise SystemExit(1)


# ============================================================
# 14. COMMAND LINE
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "agent"

    if mode == "search":
        query = " ".join(sys.argv[2:])

        for result in search_guidelines(query):
            print("\nSECTION:", result["section"])
            print(result["text"])

    elif mode == "test":
        run_tests()

    elif mode == "agent-dry":
        run_agent_job(dry=True)

    elif mode == "agent":
        run_agent_job()

    else:
        print("Available commands:")
        print("python weather_log.py search heat risk")
        print("python weather_log.py test")
        print("python weather_log.py agent-dry")
        print("python weather_log.py agent")

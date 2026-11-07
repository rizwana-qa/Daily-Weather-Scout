import os
import re
import sys
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

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

WEATHER_TIMEOUT = 30
GROQ_TIMEOUT = 90
SHEET_TIMEOUT = 30

# Conservative request limits for the current 8,000 TPM environment.
# These are safeguards, not a guarantee against provider-side rate limits.
MAX_COMPLETION_TOKENS = 700
MAX_ESTIMATED_REQUEST_TOKENS = 5000
MAX_REQUEST_CHARS = 9000
MAX_POLICY_CHARS = 5000
MAX_HISTORY_RECORDS = 3

# External emails remain disabled unless both variables are explicitly true.
AUTO_APPROVE_ALERTS = (
    os.getenv("AUTO_APPROVE_ALERTS", "false").lower() == "true"
)

ENABLE_EMAIL_ALERTS = (
    os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
)

HTTP = requests.Session()


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
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def compact_text(value, limit=250):
    if value is None:
        return ""

    text = re.sub(r"\s+", " ", str(value)).strip()

    if len(text) > limit:
        return text[:limit].rstrip() + "..."

    return text


def estimate_tokens(text):
    """
    Conservative approximation for mixed English JSON content.

    This is not an exact tokenizer. The request is also protected by
    a character limit and a maximum completion-token allowance.
    """
    return math.ceil(len(text) / 2.5)


# ============================================================
# 3. WEATHER DATA
# ============================================================

def get_weather(city):
    if city not in CITIES:
        raise ValueError(f"City not allowed: {city}")

    latitude, longitude = CITIES[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
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

    response = HTTP.get(
        url,
        params=params,
        timeout=WEATHER_TIMEOUT,
    )
    response.raise_for_status()

    raw = response.json()

    if not isinstance(raw, dict):
        raise ValueError("Weather API returned an invalid response.")

    for required_key in ("current", "hourly", "daily"):
        if not isinstance(raw.get(required_key), dict):
            raise ValueError(
                f"Weather API response is missing '{required_key}'."
            )

    current = raw["current"]
    hourly = raw["hourly"]
    daily = raw["daily"]

    current_time = local_datetime(current.get("time"))

    if current_time is None:
        raise ValueError("Weather API returned an invalid current timestamp.")

    today = current_time.date()

    # Previous completed days.
    previous_days = []

    daily_times = daily.get("time", [])

    for index, date_value in enumerate(daily_times):
        parsed_date = local_datetime(date_value)

        if parsed_date is None or parsed_date.date() >= today:
            continue

        def daily_value(key):
            values = daily.get(key, [])

            if index >= len(values):
                return None

            return safe_number(values[index])

        previous_days.append({
            "date": date_value,
            "high": daily_value("temperature_2m_max"),
            "low": daily_value("temperature_2m_min"),
            "wind": daily_value("wind_speed_10m_max"),
            "gust": daily_value("wind_gusts_10m_max"),
            "precipitation": daily_value("precipitation_sum"),
        })

    previous_days = previous_days[-5:]

    # Upcoming 24 hours.
    hourly_forecast = []

    hourly_times = hourly.get("time", [])

    hourly_fields = {
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "apparent_temperature": "apparent_temperature",
        "precipitation": "precipitation",
        "wind": "wind_speed_10m",
        "gust": "wind_gusts_10m",
        "visibility": "visibility",
    }

    for index, time_value in enumerate(hourly_times):
        forecast_time = local_datetime(time_value)

        if forecast_time is None:
            continue

        if not (
            current_time
            <= forecast_time
            < current_time + timedelta(hours=24)
        ):
            continue

        record = {"time": time_value}

        for output_key, source_key in hourly_fields.items():
            values = hourly.get(source_key, [])

            record[output_key] = (
                safe_number(values[index])
                if index < len(values)
                else None
            )

        hourly_forecast.append(record)

    # Today's daily forecast.
    today_forecast = None

    for index, date_value in enumerate(daily_times):
        parsed_date = local_datetime(date_value)

        if parsed_date is None or parsed_date.date() != today:
            continue

        def today_value(key):
            values = daily.get(key, [])

            if index >= len(values):
                return None

            return safe_number(values[index])

        today_forecast = {
            "high": today_value("temperature_2m_max"),
            "low": today_value("temperature_2m_min"),
            "wind": today_value("wind_speed_10m_max"),
            "gust": today_value("wind_gusts_10m_max"),
            "precipitation": today_value("precipitation_sum"),
        }

        break

    if not hourly_forecast and today_forecast is None:
        raise ValueError("No usable current or forecast weather data.")

    previous_winds = [
        day["wind"]
        for day in previous_days
        if safe_number(day["wind"]) is not None
    ]

    return {
        "city": city,
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude,
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
            "average_high": average(
                [day["high"] for day in previous_days]
            ),
            "average_low": average(
                [day["low"] for day in previous_days]
            ),
            "peak_wind": max(previous_winds, default=None),
        },
    }


def summarize_weather(weather):
    """
    Reduce the weather payload while preserving useful hazard indicators.
    The full hourly and historical arrays are not sent to Groq.
    """
    hourly = weather.get("hourly_forecast_next_24h", [])

    def values_for(field):
        return [
            safe_number(record.get(field))
            for record in hourly
            if safe_number(record.get(field)) is not None
        ]

    temperatures = values_for("temperature")
    apparent = values_for("apparent_temperature")
    winds = values_for("wind")
    gusts = values_for("gust")
    precipitation = values_for("precipitation")
    visibility = values_for("visibility")

    hourly_summary = {
        "forecast_hours_available": len(hourly),
        "temperature_min": min(temperatures, default=None),
        "temperature_max": max(temperatures, default=None),
        "apparent_temperature_max": max(apparent, default=None),
        "wind_max": max(winds, default=None),
        "gust_max": max(gusts, default=None),
        "precipitation_total": (
            round(sum(precipitation), 2)
            if precipitation else None
        ),
        "precipitation_hourly_max": max(
            precipitation,
            default=None,
        ),
        "visibility_min": min(visibility, default=None),
    }

    # Keep only the latest three completed days.
    recent_days = weather.get("previous_days", [])[-3:]

    return {
        "city": weather["city"],
        "timezone": weather.get("timezone"),
        "source": weather["source"],
        "retrieved_at": weather["retrieved_at"],
        "current": weather["current"],
        "today_forecast": weather.get("today_forecast"),
        "next_24_hours": hourly_summary,
        "recent_completed_days": recent_days,
        "historical_summary": weather.get("history_summary"),
    }


# ============================================================
# 4. GROQ API
# ============================================================

def ask_groq(messages, temperature=0.1):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    serialized_messages = json_text(messages)
    estimated_input_tokens = estimate_tokens(serialized_messages)

    estimated_total = (
        estimated_input_tokens + MAX_COMPLETION_TOKENS
    )

    print("\nGROQ REQUEST CHECK")
    print("Estimated input tokens:", estimated_input_tokens)
    print("Completion token allowance:", MAX_COMPLETION_TOKENS)
    print("Estimated total tokens:", estimated_total)

    if len(serialized_messages) > MAX_REQUEST_CHARS:
        raise ValueError(
            "Groq request blocked: prompt exceeds the configured "
            f"{MAX_REQUEST_CHARS}-character limit."
        )

    if estimated_total > MAX_ESTIMATED_REQUEST_TOKENS:
        raise ValueError(
            "Groq request blocked: estimated token usage exceeds "
            f"the configured {MAX_ESTIMATED_REQUEST_TOKENS}-token budget."
        )

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": "low",
        "reasoning_format": "hidden",
        "response_format": {"type": "json_object"},
    }

    response = HTTP.post(
        GROQ_URL,
        headers={
            "Authorization": "Bearer " + GROQ_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=GROQ_TIMEOUT,
    )

    if response.status_code != 200:
        # Do not automatically retry rate-limit errors.
        # Retrying immediately can make TPM exhaustion worse.
        error_body = compact_text(response.text, 700)

        raise RuntimeError(
            f"Groq API returned HTTP {response.status_code}: {error_body}"
        )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Groq returned an unexpected response structure."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq returned an empty response.")

    return content


def parse_json_response(text):
    text = (text or "").strip()

    text = text.removeprefix("```json")
    text = text.removeprefix("```")
    text = text.removesuffix("```").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Groq returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Groq response must be a JSON object.")

    return result


# ============================================================
# 5. GUIDELINES LOADING AND RETRIEVAL
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

        if not block.startswith("## "):
            continue

        title, _, body = block[3:].partition("\n")
        title = title.strip()

        if title:
            chunks[title] = body.strip()

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


# Each entry lists acceptable heading phrases for a core policy area.
# These are matched against the actual headings in guidelines.md.
MANDATORY_SECTION_ALIASES = {
    "purpose and governing principles": (
        "purpose and governing principles",
        "purpose",
        "governing principles",
    ),
    "evidence quality and confidence": (
        "evidence quality and confidence",
        "evidence quality",
        "confidence",
    ),
    "risk assessment dimensions": (
        "risk assessment dimensions",
        "risk dimensions",
        "assessment dimensions",
    ),
    "risk calculation model": (
        "risk calculation model",
        "risk calculation",
        "risk score",
    ),
    "alert decision framework": (
        "alert decision framework",
        "alert decision",
        "recommendation framework",
    ),
    "official warning override": (
        "official warning override",
        "official warning",
        "warning override",
    ),
    "data quality and assessment completeness": (
        "data quality and assessment completeness",
        "data quality",
        "assessment completeness",
    ),
    "human approval and notification governance": (
        "human approval and notification governance",
        "human approval",
        "notification governance",
    ),
    "final decision rule": (
        "final decision rule",
        "decision rule",
    ),
}


def find_mandatory_sections(chunks):
    selected = []
    missing = []

    normalized_titles = {
        title: " ".join(words(title))
        for title in chunks
    }

    for required_name, aliases in MANDATORY_SECTION_ALIASES.items():
        match = None

        for title, normalized_title in normalized_titles.items():
            for alias in aliases:
                normalized_alias = " ".join(words(alias))

                if (
                    normalized_alias
                    and normalized_alias in normalized_title
                ):
                    match = title
                    break

            if match:
                break

        if match:
            if match not in selected:
                selected.append(match)
        else:
            missing.append(required_name)

    return selected, missing


def search_guidelines(policy, query, k=3):
    """
    Search the already-loaded policy.

    This function deliberately does not read guidelines.md again.
    """
    if not isinstance(policy, dict) or not policy.get("sections"):
        raise ValueError("Loaded policy is invalid or empty.")

    chunks = policy["sections"]

    mandatory, missing = find_mandatory_sections(chunks)

    if missing:
        raise ValueError(
            "Mandatory policy sections could not be identified: "
            + "; ".join(missing)
        )

    query_words = set(words(query))

    documents = {
        title: words(title + " " + body)
        for title, body in chunks.items()
    }

    document_count = len(documents)
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
                    * (
                        math.log(
                            (document_count + 1)
                            / max(document_frequency, 1)
                        ) + 1
                    )
                )

        scores[title] = score

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected = list(mandatory)

    for title, score in ranked:
        if score > 0 and title not in selected:
            selected.append(title)

        if len(selected) >= len(mandatory) + k:
            break

    results = [
        {
            "section": title,
            "text": chunks[title],
        }
        for title in selected
    ]

    total_chars = sum(
        len(item["section"]) + len(item["text"])
        for item in results
    )

    if total_chars > MAX_POLICY_CHARS:
        raise ValueError(
            "Mandatory and relevant policy sections exceed the "
            f"{MAX_POLICY_CHARS}-character policy budget. "
            "Review guidelines.md and reduce duplication without "
            "removing safety rules."
        )

    print("Mandatory policy sections:", mandatory)
    print("Retrieved policy sections:", [item["section"] for item in results])
    print("Retrieved policy characters:", total_chars)

    return results


# ============================================================
# 6. GOOGLE SHEETS HISTORY
# ============================================================

def tool_get_history(n=3):
    if not SHEET_URL:
        return {"error": "SHEET_URL is not configured."}

    try:
        response = HTTP.get(
            SHEET_URL,
            params={
                "action": "history",
                "n": min(max(int(n), 1), MAX_HISTORY_RECORDS),
            },
            timeout=SHEET_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return {"error": "Unexpected history response format."}

        return data[:MAX_HISTORY_RECORDS]

    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}


def summarize_history(history):
    if not isinstance(history, list):
        return []

    compact = []

    for item in history[-MAX_HISTORY_RECORDS:]:
        if not isinstance(item, dict):
            continue

        compact.append({
            "timestamp": compact_text(item.get("timestamp"), 40),
            "city": compact_text(item.get("city"), 40),
            "advice": compact_text(item.get("advice"), 180),
            "alert_sent": item.get("alert_sent"),
            "alert_message": compact_text(
                item.get("alert_message"),
                120,
            ),
        })

    return compact


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

    if any(
        type(value) is not int or not 1 <= value <= 5
        for value in dimensions
    ):
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


def validate_string_list(value, field_name):
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must contain only text values.")

    return [
        compact_text(item, 300)
        for item in value
        if item.strip()
    ]


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

    missing = [
        field
        for field in required
        if field not in decision
    ]

    if missing:
        raise ValueError(
            "Missing assessment fields: " + ", ".join(missing)
        )

    recommendation = decision["alert_recommendation"]

    if recommendation not in VALID_RECOMMENDATIONS:
        raise ValueError("Invalid alert recommendation.")

    dimension_fields = (
        "likelihood_score",
        "severity_score",
        "exposure_score",
        "duration_score",
        "geographic_impact_score",
    )

    for field in dimension_fields:
        value = decision[field]

        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError(
                f"Invalid or unassessed risk dimension: {field}"
            )

    reason_fields = (
        "likelihood_reason",
        "severity_reason",
        "exposure_reason",
        "duration_reason",
        "geographic_impact_reason",
    )

    for field in reason_fields:
        if (
            not isinstance(decision[field], str)
            or not decision[field].strip()
        ):
            raise ValueError(f"Missing rationale: {field}")

    for field in ("hazard_type", "advice", "reasoning"):
        if not isinstance(decision[field], str):
            raise ValueError(f"Invalid text field: {field}")

        if not decision[field].strip():
            raise ValueError(f"Empty text field: {field}")

    decision["recommended_actions"] = validate_string_list(
        decision["recommended_actions"],
        "recommended_actions",
    )

    decision["limitations"] = validate_string_list(
        decision["limitations"],
        "limitations",
    )

    decision["missing_evidence"] = validate_string_list(
        decision["missing_evidence"],
        "missing_evidence",
    )

    decision["sources"] = validate_string_list(
        decision["sources"],
        "sources",
    )

    available_titles = {
        section["section"]
        for section in retrieved_sections
    }

    valid_sources = [
        source
        for source in decision["sources"]
        if source in available_titles
    ]

    if not valid_sources:
        raise ValueError(
            "Assessment did not identify a valid retrieved policy section."
        )

    decision["sources"] = valid_sources

    score, band = calculate_risk_score(
        decision["likelihood_score"],
        decision["severity_score"],
        decision["exposure_score"],
        decision["duration_score"],
        decision["geographic_impact_score"],
    )

    if score is None:
        raise ValueError("Risk score could not be calculated.")

    decision["risk_score"] = score
    decision["risk_band"] = band
    decision["assessment_status"] = "COMPLETE"

    return decision


def apply_official_warning_guard(decision):
    """
    Official warnings are not integrated in this version.

    Therefore, the agent cannot claim that no alert is required
    or pass through an official warning that it has not retrieved.
    """
    recommendation = decision.get("alert_recommendation")

    if recommendation == "NO_ALERT_RECOMMENDED":
        decision["alert_recommendation"] = "MONITOR"

        decision["limitations"].append(
            "A definitive no-alert conclusion is not permitted because "
            "official NCM and emergency authority warnings have not "
            "been checked."
        )

    elif recommendation == "OFFICIAL_WARNING_PASS_THROUGH":
        decision["alert_recommendation"] = "ESCALATE_FOR_REVIEW"

        decision["limitations"].append(
            "The model returned an official-warning pass-through "
            "recommendation, but no official warning feed is integrated. "
            "Manual verification is required."
        )

    return decision


# ============================================================
# 9. WEATHER ASSESSMENT
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-based weather safety assessment agent.

Follow the supplied weather safety guidelines. Do not invent weather
measurements, official warnings, regulatory requirements, thresholds,
or geographic information.

Assess only the supplied city and evidence.

Requirements:
1. Identify relevant hazards using actual measurements and forecasts.
2. Evaluate all five policy risk dimensions from 1 to 5.
3. Give a separate evidence-based rationale for each dimension.
4. Never treat missing evidence as a low score.
5. Official warning status is NOT_CHECKED. Do not claim otherwise.
6. Open-Meteo is a weather data source, not an official UAE warning authority.
7. Distinguish current observations, forecasts and historical context.
8. Do not assume all residents or workers have equal exposure.
9. Do not let previous alerts suppress a new or different hazard.
10. Do not recommend an alert solely because a value is unusual.
11. Use only recommendation values defined in the policy.
12. Explain missing evidence and limitations.
13. Cite exact section titles from the supplied policy.
14. Return a single JSON object only.

Return these exact keys:
hazard_type,
likelihood_score,
likelihood_reason,
severity_score,
severity_reason,
exposure_score,
exposure_reason,
duration_score,
duration_reason,
geographic_impact_score,
geographic_impact_reason,
alert_recommendation,
advice,
reasoning,
recommended_actions,
limitations,
missing_evidence,
sources.

All five scores must be integers from 1 to 5.
All list fields must be JSON arrays of strings.
"""


def assess_weather(weather, history, policy, retrieved_sections):
    compact_weather = summarize_weather(weather)
    compact_history = summarize_history(history)

    context = {
        "city": CITY,
        "weather": compact_weather,
        "recent_agent_history": compact_history,
        "policy_name": "Dubai Weather Safety Guidelines",
        "policy_version": policy["version"],
        "policy_hash": policy["hash"],
        "retrieved_guideline_sections": retrieved_sections,
        "official_warning_status": (
            "NOT_CHECKED. No NCM or other official warning feed "
            "is integrated."
        ),
    }

    user_prompt = (
        "Assess the supplied weather situation using the policy. "
        "Do not infer official warning status.\n\n"
        + json_text(context)
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = ask_groq(messages)
    decision = parse_json_response(response)

    decision = validate_decision(
        decision,
        retrieved_sections,
    )

    decision = apply_official_warning_guard(decision)

    return decision


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

        "advice": (
            "A reliable safety assessment could not be completed. "
            "Check official weather and emergency authority updates."
        ),

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

    policy = None
    weather = None
    history = []
    history_status = "NOT_CHECKED"

    try:
        require_configuration()

        # Load the policy exactly once.
        policy = load_guidelines()
        used.append("load_guidelines")

        # Retrieve weather.
        weather = get_weather(CITY)
        used.append("get_weather")

        # Retrieve history. History is supporting context, not a dependency
        # that can override current weather evidence.
        history_result = tool_get_history(MAX_HISTORY_RECORDS)
        used.append("get_history")

        if isinstance(history_result, dict) and "error" in history_result:
            history_status = "UNAVAILABLE"
            print("History unavailable:", history_result["error"])
            history = []
        else:
            history_status = "AVAILABLE"
            history = history_result

        # Retrieve policy sections from the already-loaded policy.
        query = (
            "Dubai weather safety heat wind dust rain flooding "
            "thunderstorms lightning cold exposure duration likelihood "
            "severity geographic impact risk calculation evidence "
            "quality alert decision official warning human approval"
        )

        retrieved_sections = search_guidelines(
            policy,
            query,
            k=3,
        )
        used.append("search_guidelines")

        if not retrieved_sections:
            raise ValueError("No policy sections were retrieved.")

        decision = assess_weather(
            weather,
            history,
            policy,
            retrieved_sections,
        )

        # Explicitly record warning verification limitations.
        decision["official_warning_status"] = "NOT_CHECKED"
        decision["official_warning_authority"] = None

        # Open-Meteo is the only weather source currently integrated.
        decision["evidence_quality"] = "E3"
        decision["evidence_confidence"] = "LOW"

        decision["policy_status"] = "AVAILABLE"
        decision["policy_version"] = policy["version"]
        decision["policy_hash"] = policy["hash"]
        decision["policy_path"] = policy["path"]

        decision["weather_source"] = weather["source"]
        decision["weather_retrieval_status"] = "AVAILABLE"
        decision["weather_data_timestamp"] = weather["retrieved_at"]

        decision["assessment_timestamp"] = (
            datetime.now().astimezone().isoformat()
        )

        decision["city"] = CITY
        decision["timezone"] = weather["timezone"]
        decision["tools_used"] = used

        decision["weather_summary"] = summarize_weather(weather)
        decision["history_checked"] = history_status == "AVAILABLE"
        decision["history_status"] = history_status

        if history_status != "AVAILABLE":
            decision["limitations"].append(
                "Recent agent history was unavailable. "
                "The assessment uses current weather evidence only."
            )

        decision["limitations"].append(
            "Official NCM and emergency authority warnings were not "
            "retrieved or verified by this implementation."
        )

        # Keep external notifications disabled unless explicitly enabled.
        authorized = (
            AUTO_APPROVE_ALERTS
            and ENABLE_EMAIL_ALERTS
        )

        decision["approval_status"] = (
            "APPROVED" if authorized else "PENDING_REVIEW"
        )

        # Do not send email for monitor, review or incomplete decisions.
        decision["send_alert"] = (
            authorized
            and decision["assessment_status"] == "COMPLETE"
            and decision["alert_recommendation"]
            == "ADVISORY_RECOMMENDED"
        )

        decision["alert_message"] = (
            compact_text(decision["advice"], 300)
            if decision["send_alert"]
            else ""
        )

        decision["notification_status"] = (
            "PENDING" if decision["send_alert"] else "NOT_SENT"
        )

        return decision, used

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

        print("\nASSESSMENT ERROR:", error_message)

        decision = incomplete_assessment(error_message)

        if isinstance(exc, FileNotFoundError):
            decision["policy_status"] = "UNAVAILABLE"
        elif policy is not None:
            decision["policy_status"] = "AVAILABLE"
        else:
            decision["policy_status"] = "UNKNOWN"

        decision["policy_version"] = (
            policy["version"] if policy else None
        )

        decision["policy_hash"] = (
            policy["hash"] if policy else None
        )

        decision["policy_path"] = str(GUIDELINES_PATH)

        decision["official_warning_status"] = "NOT_CHECKED"

        decision["evidence_quality"] = "E0"
        decision["evidence_confidence"] = "NONE"

        decision["approval_status"] = "NOT_AUTHORIZED"
        decision["send_alert"] = False
        decision["alert_message"] = ""
        decision["notification_status"] = "NOT_SENT"

        decision["weather_retrieval_status"] = (
            "AVAILABLE" if weather else "UNAVAILABLE"
        )

        decision["history_status"] = history_status

        decision["assessment_timestamp"] = (
            datetime.now().astimezone().isoformat()
        )

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
    print("OFFICIAL WARNING:", decision.get("official_warning_status"))
    print("ALERT AUTHORIZED:", decision.get("send_alert"))
    print("NOTIFICATION:", decision.get("notification_status"))

    # Preserve all original Apps Script field names.
    payload = {
        "type": "agent",
        "city": CITY,
        "tools_used": ", ".join(used),

        "advice": decision.get("advice", ""),
        "reasoning": decision.get("reasoning", ""),

        "send_alert": decision.get("send_alert") is True,
        "alert_message": decision.get("alert_message", ""),

        "sources": ", ".join(decision.get("sources", [])),

        # Additional fields. Existing Apps Script may ignore these.
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

        "geographic_impact_score": decision.get(
            "geographic_impact_score"
        ),
        "geographic_impact_reason": decision.get(
            "geographic_impact_reason"
        ),

        "risk_score": decision.get("risk_score"),
        "risk_band": decision.get("risk_band"),

        "evidence_quality": decision.get("evidence_quality"),
        "evidence_confidence": decision.get("evidence_confidence"),

        "official_warning_status": decision.get(
            "official_warning_status"
        ),
        "alert_recommendation": decision.get("alert_recommendation"),

        "assessment_timestamp": decision.get("assessment_timestamp"),
        "approval_status": decision.get("approval_status"),
        "notification_status": decision.get("notification_status"),

        "limitations": json_text(decision.get("limitations", [])),
        "missing_evidence": json_text(
            decision.get("missing_evidence", [])
        ),
    }

    if dry:
        print("\nDRY RUN: Nothing was sent to Google Sheets or email.")
        print(json_text(payload))
        return

    if not SHEET_URL:
        raise SystemExit("SHEET_URL is missing.")

    response = HTTP.post(
        SHEET_URL,
        json=payload,
        timeout=SHEET_TIMEOUT,
    )

    print("\nSHEET RESPONSE:", response.status_code, response.text)

    response.raise_for_status()

    if response.text.strip() != "OK":
        raise SystemExit(
            "Google Apps Script did not confirm success."
        )


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

    # Risk formula tests.
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
        "Out-of-range risk dimension rejected",
        score is None and band == "UNASSESSED",
    )

    # Official warning guard tests.
    test_decision = {
        "alert_recommendation": "NO_ALERT_RECOMMENDED",
        "limitations": [],
    }

    guarded = apply_official_warning_guard(test_decision)

    check(
        "Unchecked official warning prevents definitive no-alert",
        guarded["alert_recommendation"] == "MONITOR",
    )

    test_decision = {
        "alert_recommendation": "OFFICIAL_WARNING_PASS_THROUGH",
        "limitations": [],
    }

    guarded = apply_official_warning_guard(test_decision)

    check(
        "Unverified official warning is escalated",
        guarded["alert_recommendation"] == "ESCALATE_FOR_REVIEW",
    )

    # Token budget tests.
    check(
        "Token estimator returns positive estimate",
        estimate_tokens("Weather safety assessment") > 0,
    )

    check(
        "Completion token limit is conservative",
        MAX_COMPLETION_TOKENS <= 1000,
    )

    # Guidelines loading and retrieval tests.
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

        mandatory, missing = find_mandatory_sections(
            policy["sections"]
        )

        check(
            "Mandatory policy sections identified",
            len(mandatory) == len(MANDATORY_SECTION_ALIASES)
            and not missing,
        )

        results = search_guidelines(
            policy,
            "heat exposure risk assessment alert",
            k=3,
        )

        check(
            "Relevant policy sections retrieved",
            len(results) >= len(MANDATORY_SECTION_ALIASES),
        )

        check(
            "Policy retrieval stays within character budget",
            sum(
                len(item["section"]) + len(item["text"])
                for item in results
            ) <= MAX_POLICY_CHARS,
        )

    except Exception as exc:
        print("Guidelines test error:", exc)
        check("Guidelines loading and retrieval", False)

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

        try:
            policy = load_guidelines()

            for result in search_guidelines(policy, query):
                print("\nSECTION:", result["section"])
                print(result["text"])

        except Exception as exc:
            print("POLICY SEARCH ERROR:", exc)
            raise SystemExit(1)

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
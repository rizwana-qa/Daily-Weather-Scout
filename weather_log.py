import os
import re
import sys
import tempfile
import json
import math
import ast
import contextlib
import io
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

# Request budgets, sized from measured requests (see the budget tests):
# every single hazard, heat with wind, and all five hazards together
# with the conditional compliance section fit. Anything larger fails
# closed (INCOMPLETE) before a request is sent.
# The token figure uses the conservative estimator below (2.5
# characters per token), not a provider tokenizer. 8,000 is the
# documented base tokens-per-minute limit for openai/gpt-oss-20b; the
# account's own limit is unverified, and Groq's 413/429 responses are
# handled as safe failures. Revisit all three after a dry run reports
# Groq's actual usage.
MAX_COMPLETION_TOKENS = 700
MAX_ESTIMATED_REQUEST_TOKENS = 8000
MAX_REQUEST_CHARS = 19000
MAX_POLICY_CHARS = 13500
MAX_HISTORY_RECORDS = 3

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

def require_configuration(require_sheet=False):
    if CITY not in CITIES:
        raise ValueError(
            f"Unsupported city: {CITY}. Allowed cities: {list(CITIES)}"
        )

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    if require_sheet and not SHEET_URL:
        raise ValueError("SHEET_URL is not configured.")


def safe_number(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)) and math.isfinite(value):
        return value

    return None


def average(values):
    valid = [
        value for value in values
        if safe_number(value) is not None
    ]

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
    Approximate token count.

    This is a conservative estimate, not an exact provider tokenizer.
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
            "visibility,"
            "weather_code,"
            "cape"
        ),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "visibility,"
            "weather_code,"
            "cape"
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

    for key in ("current", "hourly", "daily"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(
                f"Weather API response is missing '{key}'."
            )

    current = raw["current"]
    hourly = raw["hourly"]
    daily = raw["daily"]

    current_time = local_datetime(current.get("time"))

    if current_time is None:
        raise ValueError(
            "Weather API returned an invalid current timestamp."
        )

    today = current_time.date()
    previous_days = []
    daily_times = daily.get("time", [])

    # Previous completed days.
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
        "weather_code": "weather_code",
        "cape": "cape",
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
        "source_type": (
            "Weather API, not an official UAE warning source"
        ),
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
    cape = values_for("cape")
    weather_codes = sorted({
        int(code) for code in values_for("weather_code")
    })

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
        "cape_max": max(cape, default=None),
        "weather_codes": weather_codes,
    }

    return {
        "city": weather["city"],
        "timezone": weather.get("timezone"),
        "source": weather["source"],
        "retrieved_at": weather["retrieved_at"],
        "current": weather["current"],
        "today_forecast": weather.get("today_forecast"),
        "next_24_hours": hourly_summary,
        "recent_completed_days": weather.get(
            "previous_days", []
        )[-3:],
        "historical_summary": weather.get("history_summary"),
    }


# ============================================================
# 4. GROQ API
# ============================================================

# Usage reported by the most recent Groq response (empty until a real
# response is received). Never contains credentials or message text.
LAST_GROQ_USAGE = {}


def usage_number(usage, key):
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int else None


def describe_groq_http_error(status_code, response):
    if status_code == 413:
        reason = "request too large for the model or account limit"
    elif status_code == 429:
        reason = "rate or token limit reached (RPM, TPM, RPD or TPD)"
    elif status_code == 400:
        reason = "request rejected as invalid"
    else:
        reason = "request failed"

    retry_after = response.headers.get("retry-after") if hasattr(
        response, "headers"
    ) else None

    message = f"Groq API returned HTTP {status_code} ({reason})"

    if retry_after:
        message += f"; retry-after {compact_text(retry_after, 20)}s"

    return message + f": {compact_text(response.text, 400)}"


def ask_groq(messages, temperature=0.1):
    LAST_GROQ_USAGE.clear()

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
    print("Request characters:", len(serialized_messages))

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
        # Groq documents reasoning_effort and include_reasoning for
        # openai/gpt-oss-20b; reasoning_format is not supported there.
        "reasoning_effort": "low",
        "include_reasoning": False,
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
        raise RuntimeError(
            describe_groq_http_error(response.status_code, response)
        )

    try:
        result = response.json()
        choice = result["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError,
            AttributeError) as exc:
        raise RuntimeError(
            "Groq returned an unexpected response structure."
        ) from exc

    usage = result.get("usage")

    LAST_GROQ_USAGE.update({
        "prompt_tokens": usage_number(usage, "prompt_tokens"),
        "completion_tokens": usage_number(usage, "completion_tokens"),
        "reasoning_tokens": usage_number(
            usage.get("completion_tokens_details")
            if isinstance(usage, dict) else None,
            "reasoning_tokens",
        ),
        "total_tokens": usage_number(usage, "total_tokens"),
        "finish_reason": (
            finish_reason if isinstance(finish_reason, str) else None
        ),
        "estimated_input_tokens": estimated_input_tokens,
        "completion_allowance": MAX_COMPLETION_TOKENS,
    })

    print("\nGROQ USAGE (reported by the API)")

    for key, value in LAST_GROQ_USAGE.items():
        print(f"{key}: {value}")

    if finish_reason != "stop":
        raise RuntimeError(
            "Groq response is incomplete or was cut off: "
            f"finish_reason={finish_reason!r}. The completion allowance "
            f"is {MAX_COMPLETION_TOKENS} tokens."
        )

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq returned an empty response.")

    return content


def parse_json_response(response_text):
    text = (response_text or "").strip()

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
        for word in re.findall(r"[a-z0-9]+", str(text).lower())
        if word not in STOP_WORDS
    ]


def normalize_heading(text):
    # Ignore leading section numbers such as "8." so renumbering
    # guidelines.md does not break mandatory-section matching.
    text = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", str(text))
    return " ".join(
        re.findall(r"[a-z0-9]+", text.lower())
    )


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

    # Split only on Markdown level-two headings.
    for block in re.split(r"\n(?=## )", content):
        block = block.strip()

        if not block.startswith("## "):
            continue

        title, _, body = block[3:].partition("\n")
        title = title.strip()

        if not title:
            continue

        if title in chunks:
            raise ValueError(
                f"Duplicate policy heading found: {title}"
            )

        chunks[title] = body.strip()

    if not chunks:
        raise ValueError(
            "No Markdown sections found. "
            "Expected headings beginning with ##"
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


# Core sections that are always sent to the model. They must match
# headings in guidelines.md (v2.0), ignoring section numbers. Their
# combined size must stay within MAX_POLICY_CHARS, which is itself
# bound by the Groq request limits above. Section 4 holds the 1-5
# scoring scales the model needs whenever it assigns risk scores;
# section 13 governs the advice it writes. If you rename a heading in
# guidelines.md, update it here too.
MANDATORY_SECTION_ALIASES = {
    "Risk Assessment Dimensions": (
        "Risk Assessment Dimensions",
    ),
    "Advice and Communication Requirements": (
        "Advice and Communication Requirements",
    ),
    "Alert Decision Framework": (
        "Alert Decision Framework",
    ),
    "Official Warning Override": (
        "Official Warning Override",
    ),
    "Human Approval and Notification Governance": (
        "Human Approval and Notification Governance",
    ),
    "Final Decision Rule": (
        "Final Decision Rule",
    ),
}


def find_mandatory_sections(chunks, aliases=None):
    if aliases is None:
        aliases = MANDATORY_SECTION_ALIASES

    selected = []
    missing = []

    normalized_titles = {
        normalize_heading(title): title
        for title in chunks
    }

    for required_name, alias_names in aliases.items():
        match = None

        for alias in alias_names:
            normalized_alias = normalize_heading(alias)

            if normalized_alias in normalized_titles:
                match = normalized_titles[normalized_alias]
                break

        if match:
            if match not in selected:
                selected.append(match)
        else:
            missing.append(required_name)

    return selected, missing


def section_character_count(title, body):
    return len(title) + len(body)


# Hazard-specific retrieval (guidelines.md section 6).
#
# Section 6 is split into its "###" subsections. Every subsection whose
# hazard is detected, or cannot be ruled out, is sent in full. If the
# required text does not fit MAX_POLICY_CHARS the run fails safely.
#
# The values below are RETRIEVAL-SCREENING values only. guidelines.md
# defines no numeric hazard thresholds and forbids inventing them, so
# these are deliberately conservative triggers for attaching guidance.
# They never classify risk, never lower a score and never exclude a
# hazard whose data is missing. Review them with the policy owner.
HAZARD_SCREENING = {
    "heat_temperature_c": 30.0,
    "heat_apparent_c": 32.0,
    "cold_temperature_c": 12.0,
    "wind_kmh": 20.0,
    "gust_kmh": 40.0,
    "visibility_m": 5000.0,
    "cape_jkg": 500.0,
}

# Hazard code -> section 6 subsection title (section numbers ignored).
HAZARD_POLICY_TITLES = {
    "HEAT": "Extreme Heat",
    "WIND_DUST": "Wind and Dust",
    "RAIN_FLOOD": "Rain, Thunderstorms and Flooding",
    "COLD": "Low Temperatures and Cold Conditions",
    "LIGHTNING": "Lightning and Thunderstorms",
}


# WMO weather interpretation codes reported by Open-Meteo.
WMO_RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
WMO_THUNDERSTORM_CODES = {95, 96, 97, 99}
WMO_KNOWN_CODES = {
    0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
    71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 97, 99,
}

# Physical plausibility ranges. A value outside its range, or one that
# is not a finite number, is unreliable: it is dropped and reported as
# uncertainty. It is never used as evidence of safety.
PLAUSIBLE_RANGES = {
    "highs": (-50.0, 70.0),
    "lows": (-50.0, 70.0),
    "apparent": (-60.0, 80.0),
    "wind": (0.0, 400.0),
    "gust": (0.0, 500.0),
    "precipitation": (0.0, 500.0),
    "visibility": (0.0, 200000.0),
    "cape": (0.0, 10000.0),
}


def collect_weather_values(weather):
    """
    Gather usable measurements from the current, hourly and daily
    evidence. Missing values (None) are simply absent; present but
    invalid values are counted under "unreliable".
    """
    weather = weather if isinstance(weather, dict) else {}

    current = weather.get("current")
    current = current if isinstance(current, dict) else {}

    today = weather.get("today_forecast")
    today = today if isinstance(today, dict) else {}

    hourly = weather.get("hourly_forecast_next_24h")
    hourly = [
        record for record in hourly
        if isinstance(record, dict)
    ] if isinstance(hourly, list) else []

    unreliable = {}

    def gather(name, current_key, hourly_key, today_key=None):
        raw = [current.get(current_key)]
        raw += [record.get(hourly_key) for record in hourly]

        if today_key:
            raw.append(today.get(today_key))

        low, high = PLAUSIBLE_RANGES[name]
        usable = []

        for value in raw:
            if value is None:
                continue

            number = safe_number(value)

            if number is None or not low <= number <= high:
                unreliable[name] = unreliable.get(name, 0) + 1
                continue

            usable.append(number)

        return usable

    codes = []

    raw_codes = [current.get("weather_code")]
    raw_codes += [record.get("weather_code") for record in hourly]

    for value in raw_codes:
        if value is None:
            continue

        number = safe_number(value)

        if (
            number is None
            or number != int(number)
            or int(number) not in WMO_KNOWN_CODES
        ):
            unreliable["codes"] = unreliable.get("codes", 0) + 1
            continue

        codes.append(int(number))

    return {
        "highs": gather(
            "highs", "temperature_2m", "temperature", "high"
        ),
        "lows": gather(
            "lows", "temperature_2m", "temperature", "low"
        ),
        "apparent": gather(
            "apparent", "apparent_temperature", "apparent_temperature"
        ),
        "wind": gather("wind", "wind_speed_10m", "wind", "wind"),
        "gust": gather("gust", "wind_gusts_10m", "gust", "gust"),
        "precipitation": gather(
            "precipitation", "precipitation", "precipitation",
            "precipitation",
        ),
        "visibility": gather(
            "visibility", "visibility", "visibility"
        ),
        "cape": gather("cape", "cape", "cape"),
        "codes": codes,
        "unreliable": unreliable,
    }


def identify_hazards(weather):
    """
    Return {hazard_code: [reasons]} for every hazard that the weather
    evidence indicates or cannot rule out. Reasons that start with
    "UNCERTAIN" mean essential evidence is missing or unreliable; they
    attach the guidance and are treated as an evidence gap, never as
    proof of safety. Multiple hazards can be returned together.

    Rain/flood and thunderstorm/lightning are detected separately:
    rain from precipitation and rain weather codes, thunderstorms from
    thunderstorm weather codes and from instability (CAPE) together
    with rain evidence.
    """
    values = collect_weather_values(weather)
    unreliable = values["unreliable"]
    screen = HAZARD_SCREENING
    hazards = {}

    def add(code, reason):
        hazards.setdefault(code, []).append(reason)

    def reject(name, label, codes):
        if unreliable.get(name):
            for code in codes:
                add(
                    code,
                    f"UNCERTAIN: unreliable {label} values rejected",
                )

    # Heat.
    if not values["highs"] and not values["apparent"]:
        add("HEAT", "UNCERTAIN: no temperature data")

    if values["highs"] and (
        max(values["highs"]) >= screen["heat_temperature_c"]
    ):
        add(
            "HEAT",
            f"temperature {max(values['highs'])} C reaches the "
            "heat screening value",
        )

    if values["apparent"] and (
        max(values["apparent"]) >= screen["heat_apparent_c"]
    ):
        add(
            "HEAT",
            f"apparent temperature {max(values['apparent'])} C "
            "reaches the heat screening value",
        )

    reject("highs", "temperature", ("HEAT", "COLD"))
    reject("lows", "temperature", ("HEAT", "COLD"))
    reject("apparent", "apparent temperature", ("HEAT",))

    # Cold.
    if not values["lows"]:
        add("COLD", "UNCERTAIN: no minimum temperature data")
    elif min(values["lows"]) <= screen["cold_temperature_c"]:
        add(
            "COLD",
            f"minimum temperature {min(values['lows'])} C is at or "
            "below the cold screening value",
        )

    # Wind and dust.
    if not values["wind"] and not values["gust"]:
        add("WIND_DUST", "UNCERTAIN: no wind or gust data")

    if values["wind"] and max(values["wind"]) >= screen["wind_kmh"]:
        add(
            "WIND_DUST",
            f"wind {max(values['wind'])} km/h reaches the wind "
            "screening value",
        )

    if values["gust"] and max(values["gust"]) >= screen["gust_kmh"]:
        add(
            "WIND_DUST",
            f"gust {max(values['gust'])} km/h reaches the gust "
            "screening value",
        )

    if not values["visibility"]:
        add(
            "WIND_DUST",
            "UNCERTAIN: no visibility data, dust cannot be excluded",
        )
    elif min(values["visibility"]) < screen["visibility_m"]:
        add(
            "WIND_DUST",
            f"visibility {min(values['visibility'])} m is below the "
            "visibility screening value",
        )

    reject("wind", "wind speed", ("WIND_DUST",))
    reject("gust", "wind gust", ("WIND_DUST",))
    reject("visibility", "visibility", ("WIND_DUST",))

    # Rain and flooding.
    codes = values["codes"]
    precipitation = values["precipitation"]
    thunderstorm = any(
        code in WMO_THUNDERSTORM_CODES for code in codes
    )
    rain_code = any(code in WMO_RAIN_CODES for code in codes)
    rain_signal = (
        bool(precipitation and max(precipitation) > 0)
        or rain_code
        or thunderstorm
    )

    if precipitation and max(precipitation) > 0:
        add(
            "RAIN_FLOOD",
            f"precipitation signal {max(precipitation)} mm",
        )

    if rain_code:
        add("RAIN_FLOOD", "rain weather code reported")

    if thunderstorm:
        add("RAIN_FLOOD", "thunderstorm weather code reported")

    if not precipitation and not codes:
        add(
            "RAIN_FLOOD",
            "UNCERTAIN: no precipitation or weather-code data",
        )

    reject("precipitation", "precipitation", ("RAIN_FLOOD",))

    if unreliable.get("codes"):
        add(
            "RAIN_FLOOD",
            "UNCERTAIN: unreliable weather codes rejected",
        )

    # Thunderstorms and lightning.
    if thunderstorm:
        add("LIGHTNING", "thunderstorm weather code reported")

    if rain_signal and values["cape"] and (
        max(values["cape"]) >= screen["cape_jkg"]
    ):
        add(
            "LIGHTNING",
            f"rain evidence with CAPE {max(values['cape'])} J/kg at "
            "or above the instability screening value",
        )

    if not codes:
        add(
            "LIGHTNING",
            "UNCERTAIN: no usable weather code, thunderstorms "
            "cannot be excluded",
        )

    reject("cape", "CAPE", ("LIGHTNING",))

    if unreliable.get("codes"):
        add(
            "LIGHTNING",
            "UNCERTAIN: unreliable weather codes rejected",
        )

    return hazards


def uncertain_hazards(hazards):
    """Hazards with at least one UNCERTAIN (evidence gap) reason."""
    return {
        code: [r for r in reasons if r.startswith("UNCERTAIN")]
        for code, reasons in hazards.items()
        if any(r.startswith("UNCERTAIN") for r in reasons)
    }


def apply_evidence_uncertainty(decision, hazards):
    """
    Guidelines sections 3 and 8: essential evidence that is missing or
    unreliable means the assessment is incomplete and cannot end in a
    low-risk or monitor-only result.
    """
    gaps = uncertain_hazards(hazards)

    if not gaps:
        return decision

    summary = "; ".join(
        f"{code}: {reasons[0]}" for code, reasons in gaps.items()
    )

    decision["limitations"].append(
        "Essential weather evidence is missing or unreliable ("
        + summary
        + "). Missing evidence is not evidence of safety."
    )
    decision["missing_evidence"].append(
        "Weather evidence gaps: " + summary
    )

    recommendation = decision.get("alert_recommendation")

    if recommendation in ("NO_ALERT_RECOMMENDED", "MONITOR"):
        decision["alert_recommendation"] = "UNABLE_TO_ASSESS"
    elif recommendation == "ADVISORY_RECOMMENDED":
        decision["alert_recommendation"] = "ESCALATE_FOR_REVIEW"

    enforce_assessment_status(decision)

    return decision


# Guidelines section 7: the UAE midday break generally applies from
# 15 June to 15 September.
MIDDAY_BREAK_START = (6, 15)
MIDDAY_BREAK_END = (9, 15)


def select_conditional_sections(policy, hazards, weather):
    """
    Sections retrieved only when they apply:
    - Regulatory and Workplace Compliance (section 7) when heat is
      relevant, when the date is inside the midday-break season, or
      when the date cannot be established.
    - Data Quality and Assessment Completeness (section 11) whenever
      essential evidence is missing or unreliable.
    """
    wanted = []

    current = (
        weather.get("current")
        if isinstance(weather, dict) else None
    )
    current_time = local_datetime(
        current.get("time") if isinstance(current, dict) else None
    )

    in_season = (
        current_time is None
        or MIDDAY_BREAK_START
        <= (current_time.month, current_time.day)
        <= MIDDAY_BREAK_END
    )

    if "HEAT" in hazards or in_season:
        wanted.append("Regulatory and Workplace Compliance")

    if uncertain_hazards(hazards):
        wanted.append("Data Quality and Assessment Completeness")

    aliases = {name: (name,) for name in wanted}
    selected, missing = find_mandatory_sections(
        policy["sections"], aliases
    )

    if missing:
        raise ValueError(
            "Conditional policy section is missing from "
            "guidelines.md: " + "; ".join(missing)
        )

    return [
        {"section": title, "text": policy["sections"][title]}
        for title in selected
    ]


def parse_hazard_sections(policy):
    chunks = policy["sections"]

    parent = next(
        (
            title for title in chunks
            if normalize_heading(title) == "weather hazard assessment"
        ),
        None,
    )

    if parent is None:
        raise ValueError(
            "Weather hazard guidance is missing from guidelines.md: "
            "Weather Hazard Assessment"
        )

    pieces = re.split(r"\n(?=### )", chunks[parent])
    intro = pieces[0].strip()
    blocks = {}

    for piece in pieces[1:]:
        title, _, body = piece.strip()[4:].partition("\n")
        title = title.strip()

        if not title or title in blocks:
            raise ValueError(
                f"Invalid or duplicate hazard heading: {title!r}"
            )

        blocks[title] = body.strip()

    return parent, intro, blocks


def select_hazard_guidance(policy, hazards):
    """
    Return the complete section 6 text required for the hazards:
    the section introduction, each detected hazard subsection in
    policy order, and every subsection this code does not recognise
    (unknown hazards cannot be screened, so they are always sent).
    """
    unknown_codes = [
        code for code in hazards
        if code not in HAZARD_POLICY_TITLES
    ]

    if unknown_codes:
        raise ValueError(
            "Unrecognized hazard code: " + ", ".join(unknown_codes)
        )

    parent, intro, blocks = parse_hazard_sections(policy)

    by_name = {
        normalize_heading(title): title
        for title in blocks
    }

    selected = []
    missing = []

    for code, policy_title in HAZARD_POLICY_TITLES.items():
        if code not in hazards:
            continue

        title = by_name.get(normalize_heading(policy_title))

        if title is None:
            missing.append(policy_title)
        else:
            selected.append(title)

    if missing:
        raise ValueError(
            "Hazard guidance is missing from guidelines.md: "
            + "; ".join(missing)
        )

    known = {
        normalize_heading(title)
        for title in HAZARD_POLICY_TITLES.values()
    }

    selected += [
        title for title in blocks
        if normalize_heading(title) not in known
    ]

    selected = list(dict.fromkeys(selected))

    return [{"section": parent, "text": intro}] + [
        {"section": title, "text": blocks[title]}
        for title in selected
    ]


def search_guidelines(policy, query, k=1, required_sections=None):
    """
    Retrieve all mandatory sections, every required section
    (required_sections: list of {"section", "text"}, e.g. hazard
    guidance) and up to k other relevant sections.

    Mandatory and required content is never removed or truncated; if
    it cannot fit the budget a ValueError is raised so the caller
    fails safely. Only other optional sections are excluded when the
    character budget is exceeded.
    """
    if not isinstance(policy, dict) or not policy.get("sections"):
        raise ValueError("Loaded policy is invalid or empty.")

    chunks = policy["sections"]

    mandatory, missing = find_mandatory_sections(chunks)

    if missing:
        raise ValueError(
            "Required policy headings are missing from guidelines.md: "
            + "; ".join(missing)
        )

    required = list(required_sections or [])
    texts = dict(chunks)

    for item in required:
        texts[item["section"]] = item["text"]

    # Sections that must be sent in full or the run must fail.
    mandatory = list(dict.fromkeys(
        mandatory + [item["section"] for item in required]
    ))

    query_words = set(words(query))

    # Section 6 is retrieved only through hazard subsections, so the
    # whole section is not an optional candidate (no duplicates).
    documents = {
        title: words(title + " " + body)
        for title, body in chunks.items()
        if title not in mandatory
        and normalize_heading(title) != "weather hazard assessment"
    }

    document_count = len(documents)
    scores = {}

    for title, document_words in documents.items():
        score = 0.0

        for word in query_words:
            frequency = document_words.count(word)

            if not frequency:
                continue

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
    additional_sections = []

    # Add only the requested number of relevant optional sections.
    optional_limit = max(0, int(k))

    for title, score in ranked:
        if score <= 0:
            continue

        if title in selected:
            continue

        if len(additional_sections) >= optional_limit:
            break

        additional_sections.append(title)
        selected.append(title)

    # Deduplicate while preserving order.
    selected = list(dict.fromkeys(selected))

    mandatory_chars = sum(
        section_character_count(title, texts[title])
        for title in mandatory
    )

    additional_chars = sum(
        section_character_count(title, texts[title])
        for title in additional_sections
    )

    total_chars = mandatory_chars + additional_chars

    print("\nPOLICY RETRIEVAL DIAGNOSTICS")
    print("Mandatory sections:", mandatory)
    print("Mandatory section count:", len(mandatory))
    print("Mandatory characters:", mandatory_chars)
    print("Additional sections:", additional_sections)
    print("Additional characters:", additional_chars)
    print("Final policy characters:", total_chars)
    print("Policy character limit:", MAX_POLICY_CHARS)

    # Never silently remove mandatory policy content.
    if mandatory_chars > MAX_POLICY_CHARS:
        raise ValueError(
            "Mandatory and hazard-relevant policy sections exceed the "
            f"{MAX_POLICY_CHARS}-character budget. "
            f"Required characters: {mandatory_chars}. "
            f"Required sections: {mandatory}. "
            "No required policy content was removed."
        )

    # Exclude optional sections from the end until the budget is met.
    while total_chars > MAX_POLICY_CHARS and additional_sections:
        removed = additional_sections.pop()
        selected.remove(removed)

        print("Optional section excluded to meet budget:", removed)

        additional_chars = sum(
            section_character_count(title, texts[title])
            for title in additional_sections
        )

        total_chars = mandatory_chars + additional_chars

    if total_chars > MAX_POLICY_CHARS:
        raise ValueError(
            "Policy retrieval exceeds the configured character budget. "
            "Mandatory safety sections have been preserved."
        )

    results = [
        {
            "section": title,
            "text": texts[title],
        }
        for title in selected
    ]

    # Final guarantee: every required section is present, complete
    # and appears exactly once.
    returned = [item["section"] for item in results]

    for item in mandatory:
        if returned.count(item) != 1 or texts[item] != next(
            entry["text"] for entry in results
            if entry["section"] == item
        ):
            raise ValueError(
                f"Required policy section not retrieved intact: {item}"
            )

    print("Final retrieved sections:", selected)
    print("Final retrieved policy characters:", total_chars)

    return results


# ============================================================
# 6. GOOGLE SHEETS HISTORY
# ============================================================

def tool_get_history(n=3):
    if not SHEET_URL:
        return {
            "error": "SHEET_URL is not configured. History was not retrieved."
        }

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
            return {
                "error": "Unexpected history response format."
            }

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
            "timestamp": compact_text(
                item.get("timestamp"), 32
            ),
            "city": compact_text(
                item.get("city"), 40
            ),
            "advice": compact_text(
                item.get("advice"), 120
            ),
            "alert_sent": item.get("alert_sent"),
            "alert_message": compact_text(
                item.get("alert_message"), 80
            ),
        })

    return compact


# ============================================================
# 7. PROVISIONAL RISK CALCULATION
# ============================================================

def calculate_risk_score(
    likelihood,
    severity,
    exposure,
    duration,
    geographic_impact,
):
    """
    Provisional internal calculation defined in guidelines.md
    section 5 (Risk Calculation Model). Weights and band limits must
    match that section; a test enforces this. It must not be
    represented as an official UAE or NCM risk model.

    Bands use the unrounded score; only the returned value is rounded.
    """

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

    if score <= 5:
        band = "LOW"
    elif score <= 10:
        band = "MODERATE"
    elif score <= 15:
        band = "HIGH"
    else:
        band = "CRITICAL"

    return round(score, 2), band


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
        raise ValueError(
            f"{field_name} must contain only text values."
        )

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
        field for field in required
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

    # Guideline section 5 rule 4: an essential dimension that cannot
    # be assessed is UNASSESSED (null), never a guessed score.
    # Any other non-integer or out-of-range value is malformed output.
    unassessed = []

    for field in dimension_fields:
        value = decision[field]

        if value is None or value == "UNASSESSED":
            decision[field] = None
            unassessed.append(field)
        elif type(value) is not int or not 1 <= value <= 5:
            raise ValueError(
                f"Invalid risk dimension value: {field}"
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
            raise ValueError(
                f"Invalid text field: {field} (expected text, "
                f"received {type(decision[field]).__name__})"
            )

        if not decision[field].strip():
            raise ValueError(f"Empty text field: {field}")

    for field in (
        "recommended_actions",
        "limitations",
        "missing_evidence",
        "sources",
    ):
        decision[field] = validate_string_list(
            decision[field],
            field,
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

    decision["risk_score"] = score
    decision["risk_band"] = band

    if unassessed:
        # No misleading risk classification and no low-risk or
        # advisory conclusion from incomplete scoring evidence.
        if recommendation not in (
            "ESCALATE_FOR_REVIEW",
            "UNABLE_TO_ASSESS",
        ):
            decision["alert_recommendation"] = "UNABLE_TO_ASSESS"

        decision["limitations"].append(
            "Risk score not calculated because these dimensions "
            "could not be assessed: " + ", ".join(unassessed) + "."
        )
        decision["missing_evidence"].append(
            "Evidence required to score: " + ", ".join(unassessed) + "."
        )

    # The model's preliminary analysis is not a completed safety clearance.
    decision["assessment_status"] = "PRELIMINARY"

    return decision


def enforce_assessment_status(decision):
    """
    Official warning verification is a required evidence dependency.
    An assessment can only be COMPLETE when it has been VERIFIED;
    any other warning status, including a missing one, forces
    INCOMPLETE. No official warning feed is integrated today.
    """
    if decision.get("official_warning_status") != "VERIFIED":
        decision["assessment_status"] = "INCOMPLETE"

    return decision


def apply_official_warning_guard(decision):
    """
    No official warning feed is integrated.

    The agent therefore cannot confirm that no official warning exists
    or claim that an official warning has been verified.
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

    decision["limitations"].append(
        "The five-dimension risk score is the provisional internal "
        "model in guidelines.md section 5. It is not an official "
        "UAE/NCM classification."
    )

    decision["official_warning_status"] = "NOT_CHECKED"
    enforce_assessment_status(decision)

    decision["missing_evidence"].append(
        "Current official NCM and relevant UAE emergency authority "
        "warning status has not been retrieved and verified."
    )

    return decision


# ============================================================
# 9. WEATHER ASSESSMENT
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-based weather safety assessment agent.

Follow the supplied policy sections. Never invent weather
measurements, official warnings, regulatory requirements, thresholds
or geographic information. Assess only the supplied city and evidence.

Requirements:
1. Identify hazards from actual measurements and forecasts.
2. Score the five risk dimensions provisionally, using the scale
   definitions in the supplied policy, with a separate evidence-based
   rationale for each.
3. Never treat missing evidence as a low score.
4. Official warning status is NOT_CHECKED; do not claim otherwise.
   Open-Meteo is a weather data source, not an official UAE warning
   authority.
5. Distinguish current observations, forecasts and historical context.
6. Do not assume all residents or workers have equal exposure.
7. Previous alerts must not suppress a new or different hazard. Do not
   recommend an alert solely because a value is unusual.
8. Use only recommendation values defined in the supplied policy.
9. Explain missing evidence and limitations. The assessment is never a
   completed safety clearance.
10. In sources, cite exact policy section titles.
11. Return a single JSON object only.

Return exactly these 18 keys in one JSON object.
Plain-text strings (never arrays or objects): hazard_type, advice,
reasoning, alert_recommendation, likelihood_reason, severity_reason,
exposure_reason, duration_reason, geographic_impact_reason.
Integer scores: likelihood_score, severity_score, exposure_score,
duration_score, geographic_impact_score.
Arrays of strings: recommended_actions, limitations, missing_evidence,
sources.

Each score must be an integer from 1 to 5 when evidence supports it.
If evidence for a dimension is insufficient, return null for that
score, never a guessed value, and explain why in its rationale and in
limitations.
"""


POLICY_BLOCK_START = "=== POLICY SECTION: "
POLICY_BLOCK_END = "=== END POLICY SECTION ==="


def format_policy_text(sections):
    """
    Render policy sections as plain text. Section text is sent
    verbatim (no JSON escaping); a section that contains the block
    delimiters is rejected so it cannot break out of its block.
    """
    blocks = []

    for item in sections:
        if (
            POLICY_BLOCK_START in item["text"]
            or POLICY_BLOCK_END in item["text"]
            or "\n" in item["section"]
        ):
            raise ValueError(
                "Policy section cannot be rendered safely: "
                + item["section"]
            )

        blocks.append(
            f"{POLICY_BLOCK_START}{item['section']} ===\n"
            f"{item['text']}\n{POLICY_BLOCK_END}"
        )

    return "\n\n".join(blocks)


def assess_weather(weather, history, policy, retrieved_sections):
    compact_weather = summarize_weather(weather)
    compact_history = summarize_history(history)

    context = {
        "city": CITY,
        "weather": compact_weather,
        "recent_agent_history": compact_history,
        "policy_version": policy["version"],
        "official_warning_status": (
            "NOT_CHECKED. No NCM or other official warning feed "
            "is integrated."
        ),
        "risk_score_notice": (
            "Any calculated risk score is the provisional internal "
            "model in guidelines.md section 5, not an official "
            "UAE/NCM classification."
        ),
    }

    user_prompt = (
        "Assess the supplied weather situation using the policy "
        "sections below. Do not infer official warning status. "
        "Do not claim the assessment is complete. Cite section "
        "titles exactly as written in the section headers.\n\n"
        + format_policy_text(retrieved_sections)
        + "\n\nEVIDENCE AND STATUS (JSON):\n"
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
        # A sheet URL is optional for a dry-run assessment.
        require_configuration(require_sheet=False)

        # Load policy.
        policy = load_guidelines()
        used.append("load_guidelines")

        # Retrieve weather.
        weather = get_weather(CITY)
        used.append("get_weather")

        # Retrieve recent history if a sheet URL is available.
        history_result = tool_get_history(MAX_HISTORY_RECORDS)
        used.append("get_history")

        if isinstance(history_result, dict) and "error" in history_result:
            history_status = "UNAVAILABLE"
            print("History unavailable:", history_result["error"])
            history = []
        else:
            history_status = "AVAILABLE"
            history = history_result

        # Keep the retrieval query focused on weather hazards.
        query = (
            "weather hazard categories heat wind dust rain flooding "
            "thunderstorms lightning cold weather safety"
        )

        # Identify hazards from the retrieved weather evidence and
        # attach the complete matching section 6 guidance. Anything
        # that cannot fit the budget raises and fails safely.
        hazards = identify_hazards(weather)
        used.append("identify_hazards")
        print("Hazards identified:", sorted(hazards))

        hazard_sections = select_hazard_guidance(policy, hazards)
        conditional_sections = select_conditional_sections(
            policy, hazards, weather
        )

        # Optional ranked sections are disabled (k=0): everything the
        # model receives is mandatory or required, so no optional text
        # can push the request over its budget.
        retrieved_sections = search_guidelines(
            policy,
            query,
            k=0,
            required_sections=hazard_sections + conditional_sections,
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

        # Explicitly record evidence limitations.
        decision["official_warning_status"] = "NOT_CHECKED"
        decision["official_warning_authority"] = None

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
        decision["hazard_screening"] = hazards
        decision["groq_usage"] = dict(LAST_GROQ_USAGE)

        apply_evidence_uncertainty(decision, hazards)

        decision["history_checked"] = (
            history_status == "AVAILABLE"
        )
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

        # An assessment with unverified official warnings cannot
        # authorize a notification.
        decision["approval_status"] = "PENDING_REVIEW"
        decision["send_alert"] = False
        decision["alert_message"] = ""
        decision["notification_status"] = "NOT_SENT"

        enforce_assessment_status(decision)

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
        decision["groq_usage"] = dict(LAST_GROQ_USAGE)

        decision["evidence_quality"] = "E0"
        decision["evidence_confidence"] = "NONE"

        enforce_assessment_status(decision)

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
# 12. OUTPUT AND GOOGLE SHEETS INTEGRATION
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
    print(
        "OFFICIAL WARNING:",
        decision.get("official_warning_status"),
    )
    print("ALERT AUTHORIZED:", decision.get("send_alert"))
    print("NOTIFICATION:", decision.get("notification_status"))

    print("\nLIMITATIONS:")
    for item in decision.get("limitations", []):
        print("-", item)

    print("\nMISSING EVIDENCE:")
    for item in decision.get("missing_evidence", []):
        print("-", item)

    # Preserve existing Apps Script field names.
    payload = {
        "type": "agent",
        "city": CITY,
        "tools_used": ", ".join(used),

        "advice": decision.get("advice", ""),
        "reasoning": decision.get("reasoning", ""),

        "send_alert": False,
        "alert_message": "",

        "sources": ", ".join(decision.get("sources", [])),

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
        "alert_recommendation": decision.get(
            "alert_recommendation"
        ),

        "assessment_timestamp": decision.get("assessment_timestamp"),
        "approval_status": decision.get("approval_status"),
        "notification_status": decision.get("notification_status"),

        "limitations": json_text(
            decision.get("limitations", [])
        ),
        "missing_evidence": json_text(
            decision.get("missing_evidence", [])
        ),
    }

    if dry:
        print("\nDRY RUN: No data was sent to Google Sheets.")
        print("DRY RUN: No email was sent.")
        print("\nOUTPUT PAYLOAD:")
        print(json_text(payload))
        return

    # A normal agent run can write to Sheets, but only after
    # the explicit configuration check below.
    require_configuration(require_sheet=True)

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

    # Risk calculation tests.
    score, band = calculate_risk_score(1, 1, 1, 1, 1)

    check(
        "Minimum valid provisional risk score",
        score == 1.0 and band == "LOW",
    )

    score, band = calculate_risk_score(5, 5, 5, 5, 5)

    check(
        "Maximum valid provisional risk score",
        score == 25.0 and band == "CRITICAL",
    )

    score, band = calculate_risk_score(3, 3, 3, 3, 3)

    check(
        "Provisional risk score calculation",
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
        "missing_evidence": [],
    }

    guarded = apply_official_warning_guard(test_decision)

    check(
        "Unchecked official warning prevents definitive no-alert",
        guarded["alert_recommendation"] == "MONITOR",
    )

    check(
        "Unchecked official warning keeps assessment incomplete",
        guarded["assessment_status"] == "INCOMPLETE",
    )

    test_decision = {
        "alert_recommendation": "OFFICIAL_WARNING_PASS_THROUGH",
        "limitations": [],
        "missing_evidence": [],
    }

    guarded = apply_official_warning_guard(test_decision)

    check(
        "Unverified official warning is escalated",
        guarded["alert_recommendation"] == "ESCALATE_FOR_REVIEW",
    )

    # A decision claiming COMPLETE or a verified warning cannot
    # survive while no official warning feed is integrated.
    guarded = apply_official_warning_guard({
        "alert_recommendation": "MONITOR",
        "assessment_status": "COMPLETE",
        "official_warning_status": "VERIFIED",
        "limitations": [],
        "missing_evidence": [],
    })

    check(
        "COMPLETE status is blocked without warning verification",
        guarded["assessment_status"] == "INCOMPLETE"
        and guarded["official_warning_status"] == "NOT_CHECKED",
    )

    check(
        "Status helper only allows COMPLETE when warnings verified",
        enforce_assessment_status({
            "assessment_status": "COMPLETE",
            "official_warning_status": "NOT_CHECKED",
        })["assessment_status"] == "INCOMPLETE"
        and enforce_assessment_status({
            "assessment_status": "COMPLETE",
        })["assessment_status"] == "INCOMPLETE"
        and enforce_assessment_status({
            "assessment_status": "COMPLETE",
            "official_warning_status": "VERIFIED",
        })["assessment_status"] == "COMPLETE",
    )

    # Band boundaries from guidelines.md section 5.
    check(
        "Risk band boundaries match policy (5, 10, 15)",
        [
            calculate_risk_score(1, 5, 1, 1, 1)[1],   # 5.00
            calculate_risk_score(2, 5, 1, 1, 1)[1],   # 6.75
            calculate_risk_score(2, 5, 2, 2, 2)[1],   # 10.00
            calculate_risk_score(3, 5, 3, 3, 3)[1],   # 15.00
            calculate_risk_score(4, 5, 4, 4, 4)[1],   # 20.00
        ] == ["LOW", "MODERATE", "MODERATE", "HIGH", "CRITICAL"],
    )

    # Heading matching ignores section numbers.
    check(
        "Mandatory matching tolerates renumbered headings",
        find_mandatory_sections({
            f"{index}. {name}": ""
            for index, name in enumerate(
                MANDATORY_SECTION_ALIASES, start=40
            )
        })[1] == [],
    )

    # Helper: temporarily override module globals during a check.
    def with_globals(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    # Unassessable dimensions (guidelines section 5, rule 4).
    def decision_with(**changes):
        base = {
            "hazard_type": "HEAT",
            "likelihood_score": 3,
            "likelihood_reason": "r",
            "severity_score": 3,
            "severity_reason": "r",
            "exposure_score": 3,
            "exposure_reason": "r",
            "duration_score": 3,
            "duration_reason": "r",
            "geographic_impact_score": 3,
            "geographic_impact_reason": "r",
            "alert_recommendation": "NO_ALERT_RECOMMENDED",
            "advice": "a",
            "reasoning": "r",
            "recommended_actions": [],
            "limitations": [],
            "missing_evidence": [],
            "sources": ["Final Decision Rule"],
        }
        base.update(changes)
        return base

    sections = [{"section": "Final Decision Rule", "text": "x"}]

    validated = validate_decision(
        decision_with(exposure_score=None), sections
    )

    check(
        "Unassessed dimension gives no score or risk band",
        validated["risk_score"] is None
        and validated["risk_band"] == "UNASSESSED"
        and validated["exposure_score"] is None,
    )

    check(
        "Unassessed dimension blocks low-risk or advisory outcome",
        validated["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and any(
            "exposure_score" in item
            for item in validated["limitations"]
            + validated["missing_evidence"]
        ),
    )

    check(
        "Unassessed dimension keeps ESCALATE_FOR_REVIEW",
        validate_decision(
            decision_with(
                duration_score="UNASSESSED",
                alert_recommendation="ESCALATE_FOR_REVIEW",
            ),
            sections,
        )["alert_recommendation"] == "ESCALATE_FOR_REVIEW",
    )

    for bad_value in (0, 6, 2.5, "3", True):
        try:
            validate_decision(
                decision_with(severity_score=bad_value), sections
            )
            rejected = False
        except ValueError:
            rejected = True

        check(
            f"Malformed dimension value {bad_value!r} rejected",
            rejected,
        )

    # ------------------------------------------------------------
    # Hazard retrieval, conditional policy, budgets, Groq handling.
    # Everything below is offline: no network, no real API key.
    # ------------------------------------------------------------
    check(
        "Request budgets are the documented explicit values",
        MAX_POLICY_CHARS == 13500
        and MAX_REQUEST_CHARS == 19000
        and MAX_ESTIMATED_REQUEST_TOKENS == 8000
        and MAX_COMPLETION_TOKENS == 700,
    )

    def make_weather(current=None, today=None, hourly=None):
        base_current = {
            "time": "2026-11-07T12:00",
            "temperature_2m": 22,
            "apparent_temperature": 22,
            "precipitation": 0,
            "wind_speed_10m": 5,
            "wind_gusts_10m": 10,
            "visibility": 20000,
            "weather_code": 0,
            "cape": 0,
        }
        base_today = {
            "high": 24, "low": 18, "wind": 8,
            "gust": 15, "precipitation": 0,
        }
        base_hour = {
            "time": "2026-11-07T13:00",
            "temperature": 22, "apparent_temperature": 22,
            "precipitation": 0, "wind": 5, "gust": 10,
            "visibility": 20000, "weather_code": 0, "cape": 0,
        }
        return {
            "city": CITY,
            "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-07T12:00:00+04:00",
            "current": {**base_current, **(current or {})},
            "today_forecast": {**base_today, **(today or {})},
            "previous_days": [],
            "hourly_forecast_next_24h": [
                {**base_hour, **(hourly or {})}
            ],
            "history_summary": {},
        }

    def hazards_of(**kwargs):
        return identify_hazards(make_weather(**kwargs))

    check(
        "Benign weather identifies no hazard",
        hazards_of() == {},
    )

    check(
        "High CAPE without rain evidence identifies no hazard",
        hazards_of(current={"cape": 3000}, hourly={"cape": 3000})
        == {},
    )

    individual_cases = {
        "HEAT": (
            dict(
                current={"temperature_2m": 41},
                today={"high": 42},
                hourly={"temperature": 41},
            ),
            {"HEAT"},
        ),
        "HEAT via apparent temperature": (
            dict(current={"apparent_temperature": 38}),
            {"HEAT"},
        ),
        "COLD": (
            dict(
                current={"temperature_2m": 8},
                today={"low": 7},
                hourly={"temperature": 8},
            ),
            {"COLD"},
        ),
        "WIND_DUST via wind": (
            dict(current={"wind_speed_10m": 35}),
            {"WIND_DUST"},
        ),
        "WIND_DUST via gust": (
            dict(today={"gust": 60}),
            {"WIND_DUST"},
        ),
        "WIND_DUST via visibility": (
            dict(current={"visibility": 1500}),
            {"WIND_DUST"},
        ),
        "RAIN_FLOOD via precipitation (no lightning)": (
            dict(
                hourly={"precipitation": 2.5, "weather_code": 61}
            ),
            {"RAIN_FLOOD"},
        ),
        "RAIN_FLOOD via drizzle code and precipitation": (
            dict(
                current={"precipitation": 0.1, "weather_code": 51}
            ),
            {"RAIN_FLOOD"},
        ),
        "RAIN_FLOOD via rain code alone": (
            dict(hourly={"weather_code": 63}),
            {"RAIN_FLOOD"},
        ),
        "LIGHTNING via thunderstorm code (also rain)": (
            dict(hourly={"weather_code": 95}),
            {"RAIN_FLOOD", "LIGHTNING"},
        ),
        "LIGHTNING via CAPE with rain evidence": (
            dict(
                current={"precipitation": 1.0, "weather_code": 61},
                hourly={"cape": 1500},
            ),
            {"RAIN_FLOOD", "LIGHTNING"},
        ),
    }

    for label, (kwargs, expected) in individual_cases.items():
        check(
            f"Hazard identified individually: {label}",
            set(hazards_of(**kwargs)) == expected,
        )

    check(
        "Simultaneous heat and strong wind identify both hazards",
        set(
            hazards_of(
                current={"temperature_2m": 41, "wind_gusts_10m": 70},
                today={"high": 42},
            )
        ) == {"HEAT", "WIND_DUST"},
    )

    check(
        "Every hazard can be identified at the same time",
        set(
            hazards_of(
                current={
                    "temperature_2m": 8,
                    "apparent_temperature": 40,
                    "wind_gusts_10m": 70,
                    "precipitation": 3,
                },
                today={"high": 42},
                hourly={"weather_code": 95},
            )
        ) == set(HAZARD_POLICY_TITLES),
    )

    # Missing or unreliable evidence is uncertainty, never safety.
    def only_uncertain(hazards):
        return all(
            reasons and all(r.startswith("UNCERTAIN") for r in reasons)
            for reasons in hazards.values()
        )

    missing_visibility = hazards_of(
        current={"visibility": None}, hourly={"visibility": None}
    )

    check(
        "Missing visibility makes only wind and dust uncertain",
        set(missing_visibility) == {"WIND_DUST"}
        and only_uncertain(missing_visibility),
    )

    no_rain_data = hazards_of(
        current={"precipitation": None, "weather_code": None},
        today={"precipitation": None},
        hourly={"precipitation": None, "weather_code": None},
    )

    check(
        "Missing precipitation and weather code make rain and "
        "lightning uncertain",
        set(no_rain_data) == {"RAIN_FLOOD", "LIGHTNING"}
        and only_uncertain(no_rain_data),
    )

    no_code = hazards_of(
        current={"weather_code": None}, hourly={"weather_code": None}
    )

    check(
        "Missing weather code makes only lightning uncertain",
        set(no_code) == {"LIGHTNING"} and only_uncertain(no_code),
    )

    bad_code = hazards_of(
        current={"weather_code": 12345}, hourly={"weather_code": "x"}
    )

    check(
        "Unrecognized weather codes are rejected as uncertainty",
        set(bad_code) == {"RAIN_FLOOD", "LIGHTNING"}
        and only_uncertain(bad_code),
    )

    check(
        "Implausible temperature is rejected as uncertainty",
        {"HEAT", "COLD"} <= set(
            hazards_of(
                current={"temperature_2m": 999},
                today={"high": 999, "low": -999},
                hourly={"temperature": "hot"},
            )
        ),
    )

    check(
        "Negative precipitation is rejected as uncertainty",
        "RAIN_FLOOD" in hazards_of(
            current={"precipitation": -5},
            hourly={"precipitation": -5},
            today={"precipitation": -5},
        ),
    )

    check(
        "Missing or unreadable weather makes every hazard uncertain",
        set(identify_hazards({})) == set(HAZARD_POLICY_TITLES)
        and set(identify_hazards(None)) == set(HAZARD_POLICY_TITLES)
        and set(identify_hazards({"current": "bad"}))
        == set(HAZARD_POLICY_TITLES)
        and only_uncertain(identify_hazards({})),
    )

    check(
        "uncertain_hazards separates gaps from detections",
        set(uncertain_hazards(missing_visibility)) == {"WIND_DUST"}
        and uncertain_hazards(
            hazards_of(current={"temperature_2m": 41},
                       today={"high": 41})
        ) == {},
    )

    # Evidence gaps cannot end in a low-risk or monitor-only result.
    def gap_decision(recommendation):
        return apply_evidence_uncertainty(
            {
                "alert_recommendation": recommendation,
                "assessment_status": "COMPLETE",
                "limitations": [],
                "missing_evidence": [],
            },
            missing_visibility,
        )

    check(
        "Evidence gap maps NO_ALERT and MONITOR to UNABLE_TO_ASSESS",
        gap_decision("NO_ALERT_RECOMMENDED")["alert_recommendation"]
        == "UNABLE_TO_ASSESS"
        and gap_decision("MONITOR")["alert_recommendation"]
        == "UNABLE_TO_ASSESS",
    )

    check(
        "Evidence gap maps ADVISORY to ESCALATE_FOR_REVIEW",
        gap_decision("ADVISORY_RECOMMENDED")["alert_recommendation"]
        == "ESCALATE_FOR_REVIEW"
        and gap_decision("ESCALATE_FOR_REVIEW")["alert_recommendation"]
        == "ESCALATE_FOR_REVIEW"
        and gap_decision("UNABLE_TO_ASSESS")["alert_recommendation"]
        == "UNABLE_TO_ASSESS",
    )

    gapped = gap_decision("MONITOR")

    check(
        "Evidence gap is recorded and keeps the assessment INCOMPLETE",
        gapped["assessment_status"] == "INCOMPLETE"
        and any("WIND_DUST" in item for item in gapped["limitations"])
        and any(
            "WIND_DUST" in item for item in gapped["missing_evidence"]
        ),
    )

    clean = apply_evidence_uncertainty(
        {
            "alert_recommendation": "MONITOR",
            "limitations": [],
            "missing_evidence": [],
        },
        {},
    )

    check(
        "No evidence gap leaves the recommendation unchanged",
        clean["alert_recommendation"] == "MONITOR"
        and clean["limitations"] == [],
    )

    # Complete retrieval: compare with the raw guidelines.md text,
    # found independently of the production parser.
    raw_policy = GUIDELINES_PATH.read_text(encoding="utf-8")

    def raw_block(heading):
        match = re.search(
            r"^### " + re.escape(heading) + r"\n(.*?)(?=^#{2,3} |\Z)",
            raw_policy,
            re.S | re.M,
        )
        return match.group(1).strip() if match else None

    def raw_section(heading):
        match = re.search(
            r"^## " + re.escape(heading) + r"\n(.*?)(?=^## |\Z)",
            raw_policy,
            re.S | re.M,
        )
        return match.group(1).strip() if match else None

    raw_headings = {
        "HEAT": "6.1 Extreme Heat",
        "WIND_DUST": "6.2 Wind and Dust",
        "RAIN_FLOOD": "6.3 Rain, Thunderstorms and Flooding",
        "COLD": "6.4 Low Temperatures and Cold Conditions",
        "LIGHTNING": "6.5 Lightning and Thunderstorms",
    }

    hazard_policy = load_guidelines()

    for code, heading in raw_headings.items():
        retrieved = select_hazard_guidance(hazard_policy, {code: ["t"]})
        texts_by_title = {
            item["section"]: item["text"] for item in retrieved
        }

        check(
            f"Complete policy text retrieved for {code}",
            raw_block(heading) is not None
            and texts_by_title.get(heading) == raw_block(heading)
            and len(retrieved) == 2,
        )

    heat_wind = select_hazard_guidance(
        hazard_policy, {"WIND_DUST": ["t"], "HEAT": ["t"]}
    )
    heat_wind_titles = [item["section"] for item in heat_wind]

    check(
        "Heat and wind retrieve both subsections in policy order",
        heat_wind_titles[1:] == [
            "6.1 Extreme Heat", "6.2 Wind and Dust",
        ]
        and heat_wind[0]["text"].startswith(
            "Assess each hazard independently"
        ),
    )

    check(
        "Only relevant hazard subsections are retrieved",
        not any(
            title.startswith(("6.3", "6.4", "6.5"))
            for title in heat_wind_titles
        ),
    )

    # Missing and unrecognized hazard sections.
    parent_title = next(
        title for title in hazard_policy["sections"]
        if normalize_heading(title) == "weather hazard assessment"
    )

    def policy_with_section_6(transform):
        changed = dict(hazard_policy)
        changed["sections"] = dict(hazard_policy["sections"])
        changed["sections"][parent_title] = transform(
            hazard_policy["sections"][parent_title]
        )
        return changed

    def error_of(function):
        try:
            function()
            return ""
        except ValueError as exc:
            return str(exc)

    without_wind = policy_with_section_6(
        lambda body: re.sub(
            r"### 6\.2 Wind and Dust\n.*?(?=### )", "", body,
            flags=re.S,
        )
    )

    check(
        "Missing required hazard subsection fails safely",
        "missing" in error_of(
            lambda: select_hazard_guidance(
                without_wind, {"WIND_DUST": ["t"]}
            )
        ),
    )

    check(
        "Missing subsection is irrelevant when its hazard is absent",
        len(select_hazard_guidance(without_wind, {"HEAT": ["t"]})) == 2,
    )

    with_extra = policy_with_section_6(
        lambda body: body + "\n\n### 6.6 Seismic Activity\n\nExtra text."
    )
    extra_titles = [
        item["section"]
        for item in select_hazard_guidance(with_extra, {"HEAT": ["t"]})
    ]

    check(
        "Unrecognized hazard subsection is always retrieved",
        "6.6 Seismic Activity" in extra_titles
        and "6.1 Extreme Heat" in extra_titles
        and "6.6 Seismic Activity" in [
            item["section"]
            for item in select_hazard_guidance(with_extra, {})
        ],
    )

    check(
        "Unrecognized hazard code is rejected",
        "Unrecognized" in error_of(
            lambda: select_hazard_guidance(
                hazard_policy, {"TSUNAMI": ["t"]}
            )
        ),
    )

    no_parent = dict(hazard_policy)
    no_parent["sections"] = {
        title: body
        for title, body in hazard_policy["sections"].items()
        if title != parent_title
    }

    check(
        "Missing Section 6 fails safely",
        "missing" in error_of(
            lambda: select_hazard_guidance(no_parent, {"HEAT": ["t"]})
        ),
    )

    # Conditional compliance and data-quality sections (7 and 11).
    def conditional_titles(hazards, **kwargs):
        return [
            normalize_heading(item["section"])
            for item in select_conditional_sections(
                hazard_policy, hazards, make_weather(**kwargs)
            )
        ]

    compliance = "regulatory and workplace compliance"
    data_quality = "data quality and assessment completeness"

    check(
        "Section 7 is not retrieved outside heat and the "
        "midday-break season",
        conditional_titles({}) == [],
    )

    check(
        "Section 7 is retrieved when heat is relevant",
        conditional_titles({"HEAT": ["t"]}) == [compliance],
    )

    check(
        "Section 7 is retrieved inside the midday-break season",
        conditional_titles({}, current={"time": "2026-07-01T12:00"})
        == [compliance]
        and conditional_titles(
            {}, current={"time": "2026-06-15T09:00"}
        ) == [compliance]
        and conditional_titles(
            {}, current={"time": "2026-09-15T09:00"}
        ) == [compliance]
        and conditional_titles(
            {}, current={"time": "2026-06-14T09:00"}
        ) == []
        and conditional_titles(
            {}, current={"time": "2026-09-16T09:00"}
        ) == [],
    )

    check(
        "Section 7 is retrieved when the date cannot be established",
        conditional_titles({}, current={"time": "garbage"})
        == [compliance],
    )

    check(
        "Section 11 is retrieved only when evidence is uncertain",
        conditional_titles({"WIND_DUST": ["UNCERTAIN: x"]})
        == [data_quality]
        and conditional_titles(
            {"HEAT": ["UNCERTAIN: x"]}
        ) == [compliance, data_quality]
        and data_quality not in conditional_titles({"HEAT": ["t"]}),
    )

    conditional_texts = {
        normalize_heading(item["section"]): item["text"]
        for item in select_conditional_sections(
            hazard_policy,
            {"HEAT": ["UNCERTAIN: x"]},
            make_weather(),
        )
    }

    check(
        "Sections 7 and 11 are retrieved completely",
        conditional_texts.get(compliance)
        == raw_section("7. Regulatory and Workplace Compliance")
        and conditional_texts.get(data_quality)
        == raw_section("11. Data Quality and Assessment Completeness"),
    )

    no_compliance = dict(hazard_policy)
    no_compliance["sections"] = {
        title: body
        for title, body in hazard_policy["sections"].items()
        if normalize_heading(title) != compliance
    }

    check(
        "Missing Section 7 fails safely",
        "missing" in error_of(
            lambda: select_conditional_sections(
                no_compliance, {"HEAT": ["t"]}, make_weather()
            )
        ),
    )

    # Mandatory core now includes section 4 scales and section 13.
    def retrieve(hazards, budget=None, k=0, duplicate=False,
                 weather=None):
        weather = weather or make_weather()
        required = select_hazard_guidance(hazard_policy, hazards)
        required += select_conditional_sections(
            hazard_policy, hazards, weather
        )

        if duplicate:
            required = required + required

        def run():
            return search_guidelines(
                hazard_policy, "heat wind rain", k=k,
                required_sections=required,
            )

        if budget is None:
            return run()

        return with_globals({"MAX_POLICY_CHARS": budget}, run)

    everything = {code: ["t"] for code in HAZARD_POLICY_TITLES}
    results = retrieve(everything, 100000, k=5, duplicate=True)
    titles = [item["section"] for item in results]
    texts_by_name = {
        normalize_heading(item["section"]): item["text"]
        for item in results
    }

    check(
        "Retrieved sections contain no duplicates",
        len(titles) == len(set(titles)),
    )

    check(
        "Whole Section 6 is never sent alongside its subsections",
        sum(
            1 for title in titles
            if normalize_heading(title) == "weather hazard assessment"
        ) == 1
        and next(
            len(item["text"]) for item in results
            if normalize_heading(item["section"])
            == "weather hazard assessment"
        ) < 200,
    )

    check(
        "Mandatory core is included with every hazard retrieval",
        all(
            normalize_heading(name) in texts_by_name
            for name in MANDATORY_SECTION_ALIASES
        ),
    )

    check(
        "Section 4 scoring definitions are included completely",
        texts_by_name.get("risk assessment dimensions")
        == raw_section("4. Risk Assessment Dimensions")
        and all(
            marker in texts_by_name["risk assessment dimensions"]
            for marker in (
                "Likelihood (L)", "Severity (S)", "Exposure (E)",
                "Duration (D)", "Geographic Impact (G)",
            )
        ),
    )

    check(
        "Section 4 is included even when no hazard is detected",
        "risk assessment dimensions" in [
            normalize_heading(item["section"]) for item in retrieve({})
        ],
    )

    check(
        "Section 13 advice requirements are always included",
        "advice and communication requirements" in [
            normalize_heading(item["section"]) for item in retrieve({})
        ],
    )

    check(
        "Section 12 (output schema) is enforced in code, not sent",
        "required structured assessment output" not in {
            normalize_heading(item["section"])
            for item in retrieve(everything, 100000)
        },
    )

    # Budget enforcement (never silently truncate).
    required_chars = sum(
        len(item["section"]) + len(item["text"])
        for item in retrieve({"HEAT": ["t"]}, 100000)
    )

    check(
        "Required text exactly at the character budget is accepted",
        sum(
            len(item["section"]) + len(item["text"])
            for item in retrieve({"HEAT": ["t"]}, required_chars)
        ) == required_chars,
    )

    def retrieval_error(hazards, budget):
        return error_of(lambda: retrieve(hazards, budget))

    check(
        "One character over the budget fails retrieval",
        "exceed" in retrieval_error({"HEAT": ["t"]}, required_chars - 1),
    )

    check(
        "Benign weather fits the policy budget",
        retrieval_error({}, MAX_POLICY_CHARS) == "",
    )

    check(
        "All five hazards plus sections 7 and 11 fit the policy budget",
        retrieval_error(
            {code: ["UNCERTAIN: t"] for code in HAZARD_POLICY_TITLES},
            MAX_POLICY_CHARS,
        ) == "",
    )

    check(
        "Policy growth beyond the budget fails closed",
        "exceed" in retrieval_error(
            {code: ["UNCERTAIN: t"] for code in HAZARD_POLICY_TITLES},
            MAX_POLICY_CHARS - 2000,
        ),
    )

    # Plain-text policy rendering.
    check(
        "Plain-text policy keeps text verbatim and in blocks",
        all(
            item["text"] in format_policy_text(results)
            and f"{POLICY_BLOCK_START}{item['section']} ==="
            in format_policy_text(results)
            for item in results
        )
        and "\\n" not in format_policy_text(results[:1]),
    )

    check(
        "Policy text containing a block delimiter is rejected",
        "cannot be rendered" in error_of(
            lambda: format_policy_text(
                [{"section": "S", "text": POLICY_BLOCK_END + " x"}]
            )
        )
        and "cannot be rendered" in error_of(
            lambda: format_policy_text(
                [{"section": "S\nT", "text": "x"}]
            )
        ),
    )

    # Request construction and request-size budgets.
    worst_history = [
        {
            "timestamp": "t" * 60,
            "city": "c" * 60,
            "advice": "a" * 500,
            "alert_sent": False,
            "alert_message": "m" * 500,
        }
    ] * MAX_HISTORY_RECORDS

    def build_request(hazards, weather=None):
        """Real retrieval and prompt assembly under the real limits."""
        weather = weather or make_weather()
        captured = []

        def capture(messages, temperature=0.1):
            captured.append(messages)
            raise RuntimeError("captured")

        sections = retrieve(hazards, MAX_POLICY_CHARS, weather=weather)

        def assemble():
            try:
                assess_weather(
                    weather, worst_history, hazard_policy, sections
                )
            except RuntimeError:
                pass

        with_globals({"ask_groq": capture}, assemble)

        return sections, captured[0]

    def fits_all_budgets(hazards):
        sections, messages = build_request(
            {code: ["t"] for code in hazards}
        )
        serialized = json_text(messages)

        return (
            len(serialized) <= MAX_REQUEST_CHARS
            and estimate_tokens(serialized) + MAX_COMPLETION_TOKENS
            <= MAX_ESTIMATED_REQUEST_TOKENS
        )

    check(
        "No-hazard request fits every budget with worst-case history",
        fits_all_budgets([]),
    )

    for code in HAZARD_POLICY_TITLES:
        check(
            f"Single-hazard request fits every budget: {code}",
            fits_all_budgets([code]),
        )

    check(
        "Heat with wind request fits every budget",
        fits_all_budgets(["HEAT", "WIND_DUST"]),
    )

    check(
        "All five hazards together fit every budget",
        fits_all_budgets(list(HAZARD_POLICY_TITLES)),
    )

    uncertain_all = {
        code: ["UNCERTAIN: t"] for code in HAZARD_POLICY_TITLES
    }
    _, extreme_messages = build_request(uncertain_all)

    class NoPostHttp:
        def __init__(self):
            self.posts = []

        def post(self, *args, **kwargs):
            self.posts.append(args)
            raise AssertionError("request must be blocked first")

    def groq_error(messages, **overrides):
        recorder = NoPostHttp()
        settings = {"GROQ_API_KEY": "test-key", "HTTP": recorder}
        settings.update(overrides)

        message = error_of_runtime(
            lambda: with_globals(
                settings, lambda: ask_groq(messages)
            )
        )

        return message, recorder.posts

    def error_of_runtime(function):
        try:
            function()
            return ""
        except (ValueError, RuntimeError) as exc:
            return str(exc)

    message, posts = groq_error(extreme_messages)

    check(
        "Worst case (all uncertain, sections 7 and 11) fails closed "
        "on the token estimate before sending",
        "token budget" in message and not posts,
    )

    oversized = [
        {"role": "user", "content": "x" * (MAX_REQUEST_CHARS + 1)}
    ]
    message, posts = groq_error(oversized)

    check(
        "Request over the character limit is blocked before sending",
        "character limit" in message and not posts,
    )

    message, posts = groq_error(
        [{"role": "user", "content": "short"}],
        MAX_ESTIMATED_REQUEST_TOKENS=10,
    )

    check(
        "Request over the token budget is blocked before sending",
        "token budget" in message and not posts,
    )

    # Prompt content.
    sections, messages = build_request({"HEAT": ["t"], "WIND_DUST": ["t"]})
    user_content = messages[1]["content"]

    check(
        "Prompt holds policy as plain text, not JSON-escaped",
        all(item["text"] in user_content for item in sections)
        and "retrieved_guideline_sections" not in user_content
        and "Risk Assessment Dimensions" in user_content,
    )

    check(
        "System prompt keeps its safety requirements",
        all(
            phrase in " ".join(SYSTEM_PROMPT.split())
            for phrase in (
                "NOT_CHECKED",
                "never a completed safety clearance",
                "Never treat missing evidence as a low score",
                "not an official UAE warning",
                "return null for that",
            )
        ),
    )

    check(
        "History overhead is capped per record",
        len(json_text(summarize_history(worst_history)))
        < 3 * 400,
    )

    # Pipeline with mocked weather and model (offline).
    sent_messages = []

    model_reply = json_text({
        "hazard_type": "NONE",
        "likelihood_score": 1, "likelihood_reason": "r",
        "severity_score": 1, "severity_reason": "r",
        "exposure_score": 1, "exposure_reason": "r",
        "duration_score": 1, "duration_reason": "r",
        "geographic_impact_score": 1,
        "geographic_impact_reason": "r",
        "alert_recommendation": "NO_ALERT_RECOMMENDED",
        "advice": "a", "reasoning": "r",
        "recommended_actions": [], "limitations": [],
        "missing_evidence": [],
        "sources": ["16. Final Decision Rule"],
    })

    def recording_model(messages, temperature=0.1):
        sent_messages.append(messages)
        return model_reply

    def pipeline(weather_fixture, **overrides):
        settings = {
            "require_configuration": lambda require_sheet=False: None,
            "get_weather": lambda city: weather_fixture,
            "tool_get_history": lambda n=3: [],
            "ask_groq": recording_model,
        }
        settings.update(overrides)

        return with_globals(settings, lambda: run_weather_agent()[0])

    benign_decision = pipeline(make_weather())

    check(
        "Benign weather completes retrieval and stays INCOMPLETE",
        benign_decision["hazard_screening"] == {}
        and benign_decision["assessment_status"] == "INCOMPLETE"
        and benign_decision["alert_recommendation"] == "MONITOR"
        and benign_decision["send_alert"] is False
        and len(sent_messages) == 1
        and len(json_text(sent_messages[0])) <= MAX_REQUEST_CHARS,
    )

    sent_messages.clear()

    hot_windy = make_weather(
        current={"temperature_2m": 41, "wind_gusts_10m": 70},
        today={"high": 42},
    )

    hot_decision = pipeline(hot_windy)
    hot_request = (
        sent_messages[0][1]["content"] if sent_messages else ""
    )

    check(
        "Hot and windy weather retrieves heat, wind, scales and "
        "compliance guidance",
        hot_decision["assessment_status"] == "INCOMPLETE"
        and set(hot_decision["hazard_screening"])
        == {"HEAT", "WIND_DUST"}
        and "Temperature alone is insufficient" in hot_request
        and "Distinguish sustained wind from short-duration"
        in hot_request
        and "Do not invent frost" not in hot_request
        and "Likelihood (L)" in hot_request
        and "UAE midday break" in hot_request
        and "Do not claim that all conditions are safe" in hot_request,
    )

    sent_messages.clear()

    all_missing = make_weather(
        current={
            key: None for key in make_weather()["current"]
            if key != "time"
        },
        today={key: None for key in make_weather()["today_forecast"]},
        hourly={
            key: None for key in make_weather()[
                "hourly_forecast_next_24h"
            ][0]
            if key != "time"
        },
    )

    gap_pipeline = pipeline(all_missing)
    gap_request = (
        sent_messages[0][1]["content"] if sent_messages else ""
    )

    check(
        "Missing weather evidence retrieves section 11, flags the gap "
        "and blocks a monitor-only result",
        set(gap_pipeline["hazard_screening"])
        == set(HAZARD_POLICY_TITLES)
        and "Before producing an assessment, verify" in gap_request
        and gap_pipeline["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and gap_pipeline["assessment_status"] == "INCOMPLETE"
        and any(
            "evidence gaps" in item
            for item in gap_pipeline["missing_evidence"]
        ),
    )

    sent_messages.clear()

    failed_decision = pipeline(
        hot_windy,
        MAX_POLICY_CHARS=5000,
    )

    check(
        "Policy that cannot fit the budget gives INCOMPLETE without a "
        "model call",
        failed_decision["assessment_status"] == "INCOMPLETE"
        and failed_decision["alert_recommendation"]
        == "UNABLE_TO_ASSESS"
        and failed_decision["risk_band"] == "UNASSESSED"
        and failed_decision["risk_score"] is None
        and failed_decision["official_warning_status"]
        == "NOT_CHECKED"
        and failed_decision["send_alert"] is False
        and failed_decision["notification_status"] == "NOT_SENT"
        and "exceed" in failed_decision["reasoning"]
        and not sent_messages,
    )

    # Groq response handling with a fake transport (offline).
    secret_key = "gsk_TEST_SECRET_VALUE_123"

    class FakeResponse:
        def __init__(self, status_code=200, body=None, text="",
                     headers=None):
            self.status_code = status_code
            self._body = body
            self.text = text
            self.headers = headers or {}

        def json(self):
            if self._body is None:
                raise ValueError("no json")

            return self._body

    class FakeHttp:
        def __init__(self, response):
            self.response = response
            self.requests = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.requests.append(
                {"url": url, "headers": headers, "json": json}
            )
            return self.response

    def groq_body(content=model_reply, finish="stop", usage="default"):
        if usage == "default":
            usage = {
                "prompt_tokens": 1234,
                "completion_tokens": 456,
                "total_tokens": 1690,
            }

        body = {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish,
                }
            ]
        }

        if usage is not None:
            body["usage"] = usage

        return body

    def call_groq(response):
        transport = FakeHttp(response)
        output = io.StringIO()
        outcome = {}

        def run():
            try:
                with contextlib.redirect_stdout(output):
                    outcome["content"] = ask_groq(
                        [{"role": "user", "content": "hello"}]
                    )
            except (ValueError, RuntimeError) as exc:
                outcome["error"] = str(exc)

        with_globals(
            {"GROQ_API_KEY": secret_key, "HTTP": transport}, run
        )

        outcome["usage"] = dict(LAST_GROQ_USAGE)
        outcome["output"] = output.getvalue()
        outcome["transport"] = transport

        return outcome

    ok = call_groq(FakeResponse(body=groq_body()))

    check(
        "Groq usage fields are captured and logged",
        ok.get("content") == model_reply
        and ok["usage"]["prompt_tokens"] == 1234
        and ok["usage"]["completion_tokens"] == 456
        and ok["usage"]["total_tokens"] == 1690
        and ok["usage"]["finish_reason"] == "stop"
        and "prompt_tokens: 1234" in ok["output"]
        and "finish_reason: stop" in ok["output"],
    )

    sent_payload = ok["transport"].requests[0]["json"]

    check(
        "gpt-oss reasoning configuration is supported parameters only",
        "reasoning_format" not in sent_payload
        and sent_payload["include_reasoning"] is False
        and sent_payload["reasoning_effort"] == "low"
        and sent_payload["max_completion_tokens"]
        == MAX_COMPLETION_TOKENS
        and sent_payload["model"] == "openai/gpt-oss-20b",
    )

    check(
        "API key is sent as a header but never logged or recorded",
        ok["transport"].requests[0]["headers"]["Authorization"]
        == "Bearer " + secret_key
        and secret_key not in ok["output"]
        and secret_key not in json_text(ok["usage"]),
    )

    truncated = call_groq(
        FakeResponse(body=groq_body(content="{\"hazard", finish="length"))
    )

    check(
        "Truncated response (finish_reason length) is rejected",
        "cut off" in truncated.get("error", "")
        and "content" not in truncated
        and truncated["usage"]["finish_reason"] == "length"
        and secret_key not in truncated["error"],
    )

    check(
        "Missing finish_reason is rejected",
        "cut off" in call_groq(
            FakeResponse(body=groq_body(finish=None))
        ).get("error", ""),
    )

    missing_usage = call_groq(FakeResponse(body=groq_body(usage=None)))

    check(
        "Absent usage data is recorded as null, not invented",
        missing_usage.get("content") == model_reply
        and missing_usage["usage"]["prompt_tokens"] is None
        and missing_usage["usage"]["total_tokens"] is None,
    )

    too_large = call_groq(
        FakeResponse(status_code=413, text="Request too large", body={})
    )
    limited = call_groq(
        FakeResponse(
            status_code=429, text="rate limit",
            headers={"retry-after": "12"}, body={},
        )
    )

    check(
        "Token-limit errors (413, 429) are explicit and key-free",
        "HTTP 413" in too_large["error"]
        and "too large" in too_large["error"]
        and "HTTP 429" in limited["error"]
        and "rate or token limit" in limited["error"]
        and "retry-after 12" in limited["error"]
        and secret_key not in too_large["error"] + limited["error"]
        + too_large["output"] + limited["output"],
    )

    check(
        "Empty or malformed Groq structures are rejected",
        "empty" in call_groq(
            FakeResponse(body=groq_body(content="   "))
        ).get("error", "")
        and "unexpected response structure" in call_groq(
            FakeResponse(body={"nothing": True})
        ).get("error", ""),
    )

    def groq_pipeline(response):
        transport = FakeHttp(response)

        decision = with_globals(
            {
                "GROQ_API_KEY": secret_key,
                "HTTP": transport,
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: make_weather(),
                "tool_get_history": lambda n=3: [],
            },
            lambda: run_weather_agent()[0],
        )

        return decision

    for label, response in {
        "truncated output": FakeResponse(
            body=groq_body(content="{\"hazard", finish="length")
        ),
        "malformed JSON": FakeResponse(
            body=groq_body(content="{not valid json")
        ),
        "non-object JSON": FakeResponse(
            body=groq_body(content="[1, 2, 3]")
        ),
        "token-limit error": FakeResponse(
            status_code=413, text="too large", body={}
        ),
        "rate-limit error": FakeResponse(
            status_code=429, text="slow down", body={}
        ),
    }.items():
        decision = groq_pipeline(response)

        check(
            f"Pipeline fails safely to INCOMPLETE on {label}",
            decision["assessment_status"] == "INCOMPLETE"
            and decision["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and decision["risk_band"] == "UNASSESSED"
            and decision["send_alert"] is False
            and decision["notification_status"] == "NOT_SENT"
            and decision["official_warning_status"] == "NOT_CHECKED"
            and secret_key not in json_text(decision),
        )

    good = groq_pipeline(FakeResponse(body=groq_body()))

    check(
        "Pipeline records actual Groq usage on the decision",
        good["groq_usage"]["prompt_tokens"] == 1234
        and good["groq_usage"]["finish_reason"] == "stop"
        and good["assessment_status"] == "INCOMPLETE",
    )

    # ------------------------------------------------------------
    # Regression: GitHub Actions run (main @ 587e465) failed with
    #   ValueError: Required policy headings are missing from
    #   guidelines.md: Purpose and Safety Principles; ...
    # That code still required eight legacy headings that the v2.0
    # guidelines.md does not contain. The scheduled run executed main,
    # which did not yet have the numbered-section implementation.
    # ------------------------------------------------------------
    legacy_headings = (
        "Purpose and Safety Principles",
        "Source Authority and Validation",
        "Risk Assessment and Alert Severity",
        "Alert Decision Requirements",
        "Alert Approval and Authorization Policy",
        "Failure Handling and Safe Operation",
        "Audit Trail and Compliance Controls",
        "Final Safety Rule",
    )
    legacy_aliases = {name: (name,) for name in legacy_headings}
    policy_headings = {
        normalize_heading(title)
        for title in hazard_policy["sections"]
    }
    legacy_missing = [
        name for name in legacy_headings
        if normalize_heading(name) not in policy_headings
    ]

    def legacy_requirement_error():
        def attempt():
            return error_of(
                lambda: search_guidelines(
                    hazard_policy, "heat", k=1
                )
            )

        return with_globals(
            {"MANDATORY_SECTION_ALIASES": legacy_aliases}, attempt
        )

    check(
        "Legacy heading requirement reproduces the exact Actions error",
        legacy_missing == list(legacy_headings)
        and legacy_requirement_error()
        == "Required policy headings are missing from guidelines.md: "
        + "; ".join(legacy_headings),
    )

    legacy_normalized = {
        normalize_heading(name) for name in legacy_headings
    }
    required_normalized = {
        normalize_heading(alias)
        for aliases in MANDATORY_SECTION_ALIASES.values()
        for alias in aliases
    }

    check(
        "Current mandatory sections contain none of the legacy headings",
        not required_normalized & legacy_normalized
        and required_normalized <= policy_headings,
    )

    def string_constants_outside_tests():
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        found = set()

        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "run_tests"
            ):
                continue

            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                ):
                    found.add(normalize_heading(child.value))

        return found

    production_text = string_constants_outside_tests()

    check(
        "No production code requires a legacy heading",
        not any(
            legacy in text
            for text in production_text
            for legacy in legacy_normalized
        ),
    )

    # Self-contained: own fixtures, own mocks, the real repository
    # guidelines.md and the real run_weather_agent() retrieval path.
    def run_real_retrieval_path():
        fixture = {
            "city": CITY,
            "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-07T12:00:00+04:00",
            "current": {
                "time": "2026-11-07T12:00",
                "temperature_2m": 22, "apparent_temperature": 22,
                "precipitation": 0, "wind_speed_10m": 5,
                "wind_gusts_10m": 10, "visibility": 20000,
                "weather_code": 0, "cape": 0,
            },
            "today_forecast": {
                "high": 24, "low": 18, "wind": 8,
                "gust": 15, "precipitation": 0,
            },
            "previous_days": [],
            "hourly_forecast_next_24h": [
                {
                    "time": "2026-11-07T13:00",
                    "temperature": 22, "apparent_temperature": 22,
                    "precipitation": 0, "wind": 5, "gust": 10,
                    "visibility": 20000, "weather_code": 0,
                    "cape": 0,
                }
            ],
            "history_summary": {},
        }
        reply = json_text({
            "hazard_type": "NONE",
            "likelihood_score": 1, "likelihood_reason": "r",
            "severity_score": 1, "severity_reason": "r",
            "exposure_score": 1, "exposure_reason": "r",
            "duration_score": 1, "duration_reason": "r",
            "geographic_impact_score": 1,
            "geographic_impact_reason": "r",
            "alert_recommendation": "NO_ALERT_RECOMMENDED",
            "advice": "a", "reasoning": "r",
            "recommended_actions": [], "limitations": [],
            "missing_evidence": [],
            "sources": ["16. Final Decision Rule"],
        })
        mocks = {
            "require_configuration": lambda require_sheet=False: None,
            "get_weather": lambda city: fixture,
            "tool_get_history": lambda n=3: [],
            "ask_groq": lambda messages, temperature=0.1: reply,
        }
        saved = {name: globals()[name] for name in mocks}
        printed = io.StringIO()
        globals().update(mocks)

        try:
            with contextlib.redirect_stdout(printed):
                result = run_weather_agent()[0]
        finally:
            globals().update(saved)

        return result, printed.getvalue()

    real_decision, real_output = run_real_retrieval_path()

    check(
        "Real retrieval path never reports missing required headings",
        "load_guidelines" in real_decision["tools_used"]
        and "search_guidelines" in real_decision["tools_used"]
        and real_decision["policy_status"] == "AVAILABLE"
        and real_decision["policy_path"] == str(GUIDELINES_PATH)
        and real_decision["policy_hash"] == hashlib.sha256(
            GUIDELINES_PATH.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        and "ASSESSMENT ERROR" not in real_output
        and "Required policy headings" not in real_output
        and "Required policy headings" not in json_text(real_decision),
    )

    workflow_path = (
        Path(__file__).resolve().parent
        / ".github" / "workflows" / "weather.yml"
    )
    workflow_commands = [
        line.strip() for line in
        workflow_path.read_text(encoding="utf-8").splitlines()
        if "weather_log.py" in line
    ] if workflow_path.is_file() else []

    original_directory = os.getcwd()

    with tempfile.TemporaryDirectory() as elsewhere:
        os.chdir(elsewhere)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                loaded_elsewhere = load_guidelines()
        finally:
            # Leave the directory before it is deleted (Windows).
            os.chdir(original_directory)

    check(
        "Workflow runs this weather_log.py in dry-run mode only and "
        "the policy loads from beside the script",
        workflow_commands == ["- run: python weather_log.py agent-dry"]
        and Path(loaded_elsewhere["path"]).resolve()
        == (Path(__file__).resolve().parent / "guidelines.md")
        and loaded_elsewhere["hash"] == hazard_policy["hash"],
    )

    # ------------------------------------------------------------
    # Regression: dry run failed with "ValueError: Invalid text
    # field: advice" although Groq reported finish_reason=stop with
    # 694 of 700 completion tokens. The JSON parsed and passed every
    # earlier validation step, so `advice` was present but not a
    # string. The prompt never stated that advice is plain text.
    # Everything below is self-contained and offline.
    # ------------------------------------------------------------
    def adv_response(**changes):
        body = {
            "hazard_type": "HEAT",
            "likelihood_score": 3, "likelihood_reason": "lr",
            "severity_score": 3, "severity_reason": "sr",
            "exposure_score": 3, "exposure_reason": "er",
            "duration_score": 3, "duration_reason": "dr",
            "geographic_impact_score": 3,
            "geographic_impact_reason": "gr",
            "alert_recommendation": "MONITOR",
            "advice": "Limit strenuous outdoor activity at midday.",
            "reasoning": "Temperature reaches the screening value.",
            "recommended_actions": ["Drink water"],
            "limitations": ["Official warnings not checked"],
            "missing_evidence": [],
            "sources": ["6.1 Extreme Heat"],
        }
        body.update(changes)
        return body

    adv_sections = [
        {"section": "6.1 Extreme Heat", "text": "x"},
        {"section": "16. Final Decision Rule", "text": "x"},
    ]

    def adv_validate(**changes):
        try:
            return validate_decision(adv_response(**changes), adv_sections)
        except ValueError as exc:
            return str(exc)

    def adv_raises(function):
        try:
            function()
            return ""
        except ValueError as exc:
            return str(exc)

    def adv_error(body):
        try:
            validate_decision(body, adv_sections)
            return ""
        except ValueError as exc:
            return str(exc)

    # 1. A valid response containing advice.
    adv_ok = adv_validate()

    check(
        "Valid response with advice passes validation unchanged",
        isinstance(adv_ok, dict)
        and adv_ok["advice"]
        == "Limit strenuous outdoor activity at midday."
        and adv_ok["risk_score"] == 9.0
        and adv_ok["risk_band"] == "MODERATE"
        and adv_ok["assessment_status"] == "PRELIMINARY",
    )

    # 2. Invalid types for advice are rejected and named.
    for label, bad in {
        "list": ["a", "b"],
        "dict": {"text": "a"},
        "null": None,
        "int": 3,
        "bool": True,
    }.items():
        message = adv_validate(advice=bad)

        check(
            f"advice of type {label} is rejected with the type named",
            isinstance(message, str)
            and message.startswith("Invalid text field: advice")
            and "expected text" in message
            and f"received {type(bad).__name__}" in message,
        )

    check(
        "Blank advice is rejected as empty",
        adv_validate(advice="   ") == "Empty text field: advice",
    )

    check(
        "Type errors also name hazard_type and reasoning",
        adv_validate(hazard_type=["HEAT"]).startswith(
            "Invalid text field: hazard_type"
        )
        and adv_validate(reasoning={"a": 1}).startswith(
            "Invalid text field: reasoning"
        ),
    )

    check(
        "Type error reports the type, never the model's content",
        "secret-model-text" not in adv_validate(
            advice=["secret-model-text"]
        ),
    )

    # 3. Missing required fields: every key is required.
    adv_keys = list(adv_response())

    check(
        "Eighteen response keys are required",
        len(adv_keys) == 18,
    )

    for key in adv_keys:
        body = adv_response()
        del body[key]

        check(
            f"Missing required field is rejected: {key}",
            adv_error(body) == "Missing assessment fields: " + key,
        )

    check(
        "Non-object model output is rejected",
        adv_error([1, 2]) == "AI returned an invalid assessment."
        and adv_error("text") == "AI returned an invalid assessment.",
    )

    # 4. Malformed or truncated JSON never reaches validation.
    adv_json = json_text(adv_response())
    truncated_failures = 0

    for cut in range(1, len(adv_json), 37):
        try:
            parse_json_response(adv_json[:cut])
        except ValueError as exc:
            if "invalid JSON" in str(exc):
                truncated_failures += 1

    check(
        "Truncated JSON is always rejected as invalid",
        truncated_failures == len(range(1, len(adv_json), 37)),
    )

    check(
        "Malformed and non-object JSON are rejected",
        all(
            "JSON" in adv_raises(lambda text=text: parse_json_response(text))
            for text in (
                "{not json", "", "   ", "[1, 2]", "\"text\"", "null",
                "{\"advice\": }",
            )
        ),
    )

    # Stop plus valid JSON does not prove schema completeness.
    incomplete_body = adv_response()
    del incomplete_body["sources"]

    check(
        "Valid JSON with a missing field is rejected despite a "
        "clean finish",
        "Missing assessment fields: sources" in adv_error(
            parse_json_response(json_text(incomplete_body))
        ),
    )

    # Contract: the prompt, the validator and the payload agree.
    # Parse the typed key groups; a missing group yields no keys, so
    # an old or broken prompt fails these checks instead of crashing.
    contract = re.search(
        r"Return exactly these 18 keys(.*?)Each score must be",
        SYSTEM_PROMPT,
        re.S,
    )
    prompt_text = contract.group(1) if contract else ""

    def adv_group_keys(label, following):
        match = re.search(
            re.escape(label) + r"[^:]*:(.*?)(?:" + following + r"|\Z)",
            prompt_text,
            re.S,
        )
        listed = match.group(1) if match else ""
        return [key for key in re.split(r"[\s,.]+", listed) if key]

    string_keys = adv_group_keys("Plain-text strings", "Integer scores")
    score_keys = adv_group_keys("Integer scores", "Arrays of strings")
    array_keys = adv_group_keys("Arrays of strings", r"\Z")
    prompt_keys = string_keys + score_keys + array_keys

    check(
        "Prompt lists exactly the keys the validator requires",
        sorted(prompt_keys) == sorted(adv_keys)
        and len(set(prompt_keys)) == 18,
    )

    check(
        "Prompt states the type of every key",
        "never arrays or objects" in prompt_text
        and {"hazard_type", "advice", "reasoning",
             "alert_recommendation"} <= set(string_keys)
        and set(score_keys)
        == {k for k in adv_keys if k.endswith("_score")}
        and set(array_keys)
        == {"recommended_actions", "limitations", "missing_evidence",
            "sources"}
        and all(
            isinstance(adv_response()[key], str) for key in string_keys
        )
        and all(
            type(adv_response()[key]) is int for key in score_keys
        )
        and all(
            isinstance(adv_response()[key], list) for key in array_keys
        ),
    )

    check(
        "Decision fields reach the Sheets payload as text",
        isinstance(adv_ok["advice"], str)
        and isinstance(adv_ok["reasoning"], str),
    )

    # 5. A complete valid response reaches assessment construction.
    def adv_weather():
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00",
                "temperature_2m": 35.5, "apparent_temperature": 34.3,
                "precipitation": 0, "wind_speed_10m": 5,
                "wind_gusts_10m": 10, "visibility": 20000,
                "weather_code": 0, "cape": 0,
            },
            "today_forecast": {
                "high": 36, "low": 25, "wind": 8, "gust": 15,
                "precipitation": 0,
            },
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": "2026-11-08T13:00", "temperature": 35,
                "apparent_temperature": 34, "precipitation": 0,
                "wind": 5, "gust": 10, "visibility": 20000,
                "weather_code": 0, "cape": 0,
            }],
            "history_summary": {},
        }

    def adv_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def adv_assess(reply):
        return adv_override(
            {"ask_groq": lambda messages, temperature=0.1: reply},
            lambda: assess_weather(
                adv_weather(), [], load_guidelines_quietly(),
                adv_sections,
            ),
        )

    def load_guidelines_quietly():
        with contextlib.redirect_stdout(io.StringIO()):
            return load_guidelines()

    constructed = adv_assess(json_text(adv_response()))

    check(
        "Complete valid response reaches assessment construction",
        constructed["advice"]
        == "Limit strenuous outdoor activity at midday."
        and constructed["risk_score"] == 9.0
        and constructed["risk_band"] == "MODERATE"
        and constructed["assessment_status"] == "INCOMPLETE"
        and constructed["alert_recommendation"] == "MONITOR"
        and constructed["sources"] == ["6.1 Extreme Heat"]
        and constructed["official_warning_status"]
        == "NOT_CHECKED",
    )

    check(
        "Assessment construction still fails closed on bad advice",
        "Invalid text field: advice" in adv_raises(
            lambda: adv_assess(json_text(adv_response(advice=["a"])))
        ),
    )

    # 6. Real retrieval and assessment path, mocked external services
    #    (weather, history, Groq transport). Real guidelines.md,
    #    real retrieval, real ask_groq, real validation.
    class AdvResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content, finish="stop", usage=None):
            self.content = content
            self.finish = finish
            self.usage = usage

        def json(self):
            body = {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": self.finish,
                }]
            }

            if self.usage is not None:
                body["usage"] = self.usage

            return body

    class AdvHttp:
        def __init__(self, response):
            self.response = response
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            return self.response

    incident_usage = {
        "prompt_tokens": 3239,
        "completion_tokens": 694,
        "total_tokens": 3933,
        "completion_tokens_details": {"reasoning_tokens": 301},
    }

    def adv_pipeline(content, finish="stop", usage=incident_usage):
        transport = AdvHttp(AdvResponse(content, finish, usage))
        printed = io.StringIO()

        def run():
            with contextlib.redirect_stdout(printed):
                return run_weather_agent()[0]

        result = adv_override(
            {
                "HTTP": transport,
                "GROQ_API_KEY": "gsk_adv_test_key_not_real",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: adv_weather(),
                "tool_get_history": lambda n=3: [],
            },
            run,
        )

        return result, printed.getvalue(), transport

    # Sources must be section titles actually retrieved for a heat day.
    real_sources = ["6.1 Extreme Heat", "16. Final Decision Rule"]

    good_result, good_output, good_transport = adv_pipeline(
        json_text(adv_response(sources=real_sources))
    )

    check(
        "Real path completes a valid response and records usage",
        "load_guidelines" in good_result["tools_used"]
        and "search_guidelines" in good_result["tools_used"]
        and good_result["advice"]
        == "Limit strenuous outdoor activity at midday."
        and good_result["assessment_status"] == "INCOMPLETE"
        and good_result["send_alert"] is False
        and good_result["notification_status"] == "NOT_SENT"
        and good_result["official_warning_status"] == "NOT_CHECKED"
        and good_result["groq_usage"]["prompt_tokens"] == 3239
        and good_result["groq_usage"]["completion_tokens"] == 694
        and good_result["groq_usage"]["reasoning_tokens"] == 301
        and good_result["groq_usage"]["finish_reason"] == "stop"
        and "gsk_adv_test_key_not_real" not in good_output
        and len(good_transport.posts) == 1,
    )

    incident_result, incident_output, _ = adv_pipeline(
        json_text(adv_response(
            advice=["Limit activity", "Drink water"],
            sources=real_sources,
        ))
    )

    check(
        "Incident reproduced: advice as a list fails closed with the "
        "type named and usage kept",
        incident_result["assessment_status"] == "INCOMPLETE"
        and incident_result["alert_recommendation"]
        == "UNABLE_TO_ASSESS"
        and incident_result["risk_band"] == "UNASSESSED"
        and incident_result["risk_score"] is None
        and "Invalid text field: advice (expected text, received list)"
        in incident_result["reasoning"]
        and incident_result["send_alert"] is False
        and incident_result["notification_status"] == "NOT_SENT"
        and incident_result["official_warning_status"] == "NOT_CHECKED"
        and incident_result["groq_usage"]["completion_tokens"] == 694
        and incident_result["groq_usage"]["finish_reason"] == "stop"
        and "gsk_adv_test_key_not_real"
        not in json_text(incident_result) + incident_output,
    )

    missing_result, _, _ = adv_pipeline(
        json_text({
            key: value for key, value in adv_response().items()
            if key != "advice"
        })
    )
    truncated_result, _, _ = adv_pipeline(
        json_text(adv_response())[:200], finish="length"
    )
    cut_stop_result, _, _ = adv_pipeline(
        json_text(adv_response())[:200], finish="stop"
    )

    check(
        "Real path fails closed on missing field, truncation and "
        "malformed JSON",
        "Missing assessment fields: advice" in missing_result["reasoning"]
        and "cut off" in truncated_result["reasoning"]
        and "invalid JSON" in cut_stop_result["reasoning"]
        and all(
            result["assessment_status"] == "INCOMPLETE"
            and result["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and result["send_alert"] is False
            for result in (
                missing_result, truncated_result, cut_stop_result
            )
        ),
    )

    check(
        "Absent reasoning-token detail is recorded as null",
        adv_pipeline(
            json_text(adv_response(sources=real_sources)),
            usage={"prompt_tokens": 1, "completion_tokens": 2,
                   "total_tokens": 3},
        )[0]["groq_usage"]["reasoning_tokens"] is None,
    )

    # agent-dry must not POST to Sheets or send email.
    class RecordingHttp:
        def __init__(self):
            self.posts = []

        def post(self, url, **kwargs):
            self.posts.append(url)

            class Response:
                status_code = 200
                text = "OK"

                def raise_for_status(self):
                    return None

            return Response()

    def run_job(dry):
        recorder = RecordingHttp()

        with_globals(
            {
                "HTTP": recorder,
                "SHEET_URL": "https://sheet.invalid/exec",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "run_weather_agent": lambda: (
                    {
                        **incomplete_assessment("test"),
                        "city": CITY,
                        "send_alert": False,
                    },
                    [],
                ),
            },
            lambda: run_agent_job(dry=dry),
        )

        return recorder.posts

    check(
        "agent-dry makes no HTTP POST (Sheets or otherwise)",
        run_job(dry=True) == [],
    )

    check(
        "Test harness detects a live-mode POST (control)",
        run_job(dry=False) == ["https://sheet.invalid/exec"],
    )

    imported_modules = set()

    for node in ast.walk(
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
    ):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    check(
        "No email-sending modules are imported",
        not imported_modules & {
            "smtplib", "email", "sendgrid", "yagmail", "mailbox",
        },
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
            "Actual mandatory policy headings identified",
            len(mandatory) == len(MANDATORY_SECTION_ALIASES)
            and not missing,
        )

        check(
            "All mandatory headings exist in guidelines",
            all(title in policy["sections"] for title in mandatory),
        )

        # The formula and weights in code must match section 5.
        risk_section = next(
            (
                body for title, body in policy["sections"].items()
                if normalize_heading(title)
                == "risk calculation model"
            ),
            "",
        )
        formula = re.search(
            r"Risk Score = (.+)", risk_section
        )
        weights = (
            re.findall(
                r"([0-9.]+) × \(([LEDG]) × S\)", formula.group(1)
            )
            if formula else []
        )

        check(
            "Risk formula weights match guidelines section 5",
            weights == [
                ("0.35", "L"),
                ("0.20", "E"),
                ("0.15", "D"),
                ("0.30", "G"),
            ]
            and calculate_risk_score(1, 1, 1, 1, 1)[0] == 1.0,
        )

        results = search_guidelines(
            policy,
            "heat exposure wind rain flooding weather hazards",
            k=1,
        )

        result_titles = [
            item["section"]
            for item in results
        ]

        check(
            "All mandatory sections are retrieved",
            all(title in result_titles for title in mandatory),
        )

        check(
            "Retrieved sections contain no duplicates",
            len(result_titles) == len(set(result_titles)),
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

        if not query.strip():
            query = "weather safety hazards"

        try:
            policy = load_guidelines()

            results = search_guidelines(
                policy,
                query,
                k=1,
            )

            for result in results:
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
        run_agent_job(dry=False)

    else:
        print("Available commands:")
        print("python weather_log.py search heat risk")
        print("python weather_log.py test")
        print("python weather_log.py agent-dry")
        print("python weather_log.py agent")
        raise SystemExit(1)
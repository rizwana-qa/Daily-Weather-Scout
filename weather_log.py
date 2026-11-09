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

# The guidelines are Dubai-specific, so Dubai is the only location this
# program may assess. The request asks Open-Meteo for this timezone
# explicitly instead of "auto".
DUBAI_LOCATION = {
    "city": "Dubai",
    "latitude": 25.20,
    "longitude": 55.27,
    "timezone": "Asia/Dubai",
}

CITIES = {
    "Dubai": (25.20, 55.27),
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"

GUIDELINES_PATH = Path(__file__).resolve().parent / "guidelines.md"

SHEET_URL = os.getenv("SHEET_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_TIMEOUT = 30
GROQ_TIMEOUT = 90
SHEET_TIMEOUT = 30

# Request budgets. Which hazard combinations fit is pinned by the budget
# tests; anything larger fails closed (INCOMPLETE) before a request is
# sent. The guards measure the messages plus the response schema.
# The token figure uses the component-aware estimate (see "Component-
# aware request token estimate"), or the legacy conservative estimate
# (2.5 characters per token) for a request without parts; neither is a
# provider tokenizer. 8,000 is the
# documented base tokens-per-minute limit for openai/gpt-oss-20b; the
# account's own limit is unverified, and Groq's 413/429 responses are
# handled as safe failures.
# The completion allowance covers hidden reasoning tokens plus the full
# strict-schema response: observed completions were 606 and 694 tokens,
# and 700 was exceeded once (json_validate_failed). 1,000 is capped by
# the "conservative" test and keeps that failing request (6,890
# estimated input tokens) within the 8,000 total.
MAX_COMPLETION_TOKENS = 1000
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
        raise ConfigurationError(
            f"Unsupported city: {CITY}. Allowed cities: {list(CITIES)}"
        )

    if CITY != DUBAI_LOCATION["city"] or CITIES[CITY] != (
        DUBAI_LOCATION["latitude"], DUBAI_LOCATION["longitude"]
    ):
        raise ConfigurationError(
            "The configured location does not match the approved Dubai "
            "coordinates."
        )

    if not GROQ_API_KEY:
        raise ConfigurationError("GROQ_API_KEY is not configured.")

    if require_sheet and not SHEET_URL:
        raise ConfigurationError("SHEET_URL is not configured.")


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

# ============================================================
# 2a. CLASSIFIED FAILURES
# ============================================================

class ClassifiedFailure:
    """
    Mixin for failures whose cause is known. outcome uses the guidelines
    section 11 vocabulary (INVALID or UNAVAILABLE) and failure_code is a
    fixed label. baseline_name is the exception name these failures were
    reported under before they were typed, so report text is unchanged.
    An exception without this mixin is an INTERNAL_ERROR, never one of
    these causes.
    """

    outcome = "INVALID"
    failure_code = "INTERNAL_ERROR"
    baseline_name = "Exception"


class ConfigurationError(ClassifiedFailure, ValueError):
    outcome = "INVALID"
    failure_code = "CONFIGURATION_INVALID"
    baseline_name = "ValueError"


class PolicyUnavailableError(ClassifiedFailure, FileNotFoundError):
    # Guidelines section 10: a missing policy file gives assessment_status
    # UNAVAILABLE, which is also this run outcome.
    outcome = "UNAVAILABLE"
    failure_code = "POLICY_UNAVAILABLE"
    baseline_name = "FileNotFoundError"


class PolicyInvalidError(ClassifiedFailure, ValueError):
    outcome = "INVALID"
    failure_code = "POLICY_INVALID"
    baseline_name = "ValueError"


class WeatherUnavailableError(ClassifiedFailure, ValueError):
    outcome = "UNAVAILABLE"
    failure_code = "WEATHER_UNAVAILABLE"
    baseline_name = "ValueError"

    def __init__(self, message, baseline_name=None):
        super().__init__(message)

        if baseline_name is not None:
            self.baseline_name = baseline_name


class RequestBudgetError(ClassifiedFailure, ValueError):
    outcome = "UNAVAILABLE"
    failure_code = "REQUEST_BUDGET_EXCEEDED"
    baseline_name = "ValueError"


class ResponseInvalidError(ClassifiedFailure, ValueError):
    outcome = "INVALID"
    failure_code = "RESPONSE_INVALID"
    baseline_name = "ValueError"


def exception_label(exc):
    """
    The exception name used in report text: the baseline name for a
    classified failure, so wording is unchanged, else the type name.
    """
    if isinstance(exc, ClassifiedFailure):
        return exc.baseline_name

    return type(exc).__name__


def get_weather(city):
    if city not in CITIES:
        raise ConfigurationError(f"City not allowed: {city}")

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
        "timezone": DUBAI_LOCATION["timezone"],
    }

    try:
        response = HTTP.get(
            url,
            params=params,
            timeout=WEATHER_TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Same report text as before; the cause is now classified.
        raise WeatherUnavailableError(
            str(exc), baseline_name=type(exc).__name__
        ) from exc

    if not isinstance(raw, dict):
        raise WeatherUnavailableError("Weather API returned an invalid response.")

    for key in ("current", "hourly", "daily"):
        if not isinstance(raw.get(key), dict):
            raise WeatherUnavailableError(
                f"Weather API response is missing '{key}'."
            )

    current = raw["current"]
    hourly = raw["hourly"]
    daily = raw["daily"]

    current_time = local_datetime(current.get("time"))

    if current_time is None:
        raise WeatherUnavailableError(
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
        raise WeatherUnavailableError("No usable current or forecast weather data.")

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
        "location_observation": observe_location(raw),
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


# An IANA-style zone name ("Area/City"); anything else is not reported.
_TIMEZONE_NAME = re.compile(
    r"[A-Za-z][A-Za-z0-9_+-]{0,30}(?:/[A-Za-z0-9_+-]{1,30}){0,2}"
)


def observe_location(raw):
    """
    Observe-only record of where Open-Meteo says the data is for. It is
    reported, never enforced: no tolerance for returned coordinates and
    no utc_offset_seconds value has been verified against a real
    response. Values are reduced to fixed shapes (a zone name, rounded
    numbers, booleans), so no other provider text is copied.
    """
    zone = raw.get("timezone")
    zone = (
        zone if isinstance(zone, str) and _TIMEZONE_NAME.fullmatch(zone)
        else None
    )

    def coordinate(key):
        value = raw.get(key)

        if type(value) in (int, float) and math.isfinite(value):
            return round(float(value), 3)

        return None

    return {
        "requested_timezone": DUBAI_LOCATION["timezone"],
        "returned_timezone": zone,
        "timezone_matches_request": zone == DUBAI_LOCATION["timezone"],
        "returned_latitude": coordinate("latitude"),
        "returned_longitude": coordinate("longitude"),
        "utc_offset_seconds_present": "utc_offset_seconds" in raw,
    }


# Hourly series sent to the model so duration can be judged from the
# forecast period (guidelines section 4.4). Each column keeps the
# precision Open-Meteo reports (0.1 for temperature, precipitation and
# wind; whole units for humidity, visibility in metres and weather
# codes), so no value is altered. (name, source key, decimals)
HOURLY_SERIES_FIELDS = (
    ("temp_c", "temperature", 1),
    ("apparent_c", "apparent_temperature", 1),
    ("humidity_pct", "humidity", 0),
    ("precip_mm", "precipitation", 1),
    ("wind_kmh", "wind", 1),
    ("gust_kmh", "gust", 1),
    ("visibility_m", "visibility", 0),
    ("weather_code", "weather_code", 0),
)


def hourly_series(hourly):
    """
    Compact, ordered next-24-hour series. Consecutive hourly records
    are described by a start time and a one-hour step; otherwise every
    timestamp is listed. Missing values stay null.
    """
    records = [record for record in hourly if isinstance(record, dict)]

    if not records:
        return None

    times = [record.get("time") for record in records]
    parsed = [local_datetime(value) for value in times]

    consecutive = all(parsed) and all(
        later - earlier == timedelta(hours=1)
        for earlier, later in zip(parsed, parsed[1:])
    )

    series = (
        {"start": times[0], "step_hours": 1}
        if consecutive else {"times": times}
    )

    for name, key, digits in HOURLY_SERIES_FIELDS:
        column = []

        for record in records:
            number = safe_number(record.get(key))

            if number is None:
                column.append(None)
                continue

            value = round(number, digits)
            column.append(int(value) if digits == 0 else value)

        series[name] = column

    return series


def summarize_weather(weather):
    hourly = weather.get("hourly_forecast_next_24h", [])

    def values_for(field):
        return [
            safe_number(record.get(field))
            for record in hourly
            if safe_number(record.get(field)) is not None
        ]

    # The hourly series carries every value the former min/max summary
    # was derived from; only CAPE, which the series omits, is kept here.
    cape = values_for("cape")

    return {
        "city": weather["city"],
        "timezone": weather.get("timezone"),
        "source": weather["source"],
        "retrieved_at": weather["retrieved_at"],
        "current": weather["current"],
        "today_forecast": weather.get("today_forecast"),
        "next_24_hours": {"cape_max": max(cape, default=None)},
        "next_24_hours_hourly": hourly_series(hourly),
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

# Program-generated shape summary of the last model response that
# failed validation. Holds field identifiers, counts, JSON type names
# and booleans only, never model-generated values or key names.
LAST_RESPONSE_DIAGNOSTICS = {}


def usage_number(usage, key):
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int else None


# Fixed descriptions for the HTTP statuses Groq documents. Provider
# error bodies are never copied: they can contain free text and partial
# model output (for example "failed_generation").
GROQ_STATUS_DESCRIPTIONS = {
    400: "request rejected as invalid",
    401: "authentication failed",
    403: "permission denied",
    404: "resource not found",
    413: "request too large for the model or account limit",
    422: "request could not be processed",
    424: "a dependent request failed",
    429: "rate or token limit reached (RPM, TPM, RPD or TPD)",
    498: "flex tier capacity exceeded",
    499: "request cancelled",
    500: "provider internal error",
    502: "provider gateway error",
    503: "provider unavailable",
}

# Fixed descriptions for provider error codes observed in practice.
GROQ_ERROR_CODE_DESCRIPTIONS = {
    "json_validate_failed": (
        "the provider reported that the generated output did not "
        "validate against the response schema"
    ),
}

GROQ_GENERIC_ERROR_DESCRIPTION = "provider details withheld"

# json_validate_failed has been observed with a provider message saying
# the completion limit was reached before a valid document was generated
# (a truncated generation). That message is matched against a closed
# pattern and reduced to a fixed label; its text is never copied. Any
# other message keeps the generic description above.
_COMPLETION_LIMIT_MESSAGE = re.compile(
    r"max completion tokens reached", re.IGNORECASE
)

GROQ_ERROR_REASON_DESCRIPTIONS = {
    "completion_limit_reached": (
        "the provider reported that the completion limit was reached "
        "before a valid document was generated"
    ),
}

# Provider identifiers (error.code, error.type) are passed on only when
# they look like machine identifiers; anything else is withheld.
_PROVIDER_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")
_RETRY_AFTER = re.compile(r"\d{1,6}(?:\.\d{1,3})?")


def safe_provider_identifier(value):
    if isinstance(value, str) and _PROVIDER_IDENTIFIER.fullmatch(value):
        return value

    return None


def provider_error_fields(response):
    """
    The structured error.code and error.type from a provider error
    body, each validated as an identifier, else None. The body is
    parsed defensively; message, failed_generation, any other field and
    the raw text are never read into the result.
    """
    try:
        body = response.json()
    except (ValueError, TypeError, AttributeError):
        return None, None

    error = body.get("error") if isinstance(body, dict) else None

    if not isinstance(error, dict):
        return None, None

    return (
        safe_provider_identifier(error.get("code")),
        safe_provider_identifier(error.get("type")),
    )


def provider_error_reason(response):
    """
    A fixed reason label for a provider error, else None. Only
    error.message is examined, only its first 300 characters, and only
    to match the closed pattern above; the result is a key of
    GROQ_ERROR_REASON_DESCRIPTIONS, never provider text.
    """
    try:
        body = response.json()
    except (ValueError, TypeError, AttributeError):
        return None

    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None

    if isinstance(message, str) and _COMPLETION_LIMIT_MESSAGE.search(
        message[:300]
    ):
        return "completion_limit_reached"

    return None


# Fixed provider-error categories and the guidelines section 11 outcome
# each maps to: problems with the generated output are INVALID; anything
# that left the provider unable to answer is UNAVAILABLE.
PROVIDER_ERROR_CATEGORIES = {
    "AUTHENTICATION_FAILED": "UNAVAILABLE",
    "REQUEST_TOO_LARGE": "UNAVAILABLE",
    "RATE_LIMITED": "UNAVAILABLE",
    "INVALID_REQUEST": "UNAVAILABLE",
    "PROVIDER_UNAVAILABLE": "UNAVAILABLE",
    "NETWORK_ERROR": "UNAVAILABLE",
    "PROVIDER_ERROR_UNCLASSIFIED": "UNAVAILABLE",
    "OUTPUT_TRUNCATED": "INVALID",
    "OUTPUT_VALIDATION_UNSPECIFIED": "INVALID",
    "RESPONSE_MALFORMED": "INVALID",
}


def provider_error_category(status_code, code, reason):
    """
    Fixed category for a Groq HTTP error, from the status, the validated
    error code and the closed reason label only. A json_validate_failed
    error without the completion-limit reason is not called a schema
    failure: its cause is unspecified.
    """
    status = status_code if type(status_code) is int else None

    if code == "json_validate_failed":
        return (
            "OUTPUT_TRUNCATED"
            if reason == "completion_limit_reached"
            else "OUTPUT_VALIDATION_UNSPECIFIED"
        )

    if status in (401, 403):
        return "AUTHENTICATION_FAILED"

    if status == 413:
        return "REQUEST_TOO_LARGE"

    if status in (429, 498):
        return "RATE_LIMITED"

    if status is not None and (status == 424 or 500 <= status <= 599):
        return "PROVIDER_UNAVAILABLE"

    if status is not None and 400 <= status <= 499:
        return "INVALID_REQUEST"

    return "PROVIDER_ERROR_UNCLASSIFIED"


class ProviderError(ClassifiedFailure, RuntimeError):
    """
    A Groq request that produced no usable assessment. Holds only fixed
    or validated values (category, status, identifier-shaped code and
    type, closed reason label); never provider text.
    """

    failure_code = "PROVIDER_ERROR"
    baseline_name = "RuntimeError"

    def __init__(self, message, category, http_status=None, code=None,
                 error_type=None, reason=None):
        super().__init__(message)
        self.category = (
            category if category in PROVIDER_ERROR_CATEGORIES
            else "PROVIDER_ERROR_UNCLASSIFIED"
        )
        self.outcome = PROVIDER_ERROR_CATEGORIES[self.category]
        self.http_status = http_status if type(http_status) is int else None
        self.code = safe_provider_identifier(code)
        self.error_type = safe_provider_identifier(error_type)
        self.reason = (
            reason if reason in GROQ_ERROR_REASON_DESCRIPTIONS else None
        )

    def diagnostic(self):
        return {
            "category": self.category,
            "http_status": self.http_status,
            "code": self.code,
            "type": self.error_type,
            "reason": self.reason,
        }


def describe_groq_http_error(status_code, response):
    """
    Program-generated description of a Groq HTTP error. It contains only
    the status, a fixed description, validated provider identifiers and
    a numeric retry-after; never provider-supplied text.
    """
    status = status_code if type(status_code) is int else None
    code, error_type = provider_error_fields(response)
    reason = (
        provider_error_reason(response)
        if code == "json_validate_failed" else None
    )

    headers = getattr(response, "headers", None)
    retry_after = headers.get("retry-after") if hasattr(
        headers, "get"
    ) else None

    message = (
        f"Groq API returned HTTP {status} "
        f"({GROQ_STATUS_DESCRIPTIONS.get(status, 'request failed')})"
    )

    if isinstance(retry_after, str) and _RETRY_AFTER.fullmatch(retry_after):
        message += f"; retry-after {retry_after}s"

    return (
        message
        + f"; provider code={code or 'unavailable'}, "
        f"type={error_type or 'unavailable'}: "
        + (
            GROQ_ERROR_REASON_DESCRIPTIONS.get(reason)
            or GROQ_ERROR_CODE_DESCRIPTIONS.get(
                code, GROQ_GENERIC_ERROR_DESCRIPTION
            )
        )
    )


def ask_groq(messages, temperature=0.1):
    LAST_GROQ_USAGE.clear()

    if not GROQ_API_KEY:
        raise ConfigurationError("GROQ_API_KEY is missing.")

    # Both guards measure the messages and the response schema. The
    # character guard runs first and is independent of the token guard.
    request_text = serialized_request(messages)
    estimate_error = None

    try:
        budget = request_token_budget(messages, request_text)
    except RequestBudgetError as exc:
        # An estimate that cannot be verified never bypasses the
        # guard: it is raised below, after the character guard.
        estimate_error = exc
        budget = legacy_token_budget(request_text)
        budget["method"] = "unverified"

    estimated_input_tokens = budget["estimated_input_tokens"]

    estimated_total = (
        estimated_input_tokens + MAX_COMPLETION_TOKENS
    )

    print("\nGROQ REQUEST CHECK")
    print("Estimated input tokens:", estimated_input_tokens)
    print("Completion token allowance:", MAX_COMPLETION_TOKENS)
    print("Estimated total tokens:", estimated_total)
    print("Request characters:", len(request_text))
    print("Response schema characters:", len(json_text(RESPONSE_FORMAT)))
    print_budget_components(budget)

    if len(request_text) > MAX_REQUEST_CHARS:
        raise RequestBudgetError(
            "Groq request blocked: request (messages and response "
            f"schema) exceeds the configured {MAX_REQUEST_CHARS}-"
            "character limit."
        )

    if estimate_error is not None:
        raise estimate_error

    if estimated_total > MAX_ESTIMATED_REQUEST_TOKENS:
        raise RequestBudgetError(
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
        "reasoning_effort": REASONING_EFFORT,
        "include_reasoning": False,
        # Strict structured outputs: every field required, no extras.
        "response_format": RESPONSE_FORMAT,
    }

    try:
        response = HTTP.post(
            GROQ_URL,
            headers={
                "Authorization": "Bearer " + GROQ_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=GROQ_TIMEOUT,
        )
    except requests.RequestException as exc:
        # Fixed text: the library message is not copied.
        raise ProviderError(
            "Groq request failed before a response was received ("
            + type(exc).__name__ + ").",
            "NETWORK_ERROR",
        ) from exc

    if response.status_code != 200:
        code, error_type = provider_error_fields(response)
        reason = (
            provider_error_reason(response)
            if code == "json_validate_failed" else None
        )

        raise ProviderError(
            describe_groq_http_error(response.status_code, response),
            provider_error_category(response.status_code, code, reason),
            http_status=response.status_code,
            code=code,
            error_type=error_type,
            reason=reason,
        )

    try:
        result = response.json()
        choice = result["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError,
            AttributeError) as exc:
        raise ProviderError(
            "Groq returned an unexpected response structure.",
            "RESPONSE_MALFORMED",
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
        "estimator_method": budget["method"],
        "legacy_estimated_input_tokens": (
            budget["legacy_estimated_input_tokens"]
        ),
    })

    print("\nGROQ USAGE (reported by the API)")

    for key, value in LAST_GROQ_USAGE.items():
        print(f"{key}: {value}")

    # Log only: a provider count above the estimate never changes the
    # outcome, but it must be visible.
    provider_prompt_tokens = LAST_GROQ_USAGE["prompt_tokens"]

    if (
        type(provider_prompt_tokens) is int
        and provider_prompt_tokens > estimated_input_tokens
    ):
        print(
            "ESTIMATOR UNDERRUN: provider prompt_tokens",
            provider_prompt_tokens,
            "exceeds the estimate",
            estimated_input_tokens,
        )

    if finish_reason != "stop":
        raise ProviderError(
            "Groq response is incomplete or was cut off: "
            f"finish_reason={finish_reason!r}. The completion allowance "
            f"is {MAX_COMPLETION_TOKENS} tokens.",
            "OUTPUT_TRUNCATED" if finish_reason == "length"
            else "RESPONSE_MALFORMED",
        )

    if not isinstance(content, str) or not content.strip():
        raise ProviderError(
            "Groq returned an empty response.", "RESPONSE_MALFORMED"
        )

    return content


def parse_json_response(response_text):
    text = (response_text or "").strip()

    text = text.removeprefix("```json")
    text = text.removeprefix("```")
    text = text.removesuffix("```").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseInvalidError(
            f"Groq returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ResponseInvalidError("Groq response must be a JSON object.")

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
        raise PolicyUnavailableError(
            f"guidelines.md not found at {GUIDELINES_PATH}"
        )

    content = GUIDELINES_PATH.read_text(encoding="utf-8")

    if not content.strip():
        raise PolicyInvalidError("guidelines.md is empty.")

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
            raise PolicyInvalidError(
                f"Duplicate policy heading found: {title}"
            )

        chunks[title] = body.strip()

    if not chunks:
        raise PolicyInvalidError(
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
    add_incomplete_reason(
        decision,
        "WEATHER_EVIDENCE_UNCERTAIN",
        ", ".join(sorted(gaps)),
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
        raise PolicyInvalidError(
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
        raise PolicyInvalidError(
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
            raise PolicyInvalidError(
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
        raise PolicyInvalidError(
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
        raise PolicyInvalidError("Loaded policy is invalid or empty.")

    chunks = policy["sections"]

    mandatory, missing = find_mandatory_sections(chunks)

    if missing:
        raise PolicyInvalidError(
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
        raise RequestBudgetError(
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
        raise RequestBudgetError(
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

def describe_history_error(exc):
    """
    Program-generated description of a history retrieval failure. The
    exception text is never used: requests includes the full request
    URL (the secret SHEET_URL and its query string) in its messages.
    """
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

        if type(status) is int:
            return f"history service returned HTTP {status}"

        return "history service returned an HTTP error"

    if isinstance(exc, requests.Timeout):
        return "history request timed out"

    if isinstance(exc, requests.ConnectionError):
        return "history service could not be reached"

    if isinstance(exc, ValueError):
        return "history response could not be read as JSON"

    if isinstance(exc, requests.RequestException):
        return "history request failed"

    return "history retrieval failed"


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
        return {"error": describe_history_error(exc)}


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


def add_incomplete_reason(decision, code, detail=None):
    """
    Record why an assessment is incomplete as a structured code, once
    per code. Details are generated by this program, never copied from
    model output.
    """
    reasons = decision.setdefault("incomplete_reasons", [])

    if not any(reason["code"] == code for reason in reasons):
        reasons.append({"code": code, "detail": detail})

    return decision


def validate_string_list(value, field_name):
    if not isinstance(value, list):
        raise ResponseInvalidError(f"{field_name} must be a list.")

    if any(not isinstance(item, str) for item in value):
        raise ResponseInvalidError(
            f"{field_name} must contain only text values."
        )

    return [
        compact_text(item, 300)
        for item in value
        if item.strip()
    ]


# The 16 keys the model must return. limitations and missing_evidence
# are program-owned and never requested from the model.
RESPONSE_FIELDS = (
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
    "sources",
)

SCORE_FIELDS = tuple(
    field for field in RESPONSE_FIELDS if field.endswith("_score")
)
LIST_FIELDS = ("recommended_actions", "sources")


def build_response_schema():
    """
    Strict JSON Schema for the model response, built from
    RESPONSE_FIELDS (Groq structured outputs, strict mode). Every field
    is required and no other field is allowed. Scores use Groq's
    documented union-with-null form; the 1-5 range is not expressible in
    the documented subset and stays enforced by validate_decision.
    """
    properties = {}

    for field in RESPONSE_FIELDS:
        if field in SCORE_FIELDS:
            properties[field] = {"type": ["integer", "null"]}
        elif field in LIST_FIELDS:
            properties[field] = {"type": "array", "items": {"type": "string"}}
        elif field == "alert_recommendation":
            properties[field] = {
                "type": "string",
                "enum": sorted(VALID_RECOMMENDATIONS),
            }
        else:
            properties[field] = {"type": "string"}

    return {
        "type": "object",
        "properties": properties,
        "required": list(RESPONSE_FIELDS),
        "additionalProperties": False,
    }


RESPONSE_SCHEMA_NAME = "weather_safety_assessment"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": RESPONSE_SCHEMA_NAME,
        "strict": True,
        "schema": build_response_schema(),
    },
}


def serialized_request(messages):
    """
    The text the request guards measure: the messages and the response
    schema, since both are sent to Groq.
    """
    return json_text({"messages": messages, "response_format": RESPONSE_FORMAT})


# ============================================================
# Component-aware request token estimate
# ============================================================
#
# estimate_tokens() counts 2.5 characters per token over the whole
# serialized request. Measured offline with the o200k_harmony tokenizer
# (the gpt-oss tokenizer; a development-time tool only, never imported
# here) it over-counts static prompt and policy text by about two times,
# while digit-heavy evidence JSON runs near 2.2 characters per token.
# The estimate below therefore counts each part of a request by what it
# is, and the legacy estimate stays as the fallback and for direct calls:
#
# - static text (chat template, system prompt, instructions, response
#   schema, policy blocks): token counts pinned by the SHA-256 of the
#   exact text, generated by tools/token_table.py;
# - numeric evidence JSON: an empirical bound of 13 tokens per 20 bytes
#   (worst measured 0.52 tokens per byte);
# - agent history JSON: one token per byte, an upper bound for any text
#   because every token covers at least one byte;
# - two provider-side allowances, kept separate. The historical one is
#   the largest unexplained difference between a local count and
#   recorded provider usage (JSON-object runs). The schema one is an
#   assumption for strict-mode accounting. One strict-mode record exists
#   (run #41, HEAT only: provider 4,112 tokens against a local
#   reconstruction of about 4,109); a three-hazard request has never been
#   sent to Groq, so its provider count is unverified.
#
# The harmony chat layout below is OpenAI's documented format. Groq's own
# rendering is not public, so the layout is a reconstruction whose
# unexplained difference the historical allowance covers.

PROVIDER_ALLOWANCE_HISTORICAL_TOKENS = 192
PROVIDER_ALLOWANCE_SCHEMA_TOKENS = 300

# (numerator, denominator) of tokens per byte.
NUMERIC_EVIDENCE_TOKEN_RATIO = (13, 20)
HISTORY_TOKEN_RATIO = (1, 1)

# Program-generated evidence holds few strings. More text than this (or
# any non-ASCII) is counted at one token per byte instead of the
# empirical numeric ratio.
NUMERIC_EVIDENCE_TEXT_BYTES_LIMIT = 2500

# The allowances and ratios were derived for exactly this request
# configuration. Any difference falls back to the legacy estimate.
ESTIMATOR_PINS = {
    "model": "openai/gpt-oss-20b",
    "reasoning_effort": "low",
    "response_format_sha256": (
        "ca0d29d77c08dd5f0ea03ef140952e36187778a2abf5153d34e9fc4fa913f315"
    ),
}

HARMONY_SYSTEM_TEXT = (
    "<|start|>system<|message|>You are ChatGPT, a large language model "
    "trained by OpenAI.\nKnowledge cutoff: 2024-06\n"
    "Current date: 2026-11-09\n\nReasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be "
    "included for every message.<|end|>"
)
HARMONY_DEVELOPER_PREFIX = (
    "<|start|>developer<|message|># Instructions\n\n"
)
HARMONY_USER_PREFIX = "<|end|><|start|>user<|message|>"
HARMONY_ASSISTANT_TAIL = "<|end|><|start|>assistant"

# BEGIN STATIC_TOKEN_TABLE
STATIC_TOKEN_TABLE_META = {
    "encoding": "o200k_harmony",
    "tokenizer": "tiktoken 0.14.0",
    "entries": 28,
}

STATIC_TOKEN_TABLE = {
    # chat template: system message
    "2f7aa320695440f44ba8954b669b1609ce5c2fde7da59f5a12324af443ba2f0d": 61,
    # chat template: developer prefix and system prompt
    "15c0844a9f1ccabb01d31e936fa9c53c0eb8c75fc2a3976680114711fc982d77": 311,
    # response format section
    "a0681249b32cac03fde57309a8b463bcc6e4d24465f7676980f2b04bb739eb62": 288,
    # chat template: user prefix and instructions
    "d47206cd430b829227e0e7acb6fafeee0396f0c9fd0cfffa6da8722ec77609a0": 69,
    # policy block separator
    "75a11da44c802486bc6f65640aa48a730f0f684c5c07a42ba3cd1735eb3fb070": 1,
    # evidence heading
    "7494512a07022b4975f686099bd88f35d9db63959f0ec7d48a65d3633dee1b40": 9,
    # chat template: assistant header
    "b5718bd75e24fe8c88e0b835aaaeac8aa27d6eada8aa7ae062a34b9a542699cf": 3,
    # policy block: 1. Purpose and Governing Principles
    "63756f4029b90ed364ed8ced100dabd509687b5825b6acda408ce2cb20bde2a4": 261,
    # policy block: 2. Official Sources and Evidence Hierarchy
    "cde04a58e406dad8792a79073c54288e105f892fa4f224d803cb28b10dc23236": 294,
    # policy block: 3. Evidence Quality and Confidence
    "083048653dfbeb5de0e108cf027a618b57ed8acc2af45e7af05b39c91d742294": 211,
    # policy block: 4. Risk Assessment Dimensions
    "c667cfc7f237432b3d3fee8fd869cce7aa70a2a31ae642c364020667fe6ace7c": 634,
    # policy block: 5. Risk Calculation Model
    "8b26003d25d76cb0261d7ab90ec2b11b2234a084a602a11ec772ed3fc091edb8": 384,
    # policy block: 6. Weather Hazard Assessment
    "f32394c681866719a2a743b84897fa11bdbca35ad214c0cdd643dbdd1d04b0f6": 31,
    # policy block: 7. Regulatory and Workplace Compliance
    "a1c5f7e8ce3cecd3a54cc3bd763d0b7189c955ec71801fb9d14861d6e4a816c8": 162,
    # policy block: 8. Alert Decision Framework
    "be0a36d5feb17e725ecb26005483f3ed3a8db21c22b2a1b1396f4c22a22c6927": 407,
    # policy block: 9. Official Warning Override
    "feb93cda8c408d5da1245c5120e3917e9497815b33684a4ac606cbb3ceb3f131": 144,
    # policy block: 10. Policy Retrieval and Validation
    "8385e20b46ba0c68dc3dea33ef2a074265b0507120c29a56086cadf31b46b176": 268,
    # policy block: 11. Data Quality and Assessment Completeness
    "0ad9becb5d1e3ed9bf4497321705bd13dc179f779877c1aba2b3419d01587b87": 235,
    # policy block: 12. Required Structured Assessment Output
    "0a22d109a225fd9d4a9b78633468d5ed96a0f5cd8d28dd8ebbb07fdb6bb363a9": 393,
    # policy block: 13. Advice and Communication Requirements
    "b8f5c427b77c410950707fdf6b60d5db92eba0565f032ca10f7aa0e63f31887d": 164,
    # policy block: 14. Human Approval and Notification Governance
    "e8b145fa9d06812c976e275d9992467153405bfe317bfaff409c4dc8acdc0e94": 186,
    # policy block: 15. Assessment Quality and Consistency Rules
    "4367bf0f9d2a7d55674b4b50f557c898b062630a9ab82e2b97958d10b9bcbad4": 166,
    # policy block: 16. Final Decision Rule
    "29fec544c492e7819077d841a891568a723ee73ca94c3dbff038c6f5c385afd0": 166,
    # policy block: 6.1 Extreme Heat
    "8f5a2a111efc991a7cb084151a2d6583d5c9d53b52d31a8229b9ba92c7189fcc": 188,
    # policy block: 6.2 Wind and Dust
    "f58f50675dc1752a219fb1c07a3e149acfa2d99cd3dc18916aef1444d2d21f54": 119,
    # policy block: 6.3 Rain, Thunderstorms and Flooding
    "63356fe8a7444b174460be9afb4518afe8c88dee24f9b4462d9a8769415345a4": 140,
    # policy block: 6.4 Low Temperatures and Cold Conditions
    "796ff476b73adc70448517bff60a80d8eaee11752721a96449031e80f5021693": 84,
    # policy block: 6.5 Lightning and Thunderstorms
    "e7634ca7f177fb2fb1475f2861e8893ef3817d22acf2f4d4a59f821e648f4e9b": 89,
}
# END STATIC_TOKEN_TABLE


class RequestParts:
    """
    The components a chat request is assembled from, kept so the
    estimate can count each one. The estimator rebuilds the messages from
    these parts and refuses to estimate if they differ by a single byte.
    """

    __slots__ = (
        "system", "preamble", "blocks", "evidence_text", "history_text",
    )

    def __init__(self, system, preamble, blocks, evidence_text,
                 history_text):
        self.system = system
        self.preamble = preamble
        self.blocks = tuple(blocks)
        self.evidence_text = evidence_text
        self.history_text = history_text


class BudgetedMessages(list):
    """
    A plain list of chat messages that also carries its RequestParts. It
    compares equal to, and serializes exactly like, the list it wraps.
    """

    def __init__(self, messages, parts):
        super().__init__(messages)
        self.budget_parts = parts


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_format_section_text():
    return (
        "\n\n# Response Formats\n\n## "
        + RESPONSE_SCHEMA_NAME
        + "\n\n"
        + json_text(RESPONSE_FORMAT["json_schema"]["schema"])
    )


def static_token_texts(policy):
    """
    Every static text a request can contain, as (label, text) pairs: the
    chat template, the prompts, the response schema and every policy
    block. tools/token_table.py counts these; the estimator looks them up.
    """
    items = [
        ("chat template: system message", HARMONY_SYSTEM_TEXT),
        (
            "chat template: developer prefix and system prompt",
            HARMONY_DEVELOPER_PREFIX + SYSTEM_PROMPT.strip(),
        ),
        ("response format section", response_format_section_text()),
        (
            "chat template: user prefix and instructions",
            HARMONY_USER_PREFIX + USER_PROMPT_PREAMBLE,
        ),
        ("policy block separator", POLICY_BLOCK_JOIN),
        ("evidence heading", EVIDENCE_HEADING),
        ("chat template: assistant header", HARMONY_ASSISTANT_TAIL),
    ]

    parent, intro, subsections = parse_hazard_sections(policy)

    for title, text in policy["sections"].items():
        items.append((
            f"policy block: {title}",
            policy_block_text(title, intro if title == parent else text),
        ))

    for title, text in subsections.items():
        items.append((
            f"policy block: {title}", policy_block_text(title, text)
        ))

    return items


def unverified_estimate(reason):
    return RequestBudgetError(
        "Groq request blocked: the request token estimate could not be "
        f"verified ({reason})."
    )


def verify_static_token_table():
    table = STATIC_TOKEN_TABLE

    valid = (
        type(table) is dict
        and table
        and STATIC_TOKEN_TABLE_META.get("encoding") == "o200k_harmony"
        and STATIC_TOKEN_TABLE_META.get("entries") == len(table)
        and all(
            type(key) is str
            and re.fullmatch(r"[0-9a-f]{64}", key)
            and type(value) is int
            and value > 0
            for key, value in table.items()
        )
    )

    if not valid:
        raise unverified_estimate("static token table is stale or invalid")


def static_token_count(text):
    count = STATIC_TOKEN_TABLE.get(sha256_text(text))

    if type(count) is not int or count <= 0:
        raise unverified_estimate("static token table is stale or invalid")

    return count


def estimator_pins_match():
    return (
        MODEL == ESTIMATOR_PINS["model"]
        and REASONING_EFFORT == ESTIMATOR_PINS["reasoning_effort"]
        and sha256_text(json_text(RESPONSE_FORMAT))
        == ESTIMATOR_PINS["response_format_sha256"]
    )


def ceil_ratio(byte_count, ratio):
    numerator, denominator = ratio

    return -(-byte_count * numerator // denominator)


def evidence_text_bytes(value):
    """UTF-8 bytes of every string (keys and values) in parsed JSON."""
    if isinstance(value, str):
        return len(value.encode("utf-8"))

    if isinstance(value, dict):
        return sum(
            len(str(key).encode("utf-8")) + evidence_text_bytes(item)
            for key, item in value.items()
        )

    if isinstance(value, list):
        return sum(evidence_text_bytes(item) for item in value)

    return 0


def verify_request_parts(messages, parts):
    """Raise unless the parts rebuild the messages byte for byte."""
    if type(parts) is not RequestParts:
        raise unverified_estimate("request parts are malformed")

    texts = (
        parts.system, parts.preamble, parts.evidence_text,
        parts.history_text,
    )

    if not all(type(text) is str for text in texts) or not all(
        type(block) is str for block in parts.blocks
    ):
        raise unverified_estimate("request parts are malformed")

    well_formed = (
        isinstance(messages, list)
        and len(messages) == 2
        and all(
            type(message) is dict
            and set(message) == {"role", "content"}
            and type(message["content"]) is str
            for message in messages
        )
        and [message["role"] for message in messages]
        == ["system", "user"]
    )

    if not well_formed:
        raise unverified_estimate("request parts are malformed")

    user_text = (
        parts.preamble
        + POLICY_BLOCK_JOIN.join(parts.blocks)
        + EVIDENCE_HEADING
        + parts.evidence_text
    )

    if (
        messages[0]["content"] != parts.system
        or messages[1]["content"] != user_text
    ):
        raise unverified_estimate(
            "request parts do not rebuild the messages"
        )

    try:
        evidence = json.loads(parts.evidence_text)
    except ValueError:
        raise unverified_estimate("request parts are malformed") from None

    if (
        type(evidence) is not dict
        or json_text(evidence.get("recent_agent_history"))
        != parts.history_text
        or parts.history_text not in parts.evidence_text
    ):
        raise unverified_estimate("request parts are malformed")

    return evidence


def legacy_token_budget(request_text):
    estimate = estimate_tokens(request_text)

    return {
        "method": "legacy",
        "estimated_input_tokens": estimate,
        "legacy_estimated_input_tokens": estimate,
        "reason": None,
        "components": {},
    }


def request_token_budget(messages, request_text):
    """
    The token estimate that gates a request. Plain message lists, and any
    request whose model, schema or reasoning-effort pin differs, use the
    legacy estimate. A request carrying its parts is counted by component;
    if that cannot be verified, RequestBudgetError is raised and the
    request is never sent.
    """
    budget = legacy_token_budget(request_text)
    parts = getattr(messages, "budget_parts", None)

    if parts is None:
        budget["reason"] = "no request parts"
        return budget

    if not estimator_pins_match():
        budget["reason"] = (
            "model, response schema or reasoning effort differs from the "
            "pinned configuration"
        )
        return budget

    try:
        verify_static_token_table()
        evidence = verify_request_parts(messages, parts)

        static_items = [
            HARMONY_SYSTEM_TEXT,
            HARMONY_DEVELOPER_PREFIX + parts.system,
            response_format_section_text(),
            HARMONY_USER_PREFIX + parts.preamble,
            *parts.blocks,
            *([POLICY_BLOCK_JOIN] * max(len(parts.blocks) - 1, 0)),
            EVIDENCE_HEADING,
            HARMONY_ASSISTANT_TAIL,
        ]
        static_tokens = sum(static_token_count(text) for text in static_items)

        history_bytes = len(parts.history_text.encode("utf-8"))
        numeric_text = parts.evidence_text.replace(parts.history_text, "", 1)
        numeric_bytes = len(numeric_text.encode("utf-8"))
        text_bytes = evidence_text_bytes({
            key: value for key, value in evidence.items()
            if key != "recent_agent_history"
        })
        bounded = (
            numeric_text.isascii()
            and text_bytes <= NUMERIC_EVIDENCE_TEXT_BYTES_LIMIT
        )
        numeric_tokens = ceil_ratio(
            numeric_bytes, NUMERIC_EVIDENCE_TOKEN_RATIO if bounded else (1, 1)
        )
        history_tokens = ceil_ratio(history_bytes, HISTORY_TOKEN_RATIO)
    except RequestBudgetError:
        raise
    except Exception as exc:
        raise unverified_estimate("estimate could not be computed") from exc

    budget["method"] = "component"
    budget["estimated_input_tokens"] = (
        static_tokens
        + numeric_tokens
        + history_tokens
        + PROVIDER_ALLOWANCE_HISTORICAL_TOKENS
        + PROVIDER_ALLOWANCE_SCHEMA_TOKENS
    )
    budget["components"] = {
        "policy blocks": len(parts.blocks),
        "static text tokens (pinned)": static_tokens,
        "numeric evidence bytes": numeric_bytes,
        "numeric evidence bound": (
            "empirical 0.65 tokens per byte" if bounded
            else "1 token per byte (non-ASCII or unexpected text)"
        ),
        "numeric evidence tokens": numeric_tokens,
        "history bytes": history_bytes,
        "history tokens (1 per byte)": history_tokens,
        "historical residual allowance": (
            PROVIDER_ALLOWANCE_HISTORICAL_TOKENS
        ),
        "strict-schema allowance": PROVIDER_ALLOWANCE_SCHEMA_TOKENS,
    }

    return budget


def print_budget_components(budget):
    print("Estimator method:", budget["method"])

    if budget.get("reason"):
        print("Estimator note:", budget["reason"])

    for label, value in budget["components"].items():
        print(f"  {label}: {value}")

    print(
        "Legacy estimate (2.5 characters per token):",
        budget["legacy_estimated_input_tokens"],
    )


def json_type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "other"


def schema_value_matches(field, value):
    """Whether a value has the type the response schema declares."""
    if field in SCORE_FIELDS:
        return value is None or type(value) is int
    if field in LIST_FIELDS:
        return isinstance(value, list) and all(
            isinstance(item, str) for item in value
        )
    if field == "alert_recommendation":
        return isinstance(value, str) and value in VALID_RECOMMENDATIONS
    return isinstance(value, str)


SCHEMA_TYPE_LABELS = {
    field: (
        "integer|null" if field in SCORE_FIELDS
        else "array<string>" if field in LIST_FIELDS
        else "string(enum)" if field == "alert_recommendation"
        else "string"
    )
    for field in RESPONSE_FIELDS
}


def _normalised_key(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _nested_normalised_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield _normalised_key(key)

            yield from _nested_normalised_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_normalised_keys(item)


def response_shape_diagnostics(response):
    """
    Program-generated summary of a model response's shape. It reports
    schema field identifiers, counts, JSON type names and booleans
    only. Unexpected (model-chosen) key names and all values are never
    included.
    """
    if not isinstance(response, dict):
        return {"response_type": json_type_name(response)}

    unexpected = [key for key in response if key not in RESPONSE_FIELDS]
    missing = [field for field in RESPONSE_FIELDS if field not in response]

    unexpected_types = {}

    for key in unexpected:
        type_name = json_type_name(response[key])
        unexpected_types[type_name] = unexpected_types.get(type_name, 0) + 1

    unexpected_normalised = [
        _normalised_key(key) for key in unexpected if isinstance(key, str)
    ]
    nested = set()

    for value in response.values():
        nested.update(_nested_normalised_keys(value))

    variants = {}

    for field in missing:
        target = _normalised_key(field)
        variants[field] = {
            "case_or_separator_variant": target in unexpected_normalised,
            "prefix_or_suffix_variant": any(
                key != target and len(key) >= 4
                and (
                    key.startswith(target) or key.endswith(target)
                    or target.startswith(key) or target.endswith(key)
                )
                for key in unexpected_normalised
            ),
            "found_nested": target in nested,
        }

    return {
        "top_level_key_count": len(response),
        "missing_required_count": len(missing),
        "missing_required": missing,
        "unexpected_key_count": len(unexpected),
        "unexpected_value_types": dict(sorted(unexpected_types.items())),
        "program_owned_keys": sorted(PROGRAM_OWNED_FIELDS & set(response)),
        "missing_field_variants": variants,
        "wrong_type_fields": [
            {
                "field": field,
                "expected": SCHEMA_TYPE_LABELS[field],
                "received": json_type_name(response[field]),
            }
            for field in RESPONSE_FIELDS
            if field in response
            and not schema_value_matches(field, response[field])
        ],
    }


# ------------------------------------------------------------
# Policy source citations. A citation counts only when it resolves to
# exactly one retrieved section; the stored value is always that
# section's canonical title. Nothing outside the retrieved sections can
# be accepted.
# ------------------------------------------------------------
_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")


def normalise_citation(text):
    """Remove fixed wrappers, case and spacing differences."""
    text = re.sub(r"\s+", " ", text).strip().strip("\"'`").strip()
    text = re.sub(r"^=+\s*", "", text)
    text = re.sub(r"\s*=+$", "", text)
    text = re.sub(r"^policy section\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^(?:#+|\u00a7|section)\s*", "", text, flags=re.I)
    return text.strip().lower()


def _citation_keys(heading):
    """Full heading, number alone and heading text alone."""
    full = normalise_citation(heading)
    keys = {full: "normalised_title"}
    match = _SECTION_NUMBER.match(full)

    if match:
        keys[match.group(1)] = "number"
        keys[match.group(1) + "."] = "number"
        keys[match.group(2)] = "heading"

    return keys


def citation_index(retrieved_sections):
    """{normalised key: {(canonical title, rule), ...}}"""
    index = {}

    def add(heading, title, sub_heading=False):
        for key, rule in _citation_keys(heading).items():
            # A sub-heading is citable only with its number: "4.1
            # Likelihood (L)" or "4.1", never "Likelihood (L)" alone.
            if sub_heading and rule == "heading":
                continue

            index.setdefault(key, set()).add(
                (title, "sub_heading" if sub_heading else rule)
            )

    for section in retrieved_sections:
        title = section["section"]
        add(title, title)

        # Numbered sub-headings inside the retrieved text map to it.
        for line in section["text"].splitlines():
            if line.startswith("### "):
                heading = line[4:].strip()

                if _SECTION_NUMBER.match(heading):
                    add(heading, title, sub_heading=True)

    return index


def resolve_citation(citation, retrieved_sections, index=None):
    """(canonical title or None, rule) for one citation."""
    titles = [section["section"] for section in retrieved_sections]

    if not isinstance(citation, str):
        return None, "unmatched"

    if citation in titles:
        return citation, "exact"

    index = citation_index(retrieved_sections) if index is None else index
    matches = index.get(normalise_citation(citation), set())
    resolved = {title for title, _ in matches}

    if len(resolved) == 1:
        title = resolved.pop()
        rules = sorted({rule for t, rule in matches if t == title})
        return title, rules[0]

    if len(resolved) > 1:
        return None, "ambiguous"

    if _SECTION_NUMBER.match(normalise_citation(citation)):
        return None, "unretrieved_section_reference"

    return None, "unmatched"


def resolve_policy_sources(citations, retrieved_sections):
    """Canonical titles for every resolvable citation, in order, once."""
    index = citation_index(retrieved_sections)
    resolved = []

    for citation in citations:
        title, _ = resolve_citation(citation, retrieved_sections, index)

        if title is not None and title not in resolved:
            resolved.append(title)

    return resolved


def policy_section_numbers(policy):
    """
    Every section number the loaded policy defines: the numbered
    level-two sections and their numbered "###" sub-headings.
    """
    numbers = set()

    for title, text in policy.get("sections", {}).items():
        match = _SECTION_NUMBER.match(normalise_citation(title))

        if match:
            numbers.add(match.group(1))

        for line in text.splitlines():
            if line.startswith("### "):
                sub = _SECTION_NUMBER.match(
                    normalise_citation(line[4:])
                )

                if sub:
                    numbers.add(sub.group(1))

    return numbers


def unresolved_reference_bucket(citation, index, policy_numbers):
    """
    Fixed label for a numbered citation that did not resolve. Uses only
    the number and the program's own policy data, never the text.
    """
    number = _SECTION_NUMBER.match(normalise_citation(citation)).group(1)

    if number in index:
        return "retrieved_number_title_mismatch"

    if number in policy_numbers:
        return "policy_section_not_retrieved"

    return "section_number_not_in_policy"


def source_resolution_diagnostics(citations, retrieved_sections, policy=None):
    """
    Program-generated counts per resolution rule. Never includes the
    model's citation text. With the loaded policy, numbered citations
    that did not resolve are split into fixed buckets: the number
    belongs to a retrieved section but the title differs
    (retrieved_number_title_mismatch), the section exists in the policy
    but was not sent to the model (policy_section_not_retrieved), or the
    number does not exist in the policy (section_number_not_in_policy).
    """
    if not isinstance(citations, list):
        return {"sources_type": json_type_name(citations)}

    index = citation_index(retrieved_sections)
    policy_numbers = (
        policy_section_numbers(policy) if isinstance(policy, dict) else None
    )
    counts = {}

    for citation in citations:
        _, rule = resolve_citation(citation, retrieved_sections, index)

        if (
            rule == "unretrieved_section_reference"
            and policy_numbers is not None
        ):
            rule = unresolved_reference_bucket(
                citation, index, policy_numbers
            )

        counts[rule] = counts.get(rule, 0) + 1

    return {
        "source_count": len(citations),
        "resolution_counts": dict(sorted(counts.items())),
    }


def validate_decision(decision, retrieved_sections):
    if not isinstance(decision, dict):
        raise ResponseInvalidError("AI returned an invalid assessment.")

    reserved = sorted(PROGRAM_OWNED_FIELDS & set(decision))

    if reserved:
        raise ResponseInvalidError(
            "Model output sets program-owned fields: " + ", ".join(reserved)
        )

    required = list(RESPONSE_FIELDS)

    missing = [
        field for field in required
        if field not in decision
    ]

    if missing:
        raise ResponseInvalidError(
            "Missing assessment fields: " + ", ".join(missing)
        )

    recommendation = decision["alert_recommendation"]

    if recommendation not in VALID_RECOMMENDATIONS:
        raise ResponseInvalidError("Invalid alert recommendation.")

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
            raise ResponseInvalidError(
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
            raise ResponseInvalidError(f"Missing rationale: {field}")

    for field in ("hazard_type", "advice", "reasoning"):
        if not isinstance(decision[field], str):
            raise ResponseInvalidError(
                f"Invalid text field: {field} (expected text, "
                f"received {type(decision[field]).__name__})"
            )

        if not decision[field].strip():
            raise ResponseInvalidError(f"Empty text field: {field}")

    for field in (
        "recommended_actions",
        "sources",
    ):
        decision[field] = validate_string_list(
            decision[field],
            field,
        )

    # Only citations that resolve to exactly one retrieved section are
    # kept, as canonical titles; anything else is dropped.
    valid_sources = resolve_policy_sources(
        decision["sources"], retrieved_sections
    )

    if not valid_sources:
        raise ResponseInvalidError(
            "Assessment did not identify a valid retrieved policy section."
        )

    decision["sources"] = valid_sources

    # Program-owned: filled only from validation outcomes and, later, the
    # program's own evidence and status checks. Never from model text.
    decision["limitations"] = []
    decision["missing_evidence"] = []

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
        add_incomplete_reason(
            decision,
            "RISK_DIMENSIONS_UNASSESSED",
            ", ".join(unassessed),
        )

    # The model's preliminary analysis is not a completed safety clearance.
    decision["assessment_status"] = "PRELIMINARY"

    return decision


# Dimensions whose required evidence this program does not have.
# guidelines.md 4.3: exposure needs population, workforce, vulnerable-
# group and infrastructure evidence; 4.5: geographic impact must not be
# inferred "from a single city-wide weather reading". The pipeline has
# neither, so the model's scores are never accepted: the score is set
# to null (UNASSESSED, section 5 rule 4) with a fixed reason. Remove an
# entry only when a real evidence source for that dimension exists.
# (score field -> (reason field, fixed program-generated reason))
EVIDENCE_GATED_DIMENSIONS = {
    "exposure_score": (
        "exposure_reason",
        "Not assessed: no population, workforce, vulnerable-group or "
        "infrastructure exposure evidence is available to this program.",
    ),
    "geographic_impact_score": (
        "geographic_impact_reason",
        "Not assessed: only one Dubai forecast point is available, and "
        "geographic extent cannot be inferred from a single reading.",
    ),
}


def apply_evidence_gate(decision):
    """
    Replace model-supplied scores for evidence-gated dimensions with
    null and a fixed reason, before validation and risk calculation.
    Only keys the model actually returned are touched, so a missing key
    still fails validation. Returns the names of fields where the model
    supplied a non-null value (names only, never the values).
    """
    if not isinstance(decision, dict):
        return []

    overridden = []

    for score_field, (reason_field, reason) in (
        EVIDENCE_GATED_DIMENSIONS.items()
    ):
        if score_field not in decision:
            continue

        if decision[score_field] is not None:
            overridden.append(score_field)

        decision[score_field] = None

        if reason_field in decision:
            decision[reason_field] = reason

    return overridden


class UnverifiedWarningClaim(ClassifiedFailure, ValueError):
    """
    Model text asserts an official warning state that was not checked.

    Carries only program-generated metadata: the schema field label
    (or "outside_schema"), the number (1-5) of the first matching
    pattern, the already-validated canonical policy sources and the
    evidence-gate override names. Never the matched text.
    """

    def __init__(
        self, message, field_label="unknown", pattern_number=None,
        shape=None,
    ):
        super().__init__(message)
        self.field_label = field_label
        self.pattern_number = pattern_number
        self.shape = shape
        self.sources = []
        self.gate_overrides = []

    outcome = "INVALID"
    failure_code = "UNVERIFIED_WARNING_CLAIM"
    baseline_name = "UnverifiedWarningClaim"

    def diagnostic(self):
        """"field=<label>;pattern=<n>" built from fixed values only."""
        label = (
            self.field_label
            if self.field_label in RESPONSE_FIELDS
            or self.field_label == "outside_schema"
            else "unknown"
        )
        number = (
            self.pattern_number
            if type(self.pattern_number) is int
            and 1 <= self.pattern_number
            <= len(UNVERIFIED_WARNING_CLAIM_PATTERNS)
            else None
        )

        return f"field={label};pattern={number}"

    def shape_diagnostic(self):
        """
        Fixed-vocabulary shape of a pattern 2 or 3 match, else None.
        Every value is checked against the fixed sets; anything else
        becomes "unknown" (words between and "currently": None).
        """
        if self.pattern_number not in (2, 3):
            return None

        shape = self.shape if isinstance(self.shape, dict) else {}
        between = shape.get("words_between")

        if self.pattern_number == 2:
            return {
                "lead_form": (
                    shape.get("lead_form")
                    if shape.get("lead_form") in _CLAIM_P2_LEAD_FORMS
                    else "unknown"
                ),
                "currently": (
                    shape.get("currently")
                    if type(shape.get("currently")) is bool
                    else None
                ),
                "words_between": (
                    between
                    if type(between) is int and 0 <= between <= 2
                    else None
                ),
                "term_class": (
                    shape.get("term_class")
                    if shape.get("term_class") in _CLAIM_TERM_CLASSES
                    else "unknown"
                ),
                "following_word_class": (
                    shape.get("following_word_class")
                    if shape.get("following_word_class")
                    in _CLAIM_FOLLOW_CLASSES
                    else "unknown"
                ),
            }

        return {
            "qualifier": (
                shape.get("qualifier")
                if shape.get("qualifier") in _CLAIM_QUALIFIERS
                else "unknown"
            ),
            "term_class": (
                shape.get("term_class")
                if shape.get("term_class") in _CLAIM_TERM_CLASSES
                else "unknown"
            ),
            "words_between": (
                between
                if type(between) is int and 0 <= between <= 2
                else None
            ),
            "following_word_class": (
                shape.get("following_word_class")
                if shape.get("following_word_class") in _CLAIM_FOLLOW_CLASSES
                else "unknown"
            ),
        }


_WARNING_TERMS = r"(?:warnings?|advisor(?:y|ies)|emergency\s+instructions?)"
_NOT_A_CLAIM = (
    r"(?!\s+(?:feed|feeds|source|sources|data|system|service|check|checks"
    r"|verification|status|integration|api|recommendation|recommended"
    r"|level|levels|criteria|threshold|thresholds)\b)"
)
_WARNING_STATE = (
    r"(?:in\s+effect|active|issued|in\s+place|current(?:ly)?|reported"
    r"|announced|declared|exist|existing)"
)
_WARNING_AUTHORITIES = (
    r"(?:NCM|NCEMA|Dubai\s+Police|MoHRE"
    r"|National\s+Cent(?:er|re)\s+of\s+Meteorology|authorit(?:y|ies))"
)

# Statements about whether an official warning or advisory is, or is
# not, in effect. While warnings are NOT_CHECKED either claim is
# unsupported (guidelines sections 1, 9 and 16).
UNVERIFIED_WARNING_CLAIM_PATTERNS = (
    # "no official heat advisory is in effect", "no warnings were issued"
    re.compile(
        r"\bno\s+(?:[a-z-]+\s+){0,3}" + _WARNING_TERMS + r"\b"
        + _NOT_A_CLAIM + r"(?:\s+[a-z-]+){0,4}?\s+" + _WARNING_STATE
        + r"\b",
        re.I,
    ),
    # "there are no warnings", "there's no advisory for Dubai"
    re.compile(
        r"\bthere(?:\s+(?:is|are|were|was)|['\u2019]s)\s+(?:currently\s+)?"
        r"no\s+(?:[a-z-]+\s+){0,2}" + _WARNING_TERMS + r"\b" + _NOT_A_CLAIM,
        re.I,
    ),
    # "there are no active warnings", "no official warning at this time"
    re.compile(
        r"\bno\s+(?:active|current|official|existing)\s+"
        r"(?:[a-z-]+\s+){0,2}" + _WARNING_TERMS + r"\b" + _NOT_A_CLAIM,
        re.I,
    ),
    # "a heat warning is in effect", "the advisory is not active"
    re.compile(
        _WARNING_TERMS + r"\s+(?:is|are|was|were|has|have)(?:n['\u2019]t)?\s+"
        r"(?:not\s+|no\s+longer\s+)?(?:been\s+|currently\s+)?"
        + _WARNING_STATE + r"\b(?!\s+by\b)",
        re.I,
    ),
    # "NCM has not issued", "authorities have issued"
    re.compile(
        r"\b" + _WARNING_AUTHORITIES + r"\s+(?:has|have|had)(?:n['\u2019]t)?\s+"
        r"(?:not\s+|no\s+)?(?:yet\s+)?"
        r"(?:issued|declared|announced|reported)\b",
        re.I,
    ),
)

# A match preceded by "whether" or "if" (within three words) is a
# conditional, not a claim: "verify whether an advisory is in effect".
_CONDITIONAL_LEAD = re.compile(r"\b(?:whether|if)\b(?:\s+\S+){0,3}\s*$", re.I)

# Invisible characters that could split a phrase without changing how
# it reads; removed before matching.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))

# Fields only the program may set. A model response containing any of
# them is rejected rather than trusted or silently dropped.
PROGRAM_OWNED_FIELDS = frozenset({
    "official_warning_status", "official_warning_authority",
    "assessment_status", "risk_score", "risk_band",
    "incomplete_reasons", "groq_usage", "send_alert", "alert_message",
    "approval_status", "notification_status", "evidence_quality",
    "evidence_confidence", "policy_status", "policy_version",
    "policy_hash", "policy_path", "hazard_screening", "tools_used",
    "weather_summary", "weather_source", "weather_retrieval_status",
    "weather_data_timestamp", "assessment_timestamp", "city",
    "timezone", "history_checked", "history_status",
    "response_diagnostics", "evidence_gate_overrides",
    "warning_claim_diagnostic", "limitations", "missing_evidence",
    "provider_error", "location_observation", "run_outcome",
    "failure_code", "failure_stage",
})


def find_unverified_warning_claim_pattern(text):
    """
    Number (1-5, in UNVERIFIED_WARNING_CLAIM_PATTERNS order) of the
    first pattern that asserts an official warning state in text,
    else None.
    """
    text = text.translate(_INVISIBLE)

    for number, pattern in enumerate(
        UNVERIFIED_WARNING_CLAIM_PATTERNS, start=1
    ):
        for match in pattern.finditer(text):
            lead = text[max(0, match.start() - 40):match.start()]

            if not _CONDITIONAL_LEAD.search(lead):
                return number

    return None


def find_unverified_warning_claim(text):
    """Return True when text asserts an official warning state."""
    return find_unverified_warning_claim_pattern(text) is not None


# Fixed vocabularies for the pattern 3 shape diagnostic.
_CLAIM_QUALIFIERS = ("active", "current", "official", "existing")
_CLAIM_TERM_CLASSES = ("warning", "advisory", "emergency_instruction")
_CLAIM_FOLLOW_CLASSES = (
    "end", "punctuation", "auxiliary", "preposition", "conjunction",
    "other_word", "evidence_noun", "unknown",
)
_CLAIM_P2_LEAD_FORMS = (
    "there_is", "there_are", "there_was", "there_were", "there_s",
)
# Nouns that make "no warning <noun>" a statement about evidence rather
# than about whether a warning exists. Reported separately for pattern 2.
_CLAIM_EVIDENCE_NOUNS = frozenset({
    "information", "evidence", "text", "details", "content", "input",
    "inputs", "report", "reports", "record", "records", "indication",
    "indications", "mention", "mentions", "signal", "signals", "update",
    "updates", "notice", "notices", "message", "messages", "statement",
    "statements", "entry", "entries",
})
_CLAIM_AUXILIARIES = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "will", "would", "can", "could", "should", "may", "might",
    "must", "do", "does", "did",
})
_CLAIM_PREPOSITIONS = frozenset({
    "for", "at", "in", "on", "of", "by", "to", "from", "during",
    "within", "across", "over", "under", "with", "without", "as",
    "today", "tonight", "now",
})
_CLAIM_CONJUNCTIONS = frozenset({
    "and", "or", "but", "so", "because", "since", "while", "although",
    "though", "if", "unless", "when", "that", "which",
})

# Applied only to the text already matched by pattern 3.
_CLAIM_SHAPE_PARTS = re.compile(
    r"no\s+(active|current|official|existing)\s+((?:[a-z-]+\s+){0,2})"
    r"(warnings?|advisor(?:y|ies)|emergency\s+instructions?)",
    re.I,
)
_CLAIM_NEXT_WORD = re.compile(r"\s*([A-Za-z][A-Za-z'\u2019-]*|\S)?")


def _claim_follow_class(rest):
    """Class of the text after a match, from the fixed vocabulary."""
    found = _CLAIM_NEXT_WORD.match(rest).group(1)

    if found is None:
        return "end"

    if not found[0].isalpha():
        return "punctuation"

    word = found.lower().replace("\u2019", "'")

    if word in _CLAIM_AUXILIARIES:
        return "auxiliary"

    if word in _CLAIM_PREPOSITIONS:
        return "preposition"

    if word in _CLAIM_CONJUNCTIONS:
        return "conjunction"

    return "other_word"


def find_unverified_warning_claim_shape(text):
    """
    Fixed-vocabulary shape of the first non-conditional pattern 3 match
    in text, else None. Returns only values from the fixed sets above
    and a count of 0-2; never the matched text or its surroundings.
    """
    text = text.translate(_INVISIBLE)
    unknown = {
        "qualifier": "unknown", "term_class": "unknown",
        "words_between": None, "following_word_class": "unknown",
    }

    for match in UNVERIFIED_WARNING_CLAIM_PATTERNS[2].finditer(text):
        lead = text[max(0, match.start() - 40):match.start()]

        if _CONDITIONAL_LEAD.search(lead):
            continue

        parts = _CLAIM_SHAPE_PARTS.fullmatch(match.group(0))

        if parts is None:
            return unknown

        term = parts.group(3).lower()

        return {
            "qualifier": parts.group(1).lower(),
            "term_class": (
                "warning" if term.startswith("warning")
                else "advisory" if term.startswith("advis")
                else "emergency_instruction"
            ),
            "words_between": len(parts.group(2).split()),
            "following_word_class": _claim_follow_class(
                text[match.end():]
            ),
        }

    return None


# Applied only to the text already matched by pattern 2.
_CLAIM_SHAPE_PARTS_P2 = re.compile(
    r"there(?:\s+(is|are|were|was)|['\u2019]s)\s+(currently\s+)?"
    r"no\s+((?:[a-z-]+\s+){0,2})"
    r"(warnings?|advisor(?:y|ies)|emergency\s+instructions?)",
    re.I,
)


def find_unverified_warning_claim_shape_pattern2(text):
    """
    Fixed-vocabulary shape of the first non-conditional pattern 2 match
    in text, else None. Returns only values from the fixed sets above,
    a boolean and a count of 0-2; never the matched text or its
    surroundings.
    """
    text = text.translate(_INVISIBLE)
    unknown = {
        "lead_form": "unknown", "currently": None, "words_between": None,
        "term_class": "unknown", "following_word_class": "unknown",
    }

    for match in UNVERIFIED_WARNING_CLAIM_PATTERNS[1].finditer(text):
        lead = text[max(0, match.start() - 40):match.start()]

        if _CONDITIONAL_LEAD.search(lead):
            continue

        parts = _CLAIM_SHAPE_PARTS_P2.fullmatch(match.group(0))

        if parts is None:
            return unknown

        term = parts.group(4).lower()
        rest = text[match.end():]
        follow = _claim_follow_class(rest)

        if follow == "other_word":
            word = _CLAIM_NEXT_WORD.match(rest).group(1).lower()

            if word in _CLAIM_EVIDENCE_NOUNS:
                follow = "evidence_noun"

        return {
            "lead_form": (
                "there_s" if parts.group(1) is None
                else "there_" + parts.group(1).lower()
            ),
            "currently": parts.group(2) is not None,
            "words_between": len(parts.group(3).split()),
            "term_class": (
                "warning" if term.startswith("warning")
                else "advisory" if term.startswith("advis")
                else "emergency_instruction"
            ),
            "following_word_class": follow,
        }

    return None


def find_unverified_warning_claim_shape_for(pattern_number, text):
    """Shape for the pattern that fired (2 or 3), else None."""
    if pattern_number == 3:
        return find_unverified_warning_claim_shape(text)

    if pattern_number == 2:
        return find_unverified_warning_claim_shape_pattern2(text)

    return None


def model_strings(value):
    """
    Every string inside a model value, however deeply nested,
    including dictionary key names.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key

            yield from model_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from model_strings(item)


# Official warning verification, reported separately from assessment
# validity. No warning provider is integrated, so NOT_CHECKED is the
# only status this program produces; the other values are reserved for
# a future, separately approved feed.
WARNING_VERIFICATION_STATES = (
    "NOT_CHECKED", "VERIFIED_NONE", "VERIFIED_ACTIVE", "UNAVAILABLE",
)


def official_warning_verification():
    """The single source of the official warning verification state."""
    return {"status": "NOT_CHECKED", "authority": None, "validity": None}


def reject_unverified_warning_claims(decision, official_warning_status):
    """
    Fail closed when model text states that an official warning or
    advisory is, or is not, in effect. The check runs under every
    verification status: no warning feed is integrated, so there is no
    verified state to compare a claim against. official_warning_status
    comes from the program, never from the model, and is only named in
    the error. Every string in every model field is checked, including
    keys outside the schema. The error names the field, never its
    content.
    """
    for field, value in decision.items():
        # Top-level key names are model text too. Only schema field
        # names are echoed in the error; any other key name is
        # model-chosen and is never repeated.
        location = (
            field if field in RESPONSE_FIELDS
            else "a field outside the response schema"
        )

        for text in model_strings({field: value}):
            pattern_number = find_unverified_warning_claim_pattern(text)

            if pattern_number is not None:
                raise UnverifiedWarningClaim(
                    "Model output makes an unverified official-warning "
                    f"claim in {location} while official warning status "
                    f"is {official_warning_status}.",
                    field_label=(
                        field if field in RESPONSE_FIELDS
                        else "outside_schema"
                    ),
                    pattern_number=pattern_number,
                    shape=find_unverified_warning_claim_shape_for(
                        pattern_number, text
                    ),
                )

    return decision


def enforce_assessment_status(decision):
    """
    Assessment validity, independent of official warning verification
    (guidelines section 11). An assessment is COMPLETE only when the
    policy and the weather evidence are available, a risk score was
    calculated and no incomplete reason is recorded. When the policy file
    could not be retrieved the status is UNAVAILABLE (section 10); every
    other case is INCOMPLETE. Official warning status is reported
    separately and governs alert authorization, which this program never
    grants.
    """
    complete = (
        not decision.get("incomplete_reasons")
        and decision.get("risk_score") is not None
        and decision.get("policy_status") == "AVAILABLE"
        and decision.get("weather_retrieval_status") == "AVAILABLE"
    )

    decision["assessment_status"] = (
        "COMPLETE" if complete
        else "UNAVAILABLE" if decision.get("policy_status") == "UNAVAILABLE"
        else "INCOMPLETE"
    )

    return decision


def apply_alert_authorization(decision, assessment_failed):
    """
    The only place the decision's alert fields are set. This program
    never authorizes an alert: send_alert is always False and no message
    is prepared. A failed assessment is NOT_AUTHORIZED; any other result
    waits for human review (guidelines section 14). The output payload
    separately hard-codes send_alert False.
    """
    decision["approval_status"] = (
        "NOT_AUTHORIZED" if assessment_failed else "PENDING_REVIEW"
    )
    decision["send_alert"] = False
    decision["alert_message"] = ""
    decision["notification_status"] = "NOT_SENT"

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

    decision["official_warning_status"] = (
        official_warning_verification()["status"]
    )
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
4. Official warning status is NOT_CHECKED (program-reported). Free text must not use warning, warnings, advisory or advisories, or claim their existence, absence, availability or status; describe evidence limits without that. Follow the schema. Open-Meteo is not an official UAE warning authority.
5. Distinguish current observations, forecasts and historical context.
6. Do not assume all residents or workers have equal exposure.
7. Previous alerts must not suppress a new or different hazard. Do not
   recommend an alert solely because a value is unusual.
8. Use only recommendation values defined in the supplied policy.
9. Explain missing evidence. The assessment is never a
   completed safety clearance.
10. In sources, cite exact policy section titles.
11. Return a single JSON object only.

Each score must be an integer from 1 to 5 when evidence supports it.
If evidence for a dimension is insufficient, return null for that
score, never a guessed value, and explain why in its rationale.
"""


POLICY_BLOCK_START = "=== POLICY SECTION: "
POLICY_BLOCK_END = "=== END POLICY SECTION ==="
POLICY_BLOCK_JOIN = "\n\n"
EVIDENCE_HEADING = "\n\nEVIDENCE AND STATUS (JSON):\n"

# The instructions that open the user message, before the policy blocks.
USER_PROMPT_PREAMBLE = (
    "Assess the supplied weather situation using the policy "
    "sections below. Do not infer official warning status. "
    "Do not claim the assessment is complete. In sources, give "
    "each title exactly as it appears after 'POLICY SECTION:' in a "
    "block header (for example '6.1 Extreme Heat'), without the "
    "prefix or the '===' markers.\n\n"
)


def policy_block_text(title, text):
    return f"{POLICY_BLOCK_START}{title} ===\n{text}\n{POLICY_BLOCK_END}"


def policy_blocks(sections):
    """
    Render each policy section as one plain-text block. Section text is
    sent verbatim (no JSON escaping); a section that contains the block
    delimiters is rejected so it cannot break out of its block.
    """
    blocks = []

    for item in sections:
        if (
            POLICY_BLOCK_START in item["text"]
            or POLICY_BLOCK_END in item["text"]
            or "\n" in item["section"]
        ):
            raise PolicyInvalidError(
                "Policy section cannot be rendered safely: "
                + item["section"]
            )

        blocks.append(policy_block_text(item["section"], item["text"]))

    return blocks


def format_policy_text(sections):
    return POLICY_BLOCK_JOIN.join(policy_blocks(sections))


def assess_weather(weather, history, policy, retrieved_sections):
    compact_weather = summarize_weather(weather)
    compact_history = summarize_history(history)

    context = {
        "city": CITY,
        "weather": compact_weather,
        "recent_agent_history": compact_history,
        "policy_version": policy["version"],
        "official_warning_status": (
            official_warning_verification()["status"] + " (program-reported)"
        ),
        "risk_score_notice": (
            "Any risk score is an internal provisional calculation made "
            "by the program, not an official UAE or NCM classification, "
            "and must not be treated as validated for automated public "
            "notifications."
        ),
    }

    blocks = policy_blocks(retrieved_sections)
    evidence_text = json_text(context)

    user_prompt = (
        USER_PROMPT_PREAMBLE
        + POLICY_BLOCK_JOIN.join(blocks)
        + EVIDENCE_HEADING
        + evidence_text
    )

    # The same messages, carrying the parts they were assembled from so
    # the request guard can count each part. Serialized exactly like a
    # plain list.
    messages = BudgetedMessages(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        RequestParts(
            SYSTEM_PROMPT.strip(),
            USER_PROMPT_PREAMBLE,
            blocks,
            evidence_text,
            json_text(compact_history),
        ),
    )

    LAST_RESPONSE_DIAGNOSTICS.clear()

    response = ask_groq(messages)
    decision = parse_json_response(response)

    # Computed before validation, which normalises some values.
    diagnostics = response_shape_diagnostics(decision)

    if isinstance(decision, dict) and "sources" in decision:
        diagnostics["source_resolution"] = source_resolution_diagnostics(
            decision["sources"], retrieved_sections, policy
        )

    # Evidence gate: model scores for exposure and geographic impact
    # are discarded and replaced by null plus a fixed reason.
    gate_overrides = apply_evidence_gate(decision)

    try:
        decision = validate_decision(
            decision,
            retrieved_sections,
        )
    except ValueError:
        LAST_RESPONSE_DIAGNOSTICS.update(diagnostics)
        print("\nRESPONSE SHAPE DIAGNOSTICS (program-generated; no model text)")
        print(json_text(diagnostics))
        raise

    decision["evidence_gate_overrides"] = gate_overrides

    # Checked on the model's own text, before the guard adds ours. The
    # warning status is the program's: no official feed is integrated.
    try:
        decision = reject_unverified_warning_claims(
            decision, official_warning_verification()["status"]
        )
    except UnverifiedWarningClaim as claim:
        # Validation has already run: the sources are canonical retrieved
        # titles and the override names are fixed program values.
        claim.sources = list(decision["sources"])
        claim.gate_overrides = list(gate_overrides)

        claim_diagnostics = dict(diagnostics)
        shape = claim.shape_diagnostic()

        if shape is not None:
            claim_diagnostics["warning_claim_shape"] = shape

        LAST_RESPONSE_DIAGNOSTICS.update(claim_diagnostics)
        print("\nRESPONSE SHAPE DIAGNOSTICS (program-generated; no model text)")
        print(json_text(claim_diagnostics))
        raise

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
        "incomplete_reasons": [],
    }


# ============================================================
# 11. MAIN AGENT
# ============================================================

def run_weather_agent():
    used = []

    policy = None
    weather = None
    history = []
    # None until screening runs, so a failure before it is not read as
    # "no hazards".
    hazards = None
    LAST_RESPONSE_DIAGNOSTICS.clear()

    history_status = "NOT_CHECKED"

    # Recorded for diagnosis only; the failure classification comes from
    # the exception type, never from the stage.
    stage = "configuration"

    try:
        # A sheet URL is optional for a dry-run assessment.
        require_configuration(require_sheet=False)

        # Load policy.
        stage = "policy"
        policy = load_guidelines()
        used.append("load_guidelines")

        # Retrieve weather.
        stage = "weather"
        weather = get_weather(CITY)
        used.append("get_weather")

        # Retrieve recent history if a sheet URL is available.
        stage = "history"
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
        stage = "screening"
        hazards = identify_hazards(weather)
        used.append("identify_hazards")
        print("Hazards identified:", sorted(hazards))

        stage = "retrieval"
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
            raise PolicyInvalidError("No policy sections were retrieved.")

        stage = "assessment"
        decision = assess_weather(
            weather,
            history,
            policy,
            retrieved_sections,
        )

        stage = "finalize"

        # Explicitly record evidence limitations.
        warning_verification = official_warning_verification()
        decision["official_warning_status"] = warning_verification["status"]
        decision["official_warning_authority"] = (
            warning_verification["authority"]
        )

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
        decision["location_observation"] = weather.get(
            "location_observation"
        )
        decision["hazard_screening"] = hazards
        decision["groq_usage"] = dict(LAST_GROQ_USAGE)
        decision["response_diagnostics"] = dict(LAST_RESPONSE_DIAGNOSTICS)

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
        apply_alert_authorization(decision, assessment_failed=False)

        enforce_assessment_status(decision)

        decision["run_outcome"] = (
            "COMPLETE" if decision["assessment_status"] == "COMPLETE"
            else "INCOMPLETE"
        )
        decision["failure_code"] = None
        decision["failure_stage"] = None

        return decision, used

    except Exception as exc:
        run_outcome, run_failure_code = classify_failure(exc)
        error_message = f"{exception_label(exc)}: {exc}"

        print("\nASSESSMENT ERROR:", error_message)

        decision = incomplete_assessment(error_message)

        if isinstance(exc, UnverifiedWarningClaim):
            failure_code = "UNVERIFIED_WARNING_CLAIM"
        elif isinstance(exc, PolicyUnavailableError):
            failure_code = "POLICY_UNAVAILABLE"
        else:
            failure_code = "ASSESSMENT_ERROR"

        add_incomplete_reason(
            decision,
            failure_code,
            exc.diagnostic()
            if isinstance(exc, UnverifiedWarningClaim)
            else exception_label(exc),
        )

        if isinstance(exc, UnverifiedWarningClaim):
            decision["sources"] = [
                title for title in exc.sources if isinstance(title, str)
            ]
            decision["evidence_gate_overrides"] = [
                name for name in exc.gate_overrides
                if name in EVIDENCE_GATED_DIMENSIONS
            ]
            decision["warning_claim_diagnostic"] = exc.diagnostic()

        if isinstance(exc, ProviderError):
            decision["provider_error"] = exc.diagnostic()

        if isinstance(exc, PolicyUnavailableError):
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

        decision["official_warning_status"] = (
            official_warning_verification()["status"]
        )
        decision["groq_usage"] = dict(LAST_GROQ_USAGE)
        decision["response_diagnostics"] = dict(LAST_RESPONSE_DIAGNOSTICS)
        decision["hazard_screening"] = hazards

        decision["evidence_quality"] = "E0"
        decision["evidence_confidence"] = "NONE"

        enforce_assessment_status(decision)
        apply_alert_authorization(decision, assessment_failed=True)

        decision["weather_retrieval_status"] = (
            "AVAILABLE" if weather else "UNAVAILABLE"
        )
        decision["location_observation"] = (
            weather.get("location_observation") if weather else None
        )

        decision["history_status"] = history_status

        decision["assessment_timestamp"] = (
            datetime.now().astimezone().isoformat()
        )

        decision["city"] = CITY
        decision["tools_used"] = used

        decision["run_outcome"] = run_outcome
        decision["failure_code"] = run_failure_code
        decision["failure_stage"] = stage

        return decision, used


# ============================================================
# 12. OUTPUT AND GOOGLE SHEETS INTEGRATION
# ============================================================

# ============================================================
# 11a. RUN OUTCOME REPORTING
# ============================================================

# Exit code for each run outcome (guidelines section 11 vocabulary).
# These are process exit codes: 0 means COMPLETE only, and the program
# never reports INCOMPLETE as a success (it exits 2). The workflow step
# turns exit code 2 into a passing job with one warning (weather.yml).
RUN_OUTCOME_EXIT_CODES = {
    "COMPLETE": 0,
    "INCOMPLETE": 2,
    "INVALID": 1,
    "UNAVAILABLE": 1,
}

FAILURE_CODES = (
    "CONFIGURATION_INVALID", "POLICY_UNAVAILABLE", "POLICY_INVALID",
    "WEATHER_UNAVAILABLE", "REQUEST_BUDGET_EXCEEDED", "PROVIDER_ERROR",
    "RESPONSE_INVALID", "UNVERIFIED_WARNING_CLAIM", "SHEETS_WRITE_FAILED",
    "INTERNAL_ERROR",
)

RUN_STAGES = (
    "configuration", "policy", "weather", "history", "screening",
    "retrieval", "assessment", "finalize", "report", "sheets",
)

INCOMPLETE_REASON_CODES = (
    "RISK_DIMENSIONS_UNASSESSED", "WEATHER_EVIDENCE_UNCERTAIN",
    "ASSESSMENT_ERROR", "UNVERIFIED_WARNING_CLAIM", "POLICY_UNAVAILABLE",
)

ASSESSMENT_STATUSES = ("COMPLETE", "INCOMPLETE", "INVALID", "UNAVAILABLE")

_CLAIM_DIAGNOSTIC_TEXT = re.compile(r"field=[a-z_]{1,40};pattern=[1-5]")


class SheetsWriteError(ClassifiedFailure, RuntimeError):
    outcome = "UNAVAILABLE"
    failure_code = "SHEETS_WRITE_FAILED"
    baseline_name = "RuntimeError"

    def __init__(self, message, baseline_name=None):
        super().__init__(message)

        if baseline_name is not None:
            self.baseline_name = baseline_name


def classify_failure(exc):
    """
    (outcome, failure_code) for a failure. Only classified failures map
    to a known cause; any other exception is an INTERNAL_ERROR, whatever
    stage it was raised in.
    """
    if isinstance(exc, ClassifiedFailure):
        outcome = (
            exc.outcome if exc.outcome in ("INVALID", "UNAVAILABLE")
            else "INVALID"
        )
        code = (
            exc.failure_code if exc.failure_code in FAILURE_CODES
            else "INTERNAL_ERROR"
        )

        return outcome, code

    return "INVALID", "INTERNAL_ERROR"


def run_outcome_of(decision):
    """The decision's run outcome; anything unrecognised is INVALID."""
    outcome = decision.get("run_outcome")

    return outcome if outcome in RUN_OUTCOME_EXIT_CODES else "INVALID"


def _fixed(value, allowed):
    return value if value in allowed else "unknown"


def outcome_fields(decision, outcome):
    """
    Fixed-vocabulary summary of a run for annotations and the step
    summary. Every value is checked against a closed set (or is a count
    or boolean); model text, provider text and weather values never
    appear.
    """
    reasons = [
        _fixed(reason.get("code"), INCOMPLETE_REASON_CODES)
        for reason in decision.get("incomplete_reasons", [])
        if isinstance(reason, dict)
    ]
    provider = decision.get("provider_error")
    claim = decision.get("warning_claim_diagnostic")
    observation = decision.get("location_observation")
    sources = decision.get("sources")

    def flag(key):
        value = observation.get(key) if isinstance(observation, dict) else None
        return value if type(value) is bool else None

    return {
        "outcome": _fixed(outcome, RUN_OUTCOME_EXIT_CODES),
        "exit_code": RUN_OUTCOME_EXIT_CODES.get(outcome, 1),
        "assessment_status": _fixed(
            decision.get("assessment_status"), ASSESSMENT_STATUSES
        ),
        "alert_recommendation": _fixed(
            decision.get("alert_recommendation"), VALID_RECOMMENDATIONS
        ),
        "incomplete_reasons": reasons,
        "failure_code": (
            _fixed(decision.get("failure_code"), FAILURE_CODES)
            if decision.get("failure_code") is not None else "none"
        ),
        "failure_stage": (
            _fixed(decision.get("failure_stage"), RUN_STAGES)
            if decision.get("failure_stage") is not None else "none"
        ),
        "provider_category": (
            _fixed(provider.get("category"), PROVIDER_ERROR_CATEGORIES)
            if isinstance(provider, dict) else "none"
        ),
        "warning_claim": (
            claim if isinstance(claim, str)
            and _CLAIM_DIAGNOSTIC_TEXT.fullmatch(claim)
            else "none" if claim is None else "unknown"
        ),
        "official_warning_status": _fixed(
            decision.get("official_warning_status"),
            WARNING_VERIFICATION_STATES,
        ),
        "hazards_screened": sorted(
            code for code in (decision.get("hazard_screening") or {})
            if code in HAZARD_POLICY_TITLES
        ),
        "policy_sources_cited": (
            len(sources) if isinstance(sources, list) else 0
        ),
        "location_timezone_matches": flag("timezone_matches_request"),
        "location_utc_offset_present": flag("utc_offset_seconds_present"),
    }


def _workflow_message(value):
    """Escape a workflow-command message (GitHub Actions rules)."""
    return (
        str(value).replace("%", "%25").replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def outcome_annotation(fields):
    """One GitHub Actions annotation line built from outcome_fields."""
    outcome = fields["outcome"]

    if outcome == "COMPLETE":
        level = "notice"
        message = (
            f"exit_code={fields['exit_code']}"
            f" | recommendation={fields['alert_recommendation']}"
        )
    elif outcome == "INCOMPLETE":
        # A notice, not a warning: under GitHub Actions the workflow step
        # raises the single INCOMPLETE warning from the exit code, so a
        # passing run can never lack one. This line keeps the reasons.
        level = "notice"
        message = (
            f"exit_code={fields['exit_code']}"
            f" | reasons={','.join(fields['incomplete_reasons']) or 'none'}"
            f" | warnings={fields['official_warning_status']}"
        )
    else:
        level = "error"
        message = (
            f"exit_code={fields['exit_code']}"
            f" | failure={fields['failure_code']}"
            f" | stage={fields['failure_stage']}"
            f" | provider={fields['provider_category']}"
            f" | claim={fields['warning_claim']}"
        )

    return (
        f"::{level} title=Assessment {outcome}::"
        + _workflow_message(message)
    )


def outcome_summary_markdown(fields):
    """Step-summary table built from outcome_fields only."""
    def show(value):
        if value is True:
            return "yes"

        if value is False:
            return "no"

        if value is None:
            return "unknown"

        if isinstance(value, list):
            return ", ".join(value) or "none"

        return str(value)

    rows = (
        ("Outcome", fields["outcome"]),
        ("Exit code", fields["exit_code"]),
        ("Assessment status", fields["assessment_status"]),
        ("Recommendation", fields["alert_recommendation"]),
        ("Incomplete reasons", fields["incomplete_reasons"]),
        ("Failure code", fields["failure_code"]),
        ("Failure stage", fields["failure_stage"]),
        ("Provider error", fields["provider_category"]),
        ("Warning claim", fields["warning_claim"]),
        ("Official warning status", fields["official_warning_status"]),
        ("Hazards screened", fields["hazards_screened"]),
        ("Policy sources cited", fields["policy_sources_cited"]),
        ("Location timezone matches request",
         fields["location_timezone_matches"]),
        ("UTC offset field present", fields["location_utc_offset_present"]),
    )

    return (
        "### Weather assessment run\n\n| Field | Value |\n|---|---|\n"
        + "".join(f"| {name} | {show(value)} |\n" for name, value in rows)
        + "\n"
    )


def emit_outcome(decision, outcome):
    """
    Report the run outcome: always as one program-generated log line;
    under GitHub Actions also as one annotation and a step-summary
    table. Writes only to stdout and the runner's local summary file.
    A summary write failure never changes the outcome.
    """
    fields = outcome_fields(decision, outcome)
    print("\nRUN OUTCOME (program-generated):", json_text(fields))

    if os.environ.get("GITHUB_ACTIONS") != "true":
        return fields

    print(outcome_annotation(fields))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(outcome_summary_markdown(fields))
        except OSError:
            print("STEP SUMMARY: not written (file error).")

    return fields


# Assessment status of the run that run_agent_job is currently reporting,
# so a failure while reporting can name it. Reset at the start of every
# run; None until that run has an assessment.
LAST_REPORTED_ASSESSMENT = {"status": None}


def emit_unhandled_failure(exc):
    """Report a failure raised outside run_weather_agent, then let it
    propagate (exit 1, traceback kept)."""
    outcome, failure_code = classify_failure(exc)
    stage = (
        "sheets" if isinstance(exc, SheetsWriteError)
        else "configuration" if isinstance(exc, ConfigurationError)
        else "report"
    )

    return emit_outcome(
        {
            "run_outcome": outcome, "failure_code": failure_code,
            "failure_stage": stage,
            "assessment_status": LAST_REPORTED_ASSESSMENT.get("status"),
        },
        outcome,
    )


def run_agent_job(dry=False):
    LAST_REPORTED_ASSESSMENT["status"] = None
    decision, used = run_weather_agent()
    LAST_REPORTED_ASSESSMENT["status"] = decision.get("assessment_status")

    print("\n========== WEATHER ASSESSMENT ==========")

    print("CITY:", decision.get("city"))
    print("POLICY:", decision.get("policy_status"))
    print("POLICY VERSION:", decision.get("policy_version"))
    print("ASSESSMENT:", decision.get("assessment_status"))

    for reason in decision.get("incomplete_reasons", []):
        print("INCOMPLETE REASON:", reason["code"], "|", reason["detail"])

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

    print("\nAUDIT SUMMARY (program-generated)")
    print(
        json_text(
            {
                "sources": decision.get("sources", []),
                "alert_recommendation": decision.get(
                    "alert_recommendation"
                ),
                "assessment_status": decision.get("assessment_status"),
                "risk_score_calculated": (
                    decision.get("risk_score") is not None
                ),
                "risk_band": decision.get("risk_band"),
                "incomplete_reason_codes": [
                    reason["code"]
                    for reason in decision.get("incomplete_reasons", [])
                ],
                "evidence_gate_overrides": decision.get(
                    "evidence_gate_overrides", []
                ),
                "warning_claim_diagnostic": decision.get(
                    "warning_claim_diagnostic"
                ),
                "hazards_screened": sorted(
                    decision.get("hazard_screening") or {}
                ),
            }
        )
    )

    print("\nSOURCES:", decision.get("sources"))
    print(
        "OFFICIAL WARNING:",
        decision.get("official_warning_status"),
    )
    print("ALERT AUTHORIZED:", decision.get("send_alert"))
    print(
        "LOCATION OBSERVATION (program-generated; not enforced):",
        json_text(decision.get("location_observation")),
    )
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

    outcome = run_outcome_of(decision)

    if dry:
        print("\nDRY RUN: No data was sent to Google Sheets.")
        print("DRY RUN: No email was sent.")
        # The payload stays the last output, as before.
        emit_outcome(decision, outcome)
        print("\nOUTPUT PAYLOAD:")
        print(json_text(payload))
        return outcome

    # A normal agent run can write to Sheets, but only after
    # the explicit configuration check below.
    require_configuration(require_sheet=True)

    try:
        response = HTTP.post(
            SHEET_URL,
            json=payload,
            timeout=SHEET_TIMEOUT,
        )

        print("\nSHEET RESPONSE:", response.status_code, response.text)

        response.raise_for_status()
    except requests.RequestException as exc:
        raise SheetsWriteError(
            str(exc), baseline_name=type(exc).__name__
        ) from exc

    if response.text.strip() != "OK":
        raise SheetsWriteError(
            "Google Apps Script did not confirm success."
        )

    emit_outcome(decision, outcome)
    return outcome


# ============================================================
# 13. DETERMINISTIC TESTS
# ============================================================

def run_tests():
    passed = 0
    failed = 0

    # Tests must never write workflow annotations or a step summary.
    saved_actions_env = {
        name: os.environ.pop(name, None)
        for name in ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY")
    }

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
        "Status helper never accepts a claimed COMPLETE or VERIFIED label",
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
        })["assessment_status"] == "INCOMPLETE",
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
        and MAX_COMPLETION_TOKENS == 1000,
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

    def production_input_estimate(messages, text):
        """The estimate the request guard uses; None if it cannot be
        verified (which the guard treats as blocked)."""
        try:
            return request_token_budget(messages, text)[
                "estimated_input_tokens"
            ]
        except RequestBudgetError:
            return None

    def fits_all_budgets(hazards):
        sections, messages = build_request(
            {code: ["t"] for code in hazards}
        )
        serialized = serialized_request(messages)
        estimate = production_input_estimate(messages, serialized)

        # The production estimate, as the request guard computes it.
        return (
            len(serialized) <= MAX_REQUEST_CHARS
            and estimate is not None
            and estimate + MAX_COMPLETION_TOKENS
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

    # The legacy 2.5 characters-per-token estimate blocked this request;
    # the component-aware estimate fits it. The realistic-evidence
    # five-hazard request is still blocked (see the realistic tests).
    check(
        "All five hazards together fit this fixture's budgets with the "
        "component-aware estimate",
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
            # Recorded, then raised as RuntimeError so a request that
            # wrongly reaches the transport fails the check cleanly
            # instead of aborting the whole suite.
            self.posts.append(args)
            raise RuntimeError("request must be blocked first")

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

    # The pre-send guard that trips first depends on request size: the
    # character limit is checked before the token estimate.
    check(
        "Worst case (all uncertain, sections 7 and 11) fails closed "
        "before sending",
        ("token budget" in message or "character limit" in message)
        and not posts,
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
        "recommended_actions": [],
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

    benign_decision = pipeline(
        make_weather(), EVIDENCE_GATED_DIMENSIONS={}
    )

    check(
        "Benign weather with every dimension evidenced is COMPLETE, no alert",
        benign_decision["hazard_screening"] == {}
        and benign_decision["assessment_status"] == "COMPLETE"
        and benign_decision.get("incomplete_reasons", []) == []
        and benign_decision["official_warning_status"] == "NOT_CHECKED"
        and benign_decision["alert_recommendation"] == "MONITOR"
        and benign_decision["send_alert"] is False
        and benign_decision["approval_status"] == "PENDING_REVIEW"
        and len(sent_messages) == 1
        and len(serialized_request(sent_messages[0])) <= MAX_REQUEST_CHARS,
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
        good["groq_usage"].get("prompt_tokens") == 1234
        and good["groq_usage"].get("finish_reason") == "stop"
        and good["assessment_status"] == "INCOMPLETE",
    )

    # A request blocked before Groq (a stale static token table) sends
    # nothing, records no usage and fails closed. Checks of the usage
    # record fail on this outcome; they never raise or pass.
    blocked_transport = FakeHttp(FakeResponse(body=groq_body()))
    blocked = with_globals(
        {
            "GROQ_API_KEY": secret_key,
            "HTTP": blocked_transport,
            "STATIC_TOKEN_TABLE": {},
            "require_configuration": (
                lambda require_sheet=False: None
            ),
            "get_weather": lambda city: make_weather(),
            "tool_get_history": lambda n=3: [],
        },
        lambda: run_weather_agent()[0],
    )

    check(
        "A request blocked before Groq sends nothing, records no usage "
        "and fails closed",
        blocked_transport.requests == []
        and "prompt_tokens" not in blocked["groq_usage"]
        and "could not be verified" in blocked["reasoning"]
        and blocked["assessment_status"] == "INCOMPLETE"
        and blocked["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and blocked["send_alert"] is False
        and blocked["notification_status"] == "NOT_SENT"
        and secret_key not in json_text(blocked),
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
            "recommended_actions": [],
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
    workflow_text = (
        workflow_path.read_text(encoding="utf-8")
        if workflow_path.is_file() else ""
    )
    workflow_modes = re.findall(
        r"python\s+weather_log\.py\s+(\S+)", workflow_text
    )

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
        workflow_modes == ["agent-dry"]
        and not re.search(r"weather_log\.py\s+agent(?!-dry)", workflow_text)
        and Path(loaded_elsewhere["path"]).resolve()
        == (Path(__file__).resolve().parent / "guidelines.md")
        and loaded_elsewhere["hash"] == hazard_policy["hash"],
    )

    # ------------------------------------------------------------
    # Workflow exit-code handling. weather_log.py exits 0 (COMPLETE),
    # 2 (INCOMPLETE), or 1 (INVALID, UNAVAILABLE, INTERNAL_ERROR); the
    # dry-run step must pass 0, warn on 2, and fail on everything else.
    # ------------------------------------------------------------
    def wf_run_block():
        lines = workflow_text.splitlines()

        for index, line in enumerate(lines):
            if line.strip() == "run: |":
                indent = len(line) - len(line.lstrip())
                body = []

                for item in lines[index + 1:]:
                    if item.strip() and len(item) - len(item.lstrip()) <= indent:
                        break

                    body.append(item[indent + 2:] if item.strip() else "")

                return "\n".join(body).rstrip() + "\n"

        return ""

    wf_script = wf_run_block()
    wf_warning_text = (
        "::warning title=Weather assessment INCOMPLETE::The program reported "
        "an INCOMPLETE assessment (exit code 2). This is an expected "
        "outcome, not an execution failure. This dry run sends no alert. "
        "See the job summary for the reasons."
    )

    check(
        "Workflow: exit codes 0, 1, 2 and any other are handled explicitly",
        wf_script.count("python weather_log.py agent-dry") == 1
        and "code=0\n" in wf_script
        and re.search(r'python weather_log\.py agent-dry \|\| code=\$\?', wf_script)
        is not None
        and 'case "$code" in' in wf_script
        and re.search(r"^\s*0\) ;;$", wf_script, re.M) is not None
        and re.search(r"^\s*1\) exit 1 ;;$", wf_script, re.M) is not None
        and re.search(
            r'^\s*2\) echo "::warning [^"]*" ;;$', wf_script, re.M
        ) is not None
        and re.search(
            r'^\s*\*\) echo "::error [^"]*"; exit "\$code" ;;$',
            wf_script, re.M,
        ) is not None,
    )

    check(
        "Workflow: no continue-on-error, error suppression or forced success",
        wf_script != ""
        and not any(
            fragment in workflow_text
            for fragment in (
                "continue-on-error", "|| true", "set +e", "/dev/null",
                "exit 0", "2>&1", "| tee",
            )
        ),
    )

    wf_annotations = [
        line.strip() for line in wf_script.splitlines() if "::" in line
    ]
    wf_warnings = [line for line in wf_annotations if '"::warning ' in line]

    check(
        "Workflow: INCOMPLETE gives exactly one fixed warning, never COMPLETE",
        len(wf_warnings) == 1
        and wf_script.count("::warning ") == 1
        and wf_warning_text in wf_warnings[0]
        and "$" not in wf_warnings[0]
        and "INCOMPLETE" in wf_warnings[0]
        and re.search(r"\bCOMPLETE\b", wf_warnings[0]) is None,
    )

    check(
        "Workflow: failure annotations are fixed text apart from the exit code",
        len([line for line in wf_annotations if '"::error ' in line]) == 2
        and wf_script.count("::error ") == 2
        and all(
            re.sub(r"\$code\b", "", line.split("::error", 1)[1]).count("$") == 0
            for line in wf_annotations if '"::error ' in line
        ),
    )

    check(
        "Workflow: the job summary is left to the program",
        "GITHUB_STEP_SUMMARY" not in workflow_text
        and wf_warning_text.count("job summary") == 1,
    )

    check(
        "Workflow: setup steps, secrets and dry-run mode are unchanged",
        re.findall(r"uses: (\S+)", workflow_text)
        == ["actions/checkout@v4", "actions/setup-python@v5"]
        and 'python-version: "3.12"' in workflow_text
        and "- run: pip install requests" in workflow_text
        and re.findall(r"secrets\.(\w+)", workflow_text)
        == ["SHEET_URL", "GROQ_API_KEY"]
        and "SHEET_URL: ${{ secrets.SHEET_URL }}" in workflow_text
        and "GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}" in workflow_text,
    )

    import shutil
    import subprocess

    # These behaviour checks need a POSIX bash on PATH (Git Bash on
    # Windows). Without one they are skipped and listed at the end.
    wf_bash = shutil.which("bash")
    wf_skipped = []

    def wf_run_step(exit_code, with_script=True, emit=None):
        """Run the extracted step under `bash -e` with a stub python.
        `emit` lines are printed by the stub, as the program would."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "bin").mkdir()
            shim = work / "bin" / "python"
            shim.write_text(
                "#!/bin/sh\n"
                'echo "$*" > args.txt\n'
                "[ -f emit.txt ] && cat emit.txt\n"
                "echo PYTHON-STDOUT-MARK\n"
                "echo PYTHON-STDERR-MARK >&2\n"
                'exit "$FAKE_PYTHON_EXIT"\n',
                encoding="utf-8", newline="\n",
            )
            shim.chmod(0o755)
            (work / "step.sh").write_text(
                wf_script, encoding="utf-8", newline="\n"
            )
            (work / "summary.md").write_text("PRE\n", encoding="utf-8")

            if emit:
                (work / "emit.txt").write_text(
                    "\n".join(emit) + "\n", encoding="utf-8", newline="\n"
                )

            if with_script:
                (work / "weather_log.py").write_text("", encoding="utf-8")

            env = dict(os.environ)
            env.update(
                PATH=str(work / "bin") + os.pathsep + env.get("PATH", ""),
                FAKE_PYTHON_EXIT=str(exit_code),
                GITHUB_STEP_SUMMARY="summary.md",
            )
            done = subprocess.run(
                [wf_bash, "-e", "step.sh"], cwd=work, env=env,
                capture_output=True, text=True, timeout=60,
            )
            args = work / "args.txt"

            return {
                "code": done.returncode,
                "stdout": done.stdout,
                "stderr": done.stderr,
                "args": args.read_text().strip() if args.exists() else None,
                "summary": (work / "summary.md").read_text(),
                "annotations": [
                    line.strip() for line in done.stdout.splitlines()
                    if line.startswith("::")
                ],
            }

    wf_probe = None

    if wf_bash and wf_script:
        try:
            wf_probe = wf_run_step(3)
        except (OSError, subprocess.SubprocessError):
            wf_probe = None

    if wf_probe is None or wf_probe["args"] != "weather_log.py agent-dry":
        wf_skipped.append("workflow step behaviour")
        print(
            "SKIP: workflow step behaviour checks (no usable POSIX bash). "
            "Exit-code handling is then covered structurally only."
        )
    else:
        wf_results = {
            code: wf_run_step(code) for code in (0, 1, 2, 3, 127, 139)
        }

        check(
            "Workflow step: exit 0 passes with no annotation",
            wf_results[0]["code"] == 0 and wf_results[0]["annotations"] == [],
        )

        check(
            "Workflow step: exit 2 passes with one fixed INCOMPLETE warning",
            wf_results[2]["code"] == 0
            and wf_results[2]["annotations"] == [wf_warning_text],
        )

        check(
            "Workflow step: exit 1 fails the step",
            wf_results[1]["code"] == 1 and wf_results[1]["annotations"] == [],
        )

        check(
            "Workflow step: unexpected exit codes fail with that code",
            all(
                wf_results[code]["code"] == code
                and wf_results[code]["annotations"] == [
                    "::error title=Weather assessment::Unexpected exit code "
                    f"{code} from weather_log.py (expected 0, 1 or 2)."
                ]
                for code in (3, 127, 139)
            ),
        )

        check(
            "Workflow step: python output is never suppressed",
            all(
                "PYTHON-STDOUT-MARK" in result["stdout"]
                and "PYTHON-STDERR-MARK" in result["stderr"]
                and result["args"] == "weather_log.py agent-dry"
                for result in wf_results.values()
            ),
        )

        check(
            "Workflow step: the job summary is never written by the step",
            all(result["summary"] == "PRE\n" for result in wf_results.values()),
        )

        wf_missing = wf_run_step(2, with_script=False)

        check(
            "Workflow step: a missing script is a failure, never INCOMPLETE",
            wf_missing["code"] == 1
            and wf_missing["args"] is None
            and wf_missing["annotations"] == [
                "::error title=Weather assessment::weather_log.py was not found."
            ],
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
        "Sixteen response keys are required",
        len(adv_keys) == 16,
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

    # Contract: the response schema (sent with every request and
    # enforced by Groq strict mode) and the validator agree.
    adv_schema = build_response_schema()

    check(
        "Schema lists exactly the keys the validator requires",
        sorted(adv_schema["required"]) == sorted(adv_keys)
        and sorted(adv_schema["properties"]) == sorted(adv_keys)
        and len(set(adv_schema["required"])) == 16,
    )

    adv_string_keys = [
        key for key in adv_keys
        if adv_schema["properties"][key].get("type") == "string"
    ]
    adv_score_keys = [
        key for key in adv_keys
        if adv_schema["properties"][key].get("type") == ["integer", "null"]
    ]
    adv_array_keys = [
        key for key in adv_keys
        if adv_schema["properties"][key].get("type") == "array"
    ]

    check(
        "Schema states the type of every key",
        {"hazard_type", "advice", "reasoning",
         "alert_recommendation"} <= set(adv_string_keys)
        and set(adv_score_keys)
        == {k for k in adv_keys if k.endswith("_score")}
        and set(adv_array_keys)
        == {"recommended_actions", "sources"}
        and len(adv_string_keys) + len(adv_score_keys)
        + len(adv_array_keys) == 16
        and all(
            isinstance(adv_response()[key], str) for key in adv_string_keys
        )
        and all(
            type(adv_response()[key]) is int for key in adv_score_keys
        )
        and all(
            isinstance(adv_response()[key], list) for key in adv_array_keys
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
            {
                "ask_groq": lambda messages, temperature=0.1: reply,
                "EVIDENCE_GATED_DIMENSIONS": {},
            },
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
        and good_result["groq_usage"].get("prompt_tokens") == 3239
        and good_result["groq_usage"].get("completion_tokens") == 694
        and good_result["groq_usage"].get("reasoning_tokens") == 301
        and good_result["groq_usage"].get("finish_reason") == "stop"
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
        and incident_result["groq_usage"].get("completion_tokens") == 694
        and incident_result["groq_usage"].get("finish_reason") == "stop"
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

    absent_reasoning_usage = adv_pipeline(
        json_text(adv_response(sources=real_sources)),
        usage={"prompt_tokens": 1, "completion_tokens": 2,
               "total_tokens": 3},
    )[0]["groq_usage"]

    # A missing usage record must fail this check, not satisfy "is None".
    check(
        "Absent reasoning-token detail is recorded as null",
        absent_reasoning_usage.get("prompt_tokens") == 1
        and "reasoning_tokens" in absent_reasoning_usage
        and absent_reasoning_usage["reasoning_tokens"] is None,
    )

    # ------------------------------------------------------------
    # Fixes A, B and C (self-contained, offline):
    #   A. model text must not claim any official warning state
    #      while official warnings are NOT_CHECKED;
    #   B. the hourly next-24-hour series reaches the model (duration);
    #   C. structured incomplete_reasons on every decision.
    # ------------------------------------------------------------
    def abc_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def abc_weather(hours=24, **current):
        start = datetime(2026, 11, 8, 13, 0)
        base_current = {
            "time": "2026-11-08T12:00",
            "temperature_2m": 28.3, "apparent_temperature": 32.4,
            "relative_humidity_2m": 65, "precipitation": 0,
            "wind_speed_10m": 12, "wind_gusts_10m": 20,
            "visibility": 24140, "weather_code": 1, "cape": 0,
        }
        base_current.update(current)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": base_current,
            "today_forecast": {
                "high": 31, "low": 24, "wind": 15, "gust": 27,
                "precipitation": 0,
            },
            "previous_days": [],
            "hourly_forecast_next_24h": [
                {
                    "time": (start + timedelta(hours=h)).strftime(
                        "%Y-%m-%dT%H:%M"
                    ),
                    "temperature": 31.4, "apparent_temperature": 34.6,
                    "humidity": 63, "precipitation": 0.0,
                    "wind": 13.7, "gust": 27.4, "visibility": 24140.0,
                    "weather_code": 1, "cape": 0,
                }
                for h in range(hours)
            ],
            "history_summary": {},
        }

    def abc_reply(**changes):
        body = {
            "hazard_type": "HEAT",
            "likelihood_score": 2, "likelihood_reason": "lr",
            "severity_score": 2, "severity_reason": "sr",
            "exposure_score": None,
            "exposure_reason": "No population exposure data supplied.",
            "duration_score": None,
            "duration_reason": "dr",
            "geographic_impact_score": None,
            "geographic_impact_reason": "Single city-wide point.",
            "alert_recommendation": "MONITOR",
            "advice": "Limit strenuous outdoor activity at midday. "
                      "Official warning status was not checked; "
                      "consult NCM for official warnings.",
            "reasoning": "Apparent temperature reaches the heat "
                         "screening value.",
            "recommended_actions": ["Check NCM warnings before work"],
            "sources": ["6.1 Extreme Heat"],
        }
        body.update(changes)
        return json_text(body)

    class AbcResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content):
            self.content = content

        def json(self):
            return {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 3239, "completion_tokens": 600,
                    "total_tokens": 3839,
                },
            }

    class AbcHttp:
        def __init__(self, content):
            self.content = content
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append({"url": url, "json": json})
            return AbcResponse(self.content)

    def abc_pipeline(content, weather=None, **extra):
        transport = AbcHttp(content)
        printed = io.StringIO()
        settings = {
            "HTTP": transport,
            "GROQ_API_KEY": "gsk_abc_test_key_not_real",
            "require_configuration": lambda require_sheet=False: None,
            "get_weather": lambda city: weather or abc_weather(),
            "tool_get_history": lambda n=3: [],
        }
        settings.update(extra)

        def run():
            with contextlib.redirect_stdout(printed):
                return run_weather_agent()[0]

        return abc_override(settings, run), printed.getvalue(), transport

    def abc_codes(decision):
        return [reason["code"] for reason in decision["incomplete_reasons"]]

    # A1. Claim detection: assertions of either state are claims.
    claims = (
        "No official heat advisory is in effect.",
        "There is no official heat advisory in effect for Dubai.",
        "There are no active warnings.",
        "No warnings have been issued for Dubai today.",
        "No current official weather warnings.",
        "There is no official warning at this time.",
        "no heat advisories are in place",
        "A heat warning is currently in effect.",
        "The NCM advisory is active until 6 PM.",
        "Warnings have not been issued.",
        "NCM has not issued any warning.",
        "Authorities have issued an emergency instruction.",
        "The National Center of Meteorology has not issued alerts.",
    )

    for text in claims:
        check(
            f"Warning-state claim detected: {text}",
            find_unverified_warning_claim(text),
        )

    neutral = (
        "Official warning status was not checked.",
        "Official warning status was not checked; consult NCM for "
        "official warnings.",
        "Check NCM warnings before outdoor work.",
        "Consult NCM for official warnings and advisories.",
        "Official NCM and emergency authority warnings were not "
        "retrieved or verified by this implementation.",
        "NOT_CHECKED. No NCM or other official warning feed is "
        "integrated.",
        "No immediate advisory recommendation is justified.",
        "No current advisory recommendation is made.",
        "Verify whether a heat advisory is in effect with NCM.",
        "If an official warning is issued, follow it.",
        "Heat advisories are issued by NCM.",
        "No alert is recommended.",
        "Previous alerts must not suppress a new hazard.",
        "Warning criteria are defined by NCM.",
    )

    for text in neutral:
        check(
            f"Neutral wording is not a claim: {text}",
            not find_unverified_warning_claim(text),
        )

    # A2. Every model text field is checked, without echoing content.
    def abc_claim_error(field, value):
        decision = json.loads(abc_reply())
        decision[field] = value
        try:
            reject_unverified_warning_claims(decision, "NOT_CHECKED")
            return ""
        except UnverifiedWarningClaim as exc:
            return str(exc)

    abc_text_fields = (
        "hazard_type", "advice", "reasoning",
        "likelihood_reason", "severity_reason", "exposure_reason",
        "duration_reason", "geographic_impact_reason",
        "recommended_actions", "unexpected_notes",
    )

    for field in abc_text_fields:
        value = (
            ["No official heat advisory is in effect."]
            if field == "recommended_actions"
            else "No official heat advisory is in effect."
        )
        message = abc_claim_error(field, value)

        expected_location = (
            field if field in RESPONSE_FIELDS
            else "a field outside the response schema"
        )

        check(
            f"Warning claim rejected in field: {field}",
            f"claim in {expected_location}" in message
            and "heat advisory" not in message,
        )

    try:
        reject_unverified_warning_claims(
            {"advice": "No official heat advisory is in effect."},
            "VERIFIED",
        )
        verified_claim_skipped = True
    except UnverifiedWarningClaim:
        verified_claim_skipped = False

    check(
        "Claim check is never skipped, even for a VERIFIED status",
        not verified_claim_skipped,
    )

    check(
        "Prompt forbids the terms warning and advisory in free text",
        "Free text must not use warning, warnings, advisory or "
        "advisories, or claim their existence, absence, availability "
        "or status" in " ".join(SYSTEM_PROMPT.split()),
    )

    # A3. Real path: the incident's contradictory advice fails closed.
    for label, recommendation in (
        ("MONITOR", "MONITOR"),
        ("NO_ALERT", "NO_ALERT_RECOMMENDED"),
        ("ADVISORY", "ADVISORY_RECOMMENDED"),
    ):
        claimed, claimed_output, _ = abc_pipeline(abc_reply(
            alert_recommendation=recommendation,
            advice="No official heat advisory is in effect. "
                   "Limit outdoor activity.",
        ))

        check(
            f"Contradictory advice fails closed ({label})",
            claimed["assessment_status"] == "INCOMPLETE"
            and claimed["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and claimed["risk_band"] == "UNASSESSED"
            and claimed["send_alert"] is False
            and claimed["official_warning_status"] == "NOT_CHECKED"
            and "UNVERIFIED_WARNING_CLAIM" in abc_codes(claimed)
            and "OFFICIAL_WARNINGS_NOT_VERIFIED" not in abc_codes(claimed)
            and "heat advisory is in effect"
            not in json_text(claimed) + claimed_output
            and "gsk_abc_test_key_not_real" not in claimed_output,
        )

    neutral_result, _, _ = abc_pipeline(abc_reply())

    check(
        "Neutral advice passes through the real path unchanged",
        neutral_result["advice"].startswith(
            "Limit strenuous outdoor activity at midday."
        )
        and "UNVERIFIED_WARNING_CLAIM" not in abc_codes(neutral_result),
    )

    # B1. Hourly series: complete, ordered, faithful.
    series_weather = abc_weather()
    series_weather["hourly_forecast_next_24h"][5]["humidity"] = None
    series_weather["hourly_forecast_next_24h"][7]["apparent_temperature"] = 40.04
    series = hourly_series(series_weather["hourly_forecast_next_24h"])

    check(
        "Hourly series covers all 24 hours in order",
        series["start"] == "2026-11-08T13:00"
        and series["step_hours"] == 1
        and all(
            len(series[name]) == 24
            for name, _, _ in HOURLY_SERIES_FIELDS
        ),
    )

    check(
        "Hourly series keeps values and nulls faithfully",
        series["apparent_c"][7] == 40.0
        and series["temp_c"][0] == 31.4
        and series["humidity_pct"][5] is None
        and series["humidity_pct"][0] == 63
        and series["visibility_m"][0] == 24140
        and series["wind_kmh"][0] == 13.7
        and series["gust_kmh"][0] == 27.4
        and series["weather_code"][0] == 1,
    )

    gapped = abc_weather()
    del gapped["hourly_forecast_next_24h"][3]
    gap_series = hourly_series(gapped["hourly_forecast_next_24h"])

    check(
        "Non-consecutive hours list every timestamp",
        "start" not in gap_series
        and gap_series["times"]
        == [r["time"] for r in gapped["hourly_forecast_next_24h"]]
        and len(gap_series["temp_c"]) == 23,
    )

    empty_summary = summarize_weather(abc_weather(hours=0))

    check(
        "No hourly data gives no series",
        hourly_series([]) is None
        and "next_24_hours_hourly" in empty_summary
        and empty_summary["next_24_hours_hourly"] is None,
    )

    sent_request, _, sent_transport = abc_pipeline(abc_reply())
    sent_text = json_text(
        sent_transport.posts[0]["json"]["messages"]
    ) if sent_transport.posts else ""

    check(
        "Model request carries the hourly series",
        "next_24_hours_hourly" in sent_text
        and "step_hours" in sent_text
        and "apparent_c" in sent_text,
    )

    # B2. Realistic 24-hour requests against the unchanged budgets.
    def abc_request_size(hazards):
        weather = abc_weather()
        with contextlib.redirect_stdout(io.StringIO()):
            policy = load_guidelines()
            required = select_hazard_guidance(policy, hazards)
            required += select_conditional_sections(
                policy, hazards, weather
            )
            sections = search_guidelines(
                policy, "x", k=0, required_sections=required
            )

        captured = []

        def capture(messages, temperature=0.1):
            captured.append(messages)
            raise RuntimeError("captured")

        def assemble():
            try:
                assess_weather(weather, [
                    {"timestamp": "t" * 60, "city": "c" * 60,
                     "advice": "a" * 500, "alert_sent": False,
                     "alert_message": "m" * 500},
                ] * MAX_HISTORY_RECORDS, policy, sections)
            except RuntimeError:
                pass

        abc_override({"ask_groq": capture}, assemble)
        text = serialized_request(captured[0])

        estimate = production_input_estimate(captured[0], text)

        return len(text), (
            (10 ** 9 if estimate is None else estimate)
            + MAX_COMPLETION_TOKENS
        )

    def abc_fits(hazards):
        chars, tokens = abc_request_size(hazards)
        return (
            chars <= MAX_REQUEST_CHARS
            and tokens <= MAX_ESTIMATED_REQUEST_TOKENS
        )

    check(
        "24-hour series: no-hazard request fits every budget",
        abc_fits({}),
    )

    for code in HAZARD_POLICY_TITLES:
        check(
            f"24-hour series: single hazard fits every budget: {code}",
            abc_fits({code: ["t"]}),
        )

    # The legacy estimate exceeded the 8,000-token budget for heat +
    # wind with worst-case history at the 1,000-token completion
    # allowance; the component-aware estimate fits it.
    check(
        "24-hour series: heat with wind fits at the 1,000-token "
        "completion allowance with the component-aware estimate",
        abc_fits({"HEAT": ["t"], "WIND_DUST": ["t"]}),
    )

    check(
        "24-hour series: five simultaneous hazards exceed the budget "
        "(fails closed, documented trade-off)",
        not abc_fits({code: ["t"] for code in HAZARD_POLICY_TITLES}),
    )

    # C. Structured incomplete reasons.
    check(
        "Incident shape records unassessed dimensions; warnings separately",
        abc_codes(neutral_result)[:1] == ["RISK_DIMENSIONS_UNASSESSED"]
        and "OFFICIAL_WARNINGS_NOT_VERIFIED" not in abc_codes(neutral_result)
        and neutral_result["official_warning_status"] == "NOT_CHECKED"
        and next(
            r["detail"] for r in neutral_result["incomplete_reasons"]
            if r["code"] == "RISK_DIMENSIONS_UNASSESSED"
        ) == "exposure_score, duration_score, geographic_impact_score",
    )

    # No heat here, so the request fits the budgets and reaches
    # validation, where the evidence-gap reason is recorded.
    uncertain_weather = abc_weather(
        visibility=None, temperature_2m=22, apparent_temperature=22
    )
    uncertain_weather["today_forecast"]["high"] = 24
    for record in uncertain_weather["hourly_forecast_next_24h"]:
        record["visibility"] = None
        record["temperature"] = 22
        record["apparent_temperature"] = 22

    uncertain_result, _, _ = abc_pipeline(
        abc_reply(sources=["9. Official Warning Override"]),
        weather=uncertain_weather,
    )

    check(
        "Evidence gaps are recorded as WEATHER_EVIDENCE_UNCERTAIN",
        "WEATHER_EVIDENCE_UNCERTAIN" in abc_codes(uncertain_result)
        and "WIND_DUST" in next(
            r["detail"] for r in uncertain_result["incomplete_reasons"]
            if r["code"] == "WEATHER_EVIDENCE_UNCERTAIN"
        ),
    )

    def missing_policy():
        raise PolicyUnavailableError("guidelines.md not found")

    no_policy, _, _ = abc_pipeline(
        abc_reply(), load_guidelines=missing_policy
    )
    budget_failure, _, _ = abc_pipeline(abc_reply(), MAX_POLICY_CHARS=5000)

    check(
        "Failures record POLICY_UNAVAILABLE or ASSESSMENT_ERROR",
        abc_codes(no_policy) == ["POLICY_UNAVAILABLE"]
        and abc_codes(budget_failure) == ["ASSESSMENT_ERROR"]
        and next(
            r["detail"] for r in budget_failure["incomplete_reasons"]
            if r["code"] == "ASSESSMENT_ERROR"
        ) == "ValueError",
    )

    duplicate = {}
    add_incomplete_reason(duplicate, "X", "first")
    add_incomplete_reason(duplicate, "X", "second")

    check(
        "Each incomplete reason code is recorded once",
        duplicate["incomplete_reasons"] == [
            {"code": "X", "detail": "first"}
        ],
    )

    check(
        "None of these decisions is COMPLETE or records a warning reason",
        all(
            decision["assessment_status"] == (
                "UNAVAILABLE" if decision is no_policy else "INCOMPLETE"
            )
            and "OFFICIAL_WARNINGS_NOT_VERIFIED" not in abc_codes(decision)
            for decision in (
                neutral_result, uncertain_result, no_policy,
                budget_failure, sent_request,
            )
        ),
    )

    # Dry run prints the reasons and still makes no POST.
    class AbcNoPost:
        def __init__(self):
            self.posts = []

        def post(self, *args, **kwargs):
            self.posts.append(args)

    no_post = AbcNoPost()
    dry_output = io.StringIO()

    def abc_dry_run():
        with contextlib.redirect_stdout(dry_output):
            run_agent_job(dry=True)

    abc_override(
        {
            "HTTP": no_post,
            "run_weather_agent": lambda: (neutral_result, []),
        },
        abc_dry_run,
    )

    check(
        "Dry run prints incomplete reasons and makes no POST",
        "OFFICIAL WARNING: NOT_CHECKED" in dry_output.getvalue()
        and "INCOMPLETE REASON: RISK_DIMENSIONS_UNASSESSED"
        in dry_output.getvalue()
        and no_post.posts == [],
    )

    # ------------------------------------------------------------
    # Fix B compaction, request guards and A/C bypass hardening
    # (self-contained, offline, realistic varying 24-hour data).
    # ------------------------------------------------------------
    def rv_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def rv_weather(**today):
        start = datetime(2026, 11, 8, 13, 0)
        hours = []

        for h in range(24):
            phase = math.cos((h + 1) / 24 * 2 * math.pi)
            hours.append({
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": round(31.0 + 4.3 * phase, 1),
                "apparent_temperature": round(33.6 + 5.1 * phase, 1),
                "humidity": int(58 - 17 * phase),
                "precipitation": [0.0, 0.0, 0.2, 1.4, 0.6, 0.0][h % 6],
                "wind": round(14.6 + 7.9 * phase, 1),
                "gust": round(28.3 + 13.4 * phase, 1),
                "visibility": [24140.0, 22300.0, 18760.0, 24140.0][h % 4],
                "weather_code": [1, 2, 61, 63, 2, 95][h % 6],
                "cape": [0, 120, 840, 1460, 300, 0][h % 6],
            })

        forecast = {"high": 31.4, "low": 24.1, "wind": 15.8,
                    "gust": 27.4, "precipitation": 0.0}
        forecast.update(today)

        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "interval": 900,
                "temperature_2m": 28.3, "relative_humidity_2m": 65,
                "apparent_temperature": 32.4, "precipitation": 0.0,
                "wind_speed_10m": 12.2, "wind_gusts_10m": 20.5,
                "visibility": 24140.0, "weather_code": 1, "cape": 0.0,
            },
            "today_forecast": forecast,
            "previous_days": [
                {"date": f"2026-11-0{d}", "high": 31.2, "low": 23.8,
                 "wind": 18.4, "gust": 33.1, "precipitation": 0.0}
                for d in (5, 6, 7)
            ],
            "hourly_forecast_next_24h": hours,
            "history_summary": {"average_high": 31.1,
                                "average_low": 23.9, "peak_wind": 21.6},
        }

    def rv_heat_wind_weather():
        weather = rv_weather()

        for record in weather["hourly_forecast_next_24h"]:
            record["precipitation"] = 0.0
            record["weather_code"] = 1

        return weather

    def rv_heat_only_weather():
        weather = rv_heat_wind_weather()

        for record in weather["hourly_forecast_next_24h"]:
            record["wind"] = 12.0
            record["gust"] = 25.0

        return weather

    rv_history = [{
        "timestamp": "t" * 60, "city": "c" * 60, "advice": "a" * 500,
        "alert_sent": False, "alert_message": "m" * 500,
    }] * MAX_HISTORY_RECORDS

    # B. Series keeps exact source values, order and nulls.
    rv_source = rv_weather()["hourly_forecast_next_24h"]
    rv_series = hourly_series(rv_source)

    check(
        "Series keeps every source value exactly",
        all(
            rv_series[name][index]
            == (int(record[key]) if digits == 0 else record[key])
            for name, key, digits in HOURLY_SERIES_FIELDS
            for index, record in enumerate(rv_source)
        ),
    )

    check(
        "Series keeps hourly order and timestamps",
        rv_series["start"] == rv_source[0]["time"]
        and rv_series["step_hours"] == 1
        and all(
            len(rv_series[name]) == len(rv_source)
            for name, _, _ in HOURLY_SERIES_FIELDS
        ),
    )

    rv_summary = summarize_weather(rv_weather())

    check(
        "Duplicate min/max summary removed; CAPE (not in series) kept",
        rv_summary["next_24_hours"]
        == {"cape_max": max(r["cape"] for r in rv_source)}
        and "next_24_hours_hourly" in rv_summary,
    )

    # Hazard-relevant column filtering is intentionally not applied:
    # screening values are not policy thresholds, so every column stays.
    check(
        "Every series column is sent regardless of detected hazards",
        set(rv_summary["next_24_hours_hourly"])
        == {"start", "step_hours"}
        | {name for name, _, _ in HOURLY_SERIES_FIELDS},
    )

    # Realistic request sizes under the unchanged budgets.
    def rv_request(hazards, weather):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = load_guidelines()
            required = select_hazard_guidance(policy, hazards)
            required += select_conditional_sections(
                policy, hazards, weather
            )
            sections = search_guidelines(
                policy, "x", k=0, required_sections=required
            )

        captured = []

        def capture(messages, temperature=0.1):
            captured.append(messages)
            raise RuntimeError("captured")

        def assemble():
            try:
                assess_weather(weather, rv_history, policy, sections)
            except RuntimeError:
                pass

        rv_override({"ask_groq": capture}, assemble)
        text = serialized_request(captured[0])

        estimate = production_input_estimate(captured[0], text)

        return (
            len(text) <= MAX_REQUEST_CHARS
            and estimate is not None
            and estimate + MAX_COMPLETION_TOKENS
            <= MAX_ESTIMATED_REQUEST_TOKENS
        )

    rv_four = ("HEAT", "WIND_DUST", "RAIN_FLOOD", "LIGHTNING")

    for label, hazards, expected in (
        ("no hazard", (), True),
        ("heat", ("HEAT",), True),
        ("heat + wind", ("HEAT", "WIND_DUST"), True),
        ("heat + wind + rain", ("HEAT", "WIND_DUST", "RAIN_FLOOD"), False),
        ("four hazards", rv_four, False),
        ("five hazards", tuple(HAZARD_POLICY_TITLES), False),
    ):
        check(
            f"Realistic request {'fits' if expected else 'is blocked'}: "
            f"{label}",
            rv_request({code: ["t"] for code in hazards}, rv_weather())
            is expected,
        )

    # Request guards, each verified on its own (offline transport).
    class RvResponse:
        status_code = 200
        text = ""
        headers = {}

        def json(self):
            return {
                "choices": [{
                    "message": {"content": "{}"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    class RvTransport:
        def __init__(self):
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            return RvResponse()

    rv_overhead = len(serialized_request([{"role": "user", "content": ""}]))

    def rv_message(serialized_length):
        return [{
            "role": "user",
            "content": "x" * (serialized_length - rv_overhead),
        }]

    def rv_send(serialized_length, **limits):
        transport = RvTransport()
        settings = {"GROQ_API_KEY": "gsk_rv_test_key", "HTTP": transport}
        settings.update(limits)
        outcome = {"error": ""}

        def run():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ask_groq(rv_message(serialized_length))
            except (ValueError, RuntimeError) as exc:
                outcome["error"] = str(exc)

        rv_override(settings, run)

        return outcome["error"], transport.posts

    # Character guard alone (token cap lifted).
    error, posts = rv_send(
        MAX_REQUEST_CHARS, MAX_ESTIMATED_REQUEST_TOKENS=10 ** 9
    )

    check(
        "Character guard: request at the limit is sent (control)",
        error == "" and len(posts) == 1,
    )

    error, posts = rv_send(
        MAX_REQUEST_CHARS + 1, MAX_ESTIMATED_REQUEST_TOKENS=10 ** 9
    )

    check(
        "Character guard alone blocks one character over, no POST",
        "character limit" in error and posts == [],
    )

    # Token guard alone (character limit lifted).
    rv_at_cap = int(
        (MAX_ESTIMATED_REQUEST_TOKENS - MAX_COMPLETION_TOKENS) * 2.5
    )

    error, posts = rv_send(rv_at_cap, MAX_REQUEST_CHARS=10 ** 9)

    check(
        "Token guard: request at the cap is sent (control)",
        error == "" and len(posts) == 1
        and estimate_tokens(serialized_request(rv_message(rv_at_cap)))
        + MAX_COMPLETION_TOKENS == MAX_ESTIMATED_REQUEST_TOKENS,
    )

    error, posts = rv_send(rv_at_cap + 2, MAX_REQUEST_CHARS=10 ** 9)

    check(
        "Token guard alone blocks one token over, no POST",
        "token budget" in error and posts == [],
    )

    # Under the real limits the token guard trips below the char limit.
    error, posts = rv_send(rv_at_cap + 2)

    check(
        "Token guard blocks under real limits before the char limit",
        rv_at_cap + 2 < MAX_REQUEST_CHARS
        and "token budget" in error and posts == [],
    )

    # Pipeline: blocked scenarios never reach Groq; a fitting one does.
    rv_reply = json_text({
        "hazard_type": "HEAT",
        "likelihood_score": 2, "likelihood_reason": "lr",
        "severity_score": 2, "severity_reason": "sr",
        "exposure_score": None, "exposure_reason": "er",
        "duration_score": None, "duration_reason": "dr",
        "geographic_impact_score": None,
        "geographic_impact_reason": "gr",
        "alert_recommendation": "MONITOR",
        "advice": "Limit outdoor activity. Official warning status "
                  "was not checked.",
        "reasoning": "r", "recommended_actions": [],
        "sources": ["6.1 Extreme Heat"],
    })

    class RvGroq:
        def __init__(self, content=rv_reply):
            self.content = content
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            response = RvResponse()
            content = self.content
            response.json = lambda: {
                "choices": [{
                    "message": {"content": content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3239, "completion_tokens": 600,
                          "total_tokens": 3839},
            }
            return response

    def rv_pipeline(weather, content=rv_reply, history=None):
        transport = RvGroq(content)
        history = rv_history if history is None else history

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                return run_weather_agent()[0]

        decision = rv_override(
            {
                "HTTP": transport,
                "GROQ_API_KEY": "gsk_rv_test_key",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: weather,
                "tool_get_history": lambda n=3: history,
            },
            run,
        )

        return decision, transport

    four_decision, four_transport = rv_pipeline(rv_weather())
    five_decision, five_transport = rv_pipeline(rv_weather(low=8.0))
    heat_wind_decision, heat_wind_transport = rv_pipeline(
        rv_heat_wind_weather()
    )
    heat_worst_decision, heat_worst_transport = rv_pipeline(
        rv_heat_only_weather()
    )
    fit_decision, fit_transport = rv_pipeline(
        rv_heat_only_weather(), history=[]
    )

    check(
        "Four-hazard request is rejected before Groq (INCOMPLETE)",
        set(four_decision["hazard_screening"]) == set(rv_four)
        and four_transport.posts == []
        and four_decision["assessment_status"] == "INCOMPLETE"
        and four_decision["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and (
            "token budget" in four_decision["reasoning"]
            or "character limit" in four_decision["reasoning"]
        )
        and [r["code"] for r in four_decision["incomplete_reasons"]]
        == ["ASSESSMENT_ERROR"],
    )

    check(
        "Five-hazard request is rejected before Groq (INCOMPLETE)",
        set(five_decision["hazard_screening"])
        == set(HAZARD_POLICY_TITLES)
        and five_transport.posts == []
        and five_decision["assessment_status"] == "INCOMPLETE"
        and five_decision["send_alert"] is False,
    )

    # The legacy estimate rejected both of these before Groq. The
    # component-aware estimate sends them once; the outcome is still an
    # INCOMPLETE assessment that never authorizes an alert.
    check(
        "Heat + wind with worst-case history reaches Groq once and is "
        "assessed (component-aware estimate)",
        set(heat_wind_decision["hazard_screening"])
        == {"HEAT", "WIND_DUST"}
        and len(heat_wind_transport.posts) == 1
        and heat_wind_decision["groq_usage"].get("estimator_method")
        == "component"
        and heat_wind_decision["assessment_status"] == "INCOMPLETE"
        and heat_wind_decision["send_alert"] is False,
    )

    check(
        "Heat-only with worst-case history reaches Groq once and is "
        "assessed (component-aware estimate)",
        set(heat_worst_decision["hazard_screening"]) == {"HEAT"}
        and len(heat_worst_transport.posts) == 1
        and heat_worst_decision["groq_usage"].get("estimator_method")
        == "component"
        and heat_worst_decision["assessment_status"] == "INCOMPLETE"
        and heat_worst_decision["send_alert"] is False,
    )

    check(
        "Heat-only request without history reaches Groq once and is "
        "assessed",
        set(fit_decision["hazard_screening"]) == {"HEAT"}
        and len(fit_transport.posts) == 1
        and fit_decision["groq_usage"].get("prompt_tokens") == 3239
        and fit_decision["assessment_status"] == "INCOMPLETE"
        and "RISK_DIMENSIONS_UNASSESSED"
        in [r["code"] for r in fit_decision["incomplete_reasons"]],
    )

    # A. Bypass hardening.
    def rv_reply_with(**extra):
        body = json.loads(rv_reply)
        body.update(extra)
        return json_text(body)

    spoofed, _ = rv_pipeline(rv_heat_only_weather(), rv_reply_with(
        official_warning_status="VERIFIED",
        advice="No official heat advisory is in effect.",
    ), history=[])
    injected, _ = rv_pipeline(rv_heat_only_weather(), rv_reply_with(
        incomplete_reasons=[{"code": "NONE", "detail": "all clear"}],
    ), history=[])
    hidden, _ = rv_pipeline(rv_heat_only_weather(), rv_reply_with(
        notes={"extra": ["Fine.", "There are no warnings for Dubai."]},
    ), history=[])

    check(
        "Model cannot set the warning status to skip the claim check",
        "program-owned fields: official_warning_status"
        in spoofed["reasoning"]
        and spoofed["official_warning_status"] == "NOT_CHECKED"
        and spoofed["alert_recommendation"] == "UNABLE_TO_ASSESS",
    )

    check(
        "Model cannot inject incomplete reasons",
        "program-owned fields: incomplete_reasons" in injected["reasoning"]
        and all(
            r["code"] != "NONE" for r in injected["incomplete_reasons"]
        ),
    )

    check(
        "Claims in unexpected or nested model fields are rejected",
        [r["code"] for r in hidden["incomplete_reasons"]][0]
        == "UNVERIFIED_WARNING_CLAIM"
        and "claim in a field outside the response schema"
        in hidden["reasoning"]
        and "notes" not in hidden["reasoning"]
        and "no warnings for Dubai" not in json_text(hidden),
    )

    def rv_direct_claim_error(decision):
        try:
            reject_unverified_warning_claims(decision, "NOT_CHECKED")
            return ""
        except UnverifiedWarningClaim as exc:
            return str(exc)

    check(
        "Claim check ignores a model-supplied VERIFIED status",
        "claim in advice" in rv_direct_claim_error({
            "official_warning_status": "VERIFIED",
            "advice": "No official heat advisory is in effect.",
        }),
    )

    check(
        "Claim in a top-level key name is rejected without echoing it",
        "claim in a field outside the response schema"
        in rv_direct_claim_error({
            "advice": "Limit outdoor activity.",
            "No official heat advisory is in effect": True,
        })
        and "heat advisory" not in rv_direct_claim_error({
            "No official heat advisory is in effect": True,
        }),
    )

    check(
        "Claim in a nested key name is rejected",
        "claim in a field outside the response schema"
        in rv_direct_claim_error({
            "notes": {"There are no warnings": ["ok"]},
        }),
    )

    check(
        "Schema field names are still named in claim errors",
        "claim in advice" in rv_direct_claim_error({
            "advice": "No official heat advisory is in effect.",
        })
        and "claim in recommended_actions" in rv_direct_claim_error({
            "recommended_actions": ["There are no warnings."],
        }),
    )

    keyed, _ = rv_pipeline(rv_heat_only_weather(), rv_reply_with(**{
        "NCM has not issued any warning": True,
    }), history=[])

    check(
        "Real path: key-name claim fails closed without echoing it",
        [r["code"] for r in keyed["incomplete_reasons"]][0]
        == "UNVERIFIED_WARNING_CLAIM"
        and keyed["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and keyed["send_alert"] is False
        and "NCM has not issued" not in json_text(keyed),
    )

    check(
        "Response schema constant matches the sent schema and validator",
        len(RESPONSE_FIELDS) == 16
        and len(set(RESPONSE_FIELDS)) == 16
        and list(build_response_schema()["properties"])
        == list(RESPONSE_FIELDS)
        and RESPONSE_FORMAT["json_schema"]["schema"]
        == build_response_schema(),
    )

    for text in (
        "The heat advisory isn't in effect.",
        "Warnings aren't active today.",
        "NCM hasn't issued any warning.",
        "There's no advisory for Dubai.",
        "There are no warnings.",
        "No official heat ad​visory is in ef​fect.",
        "No official warning is in effect.",
    ):
        check(
            f"Claim variant detected: {text!r}",
            find_unverified_warning_claim(text),
        )

    rv_known_codes = {
        "RISK_DIMENSIONS_UNASSESSED",
        "WEATHER_EVIDENCE_UNCERTAIN", "UNVERIFIED_WARNING_CLAIM",
        "POLICY_UNAVAILABLE", "ASSESSMENT_ERROR",
    }

    check(
        "Every recorded reason code is program-defined",
        all(
            reason["code"] in rv_known_codes
            for decision in (
                four_decision, five_decision, fit_decision, spoofed,
                injected, hidden,
            )
            for reason in decision["incomplete_reasons"]
        ),
    )

    check(
        "Every rejected or blocked assessment stays fail-closed",
        all(
            decision["assessment_status"] == "INCOMPLETE"
            and decision["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and decision["risk_band"] == "UNASSESSED"
            and decision["send_alert"] is False
            and decision["notification_status"] == "NOT_SENT"
            for decision in (
                four_decision, five_decision, spoofed, injected, hidden,
            )
        ),
    )

    # ------------------------------------------------------------
    # Strict structured outputs and schema-failure diagnostics
    # (self-contained, offline).
    # ------------------------------------------------------------
    def ss_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def ss_copy(value):
        return json.loads(json_text(value))

    ss_format = RESPONSE_FORMAT
    ss_schema = ss_format["json_schema"]["schema"]

    check(
        "Response format is strict json_schema with a name",
        ss_format["type"] == "json_schema"
        and ss_format["json_schema"]["strict"] is True
        and ss_format["json_schema"]["name"] == RESPONSE_SCHEMA_NAME
        and set(ss_format["json_schema"]) == {"name", "strict", "schema"},
    )

    check(
        "Schema requires exactly the 18 RESPONSE_FIELDS, in order",
        ss_schema["type"] == "object"
        and list(ss_schema["properties"]) == list(RESPONSE_FIELDS)
        and ss_schema["required"] == list(RESPONSE_FIELDS)
        and ss_schema["additionalProperties"] is False,
    )

    check(
        "Scores are nullable integers (documented union with null)",
        len(SCORE_FIELDS) == 5
        and all(
            ss_schema["properties"][field] == {"type": ["integer", "null"]}
            for field in SCORE_FIELDS
        ),
    )

    check(
        "alert_recommendation is limited to the permitted values",
        ss_schema["properties"]["alert_recommendation"] == {
            "type": "string", "enum": sorted(VALID_RECOMMENDATIONS),
        },
    )

    check(
        "List fields are arrays of strings; all others are strings",
        all(
            ss_schema["properties"][field]
            == {"type": "array", "items": {"type": "string"}}
            for field in LIST_FIELDS
        )
        and all(
            ss_schema["properties"][field] == {"type": "string"}
            for field in RESPONSE_FIELDS
            if field not in SCORE_FIELDS
            and field not in LIST_FIELDS
            and field != "alert_recommendation"
        ),
    )

    def ss_keywords(node):
        found = set()

        if isinstance(node, dict):
            for key, value in node.items():
                if key != "properties":
                    found.add(key)
                    found |= ss_keywords(value)
                else:
                    for item in value.values():
                        found |= ss_keywords(item)
        elif isinstance(node, list):
            for item in node:
                found |= ss_keywords(item)

        return found

    check(
        "Schema uses only documented strict-mode keywords",
        ss_keywords(ss_schema) <= {
            "type", "properties", "required", "additionalProperties",
            "items", "enum",
        },
    )

    # The documented strict-mode rules (Groq Structured Outputs, "Schema
    # Requirements"), applied to any schema so that valid and invalid
    # structures can both be exercised.
    ss_types = {
        "string", "number", "boolean", "integer", "object", "array", "null",
    }
    ss_allowed = {
        "type", "properties", "required", "additionalProperties", "items",
        "enum", "description", "anyOf", "$defs", "$ref",
    }

    def ss_problems(node, path="schema"):
        if not isinstance(node, dict):
            return [f"{path}: not an object"]

        problems = [
            f"{path}: keyword {key} is outside the documented subset"
            for key in node if key not in ss_allowed
        ]
        declared = node.get("type")
        kinds = declared if isinstance(declared, list) else [declared]

        if "type" in node and not (
            kinds and all(kind in ss_types for kind in kinds)
        ):
            problems.append(f"{path}: undocumented type")

        if "object" in kinds or "properties" in node:
            properties = node.get("properties", {})

            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties must be false")

            if set(node.get("required", [])) != set(properties):
                problems.append(f"{path}: every property must be required")

            for name, child in properties.items():
                problems += ss_problems(child, f"{path}.{name}")

        if "array" in kinds:
            if "items" not in node:
                problems.append(f"{path}: array needs items")
            else:
                problems += ss_problems(node["items"], f"{path}[]")

        if "enum" in node and not (
            isinstance(node["enum"], list) and node["enum"]
        ):
            problems.append(f"{path}: enum must be a non-empty list")

        for index, branch in enumerate(node.get("anyOf", [])):
            problems += ss_problems(branch, f"{path}|{index}")

        return problems

    ss_good = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nickname": {"type": ["string", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "level": {"type": "string", "enum": ["low", "high"]},
            "nested": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "nickname", "tags", "level", "nested"],
        "additionalProperties": False,
    }

    check(
        "Documented strict-mode rules accept valid structures and the "
        "response schema",
        ss_problems(ss_good) == [] and ss_problems(ss_schema) == [],
    )

    def ss_variant(change):
        schema = json.loads(json.dumps(ss_schema))
        change(schema)
        return ss_problems(schema)

    def ss_drop_required(schema):
        schema["required"].remove("advice")

    def ss_open_object(schema):
        schema["additionalProperties"] = True

    def ss_omit_closed(schema):
        del schema["additionalProperties"]

    def ss_add_bound(schema):
        schema["properties"]["likelihood_score"]["minimum"] = 1

    def ss_untyped_array(schema):
        del schema["properties"]["sources"]["items"]

    def ss_bad_type(schema):
        schema["properties"]["advice"]["type"] = "text"

    def ss_empty_enum(schema):
        schema["properties"]["alert_recommendation"]["enum"] = []

    def ss_optional_field(schema):
        schema["properties"]["extra"] = {"type": "string"}

    def ss_nested_open(schema):
        schema["properties"]["advice"] = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }

    check(
        "Documented strict-mode rules reject each invalid structure",
        all(
            ss_variant(change) != []
            for change in (
                ss_drop_required, ss_open_object, ss_omit_closed,
                ss_add_bound, ss_untyped_array, ss_bad_type,
                ss_empty_enum, ss_optional_field, ss_nested_open,
            )
        ),
    )

    # Schema types and the validator's type checks agree, field by field.
    ss_samples = {
        "string": "x", "integer": 3, "null": None, "array": ["x"],
        "object": {"x": "y"},
    }

    def ss_declared(field):
        declared = ss_schema["properties"][field]["type"]
        return declared if isinstance(declared, list) else [declared]

    check(
        "Every field accepts its schema type and rejects the others",
        all(
            schema_value_matches(
                field,
                sorted(VALID_RECOMMENDATIONS)[0]
                if field == "alert_recommendation" and kind == "string"
                else ss_samples[kind],
            )
            == (kind in ss_declared(field))
            for field in RESPONSE_FIELDS
            for kind in ss_samples
        ),
    )

    # The removed type paragraph is fully covered by the schema; every
    # semantic instruction remains in the prompt.
    ss_prompt = " ".join(SYSTEM_PROMPT.split())

    check(
        "Prompt no longer duplicates the schema type paragraph",
        "Return exactly these 18 keys" not in ss_prompt
        and "Plain-text strings" not in ss_prompt
        and "Arrays of strings" not in ss_prompt,
    )

    for phrase in (
        "Follow the supplied policy sections",
        "Never invent weather measurements, official warnings, "
        "regulatory requirements, thresholds or geographic information",
        "Assess only the supplied city and evidence",
        "Identify hazards from actual measurements and forecasts",
        "using the scale definitions in the supplied policy",
        "separate evidence-based rationale for each",
        "Never treat missing evidence as a low score",
        "Official warning status is NOT_CHECKED",
        "Free text must not use warning, warnings, advisory or "
        "advisories, or claim their existence, absence, availability "
        "or status",
        "Open-Meteo is not an official UAE warning authority",
        "Distinguish current observations, forecasts and historical "
        "context",
        "Do not assume all residents or workers have equal exposure",
        "Previous alerts must not suppress a new or different hazard",
        "Do not recommend an alert solely because a value is unusual",
        "Use only recommendation values defined in the supplied policy",
        "The assessment is never a completed safety clearance",
        "In sources, cite exact policy section titles",
        "Return a single JSON object only",
        "Each score must be an integer from 1 to 5 when evidence "
        "supports it",
        "return null for that score, never a guessed value, and explain "
        "why in its rationale",
    ):
        check(
            f"Prompt keeps instruction: {phrase[:60]}",
            phrase in ss_prompt,
        )

    # Independent reference checker for the documented subset.
    def ss_type_ok(value, type_name):
        return {
            "string": isinstance(value, str),
            "integer": type(value) is int,
            "null": value is None,
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }[type_name]

    def ss_conforms(value, schema):
        types = schema.get("type")
        types = types if isinstance(types, list) else [types]

        if not any(ss_type_ok(value, t) for t in types):
            return False

        if "enum" in schema and value not in schema["enum"]:
            return False

        if isinstance(value, list) and "items" in schema:
            return all(ss_conforms(item, schema["items"]) for item in value)

        if isinstance(value, dict) and "properties" in schema:
            if schema.get("additionalProperties") is False and (
                set(value) - set(schema["properties"])
            ):
                return False

            if any(field not in value for field in schema["required"]):
                return False

            return all(
                ss_conforms(value[field], schema["properties"][field])
                for field in schema["properties"]
                if field in value
            )

        return True

    def ss_response(**changes):
        body = {
            "hazard_type": "NONE",
            "likelihood_score": 1, "likelihood_reason": "lr",
            "severity_score": 1, "severity_reason": "sr",
            "exposure_score": None, "exposure_reason": "er",
            "duration_score": None, "duration_reason": "dr",
            "geographic_impact_score": None,
            "geographic_impact_reason": "gr",
            "alert_recommendation": "UNABLE_TO_ASSESS",
            "advice": "Official warning status was not checked.",
            "reasoning": "Three dimensions lack evidence.",
            "recommended_actions": [],
            "sources": ["16. Final Decision Rule"],
        }
        body.update(changes)
        return body

    ss_sections = [{"section": "16. Final Decision Rule", "text": "x"}]

    def ss_validation_error(body):
        try:
            validate_decision(ss_copy(body), ss_sections)
            return ""
        except ValueError as exc:
            return str(exc)

    ss_complete = ss_response()

    check(
        "Complete fixture conforms to the schema and validates",
        ss_conforms(ss_complete, ss_schema)
        and ss_validation_error(ss_complete) == ""
        and response_shape_diagnostics(ss_complete)["missing_required"]
        == []
        and response_shape_diagnostics(ss_complete)["wrong_type_fields"]
        == [],
    )

    for field in RESPONSE_FIELDS:
        omitted = ss_response()
        del omitted[field]
        diagnostics = response_shape_diagnostics(omitted)

        check(
            f"Omitted field is non-conforming, rejected and diagnosed: "
            f"{field}",
            not ss_conforms(omitted, ss_schema)
            and ss_validation_error(omitted)
            == "Missing assessment fields: " + field
            and diagnostics["missing_required"] == [field]
            and diagnostics["missing_required_count"] == 1
            and diagnostics["unexpected_key_count"] == 0,
        )

    for label, field, value in (
        ("advice as array", "advice", ["a"]),
        ("reasoning as object", "reasoning", {"text": "x"}),
        ("score as string", "severity_score", "3"),
        ("score as float", "likelihood_score", 2.5),
        ("score as boolean", "exposure_score", True),
        ("recommendation outside enum", "alert_recommendation", "ALERT"),
        ("list as string", "recommended_actions", "none"),
        ("list with a number", "sources", [1]),
    ):
        body = ss_response(**{field: value})
        wrong = response_shape_diagnostics(body)["wrong_type_fields"]

        check(
            f"Wrong type is non-conforming, rejected and diagnosed: "
            f"{label}",
            not ss_conforms(body, ss_schema)
            and ss_validation_error(body) != ""
            and [item["field"] for item in wrong] == [field],
        )

    extra = ss_response(notes="x")
    nested = ss_response(notes={"reasoning": "x"})

    check(
        "Extra and nested keys are non-conforming",
        not ss_conforms(extra, ss_schema)
        and not ss_conforms(nested, ss_schema),
    )

    check(
        "Null scores conform and are treated as unassessed",
        ss_conforms(ss_complete, ss_schema)
        and validate_decision(ss_copy(ss_complete), ss_sections)[
            "risk_band"
        ] == "UNASSESSED",
    )

    out_of_range = ss_response(likelihood_score=7)

    check(
        "Schema-conforming out-of-range score is still rejected locally",
        ss_conforms(out_of_range, ss_schema)
        and "Invalid risk dimension value" in ss_validation_error(
            out_of_range
        ),
    )

    # Diagnostics distinguish omission, renaming and nesting, and never
    # contain model-chosen key names or values.
    def ss_without_reasoning(**extra):
        body = ss_response(**extra)
        del body["reasoning"]
        return body

    ss_cases = {
        "omitted": ss_without_reasoning(),
        "case": ss_without_reasoning(Reasoning="SECRET-VALUE-1"),
        "prefix": ss_without_reasoning(
            assessment_reasoning="SECRET-VALUE-2"
        ),
        "nested": ss_without_reasoning(
            summary_block={"reasoning": "SECRET-VALUE-3"}
        ),
    }
    ss_diag = {
        name: response_shape_diagnostics(body)
        for name, body in ss_cases.items()
    }

    def ss_flags(name):
        return ss_diag[name]["missing_field_variants"]["reasoning"]

    check(
        "Diagnostics: plain omission shows no variant",
        ss_diag["omitted"]["missing_required"] == ["reasoning"]
        and ss_diag["omitted"]["unexpected_key_count"] == 0
        and not any(ss_flags("omitted").values()),
    )

    check(
        "Diagnostics: case variant detected",
        ss_flags("case")["case_or_separator_variant"]
        and not ss_flags("case")["found_nested"]
        and ss_diag["case"]["unexpected_value_types"] == {"string": 1},
    )

    check(
        "Diagnostics: prefix variant detected",
        ss_flags("prefix")["prefix_or_suffix_variant"]
        and not ss_flags("prefix")["case_or_separator_variant"],
    )

    check(
        "Diagnostics: nested key detected",
        ss_flags("nested")["found_nested"]
        and ss_diag["nested"]["unexpected_value_types"] == {"object": 1},
    )

    ss_diag_text = json_text(ss_diag)

    check(
        "Diagnostics never contain model key names or values",
        not any(
            marker in ss_diag_text
            for marker in (
                "SECRET-VALUE", "Reasoning", "assessment_reasoning",
                "summary_block",
            )
        ),
    )

    check(
        "Diagnostics name program-owned keys and non-object responses",
        response_shape_diagnostics(
            ss_response(official_warning_status="VERIFIED")
        )["program_owned_keys"] == ["official_warning_status"]
        and response_shape_diagnostics([1, 2])
        == {"response_type": "array"},
    )

    check(
        "response_diagnostics is a program-owned field",
        "response_diagnostics" in PROGRAM_OWNED_FIELDS,
    )

    # Payload and guards: the schema is sent and counted.
    class SsResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content):
            self.content = content

        def json(self):
            return {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3898, "completion_tokens": 606,
                          "total_tokens": 4504,
                          "completion_tokens_details": {
                              "reasoning_tokens": 105}},
            }

    class SsTransport:
        def __init__(self, content="{}"):
            self.content = content
            self.payloads = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.payloads.append(json)
            return SsResponse(self.content)

    def ss_send(messages, **limits):
        transport = SsTransport()
        settings = {"GROQ_API_KEY": "gsk_ss_test_key", "HTTP": transport}
        settings.update(limits)
        outcome = {"error": ""}

        def run():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ask_groq(messages)
            except (ValueError, RuntimeError) as exc:
                outcome["error"] = str(exc)

        ss_override(settings, run)

        return outcome["error"], transport.payloads

    _, ss_payloads = ss_send([{"role": "user", "content": "hi"}])

    check(
        "Payload sends the strict response format, not json_object",
        len(ss_payloads) == 1
        and ss_payloads[0]["response_format"] == RESPONSE_FORMAT
        and "json_object" not in json_text(ss_payloads[0])
        and ss_payloads[0]["max_completion_tokens"]
        == MAX_COMPLETION_TOKENS
        and ss_payloads[0]["reasoning_effort"] == "low",
    )

    def ss_message_measuring(guard_length):
        """Message whose guard-measured text has exactly this length."""
        base = len(serialized_request([{"role": "user", "content": ""}]))
        return [{"role": "user", "content": "x" * (guard_length - base)}]

    ss_schema_chars = (
        len(serialized_request([{"role": "user", "content": ""}]))
        - len(json_text([{"role": "user", "content": ""}]))
    )

    over_by_schema = ss_message_measuring(MAX_REQUEST_CHARS + 1)
    error, payloads = ss_send(
        over_by_schema, MAX_ESTIMATED_REQUEST_TOKENS=10 ** 9
    )

    check(
        "Character guard counts the schema (fits without it, blocked "
        "with it, no Groq call)",
        ss_schema_chars > 1000
        and len(json_text(over_by_schema)) <= MAX_REQUEST_CHARS
        and "character limit" in error
        and payloads == [],
    )

    ss_token_cap_chars = int(
        (MAX_ESTIMATED_REQUEST_TOKENS - MAX_COMPLETION_TOKENS) * 2.5
    )
    over_tokens = ss_message_measuring(ss_token_cap_chars + 2)
    error, payloads = ss_send(over_tokens, MAX_REQUEST_CHARS=10 ** 9)

    check(
        "Token guard counts the schema (fits without it, blocked with "
        "it, no Groq call)",
        estimate_tokens(json_text(over_tokens)) + MAX_COMPLETION_TOKENS
        <= MAX_ESTIMATED_REQUEST_TOKENS
        and "token budget" in error
        and payloads == [],
    )

    at_limit = ss_message_measuring(MAX_REQUEST_CHARS)
    error, payloads = ss_send(at_limit, MAX_ESTIMATED_REQUEST_TOKENS=10 ** 9)

    check(
        "Request at the character limit including the schema is sent",
        error == "" and len(payloads) == 1,
    )

    # Real pipeline: the incident response fails closed with
    # diagnostics; a complete response records empty diagnostics.
    def ss_weather():
        start = datetime(2026, 11, 8, 13, 0)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 22,
                "apparent_temperature": 22, "precipitation": 0,
                "wind_speed_10m": 5, "wind_gusts_10m": 10,
                "visibility": 24140, "weather_code": 1, "cape": 0,
            },
            "today_forecast": {"high": 24, "low": 18, "wind": 8,
                               "gust": 15, "precipitation": 0},
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": 22, "apparent_temperature": 22,
                "humidity": 55, "precipitation": 0, "wind": 6,
                "gust": 12, "visibility": 24140, "weather_code": 1,
                "cape": 0,
            } for h in range(24)],
            "history_summary": {},
        }

    def ss_pipeline(content):
        transport = SsTransport(content)
        printed = io.StringIO()

        def run():
            with contextlib.redirect_stdout(printed):
                return run_weather_agent()[0]

        decision = ss_override(
            {
                "HTTP": transport,
                "GROQ_API_KEY": "gsk_ss_test_key",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: ss_weather(),
                "tool_get_history": lambda n=3: [],
            },
            run,
        )

        return decision, printed.getvalue(), transport

    incident, incident_output, incident_transport = ss_pipeline(
        json_text(ss_without_reasoning(assessment_reasoning="SECRET-VALUE-9"))
    )

    check(
        "Incident response fails closed with diagnostics recorded",
        len(incident_transport.payloads) == 1
        and incident["assessment_status"] == "INCOMPLETE"
        and incident["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and incident["send_alert"] is False
        and "Missing assessment fields: reasoning" in incident["reasoning"]
        and incident["response_diagnostics"]["missing_required"]
        == ["reasoning"]
        and incident["response_diagnostics"]["missing_field_variants"][
            "reasoning"
        ]["prefix_or_suffix_variant"] is True
        and incident["groq_usage"].get("reasoning_tokens") == 105,
    )

    check(
        "Diagnostics are printed without model text or key",
        "RESPONSE SHAPE DIAGNOSTICS" in incident_output
        and "SECRET-VALUE" not in incident_output + json_text(incident)
        and "assessment_reasoning" not in incident_output
        and "gsk_ss_test_key" not in incident_output,
    )

    complete_run, complete_output, _ = ss_pipeline(json_text(ss_complete))

    check(
        "Complete response passes with empty diagnostics",
        complete_run["response_diagnostics"] == {}
        and "RESPONSE SHAPE DIAGNOSTICS" not in complete_output
        and complete_run["assessment_status"] == "INCOMPLETE"
        and "RISK_DIMENSIONS_UNASSESSED"
        in [r["code"] for r in complete_run["incomplete_reasons"]],
    )

    check(
        "Claim guard still applies to schema-valid responses",
        "UNVERIFIED_WARNING_CLAIM" in [
            r["code"] for r in ss_pipeline(json_text(ss_response(
                advice="No official heat advisory is in effect."
            )))[0]["incomplete_reasons"]
        ],
    )

    # ------------------------------------------------------------
    # Completion allowance (1,000 tokens) against the unchanged
    # 8,000-token and 19,000-character guards (offline).
    # ------------------------------------------------------------
    class CtResponse:
        status_code = 200
        text = ""
        headers = {}

        def json(self):
            return {
                "choices": [{
                    "message": {"content": "{}"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    class CtTransport:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.payloads.append(json)
            return CtResponse()

    def ct_message(guard_length):
        base = len(serialized_request([{"role": "user", "content": ""}]))
        return [{"role": "user", "content": "x" * (guard_length - base)}]

    def ct_send(guard_length):
        transport = CtTransport()
        saved = {name: globals()[name] for name in ("HTTP", "GROQ_API_KEY")}
        globals().update({"HTTP": transport, "GROQ_API_KEY": "gsk_ct_key"})
        error = ""

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ask_groq(ct_message(guard_length))
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        finally:
            globals().update(saved)

        return error, transport.payloads

    check(
        "Completion allowance is exactly 1,000 and never above it",
        MAX_COMPLETION_TOKENS == 1000
        and MAX_COMPLETION_TOKENS <= 1000
        and MAX_ESTIMATED_REQUEST_TOKENS == 8000
        and MAX_REQUEST_CHARS == 19000
        and MAX_POLICY_CHARS == 13500,
    )

    # The failing dry run: 17,225 request characters = 6,890 estimated
    # input tokens; 6,890 + 1,000 = 7,890 is within the 8,000 total.
    error, payloads = ct_send(17225)

    check(
        "Failing request estimate (6,890 + 1,000 = 7,890) passes the "
        "guards and is sent",
        estimate_tokens(serialized_request(ct_message(17225))) == 6890
        and 6890 + MAX_COMPLETION_TOKENS == 7890
        and error == ""
        and len(payloads) == 1
        and payloads[0]["max_completion_tokens"] == 1000,
    )

    # Exactly 8,000 (7,000 input + 1,000) is allowed; 8,001 is not.
    error, payloads = ct_send(17500)

    check(
        "Request at exactly 8,000 estimated tokens is sent",
        estimate_tokens(serialized_request(ct_message(17500)))
        + MAX_COMPLETION_TOKENS == 8000
        and error == "" and len(payloads) == 1,
    )

    error, payloads = ct_send(17502)

    check(
        "Request over 8,000 estimated tokens is rejected before Groq",
        estimate_tokens(serialized_request(ct_message(17502)))
        + MAX_COMPLETION_TOKENS == 8001
        and "token budget" in error
        and payloads == [],
    )

    error, payloads = ct_send(MAX_REQUEST_CHARS + 1)

    check(
        "Request over the character limit is still rejected before Groq",
        "character limit" in error and payloads == [],
    )

    # ------------------------------------------------------------
    # Groq HTTP error bodies are never propagated (self-contained,
    # offline). Each synthetic body carries unique marker strings.
    # ------------------------------------------------------------
    def he_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    class HeResponse:
        def __init__(self, status_code, text, headers=None):
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

        def json(self):
            return json.loads(self.text)

        def raise_for_status(self):
            return None

    class HeTransport:
        def __init__(self, response):
            self.response = response
            self.payloads = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.payloads.append(json)
            return self.response

    def he_weather():
        start = datetime(2026, 11, 8, 13, 0)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 22,
                "apparent_temperature": 22, "precipitation": 0,
                "wind_speed_10m": 5, "wind_gusts_10m": 10,
                "visibility": 24140, "weather_code": 1, "cape": 0,
            },
            "today_forecast": {"high": 24, "low": 18, "wind": 8,
                               "gust": 15, "precipitation": 0},
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": 22, "apparent_temperature": 22,
                "humidity": 55, "precipitation": 0, "wind": 6,
                "gust": 12, "visibility": 24140, "weather_code": 1,
                "cape": 0,
            } for h in range(24)],
            "history_summary": {},
        }

    def he_run(response, history=()):
        """Real pipeline + dry run + live-mode payload, all offline."""
        groq = HeTransport(response)
        pipeline_log = io.StringIO()

        def assess():
            with contextlib.redirect_stdout(pipeline_log):
                return run_weather_agent()

        decision, used = he_override(
            {
                "HTTP": groq,
                "GROQ_API_KEY": "gsk_he_test_key",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: he_weather(),
                "tool_get_history": lambda n=3: list(history),
            },
            assess,
        )

        dry_log = io.StringIO()
        sheets = HeTransport(HeResponse(200, "OK"))

        def report(dry):
            target = dry_log if dry else io.StringIO()

            with contextlib.redirect_stdout(target):
                run_agent_job(dry=dry)

        common = {
            "run_weather_agent": lambda: (decision, used),
            "HTTP": sheets,
            "SHEET_URL": "https://sheet.invalid/exec",
            "require_configuration": lambda require_sheet=False: None,
        }
        he_override(common, lambda: report(True))
        he_override(common, lambda: report(False))

        return {
            "decision": decision,
            "groq": groq,
            "pipeline_log": pipeline_log.getvalue(),
            "dry_log": dry_log.getvalue(),
            "sheets_payloads": sheets.payloads,
        }

    he_bodies = {
        "failed_generation": (400, json.dumps({"error": {
            "message": "max completion tokens reached HE-MARK-MSG1",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": "{\"advice\": \"HE-MARK-FG no official "
                                 "advisory is in effect\"",
        }}), {}),
        "error.message": (400, json.dumps({"error": {
            "message": "HE-MARK-MSG2 provider supplied text",
            "type": "invalid_request_error",
        }}), {}),
        "json_validate_failed, other message": (400, json.dumps({"error": {
            "message": "HE-MARK-MSG3 Generated JSON does not match",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": "HE-MARK-FG3",
        }}), {}),
        "unexpected nested fields": (500, json.dumps({
            "error": {
                "type": "server_error",
                "details": {"trace": ["HE-MARK-NEST"]},
                "param": "HE-MARK-PARAM",
            },
            "extra": {"deep": [{"x": "HE-MARK-TOP"}]},
        }), {}),
        "malformed JSON": (
            502, '{"error": {"message": "HE-MARK-BAD", "code": ', {}
        ),
        "plain text": (503, "HE-MARK-PLAIN service unavailable", {}),
        "unsafe identifiers and header": (429, json.dumps({"error": {
            "code": "HE-MARK-CODE not an identifier",
            "type": "HE_MARK_" + "t" * 80,
        }}), {"retry-after": "HE-MARK-RETRY"}),
    }

    he_results = {
        label: he_run(HeResponse(status, text, headers))
        for label, (status, text, headers) in he_bodies.items()
    }

    for label, result in he_results.items():
        decision = result["decision"]
        surfaces = {
            "logs": result["pipeline_log"] + result["dry_log"],
            "reasoning": decision["reasoning"],
            "limitations": json_text(decision["limitations"]),
            "payloads": json_text(result["sheets_payloads"]),
        }
        leaked = [
            surface for surface, text in surfaces.items()
            if "HE-MARK" in text or "HE_MARK" in text
        ]

        check(
            f"Provider error body never leaks ({label})",
            leaked == []
            and len(result["sheets_payloads"]) == 1
            and "OUTPUT PAYLOAD" in result["dry_log"],
        )

        check(
            f"Provider error fails closed ({label})",
            len(result["groq"].payloads) == 1
            and decision["assessment_status"] == "INCOMPLETE"
            and decision["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and decision["risk_band"] == "UNASSESSED"
            and decision["send_alert"] is False
            and decision["notification_status"] == "NOT_SENT"
            and [r["code"] for r in decision["incomplete_reasons"]]
            == ["ASSESSMENT_ERROR"]
            and "gsk_he_test_key" not in json_text(result["sheets_payloads"])
            + result["pipeline_log"] + result["dry_log"],
        )

    known = he_results["failed_generation"]["decision"]["reasoning"]

    check(
        "Known provider code stays diagnosable with a fixed description",
        "HTTP 400" in known
        and "code=json_validate_failed" in known
        and "type=invalid_request_error" in known
        and GROQ_ERROR_REASON_DESCRIPTIONS["completion_limit_reached"]
        in known,
    )

    jr_other = he_results["json_validate_failed, other message"][
        "decision"
    ]["reasoning"]

    check(
        "A json_validate_failed error without the known message keeps the "
        "generic description",
        GROQ_ERROR_CODE_DESCRIPTIONS["json_validate_failed"] in jr_other
        and GROQ_ERROR_REASON_DESCRIPTIONS["completion_limit_reached"]
        not in jr_other,
    )

    check(
        "Sanitised description is identical in reasoning and limitations",
        all(
            result["decision"]["limitations"][0]
            == result["decision"]["reasoning"]
            for result in he_results.values()
        ),
    )

    def he_describe(status, body, headers=None):
        return describe_groq_http_error(
            status, HeResponse(status, body, headers)
        )

    unknown = he_describe(400, json.dumps({"error": {
        "code": "some_new_code", "type": "invalid_request_error",
        "message": "HE-MARK-UNKNOWN",
    }}))

    check(
        "Unrecognised but well-formed code is named with the generic text",
        "code=some_new_code" in unknown
        and GROQ_GENERIC_ERROR_DESCRIPTION in unknown
        and "HE-MARK" not in unknown,
    )

    # json_validate_failed: the provider's stated reason is reduced to a
    # closed label and never copied, so a truncated generation can be
    # told apart from schema non-conformance without exposing text.
    def jr_body(message, code="json_validate_failed"):
        return json.dumps({"error": {
            "message": message, "type": "invalid_request_error",
            "code": code,
        }})

    jr_truncated = GROQ_ERROR_REASON_DESCRIPTIONS["completion_limit_reached"]
    jr_generic = GROQ_ERROR_CODE_DESCRIPTIONS["json_validate_failed"]
    jr_observed = (
        "max completion tokens reached before generating a valid document"
    )
    jr_text = he_describe(400, jr_body(jr_observed))

    check(
        "Observed truncation message is reported as a completion-limit "
        "reason",
        "HTTP 400" in jr_text
        and "code=json_validate_failed" in jr_text
        and "type=invalid_request_error" in jr_text
        and jr_text.endswith(jr_truncated)
        and jr_generic not in jr_text,
    )

    check(
        "The completion-limit reason is matched case-insensitively",
        he_describe(400, jr_body(jr_observed.upper())).endswith(jr_truncated),
    )

    jr_other_messages = (
        "Generated JSON does not match the expected schema. Please adjust "
        "your prompt.",
        "HE-MARK-JR provider supplied text",
        "",
        "x" * 400 + " max completion tokens reached",
        None,
        7,
        ["max completion tokens reached"],
    )

    check(
        "Any other json_validate_failed message keeps the generic "
        "description",
        all(
            he_describe(400, jr_body(message)).endswith(jr_generic)
            and jr_truncated not in he_describe(400, jr_body(message))
            for message in jr_other_messages
        ),
    )

    check(
        "The completion-limit reason is not used for other error codes",
        all(
            jr_truncated not in he_describe(400, jr_body(jr_observed, code))
            for code in ("some_new_code", None, "rate_limit_exceeded")
        ),
    )

    jr_marked = he_describe(
        400, jr_body("max completion tokens reached HE-MARK-JR suffix")
    )

    check(
        "The provider reason never copies provider text",
        "HE-MARK" not in jr_marked and jr_marked.endswith(jr_truncated),
    )

    check(
        "provider_error_reason returns only a fixed label or None",
        provider_error_reason(HeResponse(400, jr_body(jr_observed)))
        == "completion_limit_reached"
        and all(
            provider_error_reason(HeResponse(400, body)) is None
            for body in (
                jr_body("HE-MARK-JR"),
                "not json",
                "[]",
                "null",
                "",
                json.dumps({"error": "max completion tokens reached"}),
                json.dumps({"error": {"message": {
                    "x": "max completion tokens reached"
                }}}),
            )
        )
        and provider_error_reason(object()) is None
        and set(GROQ_ERROR_REASON_DESCRIPTIONS) == {"completion_limit_reached"},
    )

    check(
        "Malformed, plain-text and unsafe values are withheld",
        all(
            "code=unavailable, type=unavailable" in result["decision"][
                "reasoning"
            ]
            for label, result in he_results.items()
            if label in ("malformed JSON", "plain text",
                         "unsafe identifiers and header")
        )
        and "retry-after" not in he_results[
            "unsafe identifiers and header"
        ]["decision"]["reasoning"],
    )

    check(
        "Numeric retry-after and documented statuses remain",
        "retry-after 12s" in he_describe(429, "{}", {"retry-after": "12"})
        and "rate or token limit" in he_describe(429, "{}")
        and "request too large" in he_describe(413, "{}")
        and "provider unavailable" in he_describe(503, "not json")
        and "(request failed)" in he_describe(418, "{}"),
    )

    check(
        "Error description never reads the raw response text",
        "HE-MARK" not in he_describe(
            400, json.dumps({"error": {"type": "invalid_request_error"}})
            + "HE-MARK-TRAILING"
        ),
    )

    # History-dependent coverage: a request with worst-case history
    # still fits (no hazards) and completes a real round trip.
    he_history = [{
        "timestamp": "t" * 60, "city": "c" * 60, "advice": "a" * 500,
        "alert_sent": False, "alert_message": "m" * 500,
    }] * MAX_HISTORY_RECORDS
    he_reply = json_text({
        "hazard_type": "NONE",
        "likelihood_score": 1, "likelihood_reason": "lr",
        "severity_score": 1, "severity_reason": "sr",
        "exposure_score": None, "exposure_reason": "er",
        "duration_score": None, "duration_reason": "dr",
        "geographic_impact_score": None,
        "geographic_impact_reason": "gr",
        "alert_recommendation": "UNABLE_TO_ASSESS",
        "advice": "Official warning status was not checked.",
        "reasoning": "r", "recommended_actions": [],
        "sources": ["16. Final Decision Rule"],
    })

    class HeOk(HeResponse):
        def json(self):
            return {
                "choices": [{
                    "message": {"content": he_reply},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    with_history = he_run(HeOk(200, ""), history=he_history)
    sent_prompt = json_text(with_history["groq"].payloads[0]["messages"]) \
        if with_history["groq"].payloads else ""

    check(
        "Worst-case history request reaches Groq and is assessed",
        len(with_history["groq"].payloads) == 1
        and with_history["decision"]["history_status"] == "AVAILABLE"
        and "recent_agent_history" in sent_prompt
        and "a" * 120 in sent_prompt
        and with_history["decision"]["assessment_status"] == "INCOMPLETE"
        and "RISK_DIMENSIONS_UNASSESSED"
        in [r["code"] for r in with_history["decision"]["incomplete_reasons"]],
    )

    # ------------------------------------------------------------
    # Realistic-history heat assessment and history-error
    # sanitisation (self-contained, offline).
    # ------------------------------------------------------------
    def fz_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def fz_heat_weather():
        start = datetime(2026, 11, 8, 13, 0)
        hours = []

        for h in range(24):
            phase = math.cos((h + 1) / 24 * 2 * math.pi)
            hours.append({
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": round(31.0 + 4.0 * phase, 1),
                "apparent_temperature": round(33.5 + 4.5 * phase, 1),
                "humidity": int(60 - 10 * phase),
                "precipitation": 0.0,
                "wind": round(12.0 + 5.0 * phase, 1),
                "gust": round(24.0 + 9.0 * phase, 1),
                "visibility": 24140.0,
                "weather_code": 1,
                "cape": 0.0,
            })

        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "interval": 900,
                "temperature_2m": 33.4, "relative_humidity_2m": 58,
                "apparent_temperature": 36.1, "precipitation": 0.0,
                "wind_speed_10m": 14.2, "wind_gusts_10m": 27.0,
                "visibility": 24140.0, "weather_code": 1, "cape": 0.0,
            },
            "today_forecast": {"high": 35.0, "low": 27.2, "wind": 17.0,
                               "gust": 33.0, "precipitation": 0.0},
            "previous_days": [
                {"date": f"2026-11-0{d}", "high": 34.6, "low": 26.9,
                 "wind": 16.1, "gust": 30.4, "precipitation": 0.0}
                for d in (5, 6, 7)
            ],
            "hourly_forecast_next_24h": hours,
            "history_summary": {"average_high": 34.5,
                                "average_low": 26.8, "peak_wind": 17.2},
        }

    fz_history = [{
        "timestamp": f"2026-11-0{d}T04:00:12.345Z",
        "city": "Dubai",
        "advice": "Limit strenuous outdoor activity at midday and stay "
                  "hydrated. Official warning status was not checked.",
        "alert_sent": False,
        "alert_message": "",
    } for d in (5, 6, 7)]

    fz_reply = json_text({
        "hazard_type": "Extreme Heat",
        "likelihood_score": 3,
        "likelihood_reason": "Apparent temperature near 38 C forecast.",
        "severity_score": 3,
        "severity_reason": "Heat stress for outdoor activity.",
        "exposure_score": 2,
        "exposure_reason": "Outdoor workers during the afternoon.",
        "duration_score": 3,
        "duration_reason": "About five hours above 35 C apparent.",
        "geographic_impact_score": 2,
        "geographic_impact_reason": "City-wide forecast point only.",
        "alert_recommendation": "MONITOR",
        "advice": "Limit strenuous outdoor activity in the afternoon. "
                  "Official warning status was not checked.",
        "reasoning": "Heat screening triggered by the forecast.",
        "recommended_actions": ["Schedule outdoor work for mornings"],
        "sources": ["6.1 Extreme Heat", "4. Risk Assessment Dimensions"],
    })

    class FzResponse:
        def __init__(self, status_code=200, payload=None, text="",
                     json_error=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = {}
            self._json_error = json_error

        def json(self):
            if self._json_error is not None:
                raise self._json_error

            return self._payload

        def raise_for_status(self):
            return None

    def fz_groq_ok():
        return FzResponse(payload={
            "choices": [{
                "message": {"content": fz_reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 4100, "completion_tokens": 640,
                      "total_tokens": 4740,
                      "completion_tokens_details": {"reasoning_tokens": 120}},
        })

    class FzTransport:
        """Groq POSTs answered offline; history GETs answered or failed."""

        def __init__(self, history_get=None):
            self.history_get = history_get
            self.posts = []
            self.gets = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append({"url": url, "json": json})
            return fz_groq_ok()

        def get(self, url, params=None, timeout=None):
            self.gets.append(url)
            return self.history_get(url, params)

    def fz_pipeline(transport, sheet_url, history_patch=None):
        printed = io.StringIO()
        settings = {
            "HTTP": transport,
            "GROQ_API_KEY": "gsk_fz_test_key",
            "SHEET_URL": sheet_url,
            "require_configuration": lambda require_sheet=False: None,
            "get_weather": lambda city: fz_heat_weather(),
            "EVIDENCE_GATED_DIMENSIONS": {},
        }

        if history_patch is not None:
            settings["tool_get_history"] = history_patch

        def run():
            with contextlib.redirect_stdout(printed):
                return run_weather_agent()

        decision, used = fz_override(settings, run)

        dry_log = io.StringIO()
        dry_transport = FzTransport()

        def dry_run():
            with contextlib.redirect_stdout(dry_log):
                run_agent_job(dry=True)

        fz_override(
            {
                "run_weather_agent": lambda: (decision, used),
                "HTTP": dry_transport,
                "SHEET_URL": sheet_url,
            },
            dry_run,
        )

        return {
            "decision": decision,
            "log": printed.getvalue(),
            "dry_log": dry_log.getvalue(),
            "dry_posts": dry_transport.posts,
        }

    # A. Realistic three-record history + heat: fits, strict request,
    # validates locally, no external writes or email.
    fz_ok_transport = FzTransport(
        history_get=lambda url, params: FzResponse(payload=fz_history)
    )
    fz_ok = fz_pipeline(fz_ok_transport, "https://sheet.invalid/exec")
    fz_decision = fz_ok["decision"]
    fz_payload = (
        fz_ok_transport.posts[0]["json"] if fz_ok_transport.posts else {}
    )
    fz_request = serialized_request(fz_payload.get("messages", []))

    check(
        "Realistic-history heat request is within every budget",
        set(fz_decision["hazard_screening"]) == {"HEAT"}
        and fz_decision["history_status"] == "AVAILABLE"
        and len(fz_ok_transport.posts) == 1
        and len(fz_request) <= MAX_REQUEST_CHARS
        and estimate_tokens(fz_request) + MAX_COMPLETION_TOKENS
        <= MAX_ESTIMATED_REQUEST_TOKENS
        and type(fz_decision["groq_usage"].get("estimated_input_tokens"))
        is int
        and type(fz_decision["groq_usage"].get("completion_allowance"))
        is int
        and fz_decision["groq_usage"]["estimated_input_tokens"]
        + fz_decision["groq_usage"]["completion_allowance"]
        <= MAX_ESTIMATED_REQUEST_TOKENS,
    )

    check(
        "Realistic-history heat request uses the strict schema settings",
        fz_payload.get("response_format") == RESPONSE_FORMAT
        and fz_payload["response_format"]["json_schema"]["strict"] is True
        and fz_payload.get("max_completion_tokens") == 1000
        and fz_payload.get("reasoning_effort") == "low"
        and fz_payload.get("include_reasoning") is False
        and fz_payload.get("model") == MODEL,
    )

    check(
        "Realistic-history heat request carries the history evidence",
        "recent_agent_history" in fz_request
        and "Limit strenuous outdoor activity at midday" in fz_request
        and "6.1 Extreme Heat" in fz_request,
    )

    check(
        "Realistic-history heat response validates and is scored",
        fz_decision["risk_score"] == 7.5
        and fz_decision["risk_band"] == "MODERATE"
        and fz_decision["response_diagnostics"] == {}
        and fz_decision["sources"]
        == ["6.1 Extreme Heat", "4. Risk Assessment Dimensions"]
        and fz_decision["alert_recommendation"] == "MONITOR"
        and fz_decision["assessment_status"] == "COMPLETE"
        and fz_decision.get("incomplete_reasons", []) == []
        and fz_decision["official_warning_status"] == "NOT_CHECKED"
        and fz_decision["approval_status"] == "PENDING_REVIEW",
    )

    check(
        "Realistic-history heat run makes no external write or email",
        fz_ok["dry_posts"] == []
        and fz_decision["send_alert"] is False
        and fz_decision["notification_status"] == "NOT_SENT"
        and "DRY RUN: No data was sent to Google Sheets." in fz_ok["dry_log"]
        and "DRY RUN: No email was sent." in fz_ok["dry_log"]
        and "gsk_fz_test_key" not in fz_ok["log"] + fz_ok["dry_log"],
    )

    # B. A synthetic secret SHEET_URL inside requests exceptions never
    # reaches logs or assessment output.
    fz_secret_url = (
        "https://script.google.com/macros/s/FZ-SECRET-SHEET-MARKER/exec"
    )
    fz_with_query = fz_secret_url + "?action=history&n=3&FZ-QUERY-MARKER=1"

    def fz_http_error():
        response = FzResponse(status_code=404)
        return requests.HTTPError(
            f"404 Client Error: Not Found for url: {fz_with_query}",
            response=response,
        )

    fz_failures = {
        "HTTP error": (lambda: fz_http_error(),
                       "history service returned HTTP 404"),
        "connection error": (
            lambda: requests.ConnectionError(
                "HTTPSConnectionPool(host='script.google.com', port=443): "
                f"Max retries exceeded with url: {fz_with_query}"
            ),
            "history service could not be reached",
        ),
        "timeout": (
            lambda: requests.Timeout(f"Read timed out: {fz_with_query}"),
            "history request timed out",
        ),
        "generic request error": (
            lambda: requests.RequestException(f"failed {fz_with_query}"),
            "history request failed",
        ),
    }

    def fz_marker_free(text):
        return "FZ-SECRET" not in text and "FZ-QUERY" not in text

    for label, (make_error, expected) in fz_failures.items():
        def failing_get(url, params, make_error=make_error):
            raise make_error()

        direct = fz_override(
            {"SHEET_URL": fz_secret_url,
             "HTTP": FzTransport(history_get=failing_get)},
            tool_get_history,
        )
        run = fz_pipeline(FzTransport(history_get=failing_get),
                          fz_secret_url)
        surfaces = (
            run["log"] + run["dry_log"] + json_text(run["decision"])
        )

        check(
            f"History {label}: sanitised description only",
            direct == {"error": expected},
        )

        check(
            f"History {label}: secret URL never reaches logs or output",
            fz_marker_free(surfaces)
            and f"History unavailable: {expected}" in run["log"]
            and run["decision"]["history_status"] == "UNAVAILABLE"
            and any(
                "history was unavailable" in item
                for item in run["decision"]["limitations"]
            )
            and run["dry_posts"] == [],
        )

    def fz_bad_json(url, params):
        return FzResponse(json_error=ValueError(
            f"Expecting value near {fz_with_query}"
        ))

    fz_json = fz_override(
        {"SHEET_URL": fz_secret_url, "HTTP": FzTransport(fz_bad_json)},
        tool_get_history,
    )

    check(
        "History invalid JSON: sanitised description only",
        fz_json == {"error": "history response could not be read as JSON"}
        and fz_marker_free(json_text(fz_json)),
    )

    check(
        "History error descriptions never use the exception text",
        describe_history_error(
            requests.RequestException(fz_with_query)
        ) == "history request failed"
        and describe_history_error(
            requests.HTTPError(fz_with_query)
        ) == "history service returned an HTTP error"
        and fz_marker_free(
            describe_history_error(ValueError(fz_with_query))
        ),
    )

    check(
        "json_validate_failed description does not infer a cause",
        "completion" not in GROQ_ERROR_CODE_DESCRIPTIONS[
            "json_validate_failed"
        ]
        and "allowance" not in GROQ_ERROR_CODE_DESCRIPTIONS[
            "json_validate_failed"
        ]
        and "validate against the response schema"
        in GROQ_ERROR_CODE_DESCRIPTIONS["json_validate_failed"],
    )

    # ------------------------------------------------------------
    # Policy source citations resolve only to retrieved sections
    # (self-contained, offline).
    # ------------------------------------------------------------
    with contextlib.redirect_stdout(io.StringIO()):
        sr_policy = load_guidelines()
        sr_hazards = {"HEAT": ["t"]}
        sr_required = select_hazard_guidance(sr_policy, sr_hazards)
        sr_required += select_conditional_sections(
            sr_policy, sr_hazards, {"current": {"time": "2026-11-08T12:00"}}
        )
        sr_sections = search_guidelines(
            sr_policy, "x", k=0, required_sections=sr_required
        )

    sr_titles = [section["section"] for section in sr_sections]

    check(
        "Source fixture matches the failing run's nine sections",
        sr_titles == [
            "4. Risk Assessment Dimensions",
            "13. Advice and Communication Requirements",
            "8. Alert Decision Framework",
            "9. Official Warning Override",
            "14. Human Approval and Notification Governance",
            "16. Final Decision Rule",
            "6. Weather Hazard Assessment",
            "6.1 Extreme Heat",
            "7. Regulatory and Workplace Compliance",
        ],
    )

    for label, citation, expected in (
        ("exact title", "6.1 Extreme Heat", "6.1 Extreme Heat"),
        ("header prefix", "POLICY SECTION: 6.1 Extreme Heat",
         "6.1 Extreme Heat"),
        ("whole header line", "=== POLICY SECTION: 6.1 Extreme Heat ===",
         "6.1 Extreme Heat"),
        ("heading without number", "Extreme Heat", "6.1 Extreme Heat"),
        ("number only", "6.1", "6.1 Extreme Heat"),
        ("number with dot", "8.", "8. Alert Decision Framework"),
        ("Section prefix", "Section 6.1 Extreme Heat", "6.1 Extreme Heat"),
        ("section sign", "§ 9. Official Warning Override",
         "9. Official Warning Override"),
        ("markdown heading", "## 4. Risk Assessment Dimensions",
         "4. Risk Assessment Dimensions"),
        ("different case", "6.1 extreme heat", "6.1 Extreme Heat"),
        ("quoted", "'16. Final Decision Rule'", "16. Final Decision Rule"),
        ("numbered sub-heading", "4.1 Likelihood (L)",
         "4. Risk Assessment Dimensions"),
        ("parent of hazard subsection", "Weather Hazard Assessment",
         "6. Weather Hazard Assessment"),
    ):
        check(
            f"Citation resolves to its canonical title: {label}",
            resolve_policy_sources([citation], sr_sections) == [expected],
        )

    for label, citation in (
        ("section not retrieved", "5. Risk Calculation Model"),
        ("unretrieved number", "5"),
        ("unnumbered enum sub-heading", "MONITOR"),
        ("policy name", "Dubai Weather Safety Guidelines"),
        ("arbitrary text", "NCM website"),
        ("partial heading", "Heat"),
        ("empty string", ""),
    ):
        check(
            f"Citation stays rejected: {label}",
            resolve_policy_sources([citation], sr_sections) == [],
        )

    for label, citation in (
        ("numbered sub-heading", "4.1 Likelihood (L)"),
        ("numbered sub-heading as markdown", "### 4.4 Duration (D)"),
        ("sub-heading number only", "4.3"),
    ):
        check(
            f"Numbered sub-heading resolves to its parent: {label}",
            resolve_policy_sources([citation], sr_sections)
            == ["4. Risk Assessment Dimensions"],
        )

    for label, citation in (
        ("enum sub-heading", "MONITOR"),
        ("enum sub-heading as markdown", "### ADVISORY_RECOMMENDED"),
        ("enum sub-heading lower case", "escalate_for_review"),
        ("numbered sub-heading without its number", "Likelihood (L)"),
        ("another unnumbered form", "Duration (D)"),
        ("unnumbered parent-section sub-heading", "Risk bands"),
    ):
        check(
            f"Unnumbered sub-heading is rejected: {label}",
            resolve_policy_sources([citation], sr_sections) == []
            and resolve_citation(citation, sr_sections)[0] is None,
        )

    check(
        "Unnumbered sub-headings alone cannot make sources valid",
        resolve_policy_sources(
            ["MONITOR", "Likelihood (L)", "### ESCALATE_FOR_REVIEW"],
            sr_sections,
        ) == [],
    )

    check(
        "Ambiguous numbered sub-heading is rejected",
        resolve_policy_sources(["2.1 Shared"], [
            {"section": "1. A", "text": "### 2.1 Shared\nx"},
            {"section": "3. B", "text": "### 2.1 Shared\ny"},
        ]) == [],
    )

    sr_prompt_messages = []

    def sr_capture(messages, temperature=0.1):
        sr_prompt_messages.append(messages)
        raise RuntimeError("captured")

    sr_saved_ask = globals()["ask_groq"]
    globals()["ask_groq"] = sr_capture

    try:
        assess_weather(
            {"city": CITY, "source": "Open-Meteo", "retrieved_at": "x",
             "current": {}, "hourly_forecast_next_24h": []},
            [], sr_policy, sr_sections,
        )
    except RuntimeError:
        pass
    finally:
        globals()["ask_groq"] = sr_saved_ask

    sr_user_prompt = (
        sr_prompt_messages[0][1]["content"] if sr_prompt_messages else ""
    )
    sr_clarification = (
        "In sources, give each title exactly as it appears after "
        "'POLICY SECTION:' in a block header (for example '6.1 Extreme "
        "Heat'), without the prefix or the '===' markers."
    )

    check(
        "Prompt carries the source-citation clarification exactly once",
        sr_user_prompt.count(sr_clarification) == 1
        and "Cite section titles exactly as written in the section "
        "headers" not in sr_user_prompt
        and "In sources, cite exact policy section titles"
        in " ".join(SYSTEM_PROMPT.split()),
    )

    sr_ambiguous = [
        {"section": "1. Heat Advice", "text": "x"},
        {"section": "2. Heat Advice", "text": "y"},
    ]

    check(
        "Ambiguous citation stays rejected",
        resolve_policy_sources(["Heat Advice"], sr_ambiguous) == []
        and resolve_citation("Heat Advice", sr_ambiguous)
        == (None, "ambiguous"),
    )

    check(
        "Mixed citations keep only resolved canonical titles, once",
        resolve_policy_sources(
            ["Extreme Heat", "MONITOR", "6.1", "8. Alert Decision Framework",
             "5. Risk Calculation Model", "6.1 Extreme Heat"],
            sr_sections,
        ) == ["6.1 Extreme Heat", "8. Alert Decision Framework"],
    )

    def sr_response(sources):
        return {
            "hazard_type": "HEAT",
            "likelihood_score": 2, "likelihood_reason": "r",
            "severity_score": 2, "severity_reason": "r",
            "exposure_score": None, "exposure_reason": "r",
            "duration_score": None, "duration_reason": "r",
            "geographic_impact_score": None,
            "geographic_impact_reason": "r",
            "alert_recommendation": "UNABLE_TO_ASSESS",
            "advice": "Official warning status was not checked.",
            "reasoning": "r", "recommended_actions": [],
            "sources": sources,
        }

    def sr_validate_error_only(sources):
        try:
            validate_decision(sr_response(sources), sr_sections)
            return ""
        except ValueError as exc:
            return str(exc)

    def sr_validate(sources):
        try:
            return validate_decision(sr_response(sources), sr_sections)
        except ValueError as exc:
            return str(exc)

    sr_near_miss = sr_validate(
        ["POLICY SECTION: 6.1 Extreme Heat", "Extreme Heat"]
    )

    check(
        "Validator stores canonical titles for near-miss citations",
        isinstance(sr_near_miss, dict)
        and sr_near_miss["sources"] == ["6.1 Extreme Heat"],
    )

    check(
        "Validator fails closed when only unnumbered sub-headings are cited",
        sr_validate_error_only(["MONITOR", "Likelihood (L)"])
        == "Assessment did not identify a valid retrieved policy section.",
    )

    check(
        "No resolvable citation still fails closed with the same error",
        sr_validate(["5. Risk Calculation Model", "MONITOR", "NCM"])
        == "Assessment did not identify a valid retrieved policy section."
        and sr_validate([])
        == "Assessment did not identify a valid retrieved policy section.",
    )

    sr_diag = source_resolution_diagnostics(
        ["6.1 Extreme Heat", "Extreme Heat", "SR-MARKER-TEXT",
         "5. Risk Calculation Model", "4.1 Likelihood (L)"],
        sr_sections,
    )

    check(
        "Source diagnostics are counts per rule, never citation text",
        sr_diag == {
            "source_count": 5,
            "resolution_counts": {
                "exact": 1, "heading": 1, "sub_heading": 1,
                "unmatched": 1, "unretrieved_section_reference": 1,
            },
        }
        and "SR-MARKER" not in json_text(sr_diag),
    )

    # Real pipeline: an unresolvable response fails closed and records
    # the counts; a near-miss response is accepted canonically.
    def sr_weather():
        start = datetime(2026, 11, 8, 13, 0)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 33,
                "apparent_temperature": 36, "precipitation": 0,
                "wind_speed_10m": 10, "wind_gusts_10m": 20,
                "visibility": 24140, "weather_code": 1, "cape": 0,
            },
            "today_forecast": {"high": 35, "low": 27, "wind": 14,
                               "gust": 25, "precipitation": 0},
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": 33, "apparent_temperature": 36,
                "humidity": 55, "precipitation": 0, "wind": 10,
                "gust": 20, "visibility": 24140, "weather_code": 1,
                "cape": 0,
            } for h in range(24)],
            "history_summary": {},
        }

    class SrResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content):
            self.content = content

        def json(self):
            return {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    class SrTransport:
        def __init__(self, content):
            self.content = content
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            return SrResponse(self.content)

    def sr_pipeline(sources):
        transport = SrTransport(json_text(sr_response(sources)))
        printed = io.StringIO()
        saved = {name: globals()[name] for name in (
            "HTTP", "GROQ_API_KEY", "require_configuration",
            "get_weather", "tool_get_history",
        )}
        globals().update({
            "HTTP": transport,
            "GROQ_API_KEY": "gsk_sr_test_key",
            "require_configuration": lambda require_sheet=False: None,
            "get_weather": lambda city: sr_weather(),
            "tool_get_history": lambda n=3: [],
        })

        try:
            with contextlib.redirect_stdout(printed):
                decision = run_weather_agent()[0]
        finally:
            globals().update(saved)

        return decision, printed.getvalue(), transport

    sr_bad, sr_bad_log, sr_bad_transport = sr_pipeline(
        ["SR-MARKER-ONE", "5. Risk Calculation Model"]
    )

    check(
        "Real path: unresolvable sources fail closed with counts only",
        len(sr_bad_transport.posts) == 1
        and sr_bad["assessment_status"] == "INCOMPLETE"
        and sr_bad["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and sr_bad["send_alert"] is False
        and "did not identify a valid retrieved policy section"
        in sr_bad["reasoning"]
        and sr_bad["response_diagnostics"]["source_resolution"] == {
            "source_count": 2,
            "resolution_counts": {
                "unmatched": 1, "policy_section_not_retrieved": 1,
            },
        }
        and "SR-MARKER" not in sr_bad_log + json_text(sr_bad),
    )

    sr_good, _, _ = sr_pipeline(
        ["=== POLICY SECTION: 6.1 Extreme Heat ===", "4.4 Duration (D)"]
    )

    check(
        "Real path: near-miss citations are accepted as canonical titles",
        sr_good["sources"]
        == ["6.1 Extreme Heat", "4. Risk Assessment Dimensions"]
        and sr_good["response_diagnostics"] == {}
        and sr_good["assessment_status"] == "INCOMPLETE",
    )

    sr_claim, _, _ = sr_pipeline(
        ["6.1 Extreme Heat", "No official heat advisory is in effect"]
    )

    check(
        "Claim text in an unresolvable source never reaches the output",
        "advisory is in effect" not in json_text(sr_claim)
        and sr_claim["sources"] == ["6.1 Extreme Heat"],
    )

    # ------------------------------------------------------------
    # Evidence gate (exposure, geographic impact), risk-score notice
    # and dry-run audit summary (self-contained, offline).
    # ------------------------------------------------------------
    def eg_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def eg_weather():
        start = datetime(2026, 11, 8, 13, 0)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 33,
                "apparent_temperature": 36, "precipitation": 0,
                "wind_speed_10m": 10, "wind_gusts_10m": 20,
                "visibility": 24140, "weather_code": 1, "cape": 0,
            },
            "today_forecast": {"high": 35, "low": 27, "wind": 14,
                               "gust": 25, "precipitation": 0},
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": 33, "apparent_temperature": 36,
                "humidity": 55, "precipitation": 0, "wind": 10,
                "gust": 20, "visibility": 24140, "weather_code": 1,
                "cape": 0,
            } for h in range(24)],
            "history_summary": {},
        }

    def eg_reply(**changes):
        body = {
            "hazard_type": "Extreme Heat",
            "likelihood_score": 3, "likelihood_reason": "lr",
            "severity_score": 3, "severity_reason": "sr",
            "exposure_score": 4,
            "exposure_reason": "EG-MARK-EXP large population and "
                               "outdoor workforce",
            "duration_score": 3, "duration_reason": "dr",
            "geographic_impact_score": 4,
            "geographic_impact_reason": "EG-MARK-GEO city-wide",
            "alert_recommendation": "MONITOR",
            "advice": "Limit strenuous outdoor activity. Official "
                      "warning status was not checked.",
            "reasoning": "Heat screening triggered. EG-MARK-REASONING",
            "recommended_actions": [],
            "sources": ["6.1 Extreme Heat"],
        }
        body.update(changes)
        return json_text(body)

    class EgResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content):
            self.content = content

        def json(self):
            return {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    class EgTransport:
        def __init__(self, content):
            self.content = content
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            return EgResponse(self.content)

    def eg_run(content, gate=None, dry=True):
        transport = EgTransport(content)
        printed = io.StringIO()
        settings = {
            "HTTP": transport,
            "GROQ_API_KEY": "gsk_eg_test_key",
            "require_configuration": lambda require_sheet=False: None,
            "get_weather": lambda city: eg_weather(),
            "tool_get_history": lambda n=3: [],
        }

        if gate is not None:
            settings["EVIDENCE_GATED_DIMENSIONS"] = gate

        def run():
            with contextlib.redirect_stdout(printed):
                return run_weather_agent()

        decision, used = eg_override(settings, run)
        report = io.StringIO()
        sheets = EgTransport("OK")

        def job():
            with contextlib.redirect_stdout(report):
                run_agent_job(dry=True)

        eg_override(
            {"run_weather_agent": lambda: (decision, used),
             "HTTP": sheets},
            job,
        )

        return {
            "decision": decision,
            "log": printed.getvalue(),
            "report": report.getvalue(),
            "sheet_posts": sheets.posts,
        }

    # P1-1: Exposure 4 and Geographic Impact 4 are overridden.
    eg_four = eg_run(eg_reply())
    eg_d = eg_four["decision"]

    check(
        "Model Exposure 4 and Geographic Impact 4 are both overridden",
        eg_d["exposure_score"] is None
        and eg_d["geographic_impact_score"] is None
        and eg_d.get("evidence_gate_overrides")
        == ["exposure_score", "geographic_impact_score"]
        and eg_d["likelihood_score"] == 3
        and eg_d["severity_score"] == 3
        and eg_d["duration_score"] == 3,
    )

    check(
        "Overridden dimensions carry fixed program reasons, not model text",
        eg_d["exposure_reason"]
        == EVIDENCE_GATED_DIMENSIONS.get("exposure_score", (None, None))[1]
        and eg_d["geographic_impact_reason"]
        == EVIDENCE_GATED_DIMENSIONS.get(
            "geographic_impact_score", (None, None)
        )[1]
        and eg_d["exposure_reason"] is not None
        and "EG-MARK-EXP" not in json_text(eg_d)
        and "EG-MARK-GEO" not in json_text(eg_d)
        and "EG-MARK-EXP" not in eg_four["log"] + eg_four["report"]
        and "EG-MARK-GEO" not in eg_four["log"] + eg_four["report"],
    )

    # P1-3: no numerical classification or HIGH band.
    check(
        "Missing dimensions give no risk score and no band (not HIGH)",
        eg_d["risk_score"] is None
        and eg_d["risk_band"] == "UNASSESSED"
        and "10.5" not in eg_four["report"]
        and "HIGH" not in eg_four["report"].split("RISK BAND:")[1][:12]
        and eg_d["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and eg_d["assessment_status"] == "INCOMPLETE"
        and eg_d["send_alert"] is False,
    )

    check(
        "Missing dimensions are recorded with the supported reason code",
        [r["code"] for r in eg_d["incomplete_reasons"]]
        == ["RISK_DIMENSIONS_UNASSESSED"]
        and eg_d["incomplete_reasons"][0]["detail"]
        == "exposure_score, geographic_impact_score"
        and any(
            "exposure_score" in item and "geographic_impact_score" in item
            for item in eg_d["missing_evidence"]
        ),
    )

    # P1-2: other numerical values are overridden too.
    for exposure, geographic in ((1, 1), (2, 3), (5, 5)):
        run = eg_run(eg_reply(
            exposure_score=exposure, geographic_impact_score=geographic
        ))

        check(
            f"Model values ({exposure}, {geographic}) are overridden "
            "and no band results",
            run["decision"]["exposure_score"] is None
            and run["decision"]["geographic_impact_score"] is None
            and run["decision"]["risk_band"] == "UNASSESSED"
            and run["decision"]["risk_score"] is None,
        )

    only_null = eg_run(eg_reply(
        exposure_score=None, geographic_impact_score=None
    ))

    check(
        "Model nulls need no override and stay unassessed",
        only_null["decision"].get("evidence_gate_overrides") == []
        and only_null["decision"]["risk_band"] == "UNASSESSED",
    )

    # Strictness is preserved: the gate never creates missing keys.
    def eg_without(key):
        body = json.loads(eg_reply())
        del body[key]
        return json_text(body)

    for key in ("exposure_score", "geographic_impact_score",
                "exposure_reason"):
        run = eg_run(eg_without(key))

        check(
            f"Gate does not mask a missing key: {key}",
            "Missing assessment fields: " + key
            in run["decision"]["reasoning"]
            and run["decision"]["assessment_status"] == "INCOMPLETE",
        )

    # P1-4: the model cannot supply program-owned gate fields.
    for label, extra in (
        ("overrides list", {"evidence_gate_overrides": []}),
        ("incomplete reasons", {"incomplete_reasons": [
            {"code": "NONE", "detail": "all clear"}]}),
        ("risk score", {"risk_score": 10.5}),
        ("risk band", {"risk_band": "HIGH"}),
    ):
        run = eg_run(eg_reply(**extra))

        check(
            f"Model cannot inject program-owned values: {label}",
            "program-owned fields" in run["decision"]["reasoning"]
            and run["decision"]["risk_band"] == "UNASSESSED"
            and all(
                r["code"] != "NONE"
                for r in run["decision"]["incomplete_reasons"]
            ),
        )

    # The gate cannot be bypassed by wrong types in gated fields.
    garbage = eg_run(eg_reply(
        exposure_score="4", geographic_impact_score=99
    ))

    check(
        "Malformed gated values are discarded, never scored",
        garbage["decision"]["exposure_score"] is None
        and garbage["decision"]["geographic_impact_score"] is None
        and garbage["decision"]["risk_band"] == "UNASSESSED",
    )

    # P1-5: fully assessed calculations are unchanged.
    check(
        "With no gated dimension the formula still scores and bands",
        eg_run(eg_reply(), gate={})["decision"]["risk_score"] == 10.5
        and eg_run(eg_reply(), gate={})["decision"]["risk_band"] == "HIGH",
    )

    check(
        "Gate covers exactly exposure and geographic impact",
        set(EVIDENCE_GATED_DIMENSIONS)
        == {"exposure_score", "geographic_impact_score"}
        and all(
            reason_field.endswith("_reason") and len(text) > 40
            for reason_field, text in EVIDENCE_GATED_DIMENSIONS.values()
        ),
    )

    check(
        "evidence_gate_overrides is a program-owned field",
        "evidence_gate_overrides" in PROGRAM_OWNED_FIELDS,
    )

    check(
        "Apply-gate is a no-op for non-dict output and for absent keys",
        apply_evidence_gate([1, 2]) == []
        and apply_evidence_gate({"likelihood_score": 3}) == []
        and apply_evidence_gate({"exposure_score": 3})
        == ["exposure_score"],
    )

    # P2: the notice no longer points at an unsent section.
    eg_sent = []

    def eg_capture(messages, temperature=0.1):
        eg_sent.append(messages)
        raise RuntimeError("captured")

    with contextlib.redirect_stdout(io.StringIO()):
        eg_policy = load_guidelines()

    try:
        eg_override(
            {"ask_groq": eg_capture},
            lambda: assess_weather(
                eg_weather(), [], eg_policy,
                [{"section": "6.1 Extreme Heat", "text": "x"}],
            ),
        )
    except RuntimeError:
        pass

    eg_user = eg_sent[0][1]["content"] if eg_sent else ""

    check(
        "Risk-score notice no longer references guidelines.md section 5",
        "risk_score_notice" in eg_user
        and "guidelines.md" not in eg_user.split("EVIDENCE AND STATUS")[1]
        and "section 5" not in eg_user.split("EVIDENCE AND STATUS")[1]
        and "internal provisional calculation made by the program"
        in eg_user
        and "not an official UAE or NCM classification" in eg_user
        and "must not be treated as validated for automated public "
        "notifications" in eg_user,
    )

    # P3: audit summary appears in the dry-run output.
    eg_summary = None

    for line in eg_four["report"].splitlines():
        if line.startswith("{") and "risk_score_calculated" in line:
            eg_summary = json.loads(line)

    check(
        "Dry-run audit summary lists the safe audit fields",
        eg_summary is not None
        and "AUDIT SUMMARY (program-generated)" in eg_four["report"]
        and eg_summary["sources"] == ["6.1 Extreme Heat"]
        and eg_summary["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and eg_summary["assessment_status"] == "INCOMPLETE"
        and eg_summary["risk_score_calculated"] is False
        and eg_summary["risk_band"] == "UNASSESSED"
        and eg_summary["incomplete_reason_codes"]
        == ["RISK_DIMENSIONS_UNASSESSED"]
        and eg_summary["evidence_gate_overrides"]
        == ["exposure_score", "geographic_impact_score"]
        and eg_summary["hazards_screened"] == ["HEAT"],
    )

    eg_scored = eg_run(eg_reply(), gate={})
    eg_scored_summary = json.loads(next(
        line for line in eg_scored["report"].splitlines()
        if line.startswith("{") and "risk_score_calculated" in line
    ))

    check(
        "Audit summary reports when a score was calculated",
        eg_scored_summary["risk_score_calculated"] is True
        and eg_scored_summary["risk_band"] == "HIGH"
        and eg_scored_summary["alert_recommendation"] == "MONITOR",
    )

    eg_failed = eg_run(
        json_text({"not": "the schema"})
    )
    eg_failed_summary = json.loads(next(
        line for line in eg_failed["report"].splitlines()
        if line.startswith("{") and "risk_score_calculated" in line
    ))

    check(
        "Audit summary on a failed assessment holds no model text",
        eg_failed_summary["sources"] == []
        and eg_failed_summary["risk_score_calculated"] is False
        and eg_failed_summary["assessment_status"] == "INCOMPLETE"
        and eg_failed_summary["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and "ASSESSMENT_ERROR"
        in eg_failed_summary["incomplete_reason_codes"],
    )

    audit_text = "".join(
        line for line in eg_four["report"].splitlines()
        if line.startswith("{") and "risk_score_calculated" in line
    )

    check(
        "Audit summary has exactly the approved keys and no model text",
        sorted(eg_summary or {}) == sorted([
            "sources", "alert_recommendation", "assessment_status",
            "risk_score_calculated", "risk_band",
            "incomplete_reason_codes", "evidence_gate_overrides",
            "warning_claim_diagnostic", "hazards_screened",
        ])
        and "EG-MARK" not in audit_text
        and "Heat screening" not in audit_text
        and "Limit strenuous" not in audit_text,
    )

    check(
        "Audit output contains no weather payload, secret or Sheets write",
        "gsk_eg_test_key" not in eg_four["log"] + eg_four["report"]
        and "temperature" not in audit_text
        and eg_four["sheet_posts"] == [],
    )

    # ------------------------------------------------------------
    # Warning-claim failure diagnostics (self-contained, offline).
    # ------------------------------------------------------------
    def cl_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def cl_weather():
        start = datetime(2026, 11, 8, 13, 0)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 33,
                "apparent_temperature": 36, "precipitation": 0,
                "wind_speed_10m": 10, "wind_gusts_10m": 20,
                "visibility": 24140, "weather_code": 1, "cape": 0,
            },
            "today_forecast": {"high": 35, "low": 27, "wind": 14,
                               "gust": 25, "precipitation": 0},
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": 33, "apparent_temperature": 36,
                "humidity": 55, "precipitation": 0, "wind": 10,
                "gust": 20, "visibility": 24140, "weather_code": 1,
                "cape": 0,
            } for h in range(24)],
            "history_summary": {},
        }

    def cl_body(**changes):
        body = {
            "hazard_type": "Extreme Heat",
            "likelihood_score": 3, "likelihood_reason": "lr",
            "severity_score": 3, "severity_reason": "sr",
            "exposure_score": 4, "exposure_reason": "CL-MARK-EXP",
            "duration_score": 3, "duration_reason": "dr",
            "geographic_impact_score": 4,
            "geographic_impact_reason": "CL-MARK-GEO",
            "alert_recommendation": "MONITOR",
            "advice": "Limit outdoor activity. Official warning status "
                      "was not checked.",
            "reasoning": "Heat screening triggered.",
            "recommended_actions": [],
            "sources": ["POLICY SECTION: 6.1 Extreme Heat",
                        "Extreme Heat", "CL-MARK-SRC"],
        }
        body.update(changes)
        return body

    class ClResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content):
            self.content = content

        def json(self):
            return {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    class ClTransport:
        def __init__(self, content):
            self.content = content
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            return ClResponse(self.content)

    def cl_run(body):
        transport = ClTransport(json_text(body))
        printed = io.StringIO()
        report = io.StringIO()
        sheets = ClTransport("OK")

        def assess():
            with contextlib.redirect_stdout(printed):
                return run_weather_agent()

        decision, used = cl_override(
            {
                "HTTP": transport,
                "GROQ_API_KEY": "gsk_cl_test_key",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: cl_weather(),
                "tool_get_history": lambda n=3: [],
            },
            assess,
        )

        def job():
            with contextlib.redirect_stdout(report):
                run_agent_job(dry=True)

        cl_override(
            {"run_weather_agent": lambda: (decision, used),
             "HTTP": sheets},
            job,
        )

        summary = json.loads(next(
            line for line in report.getvalue().splitlines()
            if line.startswith("{") and "risk_score_calculated" in line
        ))

        return {
            "decision": decision,
            "log": printed.getvalue(),
            "report": report.getvalue(),
            "summary": summary,
            "sheet_posts": sheets.posts,
        }

    # 1. Each of the five patterns records its own number and field.
    cl_samples = {
        1: ("reasoning", "No official heat advisory is in effect."),
        2: ("advice", "There are no active warnings."),
        3: ("duration_reason", "No official warnings were checked."),
        4: ("likelihood_reason", "A heat warning is currently in effect."),
        5: ("recommended_actions", ["NCM has not issued any warning."]),
    }

    for number, (field, value) in cl_samples.items():
        text = value if isinstance(value, str) else value[0]

        check(
            f"Pattern {number} is identified by number",
            find_unverified_warning_claim_pattern(text) == number
            and find_unverified_warning_claim(text) is True,
        )

        run = cl_run(cl_body(**{field: value}))
        decision = run["decision"]
        expected = f"field={field};pattern={number}"
        reasons = {
            r["code"]: r["detail"] for r in decision["incomplete_reasons"]
        }

        check(
            f"Pattern {number} in {field} records {expected}",
            reasons.get("UNVERIFIED_WARNING_CLAIM") == expected
            and decision.get("warning_claim_diagnostic") == expected
            and run["summary"].get("warning_claim_diagnostic") == expected
            and f"INCOMPLETE REASON: UNVERIFIED_WARNING_CLAIM | {expected}"
            in run["report"],
        )

    # Non-schema key names are labelled, never echoed.
    cl_outside = cl_run(cl_body(**{
        "CL-MARK-KEY": "There are no active warnings."
    }))

    check(
        "Claim outside the schema records field=outside_schema",
        cl_outside["decision"].get("warning_claim_diagnostic")
        == "field=outside_schema;pattern=2"
        and "CL-MARK-KEY" not in json_text(cl_outside["decision"])
        + cl_outside["log"] + cl_outside["report"],
    )

    check(
        "Diagnostic is built from fixed values even if metadata is bad",
        UnverifiedWarningClaim("m", field_label="CL-MARK-LABEL",
                               pattern_number=99).diagnostic()
        == "field=unknown;pattern=None"
        and UnverifiedWarningClaim("m", "advice", 3).diagnostic()
        == "field=advice;pattern=3"
        and UnverifiedWarningClaim("m", "outside_schema", 5).diagnostic()
        == "field=outside_schema;pattern=5",
    )

    # 2-3. Validated sources and override names survive the failure.
    cl_run_one = cl_run(cl_body(reasoning="No official heat advisory is "
                                          "in effect."))
    cl_d = cl_run_one["decision"]

    check(
        "Canonical validated sources survive the warning-claim failure",
        cl_d["sources"] == ["6.1 Extreme Heat"]
        and cl_run_one["summary"]["sources"] == ["6.1 Extreme Heat"],
    )

    check(
        "Both evidence-gate override names survive the failure",
        cl_d.get("evidence_gate_overrides")
        == ["exposure_score", "geographic_impact_score"]
        and cl_run_one["summary"]["evidence_gate_overrides"]
        == ["exposure_score", "geographic_impact_score"],
    )

    # 4. No model-generated text anywhere in the failure output.
    cl_everything = (
        json_text(cl_d) + cl_run_one["log"] + cl_run_one["report"]
        + json_text(cl_run_one["summary"])
    )

    check(
        "No model text, marker or unvalidated source leaks",
        "CL-MARK" not in cl_everything
        and "advisory is in effect" not in cl_everything
        and "POLICY SECTION: 6.1" not in cl_everything
        and "gsk_cl_test_key" not in cl_everything,
    )

    check(
        "Limitations and missing evidence hold no model text",
        all(
            "advisory" not in item.lower() or "official" in item.lower()
            for item in cl_d["limitations"] + cl_d["missing_evidence"]
        )
        and "CL-MARK" not in json_text(
            cl_d["limitations"] + cl_d["missing_evidence"]
        ),
    )

    # 5. The fail-closed decision keeps every required value.
    check(
        "Warning-claim failure is fully fail-closed",
        cl_d["assessment_status"] == "INCOMPLETE"
        and cl_d["risk_score"] is None
        and cl_d["risk_band"] == "UNASSESSED"
        and cl_d["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and cl_d["send_alert"] is False
        and cl_d["alert_message"] == ""
        and cl_d["approval_status"] == "NOT_AUTHORIZED"
        and cl_d["notification_status"] == "NOT_SENT"
        and cl_d["official_warning_status"] == "NOT_CHECKED"
        and cl_d["exposure_score"] is None
        and cl_d["geographic_impact_score"] is None
        and [r["code"] for r in cl_d["incomplete_reasons"]]
        == ["UNVERIFIED_WARNING_CLAIM"]
        and cl_run_one["sheet_posts"] == [],
    )

    check(
        "Audit summary on the failure has exactly the approved keys",
        sorted(cl_run_one["summary"]) == sorted([
            "sources", "alert_recommendation", "assessment_status",
            "risk_score_calculated", "risk_band",
            "incomplete_reason_codes", "evidence_gate_overrides",
            "warning_claim_diagnostic", "hazards_screened",
        ])
        and cl_run_one["summary"]["risk_score_calculated"] is False
        and cl_run_one["summary"]["alert_recommendation"]
        == "UNABLE_TO_ASSESS",
    )

    # Failures that are not claims carry no claim diagnostic and no
    # unvalidated sources.
    cl_other = cl_run({"not": "the schema"})

    check(
        "Non-claim failure has no claim diagnostic, sources or overrides",
        "warning_claim_diagnostic" in cl_other["summary"]
        and cl_other["summary"]["warning_claim_diagnostic"] is None
        and cl_other["summary"]["sources"] == []
        and cl_other["summary"]["evidence_gate_overrides"] == []
        and "ASSESSMENT_ERROR"
        in cl_other["summary"]["incomplete_reason_codes"],
    )

    check(
        "Model cannot supply the claim diagnostic field",
        "warning_claim_diagnostic" in PROGRAM_OWNED_FIELDS
        and "program-owned fields" in cl_run(cl_body(
            warning_claim_diagnostic="field=advice;pattern=1"
        ))["decision"]["reasoning"],
    )

    # 6. Genuine claims stay blocked (including previously flagged
    # compliant-looking wording: the guard is unchanged).
    for text in (
        "No official heat advisory is in effect.",
        "There are no active warnings.",
        "A heat warning is currently in effect.",
        "NCM has not issued any warning.",
        "No official warnings were checked.",
        "Official advisories are currently unverified.",
    ):
        check(
            f"Claim stays blocked: {text}",
            find_unverified_warning_claim(text),
        )

    for text in (
        "Official warning status was not checked.",
        "Official warning status was not checked; consult NCM for "
        "official warnings.",
    ):
        check(
            f"Approved wording is not a claim: {text[:48]}",
            not find_unverified_warning_claim(text),
        )

    check(
        "The five warning-claim patterns are unchanged",
        len(UNVERIFIED_WARNING_CLAIM_PATTERNS) == 5
        and hashlib.sha256(
            "\n".join(
                f"{p.flags}:{p.pattern}"
                for p in UNVERIFIED_WARNING_CLAIM_PATTERNS
            ).encode("utf-8")
        ).hexdigest()
        == "faf42abf70960e9c7116daae877fb39a37d224d44d65268889a9bf9c9519cd87",
    )

    # ------------------------------------------------------------
    # Text-free shape diagnostic for pattern 3 matches.
    # ------------------------------------------------------------
    def cl_shape(qualifier, term, between, following):
        return {
            "qualifier": qualifier, "term_class": term,
            "words_between": between, "following_word_class": following,
        }

    cl_shape_cases = (
        ("No current advisory is warranted.",
         cl_shape("current", "advisory", 0, "auxiliary")),
        ("No existing warnings.",
         cl_shape("existing", "warning", 0, "punctuation")),
        ("No official heat advisories were considered.",
         cl_shape("official", "advisory", 1, "auxiliary")),
        ("No current alert or advisory is justified.",
         cl_shape("current", "advisory", 2, "auxiliary")),
        ("No official emergency instructions for Dubai.",
         cl_shape("official", "emergency_instruction", 0, "preposition")),
        ("NO ACTIVE WARNING and monitoring continues.",
         cl_shape("active", "warning", 0, "conjunction")),
        ("No official warning information was retrieved.",
         cl_shape("official", "warning", 0, "other_word")),
        ("Heat is high; no active advisory",
         cl_shape("active", "advisory", 0, "end")),
        ("No active warning-level hazard is indicated.",
         cl_shape("active", "warning", 0, "punctuation")),
        ("No\u200b active warning \u2019s",
         cl_shape("active", "warning", 0, "punctuation")),
    )

    for text, expected_shape in cl_shape_cases:
        check(
            f"Pattern 3 shape classified: {expected_shape['qualifier']}/"
            f"{expected_shape['term_class']}/"
            f"{expected_shape['words_between']}/"
            f"{expected_shape['following_word_class']}",
            find_unverified_warning_claim_pattern(text) == 3
            and find_unverified_warning_claim_shape(text) == expected_shape,
        )

    check(
        "No shape for conditional, non-matching or empty text",
        find_unverified_warning_claim_shape(
            "Verify whether no active warning exists later."
        ) is None
        and find_unverified_warning_claim_shape("") is None
        and find_unverified_warning_claim_shape(
            "Official warning status was not checked."
        ) is None,
    )

    # Unknown or malformed shape values fail safely.
    cl_unknown = cl_shape("unknown", "unknown", None, "unknown")

    for label, bad in (
        ("missing", None),
        ("not a dict", "CL-MARK-SHAPE"),
        ("empty", {}),
        ("free text", cl_shape("CL-MARK-Q", "CL-MARK-T", 1, "CL-MARK-F")),
        ("bad counts", cl_shape("active", "warning", 99, "end")),
    ):
        got = UnverifiedWarningClaim(
            "m", field_label="reasoning", pattern_number=3, shape=bad
        ).shape_diagnostic()
        want = {
            "bad counts": cl_shape("active", "warning", None, "end"),
            "free text": cl_shape("unknown", "unknown", 1, "unknown"),
        }.get(label, cl_unknown)

        check(
            f"Malformed shape fails safely: {label}",
            got == want and "CL-MARK" not in json_text(got),
        )

    for bad_count in (True, "2", -1, 3, 1.0):
        check(
            f"Words-between value rejected: {bad_count!r}",
            UnverifiedWarningClaim(
                "m", "reasoning", 3, cl_shape("active", "warning", bad_count,
                                              "end")
            ).shape_diagnostic()["words_between"] is None,
        )

    check(
        "Shape is only reported for patterns 2 and 3",
        all(
            UnverifiedWarningClaim(
                "m", "reasoning", number,
                cl_shape("active", "warning", 0, "end"),
            ).shape_diagnostic() is None
            for number in (1, 4, 5, None)
        )
        and all(
            UnverifiedWarningClaim(
                "m", "reasoning", number,
                cl_shape("active", "warning", 0, "end"),
            ).shape_diagnostic() is not None
            for number in (2, 3)
        ),
    )

    # End to end: marker phrase in reasoning never reaches any output.
    cl_marker = "CL-SHAPE-MARK-4417"
    cl_leak = cl_run(cl_body(
        reasoning=f"{cl_marker} says no current advisory is warranted today."
    ))
    cl_leak_dec = cl_leak["decision"]
    cl_shape_out = cl_leak_dec["response_diagnostics"].get(
        "warning_claim_shape"
    )
    cl_everything = (
        json_text(cl_leak_dec) + json.dumps(cl_leak["summary"])
        + cl_leak["log"] + cl_leak["report"]
    )

    try:
        reject_unverified_warning_claims(
            {"reasoning": f"{cl_marker} no current advisory is warranted."},
            "NOT_CHECKED",
        )
        cl_exc = None
    except UnverifiedWarningClaim as claim:
        cl_exc = claim

    check(
        "Claim failure populates response_diagnostics with the shape",
        cl_shape_out == cl_shape("current", "advisory", 0, "auxiliary")
        and cl_leak_dec["warning_claim_diagnostic"]
        == "field=reasoning;pattern=3"
        and "RESPONSE SHAPE DIAGNOSTICS" in cl_leak["log"]
        and "warning_claim_shape" in cl_leak["log"],
    )

    check(
        "Shape diagnostic holds only fixed-vocabulary values",
        isinstance(cl_shape_out, dict)
        and cl_shape_out["qualifier"] in _CLAIM_QUALIFIERS
        and cl_shape_out["term_class"] in _CLAIM_TERM_CLASSES
        and cl_shape_out["following_word_class"] in _CLAIM_FOLLOW_CLASSES
        and cl_shape_out["words_between"] in (0, 1, 2)
        and set(cl_shape_out) == {
            "qualifier", "term_class", "words_between",
            "following_word_class",
        },
    )

    check(
        "Marker phrase never appears in outputs, logs or the exception",
        cl_marker not in cl_everything
        and cl_exc is not None
        and cl_marker not in str(cl_exc) + repr(cl_exc)
        + json_text(cl_exc.shape_diagnostic())
        + cl_exc.diagnostic()
        and "advisory is warranted" not in cl_everything
        and "today" not in json_text(cl_shape_out),
    )

    check(
        "Fail-closed behaviour is unchanged by the shape diagnostic",
        cl_leak_dec["assessment_status"] == "INCOMPLETE"
        and cl_leak_dec["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and cl_leak_dec["send_alert"] is False
        and cl_leak_dec["approval_status"] == "NOT_AUTHORIZED"
        and cl_leak_dec["risk_score"] is None
        and cl_leak["summary"]["assessment_status"] == "INCOMPLETE"
        and cl_leak["summary"]["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and "UNVERIFIED_WARNING_CLAIM"
        in [r["code"] for r in cl_leak_dec["incomplete_reasons"]],
    )

    cl_other_dec = cl_run(cl_body(
        advice="NCM has not issued any warning."
    ))["decision"]

    check(
        "Patterns without a shape fill response_diagnostics without one",
        "warning_claim_shape" not in cl_other_dec["response_diagnostics"]
        and cl_other_dec["response_diagnostics"].get("missing_required")
        == []
        and cl_other_dec["warning_claim_diagnostic"]
        == "field=advice;pattern=5",
    )

    # Shape diagnostics belong to pattern 3 alone: patterns 1, 2, 4 and
    # 5 never carry one, at the guard, at the exception and end to end.
    # Patterns 1 and 2 are the dangerous cases: their sample texts also
    # match pattern 3, so a shape would be available to attach.
    for number, (field, value) in cl_samples.items():
        text = value if isinstance(value, str) else value[0]

        try:
            reject_unverified_warning_claims({field: value}, "NOT_CHECKED")
            cl_claim = None
        except UnverifiedWarningClaim as claim:
            cl_claim = claim

        cl_end = cl_run(cl_body(**{field: value}))
        cl_end_diag = cl_end["decision"]["response_diagnostics"]
        cl_has_shape = number in (2, 3)

        check(
            f"Pattern {number}: shape attached only for patterns 2 and 3",
            cl_claim is not None
            and cl_claim.pattern_number == number
            and (cl_claim.shape is not None) is cl_has_shape
            and (cl_claim.shape_diagnostic() is not None) is cl_has_shape
            and ("warning_claim_shape" in cl_end_diag) is cl_has_shape
            and ("warning_claim_shape" in cl_end["log"]) is cl_has_shape
            and (
                find_unverified_warning_claim_shape(text) is not None
            ) is (number in (1, 2, 3))
            and (
                find_unverified_warning_claim_shape_for(number, text)
                is not None
            ) is cl_has_shape,
        )

    # The shape finder follows the pattern 3 guard exactly, including
    # its exclusion list and the whether/if exemption.
    def cl_p3_hit(text):
        text = text.translate(_INVISIBLE)

        for match in UNVERIFIED_WARNING_CLAIM_PATTERNS[2].finditer(text):
            lead = text[max(0, match.start() - 40):match.start()]

            if not _CONDITIONAL_LEAD.search(lead):
                return True

        return False

    cl_corpus_total = 0
    cl_corpus_bad = []

    for lead_in in ("", "Verify whether ", "If ", "Note: ", "Heat is high. "):
        for no_word in ("no", "No", "NO"):
            for qualifier in ("active", "Current", "official", "existing",
                              "heat", "immediate"):
                for between in ("", "heat ", "alert or ", "emergency ",
                                "very heat alert ", "a b c "):
                    for term in ("warning", "warnings", "advisory",
                                 "advisories", "emergency instruction",
                                 "Emergency  Instructions"):
                        for tail in ("", " is in effect", " feed", " level",
                                     " thresholds apply", ".", " for Dubai",
                                     " and", "-level", " data source",
                                     " was issued"):
                            text = (
                                f"{lead_in}{no_word} {qualifier} "
                                f"{between}{term}{tail}"
                            )
                            number = find_unverified_warning_claim_pattern(
                                text
                            )
                            found = find_unverified_warning_claim_shape(text)
                            hit = cl_p3_hit(text)
                            cl_corpus_total += 1

                            if (found is not None) is not hit or (
                                number == 3 and found is None
                            ) or (
                                number in (4, 5, None) and found is not None
                            ) or (
                                found is not None
                                and (
                                    "unknown" in found.values()
                                    or found["words_between"]
                                    not in (0, 1, 2)
                                    or found["qualifier"]
                                    not in _CLAIM_QUALIFIERS
                                    or found["term_class"]
                                    not in _CLAIM_TERM_CLASSES
                                    or found["following_word_class"]
                                    not in _CLAIM_FOLLOW_CLASSES
                                )
                            ):
                                cl_corpus_bad.append(len(cl_corpus_bad))

    check(
        f"Shape finder matches the pattern 3 guard on {cl_corpus_total} "
        "generated phrases",
        cl_corpus_total > 10000 and cl_corpus_bad == [],
    )

    # ------------------------------------------------------------
    # limitations and missing_evidence are program-owned.
    # ------------------------------------------------------------
    po_fields = ("limitations", "missing_evidence")
    po_schema = RESPONSE_FORMAT["json_schema"]["schema"]
    po_prompt = " ".join(SYSTEM_PROMPT.split())

    check(
        "Model schema no longer requests either program-owned field",
        all(
            field not in RESPONSE_FIELDS
            and field not in LIST_FIELDS
            and field not in po_schema["properties"]
            and field not in po_schema["required"]
            and f'"{field}"' not in json_text(RESPONSE_FORMAT)
            for field in po_fields
        )
        and len(RESPONSE_FIELDS) == 16
        and sorted(po_schema["properties"]) == sorted(RESPONSE_FIELDS)
        and po_schema["additionalProperties"] is False,
    )

    check(
        "Both fields are program-owned and the prompt no longer asks "
        "for them",
        all(field in PROGRAM_OWNED_FIELDS for field in po_fields)
        and "and limitations" not in po_prompt
        and "in limitations" not in po_prompt
        and "9. Explain missing evidence. The assessment is never a "
        "completed safety clearance." in po_prompt
        and "explain why in its rationale." in po_prompt,
    )

    # Model-supplied versions are rejected as unexpected fields.
    po_mark = "PO-MARK-5521"

    for field in po_fields:
        po_direct = adv_error({**adv_response(), field: [po_mark]})
        po_bad = cl_run(cl_body(**{field: [po_mark]}))
        po_bad_d = po_bad["decision"]

        check(
            f"Model-supplied {field} is rejected as program-owned",
            po_direct
            == f"Model output sets program-owned fields: {field}"
            and [r["code"] for r in po_bad_d["incomplete_reasons"]]
            == ["ASSESSMENT_ERROR"]
            and f"program-owned fields: {field}" in po_bad_d["reasoning"]
            and po_bad_d["assessment_status"] == "INCOMPLETE"
            and po_bad_d["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and po_bad_d["send_alert"] is False
            and po_mark not in json_text(po_bad_d) + po_bad["log"]
            + po_bad["report"],
        )

    # The program fills both lists deterministically.
    po_expected_limitations = [
        "Risk score not calculated because these dimensions could not "
        "be assessed: exposure_score, geographic_impact_score.",
        "The five-dimension risk score is the provisional internal "
        "model in guidelines.md section 5. It is not an official "
        "UAE/NCM classification.",
        "Official NCM and emergency authority warnings were not "
        "retrieved or verified by this implementation.",
    ]
    po_expected_missing = [
        "Evidence required to score: exposure_score, "
        "geographic_impact_score.",
        "Current official NCM and relevant UAE emergency authority "
        "warning status has not been retrieved and verified.",
    ]
    po_run_a = cl_run(cl_body())
    po_run_b = cl_run(cl_body(
        advice="Avoid outdoor work at midday.",
        reasoning="PO-MARK-REASON " + "Heat screening triggered.",
        recommended_actions=["PO-MARK-ACTION"],
    ))
    po_a, po_b = po_run_a["decision"], po_run_b["decision"]

    check(
        "Program populates both lists deterministically from evidence "
        "and validation results",
        po_a["limitations"] == po_expected_limitations
        and po_a["missing_evidence"] == po_expected_missing
        and po_b["limitations"] == po_a["limitations"]
        and po_b["missing_evidence"] == po_a["missing_evidence"]
        and "PO-MARK" not in json_text(
            po_b["limitations"] + po_b["missing_evidence"]
        )
        and po_a["evidence_gate_overrides"]
        == ["exposure_score", "geographic_impact_score"]
        and [r["code"] for r in po_a["incomplete_reasons"]]
        == ["RISK_DIMENSIONS_UNASSESSED"],
    )

    # Present and correctly typed in every kind of assessment.
    po_claim = cl_run(cl_body(reasoning="There are no active warnings."))
    po_error = cl_run(cl_body(**{"limitations": ["x"]}))
    po_incomplete = incomplete_assessment("Assessment could not run.")

    for label, assessment in (
        ("assessment", po_a),
        ("claim failure", po_claim["decision"]),
        ("rejected response", po_error["decision"]),
        ("incomplete_assessment()", po_incomplete),
    ):
        check(
            f"Program-owned lists are present and typed: {label}",
            all(
                type(assessment.get(field)) is list
                and all(type(item) is str for item in assessment[field])
                for field in po_fields
            )
            and assessment["limitations"] != [],
        )

    check(
        "Existing program explanations survive a claim failure",
        po_claim["decision"]["official_warning_status"] == "NOT_CHECKED"
        and "UNVERIFIED_WARNING_CLAIM"
        in [r["code"] for r in po_claim["decision"]["incomplete_reasons"]]
        and any(
            "unverified official-warning claim in reasoning" in item
            for item in po_claim["decision"]["limitations"]
        )
        and po_claim["decision"]["warning_claim_diagnostic"]
        == "field=reasoning;pattern=2",
    )

    # Both lists reach the console report and the Sheets payload.
    po_payload, _ = json.JSONDecoder().raw_decode(
        po_run_a["report"].split("OUTPUT PAYLOAD:", 1)[1].lstrip()
    )
    po_report = po_run_a["report"]

    check(
        "Both lists reach the console report and the Sheets payload",
        json.loads(po_payload["limitations"]) == po_expected_limitations
        and json.loads(po_payload["missing_evidence"])
        == po_expected_missing
        and "LIMITATIONS:" in po_report
        and "MISSING EVIDENCE:" in po_report
        and all(
            f"- {item}" in po_report
            for item in po_expected_limitations + po_expected_missing
        ),
    )

    # The guard still scans every remaining model free-text field.
    for field, value in (
        ("hazard_type", "No official warning is in effect"),
        ("advice", "There are no active warnings."),
        ("reasoning", "No official heat advisory is in effect."),
        ("likelihood_reason", "A heat warning is currently in effect."),
        ("severity_reason", "NCM has not issued any warning."),
        ("duration_reason", "No official warnings were checked."),
        ("recommended_actions", ["There are no warnings."]),
    ):
        po_g = cl_run(cl_body(**{field: value}))["decision"]

        check(
            f"Claim in remaining model field still fails closed: {field}",
            po_g.get("warning_claim_diagnostic") is not None
            and po_g["warning_claim_diagnostic"].startswith(
                f"field={field};"
            )
            and "UNVERIFIED_WARNING_CLAIM"
            in [r["code"] for r in po_g["incomplete_reasons"]]
            and po_g["assessment_status"] == "INCOMPLETE"
            and po_g["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and po_g["send_alert"] is False
            and po_g["approval_status"] == "NOT_AUTHORIZED",
        )

    # ------------------------------------------------------------
    # Text-free shape diagnostic for pattern 2 matches.
    # ------------------------------------------------------------
    def d2_shape(lead, currently, between, term, following):
        return {
            "lead_form": lead, "currently": currently,
            "words_between": between, "term_class": term,
            "following_word_class": following,
        }

    d2_cases = (
        ("There are no warnings for Dubai.",
         d2_shape("there_are", False, 0, "warning", "preposition")),
        ("There is no heat advisory.",
         d2_shape("there_is", False, 1, "advisory", "punctuation")),
        ("There's currently no emergency instruction.",
         d2_shape("there_s", True, 0, "emergency_instruction",
                  "punctuation")),
        ("There\u2019s no warning today",
         d2_shape("there_s", False, 0, "warning", "preposition")),
        ("There was no warning information available.",
         d2_shape("there_was", False, 0, "warning", "evidence_noun")),
        ("There were no advisories, so none apply.",
         d2_shape("there_were", False, 0, "advisory", "punctuation")),
        ("THERE ARE NO WARNINGS and monitoring continues.",
         d2_shape("there_are", False, 0, "warning", "conjunction")),
        ("There is no warning evidence in the supplied inputs.",
         d2_shape("there_is", False, 0, "warning", "evidence_noun")),
        ("There is no advisory text provided.",
         d2_shape("there_is", False, 0, "advisory", "evidence_noun")),
        ("There is no official heat warning.",
         d2_shape("there_is", False, 2, "warning", "punctuation")),
        ("There is no warning that applies.",
         d2_shape("there_is", False, 0, "warning", "conjunction")),
        ("Heat is high; there is no advisory",
         d2_shape("there_is", False, 0, "advisory", "end")),
        ("There is no\u200b warning here",
         d2_shape("there_is", False, 0, "warning", "other_word")),
    )

    for text, expected_shape in d2_cases:
        check(
            f"Pattern 2 shape classified: {expected_shape['lead_form']}/"
            f"{expected_shape['currently']}/"
            f"{expected_shape['words_between']}/"
            f"{expected_shape['term_class']}/"
            f"{expected_shape['following_word_class']}",
            find_unverified_warning_claim_pattern(text) == 2
            and find_unverified_warning_claim_shape_pattern2(text)
            == expected_shape
            and find_unverified_warning_claim_shape_for(2, text)
            == expected_shape,
        )

    check(
        "No pattern 2 shape for conditional, excluded or empty text",
        find_unverified_warning_claim_shape_pattern2(
            "Verify whether there is no warning today."
        ) is None
        and find_unverified_warning_claim_shape_pattern2(
            "There is no warning data in the evidence."
        ) is None
        and find_unverified_warning_claim_shape_pattern2("") is None
        and find_unverified_warning_claim_shape_for(1, "x") is None
        and find_unverified_warning_claim_shape_for(None, "x") is None,
    )

    d2_unknown = d2_shape("unknown", None, None, "unknown", "unknown")

    for label, bad in (
        ("missing", None),
        ("not a dict", "D2-MARK-SHAPE"),
        ("empty", {}),
        ("free text", d2_shape("D2-MARK-L", "yes", 1, "D2-MARK-T",
                               "D2-MARK-F")),
    ):
        got = UnverifiedWarningClaim(
            "m", field_label="reasoning", pattern_number=2, shape=bad
        ).shape_diagnostic()
        want = (
            d2_shape("unknown", None, 1, "unknown", "unknown")
            if label == "free text" else d2_unknown
        )

        check(
            f"Malformed pattern 2 shape fails safely: {label}",
            got == want and "D2-MARK" not in json_text(got),
        )

    for bad_value in (True, "2", -1, 3, 1.0):
        check(
            f"Pattern 2 words-between value rejected: {bad_value!r}",
            UnverifiedWarningClaim(
                "m", "reasoning", 2,
                d2_shape("there_is", False, bad_value, "warning", "end"),
            ).shape_diagnostic()["words_between"] is None,
        )

    check(
        "Pattern 2 'currently' must be a real boolean",
        UnverifiedWarningClaim(
            "m", "reasoning", 2,
            d2_shape("there_is", 1, 0, "warning", "end"),
        ).shape_diagnostic()["currently"] is None
        and UnverifiedWarningClaim(
            "m", "reasoning", 2,
            d2_shape("there_is", True, 0, "warning", "end"),
        ).shape_diagnostic()["currently"] is True,
    )

    # The pattern 2 shape finder follows the pattern 2 guard exactly.
    def d2_p2_hit(text):
        text = text.translate(_INVISIBLE)

        for match in UNVERIFIED_WARNING_CLAIM_PATTERNS[1].finditer(text):
            lead = text[max(0, match.start() - 40):match.start()]

            if not _CONDITIONAL_LEAD.search(lead):
                return True

        return False

    d2_total = 0
    d2_bad = []

    for lead_in in ("", "Verify whether ", "If ", "Note: ", "Heat is high. "):
        for form in ("there is", "there are", "there was", "there were",
                     "there's", "There\u2019s", "THERE IS"):
            for currently in ("", "currently "):
                for between in ("", "heat ", "official heat ", "very heat "
                                "alert ", "a b c "):
                    for term in ("warning", "warnings", "advisory",
                                 "advisories", "emergency instruction",
                                 "Emergency  Instructions"):
                        for tail in ("", " feed", " level", " data source",
                                     ".", " for Dubai", " and",
                                     " information", " evidence",
                                     " is active", " today"):
                            text = (
                                f"{lead_in}{form} {currently}no "
                                f"{between}{term}{tail}"
                            )
                            number = find_unverified_warning_claim_pattern(
                                text
                            )
                            found = find_unverified_warning_claim_shape_pattern2(
                                text
                            )
                            d2_total += 1

                            if (found is not None) is not d2_p2_hit(text) or (
                                number == 2 and found is None
                            ) or (
                                found is not None
                                and (
                                    "unknown" in found.values()
                                    or found["words_between"]
                                    not in (0, 1, 2)
                                    or found["lead_form"]
                                    not in _CLAIM_P2_LEAD_FORMS
                                    or found["term_class"]
                                    not in _CLAIM_TERM_CLASSES
                                    or found["following_word_class"]
                                    not in _CLAIM_FOLLOW_CLASSES
                                    or type(found["currently"]) is not bool
                                )
                            ):
                                d2_bad.append(len(d2_bad))

    check(
        f"Pattern 2 shape finder matches the guard on {d2_total} "
        "generated phrases",
        d2_total > 10000 and d2_bad == [],
    )

    # End to end: the marker never reaches an output, the shape does.
    d2_marker = "D2-SHAPE-MARK-7743"
    d2_run = cl_run(cl_body(
        reasoning=f"{d2_marker} says there is no warning information "
                  "available today."
    ))
    d2_dec = d2_run["decision"]
    d2_everything = (
        json_text(d2_dec) + json.dumps(d2_run["summary"]) + d2_run["log"]
        + d2_run["report"]
    )

    check(
        "Pattern 2 claim failure records its shape and leaks no text",
        d2_dec.get("warning_claim_diagnostic")
        == "field=reasoning;pattern=2"
        and d2_dec["response_diagnostics"].get("warning_claim_shape")
        == d2_shape("there_is", False, 0, "warning", "evidence_noun")
        and "warning_claim_shape" in d2_run["log"]
        and d2_marker not in d2_everything
        and "information available" not in d2_everything
        and d2_dec["assessment_status"] == "INCOMPLETE"
        and d2_dec["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and d2_dec["send_alert"] is False
        and d2_dec["approval_status"] == "NOT_AUTHORIZED",
    )

    # ------------------------------------------------------------
    # Unresolved policy references are split into fixed buckets.
    # ------------------------------------------------------------
    with contextlib.redirect_stdout(io.StringIO()):
        ur_policy = load_guidelines()
        ur_required = select_hazard_guidance(ur_policy, {"HEAT": ["t"]})
        ur_required += select_conditional_sections(
            ur_policy, {"HEAT": ["t"]},
            {"current": {"time": "2026-11-08T12:00"}},
        )
        ur_sections = search_guidelines(
            ur_policy, "x", k=0, required_sections=ur_required
        )

    ur_citations = [
        "6.1 Extreme Heat",
        "13. Advice and Communication Requirements",
        "5. Risk Calculation Model",
        "3. Evidence Quality and Confidence",
        "4.1 Wrong Title",
        "4.9 UR-MARK-IMAGINARY",
        "99. UR-MARK-NOT-A-SECTION",
        "UR-MARK-FREE-TEXT",
    ]
    ur_diag = source_resolution_diagnostics(
        ur_citations, ur_sections, ur_policy
    )
    ur_legacy = source_resolution_diagnostics(ur_citations, ur_sections)

    check(
        "Unresolved numbered references are split into fixed buckets",
        ur_diag == {
            "source_count": 8,
            "resolution_counts": {
                "exact": 2,
                "policy_section_not_retrieved": 2,
                "retrieved_number_title_mismatch": 1,
                "section_number_not_in_policy": 2,
                "unmatched": 1,
            },
        }
        and "unretrieved_section_reference"
        not in ur_diag["resolution_counts"]
        and "UR-MARK" not in json_text(ur_diag),
    )

    check(
        "Without the policy the original lumped label is unchanged",
        ur_legacy == {
            "source_count": 8,
            "resolution_counts": {
                "exact": 2, "unmatched": 1,
                "unretrieved_section_reference": 5,
            },
        },
    )

    check(
        "Policy section numbers come from the loaded policy only",
        {"1", "5", "10", "16", "4.1", "4.5", "6.1", "6.5"}
        <= policy_section_numbers(ur_policy)
        and not {"4.9", "99", "17"} & policy_section_numbers(ur_policy)
        and policy_section_numbers({}) == set(),
    )

    ur_run = cl_run(cl_body(
        reasoning="There are no active warnings.",
        sources=ur_citations,
    ))
    ur_counts = ur_run["decision"].get("response_diagnostics", {}).get(
        "source_resolution", {}
    ).get("resolution_counts", {})

    check(
        "A claim failure records the unresolved-reference buckets",
        ur_counts.get("policy_section_not_retrieved") == 2
        and ur_counts.get("section_number_not_in_policy") == 2
        and ur_counts.get("retrieved_number_title_mismatch") == 1
        and sum(ur_counts.values()) == 8
        and "UR-MARK" not in json_text(ur_run["decision"])
        + ur_run["log"] + ur_run["report"]
        and ur_run["decision"].get("warning_claim_diagnostic")
        == "field=reasoning;pattern=2",
    )

    check(
        "Unresolved references still never validate as sources",
        ur_run["decision"]["sources"]
        == ["6.1 Extreme Heat",
            "13. Advice and Communication Requirements"]
        and resolve_policy_sources(ur_citations, ur_sections)
        == ["6.1 Extreme Heat",
            "13. Advice and Communication Requirements"],
    )

    # Prompt: approved sentence and rules present exactly once.
    cl_prompt = " ".join(SYSTEM_PROMPT.split())

    check(
        "Prompt carries the warning-mention ban and status rule once",
        cl_prompt.count("Official warning status is NOT_CHECKED "
                        "(program-reported)") == 1
        and cl_prompt.count("Free text must not use warning, warnings, "
                            "advisory or advisories, or claim their "
                            "existence, absence, availability or "
                            "status") == 1
        and "Do not mention official warnings" not in cl_prompt
        and "Never write the words" not in cl_prompt
        and cl_prompt.count("Open-Meteo is not an official UAE warning "
                            "authority") == 1
        and "Official warning status was not checked" not in cl_prompt
        and "if mentioned" not in cl_prompt
        and "active or inactive" not in cl_prompt,
    )

    # 7. Budgets: the limits are unchanged. The revised prompt's fit is
    # proved by the realistic-history heat checks above, which build
    # their request with this SYSTEM_PROMPT.
    check(
        "Request and token budgets are unchanged by the prompt revision",
        MAX_ESTIMATED_REQUEST_TOKENS == 8000
        and MAX_REQUEST_CHARS == 19000
        and MAX_POLICY_CHARS == 13500
        and MAX_COMPLETION_TOKENS == 1000,
    )

    # ------------------------------------------------------------
    # Variant B: the model is told not to discuss official warnings;
    # the program reports the status (self-contained, offline).
    # ------------------------------------------------------------
    def vb_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)

    def vb_weather():
        start = datetime(2026, 11, 8, 13, 0)
        return {
            "city": CITY, "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:00:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 33,
                "apparent_temperature": 36, "precipitation": 0,
                "wind_speed_10m": 10, "wind_gusts_10m": 20,
                "visibility": 24140, "weather_code": 1, "cape": 0,
            },
            "today_forecast": {"high": 35, "low": 27, "wind": 14,
                               "gust": 25, "precipitation": 0},
            "previous_days": [],
            "hourly_forecast_next_24h": [{
                "time": (start + timedelta(hours=h)).strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "temperature": 33, "apparent_temperature": 36,
                "humidity": 55, "precipitation": 0, "wind": 10,
                "gust": 20, "visibility": 24140, "weather_code": 1,
                "cape": 0,
            } for h in range(24)],
            "history_summary": {},
        }

    vb_sent = []

    def vb_capture(messages, temperature=0.1):
        vb_sent.append(messages)
        raise RuntimeError("captured")

    with contextlib.redirect_stdout(io.StringIO()):
        vb_policy = load_guidelines()
        vb_required = select_hazard_guidance(vb_policy, {"HEAT": ["t"]})
        vb_required += select_conditional_sections(
            vb_policy, {"HEAT": ["t"]},
            {"current": {"time": "2026-11-08T12:00"}},
        )
        vb_sections = search_guidelines(
            vb_policy, "x", k=0, required_sections=vb_required
        )

    try:
        vb_override(
            {"ask_groq": vb_capture},
            lambda: assess_weather(vb_weather(), [], vb_policy, vb_sections),
        )
    except RuntimeError:
        pass

    vb_user = vb_sent[0][1]["content"] if vb_sent else ""
    vb_lead_in = vb_user.split("\n\n=== POLICY SECTION")[0]
    vb_evidence = json.loads(vb_user.split("EVIDENCE AND STATUS (JSON):\n")[1])
    vb_prompt = " ".join(SYSTEM_PROMPT.split())

    # Prompt instruction validation.
    check(
        "Rule 4 wording is exactly the approved text",
        "4. Official warning status is NOT_CHECKED (program-reported). "
        "Free text must not use warning, warnings, advisory or "
        "advisories, or claim their existence, absence, availability or "
        "status; describe evidence limits without that. Follow the "
        "schema. Open-Meteo is not an official UAE warning authority. "
        "5. " in vb_prompt,
    )

    # Evidence status validation.
    check(
        "Evidence status is the short program-reported value",
        vb_evidence["official_warning_status"]
        == "NOT_CHECKED (program-reported)"
        and "feed" not in json_text(vb_evidence["official_warning_status"])
        and "No NCM" not in vb_user,
    )

    # Nothing the model is shown trips the unchanged guard.
    vb_fixed_text = " ".join(
        [SYSTEM_PROMPT, vb_lead_in]
        + [json_text({
            key: value for key, value in vb_evidence.items()
            if key not in ("weather", "recent_agent_history")
        })]
    )
    vb_sentences = re.split(r"(?<=[.!?])\s+", " ".join(vb_fixed_text.split()))

    check(
        "No fixed prompt or evidence sentence is flagged by the guard",
        [s for s in vb_sentences
         if find_unverified_warning_claim_pattern(s)] == [],
    )

    check(
        "No retrieved policy line or canonical title is flagged",
        [
            line for section in vb_sections
            for line in section["text"].splitlines()
            if find_unverified_warning_claim_pattern(line)
        ] == []
        and [
            section["section"] for section in vb_sections
            if find_unverified_warning_claim_pattern(section["section"])
        ] == [],
    )

    # Known false-positive paraphrases stay flagged (guard unchanged).
    vb_known = (
        ("No official warning information is integrated.", 3),
        ("No official warnings are integrated.", 3),
        ("No official warning is integrated, so status is unknown.", 3),
        ("Official warning status is NOT_CHECKED, so no official "
         "warnings are claimed.", 3),
        ("No official heat advisories were available for review.", 3),
        ("No official heat advisory information was retrieved.", 3),
        ("No official emergency instructions were retrieved.", 3),
        ("We do not claim any official warning is active or inactive.", 4),
    )

    for text, number in vb_known:
        check(
            f"Known paraphrase stays flagged (pattern {number}): "
            f"{text[:46]}",
            find_unverified_warning_claim_pattern(text) == number,
        )

    for text in (
        "No official warning feed is integrated.",
        "Official warning status is unknown.",
        "Warnings have not been verified by this program.",
    ):
        check(
            f"Wording outside the patterns still passes: {text[:46]}",
            find_unverified_warning_claim_pattern(text) is None,
        )

    # Genuine warning claims remain blocked.
    for text in (
        "No official heat advisory is in effect.",
        "There are no active warnings.",
        "A heat warning is currently in effect.",
        "NCM has not issued any warning.",
    ):
        check(
            f"Genuine claim stays blocked: {text}",
            find_unverified_warning_claim(text),
        )

    # Program-generated status survives a response with no warning
    # wording, and the unchanged guard still rejects wording about it.
    class VbResponse:
        status_code = 200
        text = ""
        headers = {}

        def __init__(self, content):
            self.content = content

        def json(self):
            return {
                "choices": [{
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    class VbTransport:
        def __init__(self, content):
            self.content = content
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(url)
            return VbResponse(self.content)

    def vb_reply(**changes):
        body = {
            "hazard_type": "Extreme Heat",
            "likelihood_score": 3, "likelihood_reason": "lr",
            "severity_score": 3, "severity_reason": "sr",
            "exposure_score": 3, "exposure_reason": "er",
            "duration_score": 3, "duration_reason": "dr",
            "geographic_impact_score": 3,
            "geographic_impact_reason": "gr",
            "alert_recommendation": "MONITOR",
            "advice": "Limit strenuous outdoor activity in the afternoon.",
            "reasoning": "Apparent temperature reaches the heat "
                         "screening value.",
            "recommended_actions": ["Schedule outdoor work in the morning"],
            "sources": ["6.1 Extreme Heat"],
        }
        body.update(changes)
        return json_text(body)

    def vb_run(content):
        transport = VbTransport(content)

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                return run_weather_agent()[0]

        return vb_override(
            {
                "HTTP": transport,
                "GROQ_API_KEY": "gsk_vb_test_key",
                "require_configuration": (
                    lambda require_sheet=False: None
                ),
                "get_weather": lambda city: vb_weather(),
                "tool_get_history": lambda n=3: [],
            },
            run,
        )

    vb_ok = vb_run(vb_reply())

    check(
        "Program-generated warning status is kept without model wording",
        vb_ok["official_warning_status"] == "NOT_CHECKED"
        and any(
            "warning status has not been retrieved and verified" in item
            for item in vb_ok["missing_evidence"]
        )
        and any(
            "warnings were not retrieved or verified" in item
            for item in vb_ok["limitations"]
        )
        and any(
            "warning status has not been retrieved and verified" in item
            for item in vb_ok["missing_evidence"]
        )
        and vb_ok["assessment_status"] == "INCOMPLETE"
        and vb_ok["send_alert"] is False
        and vb_ok["response_diagnostics"] == {},
    )

    check(
        "Response with no warning wording passes the guard",
        "UNVERIFIED_WARNING_CLAIM"
        not in [r["code"] for r in vb_ok["incomplete_reasons"]]
        and vb_ok["advice"]
        == "Limit strenuous outdoor activity in the afternoon.",
    )

    vb_bad = vb_run(vb_reply(
        reasoning="No official warnings are integrated, so none apply."
    ))

    check(
        "Warning wording in reasoning still fails closed with a diagnostic",
        vb_bad.get("warning_claim_diagnostic") == "field=reasoning;pattern=3"
        and vb_bad["assessment_status"] == "INCOMPLETE"
        and vb_bad["alert_recommendation"] == "UNABLE_TO_ASSESS"
        and vb_bad["risk_score"] is None
        and vb_bad["send_alert"] is False,
    )

    # Budgets with the revised prompt.
    check(
        "Budget limits are unchanged by Variant B",
        MAX_ESTIMATED_REQUEST_TOKENS == 8000
        and MAX_REQUEST_CHARS == 19000
        and MAX_POLICY_CHARS == 13500
        and MAX_COMPLETION_TOKENS == 1000,
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

    # ------------------------------------------------------------
    # B1. Preservation pins. Every expected value is a literal computed
    # from an isolated copy of be5524a (git show), never from the code
    # under test. Offline: transport-level stubs only.
    # ------------------------------------------------------------
    # Shared offline harness for the preservation pins and end-to-end tests.
    # Transport-level stubs only: nothing here reaches a network. The
    # latitude, longitude and utc_offset_seconds in ip_open_meteo() are
    # synthetic test values, not evidence of what Open-Meteo returns.
    IP_SHEET = "https://sheet.invalid/exec"
    IP_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


    def ip_weather():
        start = datetime(2026, 11, 8, 13, 0)

        return {
            "city": "Dubai",
            "timezone": "Asia/Dubai",
            "source": "Open-Meteo",
            "retrieved_at": "2026-11-08T12:05:00+04:00",
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 33,
                "relative_humidity_2m": 55, "apparent_temperature": 36,
                "precipitation": 0, "wind_speed_10m": 10,
                "wind_gusts_10m": 20, "visibility": 24140,
                "weather_code": 1, "cape": 0,
            },
            "today_forecast": {
                "high": 35, "low": 27, "wind": 14, "gust": 25,
                "precipitation": 0,
            },
            "previous_days": [],
            "hourly_forecast_next_24h": [
                {
                    "time": (start + timedelta(hours=h)).strftime(
                        "%Y-%m-%dT%H:%M"
                    ),
                    "temperature": 33, "apparent_temperature": 36,
                    "humidity": 55, "precipitation": 0, "wind": 10,
                    "gust": 20, "visibility": 24140, "weather_code": 1,
                    "cape": 0,
                }
                for h in range(24)
            ],
            "history_summary": {},
        }


    def ip_open_meteo(timezone="Asia/Dubai"):
        hours = [datetime(2026, 11, 8) + timedelta(hours=h) for h in range(48)]
        days = [datetime(2026, 11, 3) + timedelta(days=d) for d in range(7)]
        n = len(hours)

        return {
            "latitude": 25.2, "longitude": 55.25, "timezone": timezone,
            "utc_offset_seconds": 14400,
            "current": {
                "time": "2026-11-08T12:00", "temperature_2m": 33.0,
                "relative_humidity_2m": 55, "apparent_temperature": 36.0,
                "precipitation": 0.0, "wind_speed_10m": 10.0,
                "wind_gusts_10m": 20.0, "visibility": 24140.0,
                "weather_code": 1, "cape": 0.0,
            },
            "hourly": {
                "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
                "temperature_2m": [33.0] * n,
                "relative_humidity_2m": [55] * n,
                "apparent_temperature": [36.0] * n,
                "precipitation": [0.0] * n,
                "wind_speed_10m": [10.0] * n,
                "wind_gusts_10m": [20.0] * n,
                "visibility": [24140.0] * n,
                "weather_code": [1] * n,
                "cape": [0.0] * n,
            },
            "daily": {
                "time": [d.strftime("%Y-%m-%d") for d in days],
                "temperature_2m_max": [35.0] * 7,
                "temperature_2m_min": [27.0] * 7,
                "wind_speed_10m_max": [14.0] * 7,
                "wind_gusts_10m_max": [25.0] * 7,
                "precipitation_sum": [0.0] * 7,
            },
        }


    IP_REPLY = {
        "hazard_type": "Extreme Heat",
        "likelihood_score": 3,
        "likelihood_reason": "Forecast highs reach 35 C.",
        "severity_score": 3,
        "severity_reason": "Heat stress is possible outdoors.",
        "exposure_score": 4,
        "exposure_reason": "Outdoor workers.",
        "duration_score": 4,
        "duration_reason": "Hot for most daylight hours.",
        "geographic_impact_score": 4,
        "geographic_impact_reason": "City-wide.",
        "alert_recommendation": "MONITOR",
        "advice": "Limit strenuous outdoor activity at midday.",
        "reasoning": "Heat screening triggered by forecast highs.",
        "recommended_actions": ["Drink water."],
        "sources": ["6.1 Extreme Heat"],
    }


    class IpResponse:
        def __init__(self, status_code, text, headers=None):
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

        def json(self):
            return json.loads(self.text)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")


    def ip_completion(body, finish_reason="stop"):
        return IpResponse(200, json.dumps({
            "choices": [{
                "message": {"content": json.dumps(body)},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 4000, "completion_tokens": 600,
                "total_tokens": 4600,
            },
        }))


    def ip_provider_error(status, error):
        return IpResponse(status, json.dumps({"error": error}))


    class IpTransport:
        def __init__(self, groq=None, weather=None, history="[]",
                     sheets="OK", get_error=None, post_error=None):
            self.groq = groq
            self.weather = weather
            self.history = history
            self.sheets = sheets
            self.get_error = get_error
            self.post_error = post_error
            self.calls = []
            self.posts = []
            self.weather_params = None

        def get(self, url, params=None, timeout=None):
            self.calls.append(("GET", url))

            if url == IP_WEATHER_URL:
                self.weather_params = dict(params or {})

                if self.get_error is not None:
                    raise self.get_error

                return self.weather

            return IpResponse(200, self.history)

        def post(self, url, headers=None, json=None, timeout=None):
            self.calls.append(("POST", url))
            self.posts.append((url, json))

            if url == GROQ_URL:
                if self.post_error is not None:
                    raise self.post_error

                return self.groq

            return IpResponse(200, self.sheets)


    def ip_override(overrides, function):
        saved = {name: globals()[name] for name in overrides}
        globals().update(overrides)

        try:
            return function()
        finally:
            globals().update(saved)


    def ip_job(transport, dry=True, extra=None, stub_weather=True):
        """Run run_agent_job offline. Returns stdout, payload and result."""
        overrides = {
            "HTTP": transport,
            "GROQ_API_KEY": "gsk_pin_test",
            "SHEET_URL": None,
        }

        if stub_weather:
            overrides["get_weather"] = lambda city: ip_weather()
            overrides["tool_get_history"] = lambda n=3: []

        overrides.update(extra or {})
        out = io.StringIO()
        result = {}

        def run():
            with contextlib.redirect_stdout(out):
                result["value"] = run_agent_job(dry=dry)

        ip_override(overrides, run)
        text = out.getvalue()
        payload = None

        if "OUTPUT PAYLOAD:" in text:
            payload, _ = json.JSONDecoder().raw_decode(
                text.split("OUTPUT PAYLOAD:", 1)[1].lstrip()
            )
        else:
            for url, body in transport.posts:
                if url == IP_SHEET:
                    payload = body

        return {"stdout": text, "payload": payload, "value": result.get("value")}


    def ip_payload_digest(payload):
        stable = {
            key: value for key, value in payload.items()
            if key != "assessment_timestamp"
        }

        return hashlib.sha256(
            json.dumps(stable, sort_keys=True, ensure_ascii=False)
            .encode("utf-8")
        ).hexdigest()


    def ip_groq_body(transport):
        for url, body in transport.posts:
            if url == GROQ_URL:
                return body

        return None

    def ip_agent(transport, extra=None):
        overrides = {
            "HTTP": transport,
            "GROQ_API_KEY": "gsk_pin_test",
            "SHEET_URL": None,
            "get_weather": lambda city: ip_weather(),
            "tool_get_history": lambda n=3: [],
        }
        overrides.update(extra or {})
        holder = {}

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                holder["decision"] = run_weather_agent()[0]

        ip_override(overrides, run)
        return holder["decision"]

    def ip_sha(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    IP_BASELINE = {
        "system_prompt_sha": "7c3a87b5871a04dbd855c1f7fde4520a87c4fc90af591292be078deaa5739481",
        "response_format_sha": "ca0d29d77c08dd5f0ea03ef140952e36187778a2abf5153d34e9fc4fa913f315",
        "response_fields": ["hazard_type", "likelihood_score", "likelihood_reason", "severity_score", "severity_reason", "exposure_score", "exposure_reason", "duration_score", "duration_reason", "geographic_impact_score", "geographic_impact_reason", "alert_recommendation", "advice", "reasoning", "recommended_actions", "sources"],
        "groq_params": {
            "model": "openai/gpt-oss-20b",
            "temperature": 0.1,
            "max_completion_tokens": 1000,
            "reasoning_effort": "low",
            "include_reasoning": False,
        },
        "request_len": 15920,
        "request_sha": "0813e66e657d3c5baad752378caa821afe270eb44e0bca4fb71abcb044223293",
        "payload_keys": ["advice", "alert_message", "alert_recommendation", "approval_status", "assessment_status", "assessment_timestamp", "city", "duration_reason", "duration_score", "evidence_confidence", "evidence_quality", "exposure_reason", "exposure_score", "geographic_impact_reason", "geographic_impact_score", "hazard_type", "likelihood_reason", "likelihood_score", "limitations", "missing_evidence", "notification_status", "official_warning_status", "policy_hash", "policy_status", "policy_version", "reasoning", "risk_band", "risk_score", "send_alert", "severity_reason", "severity_score", "sources", "tools_used", "type"],
        "success_payload_sha": "39871a324b2591bfc3cd2eef512c2ad10a87f3960c5efe4c421bada826b5c2bb",
        "provider_payload_sha": "e2cb76ac2158e55b2d5deb156756e6d690e7272705022a9693517832f3d11307",
        "claim_payload_sha": "6a67d857a282c9c4c6e12b0367d0ca5bc1dd3467f297423a6c32b25c6bd0683d",
        "dry_calls": [
            ("GET", "https://api.open-meteo.com/v1/forecast"),
            ("GET", "https://sheet.invalid/exec"),
            ("POST", "https://api.groq.com/openai/v1/chat/completions"),
        ],
        "live_calls": [
            ("GET", "https://api.open-meteo.com/v1/forecast"),
            ("GET", "https://sheet.invalid/exec"),
            ("POST", "https://api.groq.com/openai/v1/chat/completions"),
            ("POST", "https://sheet.invalid/exec"),
        ],
    }

    check(
        "Pin: system prompt is byte-identical to be5524a",
        ip_sha(SYSTEM_PROMPT) == IP_BASELINE["system_prompt_sha"],
    )

    check(
        "Pin: response format is byte-identical to be5524a",
        ip_sha(json_text(RESPONSE_FORMAT))
        == IP_BASELINE["response_format_sha"],
    )

    check(
        "Pin: the 16 response fields are unchanged",
        list(RESPONSE_FIELDS) == IP_BASELINE["response_fields"]
        and len(RESPONSE_FIELDS) == 16,
    )

    ip_ok_transport = IpTransport(groq=ip_completion(IP_REPLY))
    ip_ok = ip_job(ip_ok_transport)
    ip_ok_body = ip_groq_body(ip_ok_transport) or {}

    check(
        "Pin: Groq model, temperature and request parameters are unchanged",
        {
            key: value for key, value in ip_ok_body.items()
            if key not in ("messages", "response_format")
        } == IP_BASELINE["groq_params"]
        and ip_sha(json_text(ip_ok_body.get("response_format")))
        == IP_BASELINE["response_format_sha"]
        and MODEL == "openai/gpt-oss-20b",
    )

    ip_request = serialized_request(ip_ok_body.get("messages", []))

    check(
        "Pin: frozen-fixture Groq request is byte-identical to be5524a",
        len(ip_request) == IP_BASELINE["request_len"]
        and ip_sha(ip_request) == IP_BASELINE["request_sha"],
    )

    check(
        "Pin: payload keys are unchanged",
        ip_ok["payload"] is not None
        and sorted(ip_ok["payload"]) == IP_BASELINE["payload_keys"],
    )

    check(
        "Pin: success payload values are unchanged",
        ip_payload_digest(ip_ok["payload"])
        == IP_BASELINE["success_payload_sha"],
    )

    ip_provider = ip_job(IpTransport(groq=ip_provider_error(400, {
        "message": "IP-MARK provider text",
        "type": "invalid_request_error",
        "code": "json_validate_failed",
        "failed_generation": "IP-MARK-FG",
    })))

    check(
        "Pin: provider-error payload keys and values are unchanged",
        sorted(ip_provider["payload"]) == IP_BASELINE["payload_keys"]
        and ip_payload_digest(ip_provider["payload"])
        == IP_BASELINE["provider_payload_sha"],
    )

    ip_claim = ip_job(IpTransport(groq=ip_completion(
        dict(IP_REPLY, reasoning="No official warnings were checked.")
    )))

    check(
        "Pin: warning-claim payload keys and values are unchanged",
        sorted(ip_claim["payload"]) == IP_BASELINE["payload_keys"]
        and ip_payload_digest(ip_claim["payload"])
        == IP_BASELINE["claim_payload_sha"],
    )

    ip_payload_literal = {}

    for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == "run_agent_job":
            for item in ast.walk(node):
                if (
                    isinstance(item, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "payload"
                        for target in item.targets
                    )
                    and isinstance(item.value, ast.Dict)
                ):
                    for key, value in zip(item.value.keys, item.value.values):
                        if isinstance(key, ast.Constant):
                            ip_payload_literal[key.value] = value

    check(
        "Pin: payload hard-codes send_alert False and an empty alert message",
        isinstance(ip_payload_literal.get("send_alert"), ast.Constant)
        and ip_payload_literal["send_alert"].value is False
        and isinstance(ip_payload_literal.get("alert_message"), ast.Constant)
        and ip_payload_literal["alert_message"].value == "",
    )

    ip_ok_decision = ip_agent(IpTransport(groq=ip_completion(IP_REPLY)))
    ip_fail_decision = ip_agent(IpTransport(groq=ip_provider_error(500, {
        "type": "server_error",
    })))

    check(
        "Pin: success and failure decisions never authorize an alert",
        ip_ok_decision["send_alert"] is False
        and ip_ok_decision["approval_status"] == "PENDING_REVIEW"
        and ip_ok_decision["notification_status"] == "NOT_SENT"
        and ip_ok_decision["alert_message"] == ""
        and ip_fail_decision["send_alert"] is False
        and ip_fail_decision["approval_status"] == "NOT_AUTHORIZED"
        and ip_fail_decision["notification_status"] == "NOT_SENT",
    )

    ip_forced = dict(ip_ok_decision)
    ip_forced.update(
        assessment_status="COMPLETE", send_alert=True,
        alert_message="IP-MARK alert", approval_status="APPROVED",
    )
    ip_forced_run = ip_job(IpTransport(), extra={
        "run_weather_agent": lambda: (dict(ip_forced), ["load_guidelines"]),
    })

    check(
        "Pin: a forced COMPLETE and approved decision still sends no alert",
        ip_forced_run["payload"]["send_alert"] is False
        and ip_forced_run["payload"]["alert_message"] == ""
        and "IP-MARK" not in json_text(ip_forced_run["payload"]),
    )

    ip_weather_ok = IpResponse(200, json.dumps(ip_open_meteo()))
    ip_dry_transport = IpTransport(
        groq=ip_completion(IP_REPLY), weather=ip_weather_ok
    )
    ip_dry = ip_job(
        ip_dry_transport, stub_weather=False, extra={"SHEET_URL": IP_SHEET}
    )

    check(
        "Pin: dry run calls weather, history and Groq only (no Sheets POST)",
        ip_dry_transport.calls == IP_BASELINE["dry_calls"]
        and ip_dry["payload"] is not None
        and "DRY RUN: No data was sent to Google Sheets." in ip_dry["stdout"],
    )

    ip_live_transport = IpTransport(
        groq=ip_completion(IP_REPLY), weather=ip_weather_ok
    )
    ip_job(
        ip_live_transport, dry=False, stub_weather=False,
        extra={"SHEET_URL": IP_SHEET},
    )

    check(
        "Pin: live-mode control posts exactly once to Sheets",
        ip_live_transport.calls == IP_BASELINE["live_calls"],
    )

    # ------------------------------------------------------------
    # B2. Typed failures: built-in base classes and report text kept.
    # ------------------------------------------------------------
    b2_types = {
        ConfigurationError: ValueError,
        PolicyUnavailableError: FileNotFoundError,
        PolicyInvalidError: ValueError,
        WeatherUnavailableError: ValueError,
        RequestBudgetError: ValueError,
        ResponseInvalidError: ValueError,
        UnverifiedWarningClaim: ValueError,
    }

    check(
        "B2: every typed failure keeps its built-in base class",
        all(
            issubclass(cls, base) and issubclass(cls, ClassifiedFailure)
            for cls, base in b2_types.items()
        ),
    )

    check(
        "B2: classified failures report under their baseline names",
        all(
            exception_label(cls("x")) == base.__name__
            for cls, base in b2_types.items()
            if cls is not UnverifiedWarningClaim
        )
        and exception_label(UnverifiedWarningClaim("x"))
        == "UnverifiedWarningClaim"
        and exception_label(TypeError("x")) == "TypeError"
        and not isinstance(TypeError("x"), ClassifiedFailure),
    )

    def b2_raised(function):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                function()
        except Exception as exc:
            return exc

        return None

    b2_dir = Path(tempfile.mkdtemp())
    (b2_dir / "empty.md").write_text("", encoding="utf-8")

    b2_cases = {
        "missing GROQ key": (
            lambda: ip_override({"GROQ_API_KEY": None},
                                lambda: require_configuration()),
            ConfigurationError,
        ),
        "city outside the allowed list": (
            lambda: get_weather("Atlantis"), ConfigurationError,
        ),
        "missing guidelines file": (
            lambda: ip_override({"GUIDELINES_PATH": b2_dir / "none.md"},
                                load_guidelines),
            PolicyUnavailableError,
        ),
        "empty guidelines file": (
            lambda: ip_override({"GUIDELINES_PATH": b2_dir / "empty.md"},
                                load_guidelines),
            PolicyInvalidError,
        ),
        "unparseable model JSON": (
            lambda: parse_json_response("not json"), ResponseInvalidError,
        ),
        "invalid assessment object": (
            lambda: validate_decision([], []), ResponseInvalidError,
        ),
        "request over the token budget": (
            lambda: ip_override(
                {"GROQ_API_KEY": "gsk_pin_test", "HTTP": IpTransport()},
                lambda: ask_groq([{"role": "user", "content": "x" * 30000}]),
            ),
            RequestBudgetError,
        ),
        "weather response without hourly data": (
            lambda: ip_override(
                {"HTTP": IpTransport(weather=IpResponse(200, json.dumps(
                    {"current": {"time": "2026-11-08T12:00"}, "daily": {}}
                )))},
                lambda: get_weather("Dubai"),
            ),
            WeatherUnavailableError,
        ),
    }

    for label, (function, expected) in b2_cases.items():
        found = b2_raised(function)
        check(
            f"B2: {label} raises {expected.__name__}",
            type(found) is expected,
        )

    b2_network = requests.ConnectionError("IP-NET connection refused")
    b2_wrapped = b2_raised(lambda: ip_override(
        {"HTTP": IpTransport(get_error=b2_network)},
        lambda: get_weather("Dubai"),
    ))

    check(
        "B2: weather transport errors are classified with unchanged text",
        isinstance(b2_wrapped, WeatherUnavailableError)
        and exception_label(b2_wrapped) == "ConnectionError"
        and str(b2_wrapped) == str(b2_network)
        and b2_wrapped.__cause__ is b2_network,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        b2_policy = load_guidelines()

    b2_internal = b2_raised(
        lambda: select_hazard_guidance(b2_policy, {"BOGUS": ["x"]})
    )

    check(
        "B2: internal consistency checks stay unclassified",
        type(b2_internal) is ValueError
        and not isinstance(b2_internal, ClassifiedFailure),
    )

    # ------------------------------------------------------------
    # B3. Provider classification through ask_groq (transport stub).
    # ------------------------------------------------------------
    def b3_ask(transport):
        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                ask_groq([{"role": "user", "content": "x"}])

        try:
            ip_override(
                {"HTTP": transport, "GROQ_API_KEY": "gsk_pin_test"}, run
            )
        except Exception as exc:
            return exc

        return None

    def b3_error(code, message="IP-MARK provider text"):
        return {
            "message": message, "type": "invalid_request_error",
            "code": code, "failed_generation": "IP-MARK-FG",
        }

    b3_cases = {
        "completion-limit 400": (
            ip_provider_error(400, b3_error(
                "json_validate_failed", "max completion tokens reached "
                "before generating a valid document IP-MARK")),
            "OUTPUT_TRUNCATED", "INVALID", 400,
        ),
        "other json_validate_failed": (
            ip_provider_error(400, b3_error("json_validate_failed")),
            "OUTPUT_VALIDATION_UNSPECIFIED", "INVALID", 400,
        ),
        "401": (ip_provider_error(401, b3_error("invalid_api_key")),
                "AUTHENTICATION_FAILED", "UNAVAILABLE", 401),
        "403": (ip_provider_error(403, {}),
                "AUTHENTICATION_FAILED", "UNAVAILABLE", 403),
        "413": (ip_provider_error(413, {}),
                "REQUEST_TOO_LARGE", "UNAVAILABLE", 413),
        "429": (ip_provider_error(429, b3_error("rate_limit_exceeded")),
                "RATE_LIMITED", "UNAVAILABLE", 429),
        "498": (ip_provider_error(498, {}),
                "RATE_LIMITED", "UNAVAILABLE", 498),
        "422": (ip_provider_error(422, {}),
                "INVALID_REQUEST", "UNAVAILABLE", 422),
        "424": (ip_provider_error(424, {}),
                "PROVIDER_UNAVAILABLE", "UNAVAILABLE", 424),
        "500": (ip_provider_error(500, {"type": "server_error"}),
                "PROVIDER_UNAVAILABLE", "UNAVAILABLE", 500),
        "503 plain text": (IpResponse(503, "IP-MARK unavailable"),
                           "PROVIDER_UNAVAILABLE", "UNAVAILABLE", 503),
        "unknown status": (ip_provider_error(600, b3_error("weird_code")),
                           "PROVIDER_ERROR_UNCLASSIFIED", "UNAVAILABLE", 600),
        "finish_reason length": (ip_completion(IP_REPLY, "length"),
                                 "OUTPUT_TRUNCATED", "INVALID", None),
        "finish_reason other": (ip_completion(IP_REPLY, "content_filter"),
                                "RESPONSE_MALFORMED", "INVALID", None),
        "no choices": (IpResponse(200, json.dumps({"x": "IP-MARK"})),
                       "RESPONSE_MALFORMED", "INVALID", None),
        "empty content": (
            IpResponse(200, json.dumps({"choices": [{
                "message": {"content": "   "}, "finish_reason": "stop",
            }]})),
            "RESPONSE_MALFORMED", "INVALID", None,
        ),
    }

    for label, (response, category, outcome, status) in b3_cases.items():
        found = b3_ask(IpTransport(groq=response))
        check(
            f"B3: {label} is classified {category}",
            isinstance(found, ProviderError)
            and found.category == category
            and found.outcome == outcome
            and found.http_status == status
            and exception_label(found) == "RuntimeError"
            and "IP-MARK" not in str(found)
            and "IP-MARK" not in json_text(found.diagnostic()),
        )

    check(
        "B3: only the completion-limit message sets the reason label",
        b3_ask(IpTransport(groq=b3_cases["completion-limit 400"][0])).reason
        == "completion_limit_reached"
        and b3_ask(IpTransport(
            groq=b3_cases["other json_validate_failed"][0]
        )).reason is None,
    )

    check(
        "B3: unknown codes are named only when identifier-shaped",
        b3_ask(IpTransport(groq=b3_cases["unknown status"][0])).code
        == "weird_code"
        and b3_ask(IpTransport(groq=ip_provider_error(
            400, b3_error("IP MARK not an identifier")
        ))).code is None,
    )

    for label, error in {
        "connection error": requests.ConnectionError("IP-NET refused"),
        "timeout": requests.Timeout("IP-NET timed out"),
    }.items():
        found = b3_ask(IpTransport(post_error=error))
        check(
            f"B3: network {label} is NETWORK_ERROR with fixed text",
            isinstance(found, ProviderError)
            and found.category == "NETWORK_ERROR"
            and found.outcome == "UNAVAILABLE"
            and "IP-NET" not in str(found)
            and str(found) == "Groq request failed before a response was "
            "received (" + type(error).__name__ + ").",
        )

    b3_decision = ip_agent(IpTransport(groq=b3_cases["completion-limit 400"][0]))

    check(
        "B3: the failure decision records the structured provider error",
        b3_decision.get("provider_error") == {
            "category": "OUTPUT_TRUNCATED", "http_status": 400,
            "code": "json_validate_failed", "type": "invalid_request_error",
            "reason": "completion_limit_reached",
        }
        and "IP-MARK" not in json_text(b3_decision)
        and b3_decision["send_alert"] is False,
    )

    check(
        "B3: the model cannot supply provider_error",
        "provider_error" in PROGRAM_OWNED_FIELDS
        and isinstance(
            b2_raised(lambda: validate_decision(
                dict(IP_REPLY, provider_error={"category": "x"}), []
            )),
            ResponseInvalidError,
        ),
    )

    # ------------------------------------------------------------
    # C. Dubai enforcement (configuration and request) and observe-only
    # response diagnostics. require_configuration and get_weather are
    # deliberately not stubbed here.
    # ------------------------------------------------------------
    check(
        "C: Dubai is the only configured location",
        CITIES == {"Dubai": (25.20, 55.27)}
        and DUBAI_LOCATION == {
            "city": "Dubai", "latitude": 25.20, "longitude": 55.27,
            "timezone": "Asia/Dubai",
        },
    )

    def c_run(transport, extra=None):
        overrides = {
            "HTTP": transport, "GROQ_API_KEY": "gsk_pin_test",
            "SHEET_URL": None,
        }
        overrides.update(extra or {})
        out = io.StringIO()
        holder = {}

        def run():
            with contextlib.redirect_stdout(out):
                holder["decision"] = run_weather_agent()[0]

        ip_override(overrides, run)
        return holder["decision"], out.getvalue()

    for label, extra in {
        "a non-Dubai city": {"CITY": "Lahore"},
        "edited Dubai coordinates": {"CITIES": {"Dubai": (25.30, 55.27)}},
    }.items():
        c_transport = IpTransport(groq=ip_completion(IP_REPLY))
        c_decision, _ = c_run(c_transport, extra)
        check(
            f"C: {label} fails closed before any network call",
            c_transport.calls == []
            and c_decision["assessment_status"] == "INCOMPLETE"
            and c_decision["alert_recommendation"] == "UNABLE_TO_ASSESS"
            and c_decision["send_alert"] is False
            and c_decision["reasoning"].startswith("ValueError: "),
        )

    def c_weather(**changes):
        raw = ip_open_meteo()
        raw.update(changes)

        return IpResponse(200, json.dumps(raw))

    c_transport = IpTransport(groq=ip_completion(IP_REPLY), weather=c_weather())
    c_decision, _ = c_run(c_transport)
    c_out = ip_job(
        IpTransport(groq=ip_completion(IP_REPLY), weather=c_weather()),
        stub_weather=False,
    )["stdout"]

    check(
        "C: the weather request asks for Dubai and Asia/Dubai explicitly",
        c_transport.weather_params is not None
        and c_transport.weather_params.get("timezone") == "Asia/Dubai"
        and c_transport.weather_params.get("latitude") == 25.20
        and c_transport.weather_params.get("longitude") == 55.27,
    )

    check(
        "C: the response location is observed and reported",
        c_decision.get("location_observation") == {
            "requested_timezone": "Asia/Dubai",
            "returned_timezone": "Asia/Dubai",
            "timezone_matches_request": True,
            "returned_latitude": 25.2,
            "returned_longitude": 55.25,
            "utc_offset_seconds_present": True,
        }
        and "LOCATION OBSERVATION (program-generated; not enforced):" in c_out
        and "location_observation" in PROGRAM_OWNED_FIELDS,
    )

    c_raw = ip_open_meteo(timezone="Asia/Karachi")
    c_raw.update(latitude=31.5, longitude=74.3)
    del c_raw["utc_offset_seconds"]
    c_mismatch_transport = IpTransport(
        groq=ip_completion(IP_REPLY), weather=IpResponse(200, json.dumps(c_raw))
    )
    c_mismatch, _ = c_run(c_mismatch_transport)

    c_mismatch_obs = c_mismatch.get("location_observation")

    check(
        "C: a mismatched response location is reported, not enforced",
        "ASSESSMENT_ERROR" not in [
            reason.get("code")
            for reason in c_mismatch.get("incomplete_reasons", [])
        ]
        and isinstance(c_mismatch_obs, dict)
        and c_mismatch_obs.get("timezone_matches_request") is False
        and c_mismatch_obs.get("returned_latitude") == 31.5
        and c_mismatch_obs.get("utc_offset_seconds_present") is False
        and c_mismatch["send_alert"] is False,
    )

    def c_marked_weather():
        return c_weather(
            timezone="Asia/Dubai::IP-MARK", latitude="IP-MARK",
            longitude=float("nan"),
        )

    c_marked, _ = c_run(IpTransport(
        groq=ip_completion(IP_REPLY), weather=c_marked_weather(),
    ))
    c_marked_out = ip_job(
        IpTransport(groq=ip_completion(IP_REPLY), weather=c_marked_weather()),
        stub_weather=False,
    )["stdout"]

    check(
        "C: the observation copies no provider text",
        isinstance(c_marked.get("location_observation"), dict)
        and all(
            key in c_marked["location_observation"]
            for key in (
                "returned_timezone", "returned_latitude", "returned_longitude",
            )
        )
        and c_marked["location_observation"]["returned_timezone"] is None
        and c_marked["location_observation"]["returned_latitude"] is None
        and c_marked["location_observation"]["returned_longitude"] is None
        and "IP-MARK" not in json_text(c_marked["location_observation"])
        and "LOCATION OBSERVATION" in c_marked_out
        and "IP-MARK" not in c_marked_out.split(
            "LOCATION OBSERVATION", 1
        )[1].splitlines()[0],
    )

    # ------------------------------------------------------------
    # D. Warning verification is separate from assessment validity;
    # alert authorization is centralized and never granted.
    # ------------------------------------------------------------
    check(
        "D: NOT_CHECKED is the only warning status this program produces",
        official_warning_verification()
        == {"status": "NOT_CHECKED", "authority": None, "validity": None}
        and WARNING_VERIFICATION_STATES == (
            "NOT_CHECKED", "VERIFIED_NONE", "VERIFIED_ACTIVE", "UNAVAILABLE",
        ),
    )

    d_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    d_production = [
        node for node in d_tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "run_tests")
    ]
    d_status_literals = []
    d_send_alert_values = []
    d_send_alert_owners = set()

    for top in d_production:
        owner = getattr(top, "name", "<module>")

        for node in ast.walk(top):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                    ):
                        if target.slice.value == "official_warning_status":
                            d_status_literals.append(
                                isinstance(node.value, ast.Constant)
                            )

                        if target.slice.value == "send_alert":
                            d_send_alert_values.append(node.value)
                            d_send_alert_owners.add(owner)

            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None)
                == "reject_unverified_warning_claims"
                and len(node.args) > 1
            ):
                d_status_literals.append(
                    isinstance(node.args[1], ast.Constant)
                )

    check(
        "D: no warning status is hard-coded outside the producer",
        d_status_literals != [] and not any(d_status_literals),
    )

    check(
        "D: the model still sees the program-reported NOT_CHECKED status",
        "NOT_CHECKED (program-reported)"
        in json_text(ip_ok_body.get("messages")),
    )

    d_claim = {"reasoning": "No official warning is in effect."}

    check(
        "D: the claim guard runs under every warning status",
        all(
            isinstance(
                b2_raised(lambda status=status: reject_unverified_warning_claims(
                    dict(d_claim), status
                )),
                UnverifiedWarningClaim,
            )
            for status in (
                "NOT_CHECKED", "VERIFIED", "VERIFIED_NONE",
                "VERIFIED_ACTIVE", "UNAVAILABLE", None,
            )
        ),
    )

    def d_status(**fields):
        decision = {
            "incomplete_reasons": [], "risk_score": 7.0,
            "policy_status": "AVAILABLE",
            "weather_retrieval_status": "AVAILABLE",
            "official_warning_status": "NOT_CHECKED",
        }
        decision.update(fields)

        return enforce_assessment_status(decision)

    check(
        "D: NOT_CHECKED warnings no longer decide assessment validity",
        d_status()["assessment_status"] == "COMPLETE"
        and d_status()["incomplete_reasons"] == [],
    )

    check(
        "D: any missing requirement keeps the assessment INCOMPLETE",
        all(
            d_status(**fields)["assessment_status"] == "INCOMPLETE"
            for fields in (
                {"incomplete_reasons": [{"code": "X", "detail": None}]},
                {"risk_score": None},
                {"weather_retrieval_status": "UNAVAILABLE"},
                {"policy_status": None},
            )
        ),
    )

    check(
        "D: an unavailable policy gives UNAVAILABLE, never COMPLETE or INCOMPLETE",
        d_status(policy_status="UNAVAILABLE")["assessment_status"]
        == "UNAVAILABLE"
        and d_status(
            policy_status="UNAVAILABLE", risk_score=7.0, incomplete_reasons=[]
        )["assessment_status"] == "UNAVAILABLE"
        and d_status(policy_status="UNKNOWN")["assessment_status"]
        == "INCOMPLETE"
        and d_status(policy_status=None)["assessment_status"] == "INCOMPLETE"
        and d_status(weather_retrieval_status="UNAVAILABLE")[
            "assessment_status"
        ] == "INCOMPLETE",
    )

    check(
        "D: with today's evidence gates no assessment is COMPLETE",
        ip_ok_decision["assessment_status"] == "INCOMPLETE"
        and "RISK_DIMENSIONS_UNASSESSED" in [
            reason["code"] for reason in ip_ok_decision["incomplete_reasons"]
        ]
        and "OFFICIAL_WARNINGS_NOT_VERIFIED" not in [
            reason["code"] for reason in ip_ok_decision["incomplete_reasons"]
        ]
        and ip_ok_decision["official_warning_status"] == "NOT_CHECKED",
    )

    d_no_alert = ip_agent(IpTransport(groq=ip_completion(
        dict(IP_REPLY, alert_recommendation="NO_ALERT_RECOMMENDED")
    )))

    check(
        "D: a no-alert conclusion never survives unchecked warnings",
        d_no_alert["alert_recommendation"] != "NO_ALERT_RECOMMENDED"
        and apply_official_warning_guard({
            "alert_recommendation": "NO_ALERT_RECOMMENDED",
            "limitations": [], "missing_evidence": [],
        })["alert_recommendation"] == "MONITOR",
    )

    check(
        "D: send_alert is only ever assigned False, in one function",
        d_send_alert_values != []
        and all(
            isinstance(value, ast.Constant) and value.value is False
            for value in d_send_alert_values
        )
        and d_send_alert_owners == {"apply_alert_authorization"},
    )

    d_complete = d_status()
    apply_alert_authorization(d_complete, assessment_failed=False)
    d_failed = apply_alert_authorization({}, assessment_failed=True)

    check(
        "D: even a COMPLETE assessment is never authorized to alert",
        d_complete["assessment_status"] == "COMPLETE"
        and d_complete["send_alert"] is False
        and d_complete["approval_status"] == "PENDING_REVIEW"
        and d_complete["alert_message"] == ""
        and d_complete["notification_status"] == "NOT_SENT"
        and d_failed == {
            "approval_status": "NOT_AUTHORIZED", "send_alert": False,
            "alert_message": "", "notification_status": "NOT_SENT",
        },
    )

    # ------------------------------------------------------------
    # E. Outcome classification, exit codes, annotations and step
    # summary, end to end through main() with transport-level stubs.
    # ------------------------------------------------------------
    check(
        "E: exit codes are COMPLETE 0, INCOMPLETE 2, INVALID/UNAVAILABLE 1",
        RUN_OUTCOME_EXIT_CODES == {
            "COMPLETE": 0, "INCOMPLETE": 2, "INVALID": 1, "UNAVAILABLE": 1,
        },
    )

    check(
        "E: the script exits with main()'s return code",
        any(
            isinstance(node, ast.If)
            and "raise SystemExit(main(sys.argv))" in ast.unparse(node)
            for node in ast.parse(
                Path(__file__).read_text(encoding="utf-8")
            ).body
        ),
    )

    check(
        "E: run_agent_job returns the run outcome",
        ip_ok["value"] == "INCOMPLETE" and ip_claim["value"] == "INVALID",
    )

    e_env_names = ("GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY")

    def e_main(transport, mode="agent-dry", extra=None, env=None, argv=None):
        overrides = {
            "HTTP": transport, "GROQ_API_KEY": "gsk_pin_test",
            "SHEET_URL": IP_SHEET,
        }
        overrides.update(extra or {})
        saved = {name: os.environ.pop(name, None) for name in e_env_names}
        os.environ.update(env or {})
        out = io.StringIO()
        result = {"code": None, "raised": None}

        def run():
            with contextlib.redirect_stdout(out):
                try:
                    result["code"] = main(argv or ["weather_log.py", mode])
                except Exception as exc:
                    result["raised"] = exc

        try:
            ip_override(overrides, run)
        finally:
            for name in e_env_names:
                os.environ.pop(name, None)

            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value

        text = out.getvalue()
        fields = None
        payload = None

        if "RUN OUTCOME (program-generated):" in text:
            fields, _ = json.JSONDecoder().raw_decode(
                text.split("RUN OUTCOME (program-generated):", 1)[1].lstrip()
            )

        if "OUTPUT PAYLOAD:" in text:
            payload, _ = json.JSONDecoder().raw_decode(
                text.split("OUTPUT PAYLOAD:", 1)[1].lstrip()
            )

        result.update(
            stdout=text, fields=fields, payload=payload,
            annotations=[
                line for line in text.splitlines() if line.startswith("::")
            ],
            sheet_posts=[url for url, _ in transport.posts if url == IP_SHEET],
            calls=list(transport.calls),
        )

        return result

    e_dir = Path(tempfile.mkdtemp())
    (e_dir / "empty.md").write_text("", encoding="utf-8")

    def e_transport(groq=None, weather=None, **kwargs):
        return IpTransport(
            groq=groq if groq is not None else ip_completion(IP_REPLY),
            weather=(
                weather if weather is not None
                else IpResponse(200, json.dumps(ip_open_meteo()))
            ),
            **kwargs,
        )

    def e_raise(error):
        def function(*args, **kwargs):
            raise error

        return function

    class EBadWeatherResponse:
        status_code = 200
        text = ""
        headers = {}

        def raise_for_status(self):
            raise TypeError("IP-MARK programming error")

        def json(self):
            return {}

    e_truncated = ip_provider_error(400, {
        "message": "max completion tokens reached before generating a "
                   "valid document IP-MARK",
        "type": "invalid_request_error", "code": "json_validate_failed",
        "failed_generation": "IP-MARK-FG",
    })
    e_claim_reply = ip_completion(
        dict(IP_REPLY, reasoning="No official warnings were checked. IP-MARK")
    )

    e_matrix = [
        ("an assessment with gated dimensions", e_transport(), {},
         "INCOMPLETE", 2, "none", "none", "none"),
        ("all dimensions evidenced (test-only gate removal)", e_transport(),
         {"EVIDENCE_GATED_DIMENSIONS": {}}, "COMPLETE", 0, "none", "none",
         "none"),
        ("a non-Dubai city", e_transport(), {"CITY": "Lahore"},
         "INVALID", 1, "CONFIGURATION_INVALID", "configuration", "none"),
        ("a missing Groq key", e_transport(), {"GROQ_API_KEY": None},
         "INVALID", 1, "CONFIGURATION_INVALID", "configuration", "none"),
        ("a missing policy file", e_transport(),
         {"GUIDELINES_PATH": e_dir / "none.md"},
         "UNAVAILABLE", 1, "POLICY_UNAVAILABLE", "policy", "none"),
        ("an empty policy file", e_transport(),
         {"GUIDELINES_PATH": e_dir / "empty.md"},
         "INVALID", 1, "POLICY_INVALID", "policy", "none"),
        ("a weather network error",
         e_transport(get_error=requests.ConnectionError("IP-MARK refused")),
         {}, "UNAVAILABLE", 1, "WEATHER_UNAVAILABLE", "weather", "none"),
        ("a weather response without hourly data",
         e_transport(weather=IpResponse(200, json.dumps({
             "current": {"time": "2026-11-08T12:00"}, "daily": {},
         }))),
         {}, "UNAVAILABLE", 1, "WEATHER_UNAVAILABLE", "weather", "none"),
        ("a request over the character budget", e_transport(),
         {"MAX_REQUEST_CHARS": 1000}, "UNAVAILABLE", 1,
         "REQUEST_BUDGET_EXCEEDED", "assessment", "none"),
        ("a completion-limit provider 400", e_transport(groq=e_truncated), {},
         "INVALID", 1, "PROVIDER_ERROR", "assessment", "OUTPUT_TRUNCATED"),
        ("another json_validate_failed",
         e_transport(groq=ip_provider_error(400, {
             "message": "IP-MARK", "type": "invalid_request_error",
             "code": "json_validate_failed",
         })),
         {}, "INVALID", 1, "PROVIDER_ERROR", "assessment",
         "OUTPUT_VALIDATION_UNSPECIFIED"),
        ("a provider rate limit",
         e_transport(groq=ip_provider_error(429, {
             "code": "rate_limit_exceeded", "message": "IP-MARK",
         })),
         {}, "UNAVAILABLE", 1, "PROVIDER_ERROR", "assessment",
         "RATE_LIMITED"),
        ("a provider network error",
         e_transport(post_error=requests.ConnectionError("IP-MARK refused")),
         {}, "UNAVAILABLE", 1, "PROVIDER_ERROR", "assessment",
         "NETWORK_ERROR"),
        ("an unknown provider status",
         e_transport(groq=ip_provider_error(600, {"message": "IP-MARK"})),
         {}, "UNAVAILABLE", 1, "PROVIDER_ERROR", "assessment",
         "PROVIDER_ERROR_UNCLASSIFIED"),
        ("a truncated completion",
         e_transport(groq=ip_completion(IP_REPLY, "length")), {},
         "INVALID", 1, "PROVIDER_ERROR", "assessment", "OUTPUT_TRUNCATED"),
        ("an invalid model response",
         e_transport(groq=ip_completion({
             key: value for key, value in IP_REPLY.items()
             if key != "reasoning"
         })),
         {}, "INVALID", 1, "RESPONSE_INVALID", "assessment", "none"),
        ("an unverified warning claim", e_transport(groq=e_claim_reply), {},
         "INVALID", 1, "UNVERIFIED_WARNING_CLAIM", "assessment", "none"),
    ]

    e_results = {}

    for label, transport, extra, outcome, code, failure, stage, provider in (
        e_matrix
    ):
        result = e_main(transport, extra=extra)
        e_results[label] = result
        fields = result["fields"] or {}
        check(
            f"E: {label} exits {code} as {outcome}",
            result["raised"] is None
            and result["code"] == code
            and fields.get("outcome") == outcome
            and fields.get("exit_code") == code
            and fields.get("failure_code") == failure
            and fields.get("failure_stage") == stage
            and fields.get("provider_category") == provider
            and result["sheet_posts"] == [],
        )

    check(
        "E: configuration and policy failures make no network call",
        all(
            e_results[label]["calls"] == []
            for label in (
                "a non-Dubai city", "a missing Groq key",
                "a missing policy file", "an empty policy file",
            )
        )
        and e_results["a request over the character budget"]["calls"]
        == IP_BASELINE["dry_calls"][:2],
    )

    check(
        "E: a missing policy gives assessment_status UNAVAILABLE (section 10)",
        e_results["a missing policy file"]["payload"]["assessment_status"]
        == "UNAVAILABLE"
        and e_results["a missing policy file"]["payload"]["policy_status"]
        == "UNAVAILABLE",
    )

    check(
        "E: the gated dry run makes the pinned calls and no Sheets POST",
        e_results["an assessment with gated dimensions"]["calls"]
        == IP_BASELINE["dry_calls"],
    )

    e_internal = [
        ("a policy-loader KeyError", e_transport(),
         {"load_guidelines": e_raise(KeyError("IP-MARK"))}, "policy"),
        ("a weather-response TypeError",
         e_transport(weather=EBadWeatherResponse()), {}, "weather"),
        ("a history ZeroDivisionError", e_transport(),
         {"tool_get_history": e_raise(ZeroDivisionError("IP-MARK"))},
         "history"),
        ("a screening AttributeError", e_transport(),
         {"identify_hazards": e_raise(AttributeError("IP-MARK"))},
         "screening"),
        ("an unknown hazard code (untyped ValueError)", e_transport(),
         {"identify_hazards": lambda weather: {"BOGUS": ["x"]}}, "retrieval"),
        ("a plain ValueError during assessment", e_transport(),
         {"summarize_weather": e_raise(ValueError("IP-MARK"))}, "assessment"),
        ("a Groq transport TypeError",
         e_transport(post_error=TypeError("IP-MARK")), {}, "assessment"),
    ]

    for label, transport, extra, stage in e_internal:
        result = e_main(transport, extra=extra)
        e_results[label] = result
        fields = result["fields"] or {}
        check(
            f"E: {label} is INTERNAL_ERROR, never a known cause",
            result["raised"] is None
            and result["code"] == 1
            and fields.get("outcome") == "INVALID"
            and fields.get("failure_code") == "INTERNAL_ERROR"
            and fields.get("failure_stage") == stage
            and fields.get("provider_category") == "none"
            and result["sheet_posts"] == [],
        )

    check(
        "E: every dry run prints the pinned payload with send_alert False",
        all(
            result["payload"] is not None
            and sorted(result["payload"]) == IP_BASELINE["payload_keys"]
            and result["payload"]["send_alert"] is False
            and result["payload"]["alert_message"] == ""
            for result in e_results.values()
        ),
    )

    check(
        "E: without GITHUB_ACTIONS no annotation is printed",
        all(result["annotations"] == [] for result in e_results.values()),
    )

    e_summary = e_dir / "summary.md"
    e_summary.write_text("PRE-EXISTING\n", encoding="utf-8")
    e_env = {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(e_summary)}
    e_incomplete = e_main(e_transport(), env=e_env)
    e_failed = e_main(e_transport(groq=e_truncated), env=e_env)
    e_claimed = e_main(e_transport(groq=e_claim_reply), env=e_env)
    e_complete = e_main(
        e_transport(), extra={"EVIDENCE_GATED_DIMENSIONS": {}}, env=e_env
    )
    e_summary_text = e_summary.read_text(encoding="utf-8")

    check(
        "E: under GitHub Actions each run prints exactly one annotation",
        e_incomplete["annotations"] == [
            "::notice title=Assessment INCOMPLETE::exit_code=2"
            " | reasons=RISK_DIMENSIONS_UNASSESSED | warnings=NOT_CHECKED"
        ]
        and e_failed["annotations"] == [
            "::error title=Assessment INVALID::exit_code=1"
            " | failure=PROVIDER_ERROR | stage=assessment"
            " | provider=OUTPUT_TRUNCATED | claim=none"
        ]
        and e_claimed["annotations"] == [
            "::error title=Assessment INVALID::exit_code=1"
            " | failure=UNVERIFIED_WARNING_CLAIM | stage=assessment"
            " | provider=none | claim=field=reasoning;pattern=3"
        ]
        and e_complete["annotations"] == [
            "::notice title=Assessment COMPLETE::exit_code=0"
            " | recommendation=MONITOR"
        ]
        and [e_incomplete["code"], e_failed["code"], e_claimed["code"],
             e_complete["code"]] == [2, 1, 1, 0],
    )

    check(
        "E: the step summary is appended, never overwritten",
        e_summary_text.startswith("PRE-EXISTING\n")
        and e_summary_text.count("### Weather assessment run") == 4,
    )

    check(
        "E: the step summary records each outcome in fixed terms",
        all(
            row in e_summary_text
            for row in (
                "| Outcome | INCOMPLETE |", "| Exit code | 2 |",
                "| Outcome | INVALID |", "| Provider error | OUTPUT_TRUNCATED |",
                "| Warning claim | field=reasoning;pattern=3 |",
                "| Outcome | COMPLETE |", "| Exit code | 0 |",
                "| Official warning status | NOT_CHECKED |",
                "| Location timezone matches request | yes |",
                "| Hazards screened | HEAT |",
            )
        ),
    )

    e_sections = e_summary_text.split("### Weather assessment run")[1:]

    def e_rows(section):
        return {
            line.split("|")[1].strip(): line.split("|")[2].strip()
            for line in section.splitlines()
            if line.startswith("| ") and line.count("|") == 3
        }

    e_summary_rows = [e_rows(section) for section in e_sections]
    e_expected_rows = [
        {   # INCOMPLETE: reasons kept, never labelled COMPLETE
            "Outcome": "INCOMPLETE", "Exit code": "2",
            "Assessment status": "INCOMPLETE",
            "Incomplete reasons": "RISK_DIMENSIONS_UNASSESSED",
            "Failure code": "none", "Failure stage": "none",
        },
        {   # provider output truncated
            "Outcome": "INVALID", "Exit code": "1",
            "Assessment status": "INCOMPLETE",
            "Failure code": "PROVIDER_ERROR", "Failure stage": "assessment",
            "Provider error": "OUTPUT_TRUNCATED",
        },
        {   # unverified warning claim
            "Outcome": "INVALID", "Exit code": "1",
            "Assessment status": "INCOMPLETE",
            "Failure code": "UNVERIFIED_WARNING_CLAIM",
            "Failure stage": "assessment",
            "Warning claim": "field=reasoning;pattern=3",
        },
        {   # COMPLETE (test-only gate removal)
            "Outcome": "COMPLETE", "Exit code": "0",
            "Assessment status": "COMPLETE",
            "Incomplete reasons": "none", "Failure code": "none",
        },
    ]

    check(
        "E: each step summary states its own status, reasons and failure code",
        len(e_summary_rows) == 4
        and all(
            all(rows.get(key) == value for key, value in expected.items())
            for rows, expected in zip(e_summary_rows, e_expected_rows)
        )
        and [rows.get("Assessment status") for rows in e_summary_rows]
        == ["INCOMPLETE", "INCOMPLETE", "INCOMPLETE", "COMPLETE"],
    )

    e_public = (
        "\n".join(
            line
            for result in list(e_results.values())
            + [e_incomplete, e_failed, e_claimed, e_complete]
            for line in result["annotations"]
        )
        + e_summary_text
        + json_text([result["fields"] for result in e_results.values()])
    )

    check(
        "E: annotations and summaries carry no model or provider text",
        all(
            marker not in e_public
            for marker in (
                "IP-MARK", "max completion tokens", "No official warnings",
                "Heat screening triggered", "Limit strenuous", "gsk_",
                "sheet.invalid", "open-meteo",
            )
        ),
    )

    check(
        "E: every annotation line uses only the fixed format",
        all(
            re.fullmatch(
                r"::(notice|warning|error) title=Assessment "
                r"(COMPLETE|INCOMPLETE|INVALID|UNAVAILABLE)::"
                r"[A-Za-z0-9_=|;, -]+",
                line,
            )
            for result in (e_incomplete, e_failed, e_claimed, e_complete)
            for line in result["annotations"]
        ),
    )

    e_bogus = outcome_fields({
        "assessment_status": "DONE::X",
        "alert_recommendation": "SEND%0ANOW",
        "incomplete_reasons": [
            {"code": "EVIL::%0A"}, {"code": "RISK_DIMENSIONS_UNASSESSED"},
            "not a dict",
        ],
        "failure_code": "PWNED", "failure_stage": "space",
        "provider_error": {"category": "IP-MARK"},
        "warning_claim_diagnostic": "field=x;pattern=9::IP-MARK",
        "official_warning_status": "VERIFIED",
        "hazard_screening": {"EVIL": [], "HEAT": []},
        "sources": "not a list",
        "location_observation": {
            "timezone_matches_request": "yes",
            "utc_offset_seconds_present": True,
        },
    }, "BOGUS")

    check(
        "E: the outcome formatter only emits closed-vocabulary values",
        e_bogus == {
            "outcome": "unknown", "exit_code": 1,
            "assessment_status": "unknown",
            "alert_recommendation": "unknown",
            "incomplete_reasons": ["unknown", "RISK_DIMENSIONS_UNASSESSED"],
            "failure_code": "unknown", "failure_stage": "unknown",
            "provider_category": "unknown", "warning_claim": "unknown",
            "official_warning_status": "unknown",
            "hazards_screened": ["HEAT"], "policy_sources_cited": 0,
            "location_timezone_matches": None,
            "location_utc_offset_present": True,
        }
        and "IP-MARK" not in outcome_annotation(e_bogus)
        and "IP-MARK" not in outcome_summary_markdown(e_bogus)
        and _workflow_message("a%b\r\nc") == "a%25b%0D%0Ac",
    )

    e_bad_summary = e_main(e_transport(), env={
        "GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(e_dir),
    })

    check(
        "E: a step-summary write failure does not change the outcome",
        e_bad_summary["code"] == 2
        and "STEP SUMMARY: not written (file error)." in e_bad_summary["stdout"]
        and len(e_bad_summary["annotations"]) == 1,
    )

    e_live_transport = e_transport()
    e_live = e_main(e_live_transport, mode="agent")
    e_live_payload = [body for url, body in e_live_transport.posts
                      if url == IP_SHEET]

    check(
        "E: live mode writes one Sheets record, then exits 2 for INCOMPLETE",
        e_live["raised"] is None
        and e_live["code"] == 2
        and e_live["sheet_posts"] == [IP_SHEET]
        and len(e_live_payload) == 1
        and sorted(e_live_payload[0]) == IP_BASELINE["payload_keys"]
        and e_live_payload[0]["send_alert"] is False
        and e_live["fields"]["outcome"] == "INCOMPLETE",
    )

    class ESheetDown(IpTransport):
        def post(self, url, headers=None, json=None, timeout=None):
            if url == IP_SHEET:
                self.calls.append(("POST", url))
                self.posts.append((url, json))
                return IpResponse(503, "IP-MARK down")

            return IpTransport.post(self, url, headers, json, timeout)

    for label, transport in {
        "an HTTP error": ESheetDown(
            groq=ip_completion(IP_REPLY),
            weather=IpResponse(200, json.dumps(ip_open_meteo())),
        ),
        "an unconfirmed write": e_transport(sheets="IP-MARK not ok"),
    }.items():
        result = e_main(transport, mode="agent")
        check(
            f"E: a Sheets write with {label} is SHEETS_WRITE_FAILED, re-raised",
            isinstance(result["raised"], SheetsWriteError)
            and result["code"] is None
            and result["fields"]["outcome"] == "UNAVAILABLE"
            and result["fields"]["failure_code"] == "SHEETS_WRITE_FAILED"
            and result["fields"]["failure_stage"] == "sheets"
            and "IP-MARK" not in json_text(result["fields"]),
        )

    e_test_calls = []
    e_test_mode = e_main(
        IpTransport(), argv=["weather_log.py", "test"],
        extra={"run_tests": lambda: e_test_calls.append("run")},
    )

    try:
        ip_override(
            {"run_tests": e_raise(SystemExit(1))},
            lambda: main(["weather_log.py", "test"]),
        )
        e_failing_exit = None
    except SystemExit as exc:
        e_failing_exit = exc.code

    e_usage = e_main(IpTransport(), argv=["weather_log.py", "unknown-mode"])
    e_search_transport = IpTransport()
    e_search = e_main(
        e_search_transport, argv=["weather_log.py", "search", "heat"]
    )

    check(
        "E: test, search and usage modes keep their exit behaviour",
        e_test_mode["code"] == 0 and e_test_calls == ["run"]
        and e_failing_exit == 1
        and e_usage["code"] == 1 and "Available commands:" in e_usage["stdout"]
        and e_search["code"] == 0 and "SECTION:" in e_search["stdout"]
        and e_search_transport.calls == [],
    )

    # ------------------------------------------------------------
    # Review corrections L1-L5.
    # ------------------------------------------------------------
    # L1: only the typed policy failure is reported as a missing policy.
    l1_untyped = ip_agent(
        IpTransport(groq=ip_completion(IP_REPLY)),
        extra={"summarize_weather": e_raise(FileNotFoundError("IP-MARK"))},
    )

    check(
        "L1: an untyped FileNotFoundError is INTERNAL_ERROR, not a missing policy",
        l1_untyped["run_outcome"] == "INVALID"
        and l1_untyped["failure_code"] == "INTERNAL_ERROR"
        and [r["code"] for r in l1_untyped["incomplete_reasons"]]
        == ["ASSESSMENT_ERROR"]
        and l1_untyped["incomplete_reasons"][0]["detail"] == "FileNotFoundError"
        and l1_untyped["policy_status"] == "AVAILABLE",
    )

    l1_missing = ip_agent(
        IpTransport(), extra={"GUIDELINES_PATH": e_dir / "absent.md"}
    )

    check(
        "L1: a missing policy file is POLICY_UNAVAILABLE in every label",
        l1_missing["run_outcome"] == "UNAVAILABLE"
        and l1_missing["failure_code"] == "POLICY_UNAVAILABLE"
        and l1_missing["failure_stage"] == "policy"
        and [r["code"] for r in l1_missing["incomplete_reasons"]]
        == ["POLICY_UNAVAILABLE"]
        and l1_missing["policy_status"] == "UNAVAILABLE",
    )

    l1_stub = ip_agent(
        IpTransport(),
        extra={"load_guidelines": e_raise(FileNotFoundError("IP-MARK"))},
    )

    check(
        "L1: an untyped loader error never claims the policy is missing",
        l1_stub["failure_code"] == "INTERNAL_ERROR"
        and l1_stub["policy_status"] == "UNKNOWN"
        and "POLICY_UNAVAILABLE" not in [
            r["code"] for r in l1_stub["incomplete_reasons"]
        ],
    )

    # L2: a reporting failure names the real assessment status.
    l2_incomplete = e_main(e_transport(sheets="IP-MARK not ok"), mode="agent")
    l2_complete = e_main(
        e_transport(sheets="IP-MARK not ok"), mode="agent",
        extra={"EVIDENCE_GATED_DIMENSIONS": {}},
    )

    check(
        "L2: a reporting failure names the actual assessment status",
        isinstance(l2_incomplete["raised"], SheetsWriteError)
        and l2_incomplete["fields"]["assessment_status"] == "INCOMPLETE"
        and isinstance(l2_complete["raised"], SheetsWriteError)
        and l2_complete["fields"]["assessment_status"] == "COMPLETE",
    )

    check(
        "L2: naming the status leaves the outcome classification unchanged",
        all(
            result["code"] is None
            and result["fields"]["outcome"] == "UNAVAILABLE"
            and result["fields"]["failure_code"] == "SHEETS_WRITE_FAILED"
            and result["fields"]["failure_stage"] == "sheets"
            for result in (l2_incomplete, l2_complete)
        ),
    )

    with contextlib.redirect_stdout(io.StringIO()):
        LAST_REPORTED_ASSESSMENT["status"] = "EVIL::X"
        l2_bogus = emit_unhandled_failure(SheetsWriteError("x"))
        LAST_REPORTED_ASSESSMENT["status"] = None
        l2_none = emit_unhandled_failure(SheetsWriteError("x"))

    check(
        "L2: an unrecognised or missing status is reported as unknown",
        l2_bogus["assessment_status"] == "unknown"
        and l2_none["assessment_status"] == "unknown",
    )

    # L3: the realistic-history request (3 history records) is pinned too.
    check(
        "Pin: realistic-history Groq request is byte-identical to be5524a",
        len(fz_request) == 17240
        and ip_sha(fz_request)
        == "f93405f17b7576003b75d4a9624e3158fabc3de428edf6abad497eab7627b081",
    )

    # L4: the dry-run payload stays the last output.
    check(
        "L4: the dry-run payload is still the last output",
        ip_dry["stdout"].rstrip().endswith(json_text(ip_dry["payload"]))
        and all(
            result["stdout"].rstrip().endswith(json_text(result["payload"]))
            for result in e_results.values()
        ),
    )

    # L5: an unclassified exception while reporting is INTERNAL_ERROR at
    # the report stage, gets one safe annotation and is re-raised as is.
    LAST_REPORTED_ASSESSMENT["status"] = "COMPLETE"
    l5_summary = e_dir / "l5_summary.md"
    l5 = e_main(
        IpTransport(),
        extra={"run_weather_agent": lambda: (
            {"incomplete_reasons": [{"no_code": "IP-MARK"}], "sources": []},
            [],
        )},
        env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(l5_summary)},
    )
    l5_summary_text = (
        l5_summary.read_text(encoding="utf-8") if l5_summary.exists() else ""
    )

    check(
        "L5: an unclassified reporting error is INTERNAL_ERROR at report",
        type(l5["raised"]) is KeyError
        and l5["raised"].args == ("code",)
        and l5["code"] is None
        and l5["fields"] is not None
        and l5["fields"]["outcome"] == "INVALID"
        and l5["fields"]["exit_code"] == 1
        and l5["fields"]["failure_code"] == "INTERNAL_ERROR"
        and l5["fields"]["failure_stage"] == "report"
        and l5["fields"]["assessment_status"] == "unknown",
    )

    check(
        "L5: the reporting error gets one safe annotation and a summary",
        l5["annotations"] == [
            "::error title=Assessment INVALID::exit_code=1"
            " | failure=INTERNAL_ERROR | stage=report | provider=none"
            " | claim=none"
        ]
        and "| Failure stage | report |" in l5_summary_text
        and "IP-MARK" not in l5_summary_text + "".join(l5["annotations"]),
    )

    # ------------------------------------------------------------
    # Approved corrections: policy-unavailable status consistency and
    # one authoritative INCOMPLETE warning.
    # ------------------------------------------------------------
    p1_summary = e_dir / "p1_summary.md"
    p1_summary.write_text("", encoding="utf-8")
    p1 = e_main(
        e_transport(), extra={"GUIDELINES_PATH": e_dir / "absent.md"},
        env={"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(p1_summary)},
    )
    p1_text = p1_summary.read_text(encoding="utf-8")

    check(
        "P1: a missing policy reads UNAVAILABLE on every surface",
        p1["raised"] is None
        and p1["code"] == 1
        and p1["payload"].get("assessment_status") == "UNAVAILABLE"
        and p1["payload"].get("policy_status") == "UNAVAILABLE"
        and p1["payload"].get("alert_recommendation") == "UNABLE_TO_ASSESS"
        and p1["payload"].get("send_alert") is False
        and p1["payload"].get("alert_message") == ""
        and p1["payload"].get("approval_status") == "NOT_AUTHORIZED"
        and p1["payload"].get("notification_status") == "NOT_SENT"
        and "ASSESSMENT: UNAVAILABLE" in p1["stdout"]
        and p1["fields"]["outcome"] == "UNAVAILABLE"
        and p1["fields"]["assessment_status"] == "UNAVAILABLE"
        and p1["fields"]["failure_code"] == "POLICY_UNAVAILABLE"
        and p1["fields"]["exit_code"] == 1
        and "| Assessment status | UNAVAILABLE |" in p1_text
        and "| Outcome | UNAVAILABLE |" in p1_text
        and p1["annotations"] == [
            "::error title=Assessment UNAVAILABLE::exit_code=1"
            " | failure=POLICY_UNAVAILABLE | stage=policy | provider=none"
            " | claim=none"
        ]
        and p1["sheet_posts"] == []
        and p1["calls"] == [],
    )

    check(
        "P1: every other failure keeps assessment_status INCOMPLETE",
        all(
            e_results[label]["payload"].get("assessment_status") == "INCOMPLETE"
            and e_results[label]["fields"].get("assessment_status") == "INCOMPLETE"
            for label in (
                "an empty policy file", "a weather network error",
                "a provider rate limit", "an invalid model response",
                "an unverified warning claim", "a policy-loader KeyError",
                "a request over the character budget",
            )
        )
        and l1_untyped.get("assessment_status") == "INCOMPLETE"
        and l1_stub.get("assessment_status") == "INCOMPLETE",
    )

    p1_policy = GUIDELINES_PATH.read_text(encoding="utf-8")
    p1_unavailable = p1_policy.split(
        "### If the guidelines file is unavailable", 1
    )[1].split("### If the guidelines file is available", 1)[0]
    p1_statuses = p1_policy.split("### Assessment status", 1)[1].split(
        "A complete assessment requires", 1
    )[0]

    check(
        "P1: guidelines section 10 and the code agree on the status",
        re.findall(r"`assessment_status = (\w+)`", p1_unavailable)
        == ["UNAVAILABLE"]
        and re.findall(r"`assessment_status = (\w+)`", p1_unavailable)[0]
        == p1["payload"].get("assessment_status")
        and re.findall(r"^- `(\w+)`$", p1_statuses, re.M)
        == ["COMPLETE", "INCOMPLETE", "INVALID", "UNAVAILABLE"]
        and "`UNAVAILABLE` applies when the policy file cannot be retrieved "
        "(section 10); no assessment is attempted." in p1_statuses,
    )

    # Duplicate and missing warnings: the program's own annotations are
    # replayed by the stub, so the combined output is what the run shows.
    p2_usable = (
        wf_probe is not None and wf_probe["args"] == "weather_log.py agent-dry"
    )

    check(
        "P2: the program reports INCOMPLETE as a notice with its reasons",
        e_incomplete["annotations"] != []
        and not any(
            line.startswith("::warning") for line in e_incomplete["annotations"]
        )
        and all(
            line.startswith("::notice title=Assessment INCOMPLETE::")
            and "reasons=RISK_DIMENSIONS_UNASSESSED" in line
            for line in e_incomplete["annotations"]
        ),
    )

    if not p2_usable:
        wf_skipped.append("duplicate-warning")
        print(
            "SKIP: duplicate-warning checks (no usable POSIX bash). "
            "The INCOMPLETE warning count is then not exercised."
        )
    else:
        p2_both = wf_run_step(2, emit=e_incomplete["annotations"])
        p2_silent = wf_run_step(2)
        p2_complete = wf_run_step(0, emit=e_complete["annotations"])
        p2_failed = wf_run_step(1, emit=p1["annotations"])

        def p2_levels(result, level):
            return [
                line for line in result["annotations"]
                if line.startswith("::" + level)
            ]

        check(
            "P2: an INCOMPLETE run shows exactly one warning, reasons kept",
            p2_both["code"] == 0
            and p2_levels(p2_both, "warning") == [wf_warning_text]
            and p2_levels(p2_both, "notice") == e_incomplete["annotations"]
            and p2_levels(p2_both, "error") == [],
        )

        check(
            "P2: exit 2 without any program annotation still warns once",
            p2_silent["code"] == 0
            and p2_levels(p2_silent, "warning") == [wf_warning_text]
            and len(p2_silent["annotations"]) == 1,
        )

        check(
            "P2: COMPLETE and failed runs show no INCOMPLETE warning",
            p2_complete["code"] == 0
            and p2_levels(p2_complete, "warning") == []
            and p2_levels(p2_complete, "notice") == e_complete["annotations"]
            and p2_failed["code"] == 1
            and p2_levels(p2_failed, "warning") == []
            and p2_levels(p2_failed, "error") == p1["annotations"],
        )

    # ------------------------------------------------------------
    # Component-aware request estimate (BE groups). Offline. The
    # tokenizer groups need tiktoken and WEATHER_TOKENIZER_TESTS=1 and
    # are reported as skipped otherwise, never as passed.
    # ------------------------------------------------------------
    import itertools
    import random
    import string

    be_json = json
    be_skipped = []
    be_encoding = None

    if os.getenv("WEATHER_TOKENIZER_TESTS") == "1":
        try:
            import tiktoken

            be_encoding = tiktoken.get_encoding("o200k_harmony")
        except Exception as exc:
            be_skipped.append(f"tokenizer unavailable ({type(exc).__name__})")
    else:
        be_skipped.append("tokenizer groups (set WEATHER_TOKENIZER_TESTS=1)")

    def be_tokens(text):
        return len(be_encoding.encode(text, allowed_special="all"))

    def be_plain_tokens(text):
        return len(
            be_encoding.encode(
                text, allowed_special=set(), disallowed_special=()
            )
        )

    def be_exact_tokens(messages):
        """Documented harmony layout, counted by the development tokenizer."""
        return be_tokens(
            HARMONY_SYSTEM_TEXT
            + HARMONY_DEVELOPER_PREFIX + messages[0]["content"]
            + response_format_section_text()
            + HARMONY_USER_PREFIX + messages[1]["content"]
            + HARMONY_ASSISTANT_TAIL
        )

    with contextlib.redirect_stdout(io.StringIO()):
        be_policy = load_guidelines()

    be_hazard_codes = list(HAZARD_POLICY_TITLES)
    be_combos = [()] + [
        combo
        for size in range(1, len(be_hazard_codes) + 1)
        for combo in itertools.combinations(be_hazard_codes, size)
    ]
    be_rng = random.Random(7)
    be_prose = (
        "Limit strenuous outdoor work during the midday heat, drink water "
        "often and rest in shade. "
    )
    be_arabic = "".join(chr(code) for code in range(0x0621, 0x064B))

    def be_field(kind, size):
        if kind == "English":
            return (be_prose * 6)[:size]
        if kind == "Arabic":
            return "".join(be_rng.choice(be_arabic + "   ") for _ in range(size))
        if kind == "ASCII":
            alphabet = string.ascii_letters + string.digits + string.punctuation
            return "".join(be_rng.choice(alphabet) for _ in range(size))

        return "".join(
            chr(be_rng.choice([
                be_rng.randint(0x0621, 0x064A),
                be_rng.randint(0x4E00, 0x9FFF),
                be_rng.randint(0x1F300, 0x1F5FF),
            ]))
            for _ in range(size)
        )

    def be_history(variant):
        if variant == "typical":
            return [{
                "timestamp": "2026-11-09T04:10:11.417000+00:00",
                "city": "Dubai", "advice": (be_prose * 6)[:86],
                "alert_sent": False, "alert_message": "",
            } for _ in range(MAX_HISTORY_RECORDS)]

        return [{
            "timestamp": "2026-11-09T04:10:11.417000+00:00",
            "city": be_field(variant, 50), "advice": be_field(variant, 130),
            "alert_sent": True, "alert_message": be_field(variant, 90),
        } for _ in range(MAX_HISTORY_RECORDS)]

    be_variants = {
        "typical": be_history("typical"),
        "English at caps": be_history("English"),
        "Arabic at caps": be_history("Arabic"),
        "ASCII at caps": be_history("ASCII"),
        "CJK/emoji at caps": be_history("CJK"),
    }
    be_weather = rv_weather()
    be_sections = {}

    def be_sections_for(combo):
        if combo not in be_sections:
            hazards = {code: ["t"] for code in combo}

            with contextlib.redirect_stdout(io.StringIO()):
                required = select_hazard_guidance(be_policy, hazards)
                required += select_conditional_sections(
                    be_policy, hazards, be_weather
                )
                be_sections[combo] = search_guidelines(
                    be_policy, "x", k=0, required_sections=required
                )

        return be_sections[combo]

    def be_build(combo, history, weather=None):
        captured = []

        def capture(messages, temperature=0.1):
            captured.append(messages)
            raise RuntimeError("captured")

        def assemble():
            try:
                assess_weather(
                    weather or be_weather, history, be_policy,
                    be_sections_for(combo),
                )
            except RuntimeError:
                pass

        with_globals({"ask_groq": capture}, assemble)

        return captured[0]

    be_requests = {
        (combo, name): be_build(combo, history)
        for combo in be_combos
        for name, history in be_variants.items()
    }

    be_unverified_components = {
        "policy blocks": 0, "static text tokens (pinned)": 0,
        "numeric evidence bytes": 0, "numeric evidence bound": "",
        "numeric evidence tokens": 0, "history bytes": 0,
        "history tokens (1 per byte)": 0,
        "historical residual allowance": 0, "strict-schema allowance": 0,
    }

    def be_budget(messages):
        """The production estimate. An estimate that cannot be verified
        is reported as an impossibly large one so the checks fail by
        assertion instead of crashing."""
        text = serialized_request(messages)

        try:
            return request_token_budget(messages, text)
        except RequestBudgetError:
            return {
                "method": "unverified",
                "estimated_input_tokens": 10 ** 9,
                "legacy_estimated_input_tokens": estimate_tokens(text),
                "reason": None,
                "components": dict(be_unverified_components),
            }

    be_budgets = {
        key: be_budget(messages) for key, messages in be_requests.items()
    }
    be_limit = MAX_ESTIMATED_REQUEST_TOKENS - MAX_COMPLETION_TOKENS

    class BeTransport:
        """Recording stand-in for the HTTP session (no network)."""

        def __init__(self):
            self.posts = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.posts.append(json)

            return IpResponse(200, be_json.dumps({
                "choices": [{
                    "message": {"content": be_json.dumps(IP_REPLY)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4300, "completion_tokens": 700,
                          "total_tokens": 5000},
            }))

    be_transport = BeTransport()

    def be_send(messages, **overrides):
        """The real ask_groq with a recording transport. Returns the error
        text (or None) and the payloads posted."""
        be_transport.posts.clear()
        error = None
        settings = {"HTTP": be_transport, "GROQ_API_KEY": "gsk_be_test"}
        settings.update(overrides)

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                return ask_groq(messages)

        try:
            with_globals(settings, run)
        except RequestBudgetError as exc:
            error = str(exc)

        return error, list(be_transport.posts)

    # BE1: the parts rebuild the messages byte for byte, and the wrapper
    # changes no serialized byte.
    def be_verified(messages):
        try:
            return verify_request_parts(
                messages, messages.budget_parts
            ) is not None
        except RequestBudgetError:
            return False

    def be_rebuilds(messages):
        parts = messages.budget_parts

        return (
            type(messages) is BudgetedMessages
            and messages[0]["content"] == parts.system
            and messages[1]["content"]
            == parts.preamble
            + POLICY_BLOCK_JOIN.join(parts.blocks)
            + EVIDENCE_HEADING
            + parts.evidence_text
            and be_verified(messages)
            and serialized_request(messages)
            == serialized_request(list(messages))
            and json_text(messages) == json_text(list(messages))
            and messages == list(messages)
        )

    check(
        "BE1: parts rebuild the messages exactly for 32 hazard "
        "combinations x 5 history variants",
        len(be_requests) == 160
        and all(be_rebuilds(m) for m in be_requests.values()),
    )

    be_triple = ("HEAT", "RAIN_FLOOD", "LIGHTNING")
    be_triple_messages = be_requests[(be_triple, "typical")]
    be_guard = serialized_request(be_triple_messages)
    be_escape = lambda text: be_json.dumps(text, ensure_ascii=False)[1:-1]

    check(
        "BE1: the guard text is exactly the sum of its serialized pieces",
        len(be_guard)
        == len('{"messages":[{"role":"system","content":"')
        + len(be_escape(be_triple_messages[0]["content"]))
        + len('"},{"role":"user","content":"')
        + len(be_escape(be_triple_messages[1]["content"]))
        + len('"}],"response_format":')
        + len(json_text(RESPONSE_FORMAT))
        + len("}"),
    )

    be_error, be_posts = be_send(be_triple_messages)
    be_error_plain, be_posts_plain = be_send(
        list(be_triple_messages), MAX_ESTIMATED_REQUEST_TOKENS=10 ** 9
    )

    check(
        "BE1: the posted payload is byte-identical with and without the "
        "parts wrapper",
        be_error is None and be_error_plain is None
        and len(be_posts) == 1 and len(be_posts_plain) == 1
        and be_json.dumps(be_posts[0], allow_nan=False)
        == be_json.dumps(be_posts_plain[0], allow_nan=False),
    )

    # BE2: the static token table matches the static texts exactly.
    be_texts = static_token_texts(be_policy)
    be_digests = [sha256_text(text) for _, text in be_texts]

    check(
        "BE2: the static token table covers exactly the static texts",
        len(set(be_digests)) == len(be_digests)
        and set(be_digests) == set(STATIC_TOKEN_TABLE)
        and STATIC_TOKEN_TABLE_META["entries"] == len(STATIC_TOKEN_TABLE)
        and STATIC_TOKEN_TABLE_META["encoding"] == "o200k_harmony"
        and all(
            type(value) is int and value > 0
            for value in STATIC_TOKEN_TABLE.values()
        ),
    )

    check(
        "BE2: every block any request carries is in the table",
        all(
            sha256_text(block) in STATIC_TOKEN_TABLE
            for messages in be_requests.values()
            for block in messages.budget_parts.blocks
        ),
    )

    def be_expected_static(messages):
        parts = messages.budget_parts
        texts = [
            HARMONY_SYSTEM_TEXT,
            HARMONY_DEVELOPER_PREFIX + SYSTEM_PROMPT.strip(),
            response_format_section_text(),
            HARMONY_USER_PREFIX + USER_PROMPT_PREAMBLE,
            *parts.blocks,
            *([POLICY_BLOCK_JOIN] * (len(parts.blocks) - 1)),
            EVIDENCE_HEADING,
            HARMONY_ASSISTANT_TAIL,
        ]

        return sum(
            STATIC_TOKEN_TABLE.get(sha256_text(text), 0) for text in texts
        )

    check(
        "BE2: the static count is exactly the pinned counts of every "
        "static text of the request, none omitted or repeated (160 "
        "requests)",
        all(
            be_budgets[key]["components"]["static text tokens (pinned)"]
            == be_expected_static(messages)
            for key, messages in be_requests.items()
        ),
    )

    check(
        "BE2: the estimate is exactly static + numeric evidence + "
        "history + the two separate allowances (160 requests)",
        all(
            budget["estimated_input_tokens"]
            == budget["components"]["static text tokens (pinned)"]
            + budget["components"]["numeric evidence tokens"]
            + budget["components"]["history tokens (1 per byte)"]
            + 192 + 300
            and budget["components"]["historical residual allowance"]
            == 192
            and budget["components"]["strict-schema allowance"] == 300
            for budget in be_budgets.values()
        ),
    )

    if be_encoding is not None:
        check(
            "BE2: every pinned count equals the development tokenizer's",
            all(
                STATIC_TOKEN_TABLE.get(sha256_text(text)) == be_tokens(text)
                for _, text in be_texts
            ),
        )
    else:
        print("SKIP: BE2 pinned counts versus the development tokenizer")

    # BE3: fit and block verdicts for every combination and history type.
    be_verdict = {
        key: (
            be_budgets[key]["estimated_input_tokens"] <= be_limit,
            len(serialized_request(messages)) <= MAX_REQUEST_CHARS,
        )
        for key, messages in be_requests.items()
    }
    be_realistic = (
        "typical", "English at caps", "Arabic at caps", "ASCII at caps"
    )

    check(
        "BE3: all 32 combinations fit the token budget for typical, "
        "English, Arabic and ASCII history",
        all(
            be_verdict[(combo, name)][0]
            for combo in be_combos for name in be_realistic
        ),
    )

    be_cjk_fit = {
        combo for combo in be_combos
        if be_verdict[(combo, "CJK/emoji at caps")][0]
    }

    check(
        "BE3: CJK/emoji-heavy history at caps fails closed for every "
        "combination containing heat, and for most others",
        all("HEAT" not in combo for combo in be_cjk_fit)
        and len(be_cjk_fit) < len(be_combos) // 2,
    )

    print(
        "BE3 token verdict fits (of 32):",
        {name: sum(be_verdict[(c, name)][0] for c in be_combos)
         for name in be_variants},
        "| over the 19,000-character cap (of 32):",
        {name: sum(not be_verdict[(c, name)][1] for c in be_combos)
         for name in be_variants},
    )

    be_cap_only = [
        key for key, (tokens_ok, chars_ok) in be_verdict.items()
        if tokens_ok and not chars_ok
    ]

    check(
        "BE3: the character cap blocks on its own where the token estimate "
        "fits",
        len(be_cap_only) > 0
        and all(
            "character limit" in (be_send(be_requests[key])[0] or "")
            and be_send(be_requests[key])[1] == []
            for key in be_cap_only[:3]
        ),
    )

    # BE9: the production three-hazard request (offline reconstruction).
    be_triple_titles = [s["section"] for s in be_sections_for(be_triple)]
    be_triple_budget = be_budgets[(be_triple, "typical")]

    check(
        "BE9: the three-hazard request carries all 11 sections, unchanged",
        len(be_triple_titles) == 11
        and be_triple_messages.budget_parts.blocks
        == tuple(policy_blocks(be_sections_for(be_triple)))
        and all(
            item["text"] in be_triple_messages[1]["content"]
            for item in be_sections_for(be_triple)
        )
        and any(t.startswith("6.1 ") for t in be_triple_titles)
        and any(t.startswith("6.3 ") for t in be_triple_titles)
        and any(t.startswith("6.5 ") for t in be_triple_titles)
        and any(t.startswith("7. ") for t in be_triple_titles),
    )

    check(
        "BE9: the three-hazard request is sent once under the real limits",
        be_error is None
        and len(be_posts) == 1
        and be_triple_budget["method"] == "component"
        and be_triple_budget["estimated_input_tokens"] <= be_limit
        and len(be_guard) <= MAX_REQUEST_CHARS
        and be_posts[0]["messages"] == list(be_triple_messages),
    )

    # BE7: pins, strict schema and the legacy fallback.
    check(
        "BE7: the pinned model, reasoning effort and response format "
        "match the request configuration",
        estimator_pins_match()
        and ESTIMATOR_PINS["model"] == MODEL == "openai/gpt-oss-20b"
        and ESTIMATOR_PINS["reasoning_effort"] == REASONING_EFFORT == "low"
        and len(be_posts) == 1
        and be_posts[0]["reasoning_effort"] == "low"
        and be_posts[0]["response_format"] == RESPONSE_FORMAT
        and be_posts[0]["model"] == MODEL
        and be_posts[0]["temperature"] == 0.1
        and be_posts[0]["max_completion_tokens"] == 1000,
    )

    check(
        "BE7: the response-format section is the real strict schema",
        response_format_section_text()
        == "\n\n# Response Formats\n\n## weather_safety_assessment\n\n"
        + be_json.dumps(
            RESPONSE_FORMAT["json_schema"]["schema"],
            separators=(",", ":"),
        )
        and RESPONSE_FORMAT["json_schema"]["strict"] is True,
    )

    be_changed_format = be_json.loads(json_text(RESPONSE_FORMAT))
    be_changed_format["json_schema"]["schema"]["properties"]["extra"] = {
        "type": "string"
    }

    for label, overrides in {
        "model": {"MODEL": "other/model"},
        "reasoning effort": {"REASONING_EFFORT": "high"},
        "response schema": {"RESPONSE_FORMAT": be_changed_format},
    }.items():
        budget, legacy = with_globals(
            overrides,
            lambda: (
                be_budget(be_triple_messages),
                estimate_tokens(serialized_request(be_triple_messages)),
            ),
        )
        error, posts = be_send(be_triple_messages, **overrides)

        check(
            f"BE7: a different {label} falls back to the legacy estimate",
            budget["method"] == "legacy"
            and budget["estimated_input_tokens"] == legacy
            and "token budget" in (error or "")
            and posts == [],
        )

    check(
        "BE7: a plain message list uses the legacy estimate",
        be_budget(list(be_triple_messages))["method"] == "legacy",
    )

    # BE12: fail-closed behaviour.
    def be_blocked(messages, **overrides):
        error, posts = be_send(messages, **overrides)

        return (
            error is not None
            and "could not be verified" in error
            and posts == []
        )

    be_parts = be_triple_messages.budget_parts

    def be_consistent(parts):
        """Messages rebuilt from the given parts, so only the parts'
        own defect is under test."""
        return BudgetedMessages(
            [
                {"role": "system", "content": parts.system},
                {"role": "user", "content": parts.preamble
                 + POLICY_BLOCK_JOIN.join(parts.blocks) + EVIDENCE_HEADING
                 + parts.evidence_text},
            ],
            parts,
        )

    def be_parts_with(**changes):
        values = {
            "system": be_parts.system, "preamble": be_parts.preamble,
            "blocks": be_parts.blocks,
            "evidence_text": be_parts.evidence_text,
            "history_text": be_parts.history_text,
        }
        values.update(changes)

        return RequestParts(**values)

    be_mutated = BudgetedMessages(
        [
            dict(be_triple_messages[0]),
            {"role": "user",
             "content": be_triple_messages[1]["content"] + " "},
        ],
        be_parts,
    )
    be_extra_key = BudgetedMessages(
        [dict(be_triple_messages[0], extra="x"), be_triple_messages[1]],
        be_parts,
    )

    check(
        "BE12: malformed or mismatching parts fail closed before sending",
        be_blocked(BudgetedMessages(
            list(be_triple_messages), be_parts_with(system=be_parts.system + "x")
        ))
        and be_blocked(be_mutated)
        and be_blocked(be_extra_key)
        and be_blocked(BudgetedMessages(
            list(be_triple_messages), be_parts_with(blocks=be_parts.blocks[:-1])
        ))
        and be_blocked(BudgetedMessages(
            list(be_triple_messages),
            be_parts_with(blocks=list(be_parts.blocks) + [1]),
        ))
        and be_blocked(BudgetedMessages(list(be_triple_messages), "parts"))
        and be_blocked(BudgetedMessages(list(be_triple_messages), object())),
    )

    check(
        "BE12: evidence that is not valid JSON, or a history that is not "
        "the evidence's own, fails closed",
        be_blocked(be_consistent(
            be_parts_with(evidence_text='{"city": "Dubai"', history_text="[]")
        ))
        and be_blocked(be_consistent(be_parts_with(history_text="[]")))
        and be_blocked(be_consistent(
            be_parts_with(history_text=be_parts.history_text + " ")
        ))
        and be_blocked(be_consistent(
            be_parts_with(evidence_text="[1, 2]", history_text="[]")
        )),
    )

    be_first = next(iter(STATIC_TOKEN_TABLE))
    be_without_first = {
        key: value for key, value in STATIC_TOKEN_TABLE.items()
        if key != be_first
    }

    check(
        "BE12: a stale, incomplete or invalid static table fails closed",
        be_blocked(
            be_triple_messages,
            STATIC_TOKEN_TABLE=be_without_first,
            STATIC_TOKEN_TABLE_META=dict(
                STATIC_TOKEN_TABLE_META, entries=len(be_without_first)
            ),
        )
        and be_blocked(be_triple_messages, STATIC_TOKEN_TABLE={})
        and be_blocked(
            be_triple_messages,
            STATIC_TOKEN_TABLE=dict(STATIC_TOKEN_TABLE, bad=1),
        )
        and be_blocked(
            be_triple_messages,
            STATIC_TOKEN_TABLE_META=dict(
                STATIC_TOKEN_TABLE_META, encoding="cl100k_base"
            ),
        )
        and be_blocked(be_triple_messages, STATIC_TOKEN_TABLE=None)
        and be_blocked(
            be_triple_messages,
            STATIC_TOKEN_TABLE=dict(STATIC_TOKEN_TABLE, **{be_first: 0}),
        ),
    )

    check(
        "BE12: an unverified estimate is a typed REQUEST_BUDGET_EXCEEDED "
        "failure with the existing outcome",
        issubclass(RequestBudgetError, ClassifiedFailure)
        and RequestBudgetError.failure_code == "REQUEST_BUDGET_EXCEEDED"
        and RequestBudgetError.outcome == "UNAVAILABLE"
        and unverified_estimate("x").failure_code
        == "REQUEST_BUDGET_EXCEEDED",
    )

    be_oversize_error, be_oversize_posts = be_send(
        BudgetedMessages(
            list(be_triple_messages), be_parts_with(system="x")
        ),
        MAX_REQUEST_CHARS=100,
    )

    check(
        "BE12: the character guard runs before an unverifiable estimate",
        "character limit" in (be_oversize_error or "")
        and be_oversize_posts == [],
    )

    be_unicode_weather = rv_weather()
    be_unicode_weather["timezone"] = "Asia/Dubaï"
    be_unicode_budget = be_budget(
        be_build(be_triple, be_variants["typical"], be_unicode_weather)
    )
    be_text_weather = rv_weather()
    be_text_weather["timezone"] = "A" * (NUMERIC_EVIDENCE_TEXT_BYTES_LIMIT + 5)
    be_text_budget = be_budget(
        be_build(be_triple, be_variants["typical"], be_text_weather)
    )

    check(
        "BE12: non-ASCII or text-heavy numeric evidence is counted at one "
        "token per byte, normal evidence at the empirical ratio",
        "1 token per byte"
        in be_unicode_budget["components"]["numeric evidence bound"]
        and be_unicode_budget["components"]["numeric evidence tokens"]
        == be_unicode_budget["components"]["numeric evidence bytes"]
        and "1 token per byte"
        in be_text_budget["components"]["numeric evidence bound"]
        and be_text_budget["components"]["numeric evidence tokens"]
        == be_text_budget["components"]["numeric evidence bytes"]
        and be_triple_budget["components"]["numeric evidence bound"]
        == "empirical 0.65 tokens per byte",
    )

    be_cjk = be_requests[(be_triple, "CJK/emoji at caps")]
    be_cjk_error, be_cjk_posts = be_send(be_cjk)

    check(
        "BE12: CJK/emoji-heavy history over the budget fails closed with "
        "the existing token-budget error and no truncation",
        be_budgets[(be_triple, "CJK/emoji at caps")][
            "estimated_input_tokens"
        ] > be_limit
        and "token budget" in (be_cjk_error or "")
        and be_cjk_posts == []
        and be_cjk.budget_parts.history_text
        == json_text(summarize_history(be_variants["CJK/emoji at caps"]))
        and be_cjk.budget_parts.blocks
        == tuple(policy_blocks(be_sections_for(be_triple))),
    )

    # BE10 and BE11: boundaries, on requests whose padding is exactly one
    # estimated token and one character per ASCII byte of history.
    be_blocks = list(be_triple_messages.budget_parts.blocks)
    be_context = be_json.loads(be_triple_messages.budget_parts.evidence_text)

    def be_padded(pad):
        history = [{"advice": "a" * pad}]
        evidence_text = json_text(
            dict(be_context, recent_agent_history=history)
        )

        return be_consistent(RequestParts(
            SYSTEM_PROMPT.strip(), USER_PROMPT_PREAMBLE, be_blocks,
            evidence_text, json_text(history),
        ))

    be_pad_zero = be_budget(be_padded(0))["estimated_input_tokens"]
    be_pad_one = be_budget(be_padded(1))["estimated_input_tokens"]
    be_token_pad = be_limit - be_pad_zero
    be_relax_chars = {"MAX_REQUEST_CHARS": 10 ** 9}
    be_relax_tokens = {"MAX_ESTIMATED_REQUEST_TOKENS": 10 ** 9}

    check(
        "BE10: one byte of history is exactly one estimated token",
        be_pad_one - be_pad_zero == 1 and be_token_pad > 0,
    )

    be_below = be_send(be_padded(be_token_pad - 1), **be_relax_chars)
    be_at = be_send(be_padded(be_token_pad), **be_relax_chars)
    be_above = be_send(be_padded(be_token_pad + 1), **be_relax_chars)

    check(
        "BE10: 7,999 and 8,000 estimated total tokens are sent, 8,001 is "
        "blocked",
        be_budget(be_padded(be_token_pad - 1))["estimated_input_tokens"]
        + MAX_COMPLETION_TOKENS == MAX_ESTIMATED_REQUEST_TOKENS - 1
        and be_budget(be_padded(be_token_pad))["estimated_input_tokens"]
        + MAX_COMPLETION_TOKENS == MAX_ESTIMATED_REQUEST_TOKENS
        and be_below[0] is None and len(be_below[1]) == 1
        and be_at[0] is None and len(be_at[1]) == 1
        and "token budget" in (be_above[0] or "") and be_above[1] == [],
    )

    be_base_chars = len(serialized_request(be_padded(0)))

    def be_chars_target(target):
        return be_padded(target - be_base_chars)

    be_c_below = be_send(be_chars_target(MAX_REQUEST_CHARS - 1), **be_relax_tokens)
    be_c_at = be_send(be_chars_target(MAX_REQUEST_CHARS), **be_relax_tokens)
    be_c_above = be_send(be_chars_target(MAX_REQUEST_CHARS + 1), **be_relax_tokens)

    check(
        "BE11: 18,999 and 19,000 request characters are sent, 19,001 is "
        "blocked",
        len(serialized_request(be_chars_target(MAX_REQUEST_CHARS - 1)))
        == MAX_REQUEST_CHARS - 1
        and len(serialized_request(be_chars_target(MAX_REQUEST_CHARS)))
        == MAX_REQUEST_CHARS
        and len(serialized_request(be_chars_target(MAX_REQUEST_CHARS + 1)))
        == MAX_REQUEST_CHARS + 1
        and be_c_below[0] is None and len(be_c_below[1]) == 1
        and be_c_at[0] is None and len(be_c_at[1]) == 1
        and "character limit" in (be_c_above[0] or "")
        and be_c_above[1] == [],
    )

    check(
        "BE11: each guard blocks on its own, and the character guard "
        "comes first when both trip",
        "token budget" in (
            be_send(be_padded(be_token_pad + 1), **be_relax_chars)[0] or ""
        )
        and "character limit" in (
            be_send(
                be_chars_target(MAX_REQUEST_CHARS + 1), **be_relax_tokens
            )[0] or ""
        )
        and "character limit" in (
            be_send(
                be_padded(be_token_pad + 1), MAX_REQUEST_CHARS=1000
            )[0] or ""
        ),
    )

    # The provider count above the estimate is logged and never fatal.
    be_real_estimate = request_token_budget

    def be_small_estimate(messages, request_text):
        budget = be_real_estimate(messages, request_text)
        budget["estimated_input_tokens"] = 100

        return budget

    be_underrun_out = io.StringIO()

    def be_underrun_run():
        with contextlib.redirect_stdout(be_underrun_out):
            return ask_groq(be_triple_messages)

    be_underrun_error = None

    try:
        with_globals(
            {"HTTP": be_transport, "GROQ_API_KEY": "gsk_be_test",
             "request_token_budget": be_small_estimate},
            be_underrun_run,
        )
    except Exception as exc:
        be_underrun_error = exc

    check(
        "BE12: a provider count above the estimate is logged, not fatal; "
        "the usage record carries the estimator method",
        be_underrun_error is None
        and "ESTIMATOR UNDERRUN" in be_underrun_out.getvalue()
        and LAST_GROQ_USAGE.get("estimator_method") == "component"
        and LAST_GROQ_USAGE.get("legacy_estimated_input_tokens")
        == estimate_tokens(be_guard),
    )

    # BE8: recorded provider usage. Run #41 (be5524a, strict schema, HEAT
    # only): the legacy estimate was 6,889 and Groq reported 4,112 prompt
    # tokens. A HEAT-only request with 3 history records must be counted
    # at no less than that.
    be_run41 = {"legacy": 6889, "prompt_tokens": 4112}
    be_heat_budget = be_budgets[(("HEAT",), "typical")]
    be_allowances = (
        PROVIDER_ALLOWANCE_HISTORICAL_TOKENS + PROVIDER_ALLOWANCE_SCHEMA_TOKENS
    )

    check(
        "BE8: a HEAT-only request is counted at no less than the recorded "
        "run #41 provider usage, before and after the allowances",
        be_heat_budget["estimated_input_tokens"] - be_allowances
        >= be_run41["prompt_tokens"]
        and be_heat_budget["estimated_input_tokens"]
        >= be_run41["prompt_tokens"] + be_allowances
        and math.ceil(17221 / 2.5) == be_run41["legacy"]
        and math.ceil(17222 / 2.5) == be_run41["legacy"],
    )

    # BE4 to BE6: tokenizer-dependent groups.
    if be_encoding is not None:
        check(
            "BE4: the component estimate never falls below the exact local "
            "token count (160 requests)",
            all(
                be_budgets[key]["estimated_input_tokens"] - be_allowances
                >= be_exact_tokens(messages)
                for key, messages in be_requests.items()
            ),
        )

        check(
            "BE4: the allowances sit on top of an estimate that already "
            "covers the exact count (160 requests)",
            all(
                be_budgets[key]["estimated_input_tokens"]
                >= be_exact_tokens(messages) + be_allowances
                for key, messages in be_requests.items()
            ),
        )

        def be_numeric_case(weather):
            messages = be_build(("HEAT",), [], weather)
            text = messages.budget_parts.evidence_text.replace(
                messages.budget_parts.history_text, "", 1
            )
            size = len(text.encode("utf-8"))

            return (
                be_plain_tokens(text),
                -(-size * NUMERIC_EVIDENCE_TOKEN_RATIO[0]
                  // NUMERIC_EVIDENCE_TOKEN_RATIO[1]),
                size,
            )

        def be_random_weather(rng):
            weather = rv_weather()

            for record in weather["hourly_forecast_next_24h"]:
                record["temperature"] = round(rng.uniform(5, 49.9), 1)
                record["apparent_temperature"] = round(rng.uniform(5, 59.9), 1)
                record["humidity"] = rng.randint(1, 100)
                record["precipitation"] = rng.choice(
                    [0.0, round(rng.uniform(0.1, 9.9), 1),
                     round(rng.uniform(10, 99.9), 1)]
                )
                record["wind"] = round(rng.uniform(0, 99.9), 1)
                record["gust"] = round(rng.uniform(0, 149.9), 1)
                record["visibility"] = round(rng.uniform(0, 99999.9), 1)
                record["weather_code"] = rng.choice([0, 1, 3, 61, 95, 99])
                record["cape"] = round(rng.uniform(0, 9999.9), 1)

            return weather

        be_evidence_rng = random.Random(11)
        be_cases = [rv_weather()] + [
            be_random_weather(be_evidence_rng) for _ in range(120)
        ]
        be_results = [be_numeric_case(weather) for weather in be_cases]

        check(
            "BE5: numeric evidence stays within the empirical bound for "
            "the realistic and 120 random payloads",
            all(tokens <= bound for tokens, bound, _ in be_results)
            and max(tokens / size for tokens, _, size in be_results) < 0.6,
        )

        be_classes = {
            "English": lambda size: (be_prose * 6)[:size],
            "letters and spaces": lambda size: "".join(
                be_rng.choice(string.ascii_letters + "   ")
                for _ in range(size)),
            "alphanumeric": lambda size: "".join(
                be_rng.choice(string.ascii_letters + string.digits)
                for _ in range(size)),
            "punctuation": lambda size: "".join(
                be_rng.choice(string.punctuation) for _ in range(size)),
            "printable ASCII": lambda size: "".join(
                be_rng.choice(string.printable.strip()) for _ in range(size)),
            "Arabic": lambda size: be_field("Arabic", size),
            "CJK/emoji mix": lambda size: be_field("CJK", size),
        }
        be_fuzz_ok = True
        be_fuzz_covered = True

        for label, make in be_classes.items():
            for _ in range(40):
                history = [{
                    "timestamp": "2026-11-09T04:10:11.417000+00:00",
                    "city": make(45), "advice": make(125),
                    "alert_sent": False, "alert_message": make(85),
                }] * MAX_HISTORY_RECORDS
                messages = be_build(("HEAT",), history)
                text = messages.budget_parts.history_text
                tokens = be_plain_tokens(text)
                budget = be_budget(messages)

                be_fuzz_ok &= tokens <= len(text.encode("utf-8"))
                be_fuzz_covered &= (
                    budget["components"]["history tokens (1 per byte)"]
                    >= tokens
                )

        check(
            "BE6: history tokens never exceed history bytes, and the "
            "estimate covers them, across 7 text classes x 40 samples",
            be_fuzz_ok and be_fuzz_covered,
        )
    else:
        for name in (
            "BE4 estimate versus the exact local count",
            "BE5 numeric evidence bound over payloads",
            "BE6 history token fuzz and Unicode classes",
        ):
            print("SKIP:", name)

    # BE13: preservation specific to this change.
    be_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    be_imported = {
        alias.name.split(".")[0]
        for node in be_tree.body if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in be_tree.body if isinstance(node, ast.ImportFrom)
    }
    be_function_imports = {
        alias.name.split(".")[0]
        for node in be_tree.body
        if isinstance(node, ast.FunctionDef) and node.name != "run_tests"
        for sub in ast.walk(node) if isinstance(sub, ast.Import)
        for alias in sub.names
    } | {
        (sub.module or "").split(".")[0]
        for node in be_tree.body
        if isinstance(node, ast.FunctionDef) and node.name != "run_tests"
        for sub in ast.walk(node) if isinstance(sub, ast.ImportFrom)
    }

    check(
        "BE13: no runtime tokenizer dependency: tiktoken is not imported "
        "at module level or inside any production function",
        "tiktoken" not in be_imported
        and "tiktoken" not in be_function_imports,
    )

    be_alert_writes = [
        sub for node in ast.walk(be_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in (
            "request_token_budget", "verify_request_parts", "ask_groq",
            "assess_weather", "print_budget_components",
            "static_token_texts", "verify_static_token_table",
            "legacy_token_budget", "static_token_count",
        )
        for sub in ast.walk(node)
        if isinstance(sub, ast.Subscript)
        and isinstance(sub.slice, ast.Constant)
        and sub.slice.value in (
            "send_alert", "alert_message", "approval_status",
            "notification_status",
        )
        and isinstance(sub.ctx, ast.Store)
    ]

    check(
        "BE13: the estimator, guard and request assembly never write an "
        "alert field",
        not be_alert_writes,
    )

    check(
        "BE13: the budgets and completion allowance are unchanged",
        MAX_COMPLETION_TOKENS == 1000
        and MAX_ESTIMATED_REQUEST_TOKENS == 8000
        and MAX_REQUEST_CHARS == 19000
        and MAX_POLICY_CHARS == 13500
        and PROVIDER_ALLOWANCE_HISTORICAL_TOKENS == 192
        and PROVIDER_ALLOWANCE_SCHEMA_TOKENS == 300,
    )

    if be_skipped:
        print("Skipped (tokenizer tests):", "; ".join(be_skipped))

    for name, value in saved_actions_env.items():
        if value is not None:
            os.environ[name] = value

    print("\n========== TEST RESULTS ==========")
    print("Passed:", passed)
    print("Failed:", failed)
    print("Total:", passed + failed)

    if wf_skipped:
        print("Skipped (no usable bash):", ", ".join(wf_skipped))

    if failed:
        raise SystemExit(1)


# ============================================================
# 14. COMMAND LINE
# ============================================================

def main(argv):
    """Command-line entry point. Returns the process exit code."""
    mode = argv[1] if len(argv) > 1 else "agent"

    if mode == "search":
        query = " ".join(argv[2:])

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
            return 1

        return 0

    if mode == "test":
        run_tests()
        return 0

    if mode in ("agent-dry", "agent"):
        try:
            outcome = run_agent_job(dry=(mode == "agent-dry"))
        except Exception as exc:
            emit_unhandled_failure(exc)
            raise

        return RUN_OUTCOME_EXIT_CODES[outcome]

    print("Available commands:")
    print("python weather_log.py search heat risk")
    print("python weather_log.py test")
    print("python weather_log.py agent-dry")
    print("python weather_log.py agent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

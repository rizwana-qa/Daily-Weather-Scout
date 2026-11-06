"""
Daily Weather Scout - tool-using weather agent.

Required environment variables:
    GROQ_API_KEY : Groq API key
    SHEET_URL    : Google Apps Script web-app URL

Usage:
    python weather_log.py                 # run agent and write result to Sheet
    python weather_log.py agent-dry       # run agent without writing to Sheet
    python weather_log.py ask <question>  # ask a weather question
    python weather_log.py test            # run five live-model scenario tests

The agent can only call the explicitly allow-listed tools below. The Python
application validates tool arguments and applies final alert guardrails.
"""

import json
import os
import sys
import time
from typing import Any, Callable

import requests


# ---------- Settings ----------
CITY = os.getenv("WEATHER_CITY", "Lahore")
CITIES = {
    "Lahore": (31.52, 74.32),
    "Karachi": (24.86, 67.01),
    "Islamabad": (33.69, 73.05),
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_STEPS = 6
WEATHER_TIMEOUT = 30
GROQ_TIMEOUT = 60
SHEET_TIMEOUT = 30
MAX_HISTORY_ITEMS = 10
ALERT_MESSAGE_MAX_LENGTH = 300

if CITY not in CITIES:
    raise SystemExit(
        f"WEATHER_CITY must be one of: {', '.join(CITIES)}. Received: {CITY}"
    )


# ---------- Shared HTTP / configuration helpers ----------
def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def post_to_sheet(payload: dict[str, Any]) -> None:
    sheet_url = require_env("SHEET_URL")
    response = requests.post(sheet_url, json=payload, timeout=SHEET_TIMEOUT)
    response.raise_for_status()
    if response.text.strip() != "OK":
        raise RuntimeError(f"Google Sheet script returned an unexpected response: {response.text[:300]}")


# ---------- 1. Weather data: fetched by Python, never invented by the AI ----------
def get_weather(city: str) -> dict[str, Any]:
    if city not in CITIES:
        raise ValueError(f"City is not allowed. Choose one of: {', '.join(CITIES)}")

    latitude, longitude = CITIES[city]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "timezone": "auto",
        "past_days": 5,
        "forecast_days": 1,
    }
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=WEATHER_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()

    daily = body.get("daily")
    if not isinstance(daily, dict):
        raise RuntimeError("Weather service response did not contain daily data.")

    required_fields = (
        "time",
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_max",
    )
    if any(not isinstance(daily.get(field), list) for field in required_fields):
        raise RuntimeError("Weather service response is missing required daily fields.")

    # Use the five completed days. The current day may be incomplete.
    days = []
    for index in range(min(5, len(daily["time"]))):
        high = daily["temperature_2m_max"][index]
        low = daily["temperature_2m_min"][index]
        wind = daily["wind_speed_10m_max"][index]
        if high is None or low is None or wind is None:
            continue
        days.append({
            "date": daily["time"][index],
            "high": float(high),
            "low": float(low),
            "wind": float(wind),
        })

    if len(days) != 5:
        raise RuntimeError(
            f"Expected five complete days of weather data; received {len(days)}."
        )

    return {
        "city": city,
        "days": days,
        "avg_high": round(sum(item["high"] for item in days) / len(days), 1),
        "avg_low": round(sum(item["low"] for item in days) / len(days), 1),
        "peak_wind": round(max(item["wind"] for item in days), 1),
    }


# ---------- 2. Groq API ----------
def ask_groq(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    api_key = require_env("GROQ_API_KEY")
    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            **extra,
        },
        timeout=GROQ_TIMEOUT,
    )
    response.raise_for_status()

    body = response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Groq response did not contain a completion choice.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Groq response did not contain a valid assistant message.")
    return message


def parse_decision(content: str | None) -> dict[str, Any] | None:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(decision, dict):
        return None

    expected_keys = {"advice", "reasoning", "send_alert", "alert_message"}
    if set(decision) != expected_keys:
        return None

    if not (
        isinstance(decision["advice"], str)
        and isinstance(decision["reasoning"], str)
        and isinstance(decision["send_alert"], bool)
        and isinstance(decision["alert_message"], str)
    ):
        return None

    if not decision["advice"].strip() or not decision["reasoning"].strip():
        return None

    return decision


# ---------- 3. Allow-listed tools ----------
def tool_get_weather(city: str) -> dict[str, Any]:
    try:
        data = get_weather(city)
        return {
            "city": data["city"],
            "days": data["days"],
            "avg_high": data["avg_high"],
            "avg_low": data["avg_low"],
            "peak_wind": data["peak_wind"],
            "daily_highs": [item["high"] for item in data["days"]],
        }
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        return {"error": f"Weather data unavailable: {exc}"}


def tool_get_history(n: int = 5) -> dict[str, Any]:
    try:
        count = max(1, min(int(n), MAX_HISTORY_ITEMS))
    except (TypeError, ValueError):
        count = 5

    try:
        sheet_url = require_env("SHEET_URL")
        response = requests.get(
            sheet_url,
            params={"action": "history", "n": count},
            timeout=SHEET_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("error"):
            return {"error": str(result["error"])}
        if not isinstance(result, (dict, list)):
            return {"error": "History endpoint returned an unsupported response."}
        return result
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        return {"error": f"History unavailable: {exc}"}


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the five most recent completed days of weather for an allowed city, "
                "including daily highs/lows, average high/low and peak wind in km/h."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": list(CITIES)}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": (
                "Read recent decisions from the Google Sheet Agent history, including "
                "whether a similar alert was already sent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_HISTORY_ITEMS,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
]

REAL_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_weather": tool_get_weather,
    "get_history": tool_get_history,
}


# ---------- 4. Agent loop ----------
def run_weather_agent(
    question: str | None = None,
    tools: dict[str, Callable[..., dict[str, Any]]] | None = None,
    max_steps: int = MAX_STEPS,
) -> tuple[dict[str, Any], list[str]]:
    available_tools = tools or REAL_TOOLS
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    system_prompt = (
        f"You are a cautious weather agent for {CITY}. Use get_weather and get_history "
        "before making an alert decision. Weather facts must come only from tool results. "
        "The weather tool reports the five most recent completed days, not a live observation. "
        "Consider unusually high temperatures, strong wind, large changes and previous alerts. "
        "Do not repeat an alert for the same continuing event if history shows it was already sent. "
        "If weather data is unavailable, do not alert. If history is unavailable, do not send an "
        "alert because duplicate-alert status cannot be checked. Never invent numbers or claim "
        "that an alert was sent. When finished, return JSON only with exactly these keys: "
        "advice (one or two concise sentences), reasoning (two or three sentences using available "
        "numbers), send_alert (boolean), alert_message (short message or empty string)."
    )
    user_prompt = question or (
        f"Check the five most recent completed days for {CITY}. Review recent agent history, "
        "then decide whether advice or an alert is warranted."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    used: list[str] = []
    weather_failed = False
    history_failed = False

    for step in range(1, max_steps + 1):
        message = ask_groq(messages, tools=AGENT_TOOLS, temperature=0.2)
        calls = message.get("tool_calls")

        if not calls:
            decision = parse_decision(message.get("content"))
            if decision is not None:
                if weather_failed or history_failed:
                    decision["send_alert"] = False
                    decision["alert_message"] = ""
                return decision, used

            messages.append({
                "role": "assistant",
                "content": message.get("content") or "",
            })
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was not valid. Return one JSON object only, "
                    "with exactly: advice, reasoning, send_alert, alert_message. "
                    "Use a boolean for send_alert."
                ),
            })
            continue

        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:
            function_data = call.get("function", {})
            name = function_data.get("name")
            raw_arguments = function_data.get("arguments") or "{}"

            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
            except (json.JSONDecodeError, ValueError):
                result = {"error": "Invalid tool arguments."}
                name = str(name or "unknown")
            else:
                print(f"STEP {step} -> {name} {arguments}")
                function = available_tools.get(name)
                if function is None:
                    result = {"error": "Tool is not allowed."}
                else:
                    try:
                        if name == "get_weather":
                            city = arguments.get("city")
                            if city not in CITIES:
                                raise ValueError("City is not allowed.")
                        elif name == "get_history":
                            arguments = {"n": arguments.get("n", 5)}
                        result = function(**arguments)
                        if not isinstance(result, dict):
                            result = {"error": "Tool returned an invalid result."}
                    except (TypeError, ValueError, requests.RequestException) as exc:
                        result = {"error": str(exc)}

            if name == "get_weather" and "error" in result:
                weather_failed = True
            if name == "get_history" and "error" in result:
                history_failed = True

            used.append(str(name or "unknown"))
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })

    raise RuntimeError(f"The agent did not finish within {max_steps} steps.")


# ---------- 5. Guardrails and Sheet logging ----------
def apply_guardrails(decision: dict[str, Any]) -> dict[str, Any]:
    """Validate and constrain the AI's proposed decision before any side effect."""
    advice = decision["advice"].strip()[:1000]
    reasoning = decision["reasoning"].strip()[:1500]
    message = decision["alert_message"].strip()[:ALERT_MESSAGE_MAX_LENGTH]

    send_alert = decision["send_alert"] is True and bool(message)
    return {
        "advice": advice,
        "reasoning": reasoning,
        "send_alert": send_alert,
        "alert_message": message if send_alert else "",
    }


def run_agent_job(dry: bool = False) -> dict[str, Any]:
    raw_decision, used_tools = run_weather_agent()
    decision = apply_guardrails(raw_decision)

    print("\nADVICE:", decision["advice"])
    print("REASONING:", decision["reasoning"])
    print("ALERT:", decision["send_alert"], decision["alert_message"])

    payload = {
        "type": "agent",
        "city": CITY,
        "tools_used": ", ".join(used_tools),
        **decision,
    }

    if dry:
        print("\nDry run: nothing was sent to the Google Sheet.")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return payload

    post_to_sheet(payload)
    print("Google Sheet updated successfully.")
    return payload


# ---------- 6. Five scenario tests ----------
SCENARIOS = [
    {
        "name": "Normal day",
        "weather": {
            "city": CITY, "avg_high": 31.0, "avg_low": 21.0, "peak_wind": 12.0,
            "daily_highs": [30, 31, 31, 32, 31],
        },
        "history": [],
        "expect_alert": False,
    },
    {
        "name": "Heatwave",
        "weather": {
            "city": CITY, "avg_high": 43.0, "avg_low": 30.0, "peak_wind": 14.0,
            "daily_highs": [38, 41, 43, 45, 48],
        },
        "history": [],
        "expect_alert": True,
    },
    {
        "name": "Storm winds",
        "weather": {
            "city": CITY, "avg_high": 29.0, "avg_low": 22.0, "peak_wind": 75.0,
            "daily_highs": [29, 28, 29, 30, 29],
        },
        "history": [],
        "expect_alert": True,
    },
    {
        "name": "Same heatwave, already alerted",
        "weather": {
            "city": CITY, "avg_high": 44.0, "avg_low": 31.0, "peak_wind": 14.0,
            "daily_highs": [44, 45, 44, 45, 46],
        },
        "history": [{
            "time": "yesterday",
            "city": CITY,
            "advice": "Extreme heat",
            "alert_sent": True,
            "alert_message": "Heatwave: highs above 44 C. Stay indoors at midday.",
        }],
        "expect_alert": False,
    },
    {
        "name": "Weather service down",
        "weather": {"error": "service unavailable"},
        "history": [],
        "expect_alert": False,
    },
]


def run_tests() -> None:
    """Run the five scenarios through the live Groq model with mocked data tools.

    This uses Groq API calls, but does not write to the Google Sheet.
    Results can vary because the agent decision is model-generated.
    """
    passed = 0

    for scenario in SCENARIOS:
        def fake_weather(city: str, current: dict[str, Any] = scenario) -> dict[str, Any]:
            return current["weather"]

        def fake_history(n: int = 5, current: dict[str, Any] = scenario) -> dict[str, Any]:
            return {"items": current["history"]}

        fake_tools = {
            "get_weather": fake_weather,
            "get_history": fake_history,
        }

        try:
            raw_decision, _ = run_weather_agent(
                question=(
                    f"Scenario: {scenario['name']}. Evaluate the supplied tool data. "
                    "Make the alert decision according to your instructions."
                ),
                tools=fake_tools,
            )
            decision = apply_guardrails(raw_decision)
            actual = decision["send_alert"]
            ok = actual is scenario["expect_alert"]
            if ok:
                passed += 1
            print(
                f"{'PASS' if ok else 'FAIL'} | {scenario['name']} | "
                f"expected alert {scenario['expect_alert']}, got {actual}"
            )
            print("      ", decision["reasoning"])
        except Exception as exc:
            print(f"FAIL | {scenario['name']} | test error: {exc}")

        time.sleep(2)  # Be considerate of API rate limits.

    print(f"\n{passed} of {len(SCENARIOS)} scenarios passed.")
    if passed != len(SCENARIOS):
        raise SystemExit(1)


# ---------- 7. Command-line entry point ----------
def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "agent"

    try:
        if mode == "ask":
            question = " ".join(sys.argv[2:]).strip()
            if not question:
                raise SystemExit('Usage: python weather_log.py ask "weather question"')
            answer, tools_used = run_weather_agent(question=question)
            print(answer)
            print("Tools used:", ", ".join(tools_used))
        elif mode == "dry":
            # Backward-compatible alias for a no-write agent run.
            run_agent_job(dry=True)
        elif mode == "agent-dry":
            run_agent_job(dry=True)
        elif mode == "agent":
            run_agent_job(dry=False)
        elif mode == "test":
            run_tests()
        else:
            raise SystemExit(
                "Unknown mode. Use: agent, agent-dry, dry, ask, or test."
            )
    except requests.RequestException as exc:
        raise SystemExit(f"Network/API request failed: {exc}") from exc
    except (RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Weather agent failed safely: {exc}") from exc


if __name__ == "__main__":
    main()

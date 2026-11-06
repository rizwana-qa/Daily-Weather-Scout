
import os
import sys
import json
import requests
from time import sleep

# ============================================================
# 1. SETTINGS
# ============================================================

CITY = "Lahore"

CITIES = {
    "Lahore": (31.52, 74.32),
    "Karachi": (24.86, 67.01),
    "Islamabad": (33.69, 73.05),
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

MAX_STEPS = 6
REQUEST_TIMEOUT = 30


# ============================================================
# 2. ENVIRONMENT VALIDATION
# ============================================================

def validate_environment(require_sheet=False):
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Missing GROQ_API_KEY environment variable.")

    if require_sheet and not os.getenv("SHEET_URL"):
        raise SystemExit("Missing SHEET_URL environment variable.")


# ============================================================
# 3. WEATHER DATA
# ============================================================

def get_weather(city):
    if city not in CITIES:
        raise ValueError(f"City not supported: {city}")

    lat, lon = CITIES[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "wind_speed_10m_max"
        ),
        "timezone": "auto",
        "past_days": 5,
        "forecast_days": 1,
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    result = response.json()

    daily = result.get("daily", {})

    required_fields = [
        "time",
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_max",
    ]

    if not all(field in daily for field in required_fields):
        raise ValueError("Weather API returned incomplete data.")

    # Exclude today because today's data may be incomplete.
    days = []

    for i in range(min(5, len(daily["time"]))):
        high = daily["temperature_2m_max"][i]
        low = daily["temperature_2m_min"][i]
        wind = daily["wind_speed_10m_max"][i]

        if high is None or low is None or wind is None:
            continue

        days.append({
            "date": daily["time"][i],
            "high": high,
            "low": low,
            "wind": wind,
        })

    if not days:
        raise ValueError("No complete weather records available.")

    return {
        "city": city,
        "days": days,
        "avg_high": round(
            sum(x["high"] for x in days) / len(days), 1
        ),
        "avg_low": round(
            sum(x["low"] for x in days) / len(days), 1
        ),
        "peak_wind": max(x["wind"] for x in days),
    }


# ============================================================
# 4. GROQ API
# ============================================================

def ask_groq(messages, **extra):
    validate_environment()

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            **extra,
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Groq API error ({response.status_code}): "
            f"{response.text[:1000]}"
        )

    result = response.json()

    if not result.get("choices"):
        raise RuntimeError("Groq returned no choices.")

    return result["choices"][0]["message"]


# ============================================================
# 5. JSON PARSING
# ============================================================

def extract_json(text):
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return json.loads(text)


def get_json(prompt, keys):
    for attempt in range(2):

        message = ask_groq([
            {
                "role": "system",
                "content": "Return valid JSON only. No Markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ], temperature=0.2)

        try:
            data = extract_json(message.get("content"))

            if isinstance(data, dict) and all(
                key in data for key in keys
            ):
                return data

        except (json.JSONDecodeError, TypeError):
            pass

        print(f"Invalid JSON response. Attempt {attempt + 1}/2")

    raise RuntimeError("AI did not return valid JSON.")


# ============================================================
# 6. GOOGLE SHEETS
# ============================================================

def send_to_sheet(payload):
    validate_environment(require_sheet=True)

    response = requests.post(
        os.environ["SHEET_URL"],
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    if response.text.strip() != "OK":
        raise RuntimeError(
            f"Google Sheets script returned: {response.text}"
        )

    print("Successfully sent data to Google Sheets.")


def get_history(n=5):
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 10:
        raise ValueError("History limit must be between 1 and 10.")

    validate_environment(require_sheet=True)

    response = requests.get(
        os.environ["SHEET_URL"],
        params={"action": "history", "n": n},
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    try:
        return response.json()
    except ValueError:
        return {
            "error": "Could not read history: " + response.text[:100]
        }


# ============================================================
# 7. BASIC WEATHER JOB
# ============================================================

def run_job(dry=False):
    data = get_weather(CITY)

    table = "\n".join(
        f"{x['date']}: high {x['high']} C, "
        f"low {x['low']} C, wind {x['wind']} km/h"
        for x in data["days"]
    )

    prompt = (
        f"Weather for {CITY}, recent five complete days:\n{table}\n\n"
        "Return JSON with exactly these keys:\n"
        "trend: warming, cooling or steady\n"
        "wind_risk: low, medium or high\n"
        "advice: one sentence\n"
        "send_alert: boolean\n"
        "reason: one sentence\n"
        "Use only the supplied weather data."
    )

    result = get_json(
        prompt,
        ["trend", "wind_risk", "advice", "send_alert", "reason"],
    )

    if result["trend"] not in ("warming", "cooling", "steady"):
        raise ValueError("Invalid trend returned by AI.")

    if result["wind_risk"] not in ("low", "medium", "high"):
        raise ValueError("Invalid wind risk returned by AI.")

    if not isinstance(result["send_alert"], bool):
        raise ValueError("Invalid send_alert value.")

    payload = {
        "city": CITY,
        "period": (
            data["days"][0]["date"]
            + " to "
            + data["days"][-1]["date"]
        ),
        "avg_max": data["avg_high"],
        "avg_min": data["avg_low"],
        "peak_wind": data["peak_wind"],
        "trend": result["trend"],
        "wind_risk": result["wind_risk"],
        "advice": result["advice"],
        "send_alert": result["send_alert"],
        "reason": result["reason"],
    }

    print(json.dumps(payload, indent=2))

    if dry:
        print("Dry run. Nothing was sent.")
        return

    send_to_sheet(payload)


# ============================================================
# 8. AGENT TOOLS
# ============================================================

def tool_get_weather(city):
    if city not in CITIES:
        return {"error": "City not allowed."}

    data = get_weather(city)

    return {
        "city": data["city"],
        "avg_high": data["avg_high"],
        "avg_low": data["avg_low"],
        "peak_wind": data["peak_wind"],
        "daily_highs": [
            x["high"] for x in data["days"]
        ],
        "dates": [
            x["date"] for x in data["days"]
        ],
    }


def tool_get_history(n=5):
    return get_history(n)


REAL_TOOLS = {
    "get_weather": tool_get_weather,
    "get_history": tool_get_history,
}


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get recent weather data for a supported city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "enum": list(CITIES),
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": (
                "Get previous weather decisions and alerts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    }
                },
            },
        },
    },
]


# ============================================================
# 9. AGENT DECISION PARSING
# ============================================================

def parse_decision(text):
    try:
        data = extract_json(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    required = {
        "advice": str,
        "reasoning": str,
        "send_alert": bool,
        "alert_message": str,
    }

    for key, expected_type in required.items():
        if not isinstance(data.get(key), expected_type):
            return None

    return data


def apply_guardrails(decision):
    advice = decision["advice"].strip()
    reasoning = decision["reasoning"].strip()
    message = decision["alert_message"].strip()[:300]

    send = (
        decision["send_alert"] is True
        and bool(message)
    )

    return {
        "advice": advice,
        "reasoning": reasoning,
        "send_alert": send,
        "alert_message": message if send else "",
    }


# ============================================================
# 10. WEATHER AGENT
# ============================================================

def run_weather_agent(tools=None, max_steps=MAX_STEPS):

    if tools is None:
        tools = REAL_TOOLS

    system = (
        f"You are a weather agent for {CITY}. "
        "Use tools to obtain weather data and recent alert history. "
        "Compare weather conditions with previous days. "
        "Alert only when there is a significant or potentially "
        "dangerous condition. Avoid duplicate alerts. "
        "Do not invent numbers or facts. "
        "If weather or history data is unavailable, do not alert. "
        "Do not treat a missing history response as an empty history. "
        "When finished, return JSON only with exactly these keys: "
        "advice, reasoning, send_alert, alert_message. "
        "send_alert must be a JSON boolean."
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Check the weather for {CITY} and decide what to do.",
        },
    ]

    used = []

    for step in range(1, max_steps + 1):

        msg = ask_groq(
            messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        calls = msg.get("tool_calls")

        if not calls:

            decision = parse_decision(msg.get("content"))

            if decision:
                return decision, used

            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
            })

            messages.append({
                "role": "user",
                "content": (
                    "Return valid JSON only with the four required keys. "
                    "Do not add Markdown."
                ),
            })

            continue

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:

            name = call["function"]["name"]

            try:
                args = json.loads(
                    call["function"].get("arguments") or "{}"
                )
            except json.JSONDecodeError:
                args = None

            print(f"STEP {step} -> {name} {args}")

            fn = tools.get(name)

            try:
                if fn is None or not isinstance(args, dict):
                    raise ValueError("Tool or arguments not allowed.")

                if name == "get_weather":
                    if set(args) != {"city"}:
                        raise ValueError("Invalid weather arguments.")

                elif name == "get_history":
                    if set(args) - {"n"}:
                        raise ValueError("Invalid history arguments.")

                    args.setdefault("n", 5)

                result = fn(**args)

            except Exception as error:
                result = {"error": str(error)}

            used.append(name)

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            })

    raise RuntimeError("Agent exceeded its maximum steps.")


# ============================================================
# 11. AGENT JOB
# ============================================================

def run_agent_job(dry=False):

    decision, used = run_weather_agent()

    decision = apply_guardrails(decision)

    print("\nADVICE:", decision["advice"])
    print("REASONING:", decision["reasoning"])
    print("ALERT:", decision["send_alert"])
    print("MESSAGE:", decision["alert_message"])

    payload = {
        "type": "agent",
        "city": CITY,
        "tools_used": ", ".join(used),
        **decision,
    }

    if dry:
        print("\nDry run. Nothing was sent to the sheet or email.")
        return

    send_to_sheet(payload)


# ============================================================
# 12. TEST SCENARIOS
# ============================================================

SCENARIOS = [
    {
        "name": "Normal day",
        "weather": {
            "city": CITY,
            "avg_high": 31.0,
            "avg_low": 21.0,
            "peak_wind": 12.0,
            "daily_highs": [30, 31, 31, 32, 31],
        },
        "history": [],
        "expect_alert": False,
    },
    {
        "name": "Heatwave",
        "weather": {
            "city": CITY,
            "avg_high": 43.0,
            "avg_low": 30.0,
            "peak_wind": 14.0,
            "daily_highs": [38, 41, 43, 45, 48],
        },
        "history": [],
        "expect_alert": True,
    },
    {
        "name": "Storm winds",
        "weather": {
            "city": CITY,
            "avg_high": 29.0,
            "avg_low": 22.0,
            "peak_wind": 75.0,
            "daily_highs": [29, 28, 29, 30, 29],
        },
        "history": [],
        "expect_alert": True,
    },
    {
        "name": "Already alerted",
        "weather": {
            "city": CITY,
            "avg_high": 44.0,
            "avg_low": 31.0,
            "peak_wind": 14.0,
            "daily_highs": [44, 45, 44, 45, 46],
        },
        "history": [
            {
                "time": "yesterday",
                "city": CITY,
                "advice": "Extreme heat",
                "alert_sent": True,
                "alert_message": (
                    "Heatwave: highs above 44 C."
                ),
            }
        ],
        "expect_alert": False,
    },
    {
        "name": "Weather service down",
        "weather": {"error": "service unavailable"},
        "history": [],
        "expect_alert": False,
    },
]


# ============================================================
# 13. TEST RUNNER
# ============================================================

def run_tests():

    passed = 0

    for scenario in SCENARIOS:

        tools = {
            "get_weather": (
                lambda city, s=scenario: s["weather"]
            ),
            "get_history": (
                lambda n=5, s=scenario: s["history"]
            ),
        }

        try:
            decision, _ = run_weather_agent(tools=tools)

            decision = apply_guardrails(decision)

            ok = (
                decision["send_alert"]
                == scenario["expect_alert"]
            )

        except Exception as error:
            print(f"ERROR | {scenario['name']} | {error}")
            ok = scenario["expect_alert"] is False

        if ok:
            passed += 1

        print(
            f"{'PASS' if ok else 'FAIL'} | "
            f"{scenario['name']} | "
            f"Expected: {scenario['expect_alert']} | "
            f"Actual: {ok}"
        )

        sleep(2)

    print(f"\n{passed} of {len(SCENARIOS)} scenarios passed.")


# ============================================================
# 14. INTERACTIVE WEATHER QUESTION
# ============================================================

def run_agent(question):

    messages = [
        {
            "role": "system",
            "content": (
                "You answer weather questions. "
                "Use tools for weather facts. Never guess numbers."
            ),
        },
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):

        msg = ask_groq(
            messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        calls = msg.get("tool_calls")

        if not calls:
            return msg.get("content", "")

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:

            name = call["function"]["name"]

            try:
                args = json.loads(
                    call["function"].get("arguments") or "{}"
                )

                if name == "get_weather" and set(args) == {"city"}:
                    result = tool_get_weather(args["city"])

                elif name == "get_history" and set(args) <= {"n"}:
                    result = tool_get_history(args.get("n", 5))

                else:
                    result = {"error": "Tool or arguments not allowed."}

            except Exception as error:
                result = {"error": str(error)}

            print(f"STEP {step} -> {name}")

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            })

    raise RuntimeError("Question agent exceeded maximum steps.")


# ============================================================
# 15. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mode = sys.argv[1] if len(sys.argv) > 1 else "job"

    try:

        if mode == "ask":
            question = " ".join(sys.argv[2:])

            if not question:
                raise ValueError("Provide a weather question.")

            print(run_agent(question))

        elif mode == "dry":
            run_job(dry=True)

        elif mode == "agent":
            run_agent_job()

        elif mode == "agent-dry":
            run_agent_job(dry=True)

        elif mode == "test":
            run_tests()

        elif mode == "job":
            run_job()

        else:
            raise ValueError(
                "Unknown mode. Use: job, dry, ask, agent, agent-dry, test."
            )

    except (requests.RequestException, RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

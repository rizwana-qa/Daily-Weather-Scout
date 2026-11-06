import os
import sys
import json
import requests
# ---------- Settings ----------
CITY = "Lahore"   # change to your city (must be in CITIES below)
CITIES = {
   "Lahore": (31.52, 74.32),
   "Karachi": (24.86, 67.01),
   "Islamabad": (33.69, 73.05),
}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
# ---------- 1. Get weather data (your code, not the AI) ----------
def get_weather(city):
   lat, lon = CITIES[city]
   url = (
       "https://api.open-meteo.com/v1/forecast"
       f"?latitude={lat}&longitude={lon}"
       "&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max"
       "&timezone=auto&past_days=5&forecast_days=1"
   )
   d = requests.get(url, timeout=30).json()["daily"]
   days = []
   for i in range(5):  # the first 5 days; today is not finished, so we skip it
       days.append({
           "date": d["time"][i],
           "high": d["temperature_2m_max"][i],
           "low": d["temperature_2m_min"][i],
           "wind": d["wind_speed_10m_max"][i],
       })
   return {
       "city": city,
       "days": days,
       "avg_high": round(sum(x["high"] for x in days) / 5, 1),
       "avg_low": round(sum(x["low"] for x in days) / 5, 1),
       "peak_wind": max(x["wind"] for x in days),
   }
# ---------- 2. Talk to the AI ----------
def ask_groq(messages, **extra):
   r = requests.post(
       GROQ_URL,
       headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
       json={"model": MODEL, "messages": messages, **extra},
       timeout=60,
   )
   if r.status_code != 200:
       raise SystemExit("Groq error: " + r.text)
   return r.json()["choices"][0]["message"]
def get_json(prompt, keys):
   # Ask for JSON. If the reply is broken, try once more.
   for attempt in range(2):
       text = ask_groq(
           [{"role": "system", "content": "Reply with JSON only. No extra text."},
            {"role": "user", "content": prompt}],
           temperature=1.5,
       )["content"].strip()
       text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
       try:
           data = json.loads(text)
       except json.JSONDecodeError:
           print("Bad JSON, trying again")
           continue
       if all(k in data for k in keys):
           return data
       print("Missing fields, trying again")
   raise SystemExit("The AI did not return valid JSON")
# ---------- 3. The daily job ----------
def run_job(dry=False):
   data = get_weather(CITY)
   table = "\n".join(
       f"{x['date']}: high {x['high']}C, low {x['low']}C, max wind {x['wind']} km/h"
       for x in data["days"]
   )
   prompt = (
       f"Weather for {CITY}, last 5 days:\n{table}\n\n"
       "Return JSON with exactly these keys: "
       "trend (warming, cooling or steady), "
       "wind_risk (low, medium or high), "
       "advice (one sentence), "
       "send_alert (true only if something unusual needs attention, otherwise false), "
       "reason (one sentence). "
       "Use only the numbers above. Do not invent anything."
   )
   result = get_json(prompt, ["trend", "wind_risk", "advice", "send_alert", "reason"])
   print(result)
   payload = {
       "city": CITY,
       "period": data["days"][0]["date"] + " to " + data["days"][-1]["date"],
       "avg_max": data["avg_high"],
       "avg_min": data["avg_low"],
       "peak_wind": data["peak_wind"],
       "trend": result["trend"],
       "wind_risk": result["wind_risk"],
       "advice": result["advice"],
       "send_alert": result["send_alert"] is True,
       "reason": result["reason"],
   }
   if dry:
       print("Dry run. Nothing was sent to the sheet.")
       return
   response = requests.post(os.environ["SHEET_URL"], json=payload, timeout=30)
   print(response.status_code, response.text)
   if response.text != "OK":
       raise SystemExit("The sheet script said: " + response.text)
# ---------- 4. The agent: the AI chooses a tool, your code runs it ----------
TOOLS = [{
   "type": "function",
   "function": {
       "name": "get_weather",
       "description": "Get the 5 day weather summary (average high, average low, peak wind) for one city",
       "parameters": {
           "type": "object",
           "properties": {"city": {"type": "string", "enum": list(CITIES)}},
           "required": ["city"],
       },
   },
}]
def run_agent(question, max_steps=5):
   messages = [
       {"role": "system", "content": "You answer weather questions. Use the tool for facts. Never guess numbers."},
       {"role": "user", "content": question},
   ]
   for step in range(1, max_steps + 1):
       msg = ask_groq(messages, tools=TOOLS, temperature=0.2)
       calls = msg.get("tool_calls")
       if not calls:
           return msg["content"]          # no tool needed, this is the final answer
       messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
       for call in calls:
           name = call["function"]["name"]
           args = json.loads(call["function"]["arguments"])
           print("STEP", step, "->", name, args)       # log every decision
           if name == "get_weather" and args.get("city") in CITIES:
               data = get_weather(args["city"])
               result = {k: v for k, v in data.items() if k != "days"}
           else:
               result = {"error": "tool or city not allowed"}
           messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
   return "Stopped: too many steps"
# ---------- Start here ----------


# ---------- 5. The weather agent: the AI decides what to check and whether to alert ----------
from time import sleep
MAX_STEPS = 6   # the agent must finish within this many steps
def tool_get_weather(city):
   if city not in CITIES:
       return {"error": "city not allowed"}
   data = get_weather(city)
   summary = {k: v for k, v in data.items() if k != "days"}
   summary["daily_highs"] = [x["high"] for x in data["days"]]
   return summary
def tool_get_history(n=5):
   # Reads the agent's own recent decisions from your Google Sheet (Agent tab)
   r = requests.get(os.environ["SHEET_URL"], params={"action": "history", "n": n}, timeout=30)
   try:
       return r.json()
   except ValueError:
       return {"error": "could not read history: " + r.text[:100]}
REAL_TOOLS = {"get_weather": tool_get_weather, "get_history": tool_get_history}
AGENT_TOOLS = [
   {"type": "function", "function": {
       "name": "get_weather",
       "description": "Get the last 5 days of weather for a city: average high, average low, peak wind and the daily highs",
       "parameters": {"type": "object",
                      "properties": {"city": {"type": "string", "enum": list(CITIES)}},
                      "required": ["city"]}}},
   {"type": "function", "function": {
       "name": "get_history",
       "description": "Get your own most recent decisions from previous runs: time, city, advice, whether an alert was sent, and the alert message",
       "parameters": {"type": "object",
                      "properties": {"n": {"type": "integer", "description": "how many past runs, 1 to 10"}}}}},
]
def parse_decision(text):
   text = (text or "").strip()
   text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
   try:
       d = json.loads(text)
   except json.JSONDecodeError:
       return None
   ok = (isinstance(d, dict) and isinstance(d.get("advice"), str) and isinstance(d.get("reasoning"), str)
         and isinstance(d.get("send_alert"), bool) and isinstance(d.get("alert_message"), str))
   return d if ok else None
def apply_guardrails(d):
   # The AI recommends. Your code decides what is allowed.
   message = d["alert_message"].strip()[:300]            # a short alert only
   send = d["send_alert"] is True and message != ""      # no alert without a message
   return {"advice": d["advice"].strip(), "reasoning": d["reasoning"].strip(),
           "send_alert": send, "alert_message": message if send else ""}
def run_weather_agent(tools=REAL_TOOLS, max_steps=MAX_STEPS):
   system = (
       f"You are a weather agent for {CITY}. Use your tools to check the weather and your own recent decisions. "
       "Compare today with the recent days and with your past alerts. "
       "Send an alert only if something unusual or dangerous needs attention, such as extreme heat, "
       "strong wind or a big change, and only if the history does not show you already alerted about the same thing. "
       "Never invent numbers. If a tool returns an error, do not guess: say the data is unavailable and do not alert. "
       "When you are done, reply with JSON only, with exactly these keys: "
       "advice (one or two sentences for a general reader), "
       "reasoning (two or three sentences explaining your decision with numbers), "
       "send_alert (true or false), "
       "alert_message (a short message, or an empty string if there is no alert)."
   )
   messages = [{"role": "system", "content": system},
               {"role": "user", "content": f"Check the weather for {CITY} and decide what to do."}]
   used = []
   for step in range(1, max_steps + 1):
       msg = ask_groq(messages, tools=AGENT_TOOLS, temperature=0.2)
       calls = msg.get("tool_calls")
       if not calls:
           decision = parse_decision(msg.get("content"))
           if decision:
               return decision, used
           messages.append({"role": "assistant", "content": msg.get("content") or ""})
           messages.append({"role": "user", "content": "Reply with the JSON object only, with the 4 keys."})
           continue
       messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
       for call in calls:
           name = call["function"]["name"]
           args = json.loads(call["function"]["arguments"] or "{}")
           print("STEP", step, "->", name, args)
           fn = tools.get(name)
           if fn is None:
               result = {"error": "tool not allowed"}
           else:
               try:
                   result = fn(**args)
               except TypeError:
                   result = {"error": "bad arguments"}
           used.append(name)
           messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
   raise SystemExit("The agent did not finish in time")
def run_agent_job(dry=False):
   decision, used = run_weather_agent()
   decision = apply_guardrails(decision)
   print("\nADVICE:", decision["advice"])
   print("REASONING:", decision["reasoning"])
   print("ALERT:", decision["send_alert"], decision["alert_message"])
   payload = {"type": "agent", "city": CITY, "tools_used": ", ".join(used), **decision}
   if dry:
       print("\nDry run. Nothing was sent to the sheet or email.")
       return
   response = requests.post(os.environ["SHEET_URL"], json=payload, timeout=30)
   print(response.status_code, response.text)
   if response.text != "OK":
       raise SystemExit("The sheet script said: " + response.text)
# ---------- 6. Tests: does the agent make the right decision? ----------
SCENARIOS = [
   {"name": "Normal day",
    "weather": {"city": CITY, "avg_high": 31.0, "avg_low": 21.0, "peak_wind": 12.0,
                "daily_highs": [30, 31, 31, 32, 31]},
    "history": [], "expect_alert": False},
   {"name": "Heatwave",
    "weather": {"city": CITY, "avg_high": 43.0, "avg_low": 30.0, "peak_wind": 14.0,
                "daily_highs": [38, 41, 43, 45, 48]},
    "history": [], "expect_alert": True},
   {"name": "Storm winds",
    "weather": {"city": CITY, "avg_high": 29.0, "avg_low": 22.0, "peak_wind": 75.0,
                "daily_highs": [29, 28, 29, 30, 29]},
    "history": [], "expect_alert": True},
   {"name": "Same heatwave, already alerted",
    "weather": {"city": CITY, "avg_high": 44.0, "avg_low": 31.0, "peak_wind": 14.0,
                "daily_highs": [44, 45, 44, 45, 46]},
    "history": [{"time": "yesterday", "city": CITY, "advice": "Extreme heat", "alert_sent": True,
                 "alert_message": "Heatwave: highs above 44 C. Stay indoors at midday."}],
    "expect_alert": False},
   {"name": "Weather service down",
    "weather": {"error": "service unavailable"},
    "history": [], "expect_alert": False},
]
def run_tests():
   passed = 0
   for s in SCENARIOS:
       tools = {"get_weather": lambda city, s=s: s["weather"],
                "get_history": lambda n=5, s=s: s["history"]}
       decision = apply_guardrails(run_weather_agent(tools)[0])
       ok = decision["send_alert"] == s["expect_alert"]
       passed += ok
       print(f"{'PASS' if ok else 'FAIL'} | {s['name']} | expected alert {s['expect_alert']}, "
             f"got {decision['send_alert']}")
       print("     ", decision["reasoning"])
       sleep(2)   # be gentle with the free limits
   print(f"\n{passed} of {len(SCENARIOS)} passed")


# ---------- Start here ----------
if __name__ == "__main__":
   mode = sys.argv[1] if len(sys.argv) > 1 else "job"
   if mode == "ask":
       print(run_agent(" ".join(sys.argv[2:])))
   elif mode == "dry":
       run_job(dry=True)
   elif mode == "agent":
       run_agent_job()
   elif mode == "agent-dry":
       run_agent_job(dry=True)
   elif mode == "test":
       run_tests()
   else:
       run_job()

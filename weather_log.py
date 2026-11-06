import os
import requests
 
url = "https://api.open-meteo.com/v1/forecast?latitude=25.27&longitude=55.29&current_weather=true"
weather = requests.get(url, timeout=30).json()["current_weather"]
 
payload = {
    "city": "Dubai- UAE",
    "temperature": weather["temperature"],
    "windspeed": weather["windspeed"],
}
 
response = requests.post(os.environ["SHEET_URL"], json=payload, timeout=30)
print(response.status_code, response.text)
 
if response.text != "OK":
    raise SystemExit("The sheet script said: " + response.text)

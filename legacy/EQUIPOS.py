import requests

API_KEY = "2ea1ff78a5c5f279d729509d550bcf61"

url = "https://v3.football.api-sports.io/teams"

headers = {
    "x-apisports-key": API_KEY
}

params = {
    "search": "Recoleta"
}

response = requests.get(url, headers=headers, params=params)

data = response.json()

print("Resultados:", data["results"])

for item in data["response"]:
    team = item["team"]

    print("ID:", team["id"])
    print("NAME:", team["name"])
    print("CODE:", team["code"])
    print("COUNTRY:", team["country"])
    print("LOGO:", team["logo"])
    print("-" * 50)
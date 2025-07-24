import requests
import pandas as pd
import time

API_KEY = "1ccc14e8da5a40c0575ae0c272645ecf"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

LEAGUES = {
    "France_Ligue1": 61,
    "France_Ligue2": 62,
    "England_PremierLeague": 39,
    "England_Championship": 40,
    "England_LeagueOne": 41,
    "England_LeagueTwo": 42,
    "England_Conference": 43,
    "Scotland_Premiership": 179,
    "Scotland_Division1": 180,
    "Scotland_Division2": 181,
    "Scotland_Division3": 182,
    "Germany_Bundesliga1": 78,
    "Germany_Bundesliga2": 79,
    "Italy_SerieA": 135,
    "Italy_SerieB": 136,
    "Spain_LaLiga1": 140,
    "Spain_LaLiga2": 141,
    "Netherlands_Eredivisie": 88,
    "Belgium_Jupiler": 94,
    "Portugal_Liga1": 94,
    "Turkey_SuperLig": 203,
    "Greece_SuperLeague": 197
}

SEASON = 2024
BOOKMAKER_ID = 2  # Bet365
BET_TYPE_ID = 1   # Match Winner (1X2)
all_matches = []

def get_fixtures_by_league(league_id):
    url = f"{BASE_URL}/fixtures"
    params = {"league": league_id, "season": SEASON}
    response = requests.get(url, headers=HEADERS, params=params)
    return response.json().get("response", [])

def get_odds_for_fixture(fixture_id):
    url = f"{BASE_URL}/odds"
    params = {
        "fixture": fixture_id,
        "bookmaker": BOOKMAKER_ID,
        "bet": BET_TYPE_ID
    }
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json().get("response", [])
    if data:
        try:
            odds = data[0]['bookmakers'][0]['bets'][0]['values']
            odds_dict = {item['value']: item['odd'] for item in odds}
            return {
                "odds_home_win": odds_dict.get("Home"),
                "odds_draw": odds_dict.get("Draw"),
                "odds_away_win": odds_dict.get("Away")
            }
        except Exception:
            return {}
    return {}

# Récupération des données
for league_name, league_id in LEAGUES.items():
    print(f"⏳ Chargement : {league_name}...")
    fixtures = get_fixtures_by_league(league_id)

    for fix in fixtures:
        if fix['fixture']['status']['short'] != "FT":
            continue

        fixture_id = fix['fixture']['id']
        score_home = fix['goals']['home']
        score_away = fix['goals']['away']
        halftime = fix.get("score", {}).get("halftime", {})

        if None in [score_home, score_away, halftime.get('home'), halftime.get('away')]:
            continue

        home_1st = halftime['home']
        away_1st = halftime['away']
        home_2nd = score_home - home_1st
        away_2nd = score_away - away_1st

        # Appel pour récupérer les cotes
        odds = get_odds_for_fixture(fixture_id)
        time.sleep(1)  # éviter le blocage API

        record = {
            "fixture_id": fixture_id,
            "league": league_name,
            "date": fix['fixture']['date'],
            "home_team": fix['teams']['home']['name'],
            "away_team": fix['teams']['away']['name'],
            "score_home": score_home,
            "score_away": score_away,
            "home_goals_1st_half": home_1st,
            "home_goals_2nd_half": home_2nd,
            "away_goals_1st_half": away_1st,
            "away_goals_2nd_half": away_2nd,
            "odds_home_win": odds.get("odds_home_win"),
            "odds_draw": odds.get("odds_draw"),
            "odds_away_win": odds.get("odds_away_win")
        }

        all_matches.append(record)

# Export
df = pd.DataFrame(all_matches)
df.to_csv("results_with_halves_and_odds.csv", index=False)
print("✅ Données enregistrées dans 'results_with_halves_and_odds.csv'")

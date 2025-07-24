#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:39:21 2025

@author: bobunda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 10:02:57 2024

@author: bobunda
"""

import requests
import json

# lien d'accès

url_base='http://127.0.0.1:5000'


#url_base='https://model-scv.onrender.com'

# Test de point d'accès d'accueil
#reponse=requests.get(f"{url_base}/")
##
#print("reponse de point d'accès:", reponse.text) 

# Données d'exemple pour la prédiction

data={

    "matches": [
        
        {
            "HomeTeam": "Fulham",
            "AwayTeam": "Everton",
            "comp": "pl",
            "odds_home":1.82,
            "odds_draw":4.50,
            "odds_away":3.40,
            "match_Date":'2025-05-10'
        },
        {
            "HomeTeam": "Southampton",
            "AwayTeam": "Manchester City",
            "comp": "pl",
            "odds_home":11.00,
            "odds_draw":1.20,
            "odds_away":7.50,
            "match_Date":'2025-05-10'
        },
        {
            "HomeTeam": "Borussia Mönchengladbach",
            "AwayTeam": "VfL Wolfsburg",
            "comp": "bl",
            "odds_home":2.20,
            "odds_draw":2.80,
            "odds_away":4.20,
            "match_Date":'2025-05-17'
        },
        {
            "HomeTeam": "FSV Mainz 05",
            "AwayTeam": "Bayer Leverkusen",
            "comp": "bl",
            "odds_home":2.15,
            "odds_draw":3.00,
            "odds_away":3.90,
            "match_Date":'2025-05-17'
        },
        {
            "HomeTeam": "Genoa",
            "AwayTeam": "Atalanta",
            "comp": "sa",
            "odds_home":4.50,
            "odds_draw":1.75,
            "odds_away":3.90,
            "match_Date":'2025-05-17'
        },
        {
            "HomeTeam": "Cagliari",
            "AwayTeam": "Venezia",
            "comp": "sa",
            "odds_home":2.60,
            "odds_draw":3.00,
            "odds_away":3.00,
            "match_Date":'2025-05-18'
        },
        {
            "HomeTeam": "Juventus",
            "AwayTeam": "Udinese",
            "comp": "sa",
            "odds_home":1.38,
            "odds_draw":8.50,
            "odds_away":4.75,
            "match_Date":'2025-05-18'
        },

        {
            "HomeTeam": "Las Palmas",
            "AwayTeam": "Leganes",
            "comp": "lg",
            "odds_home":2.90,
            "odds_draw":2.35,
            "odds_away":3.50,
            "match_Date":'2025-05-18'
        },
        {
            "HomeTeam": "Sevilla",
            "AwayTeam": "Real Madrid",
            "comp": "lg",
            "odds_home":4.20,
            "odds_draw":1.75,
            "odds_away":4.00,
            "match_Date":'2025-05-18'
        },
        {
            "HomeTeam": "Valladolid",
            "AwayTeam": "Alaves",
            "comp": "lg",
            "odds_home":6.00,
            "odds_draw":1.60,
            "odds_away":3.75,
            "match_Date":'2025-05-18'
        },
        
        {
            "HomeTeam": "Lyon",
            "AwayTeam": "Angers",
            "comp": "fl",
            "odds_home":1.29,
            "odds_draw":9.00,
            "odds_away":6.00,
            "match_Date":'2025-05-17'
        },
        {
            "HomeTeam": "Marseille",
            "AwayTeam": "Rennes",
            "comp": "fl",
            "odds_home":1.75,
            "odds_draw":4.20,
            "odds_away":3.90,
            "match_Date":'2025-05-17'
        },
        {
            "HomeTeam": "Strasbourg",
            "AwayTeam": "Le Havre",
            "comp": "fl",
            "odds_home":1.57,
            "odds_draw":5.00,
            "odds_away":4.75,
            "match_Date":'2025-05-17'
        },
        {
            "HomeTeam": "Saint Etienne",
            "AwayTeam": "Toulouse",
            "comp": "fl",
            "odds_home":1.97,
            "odds_draw":3.25,
            "odds_away":4.20,
            "match_Date":'2025-05-17'
        }
        ,
        {
            "HomeTeam": "Almere City FC",
            "AwayTeam": "Sparta Rotterdam",
            "comp": "au",
            "odds_home":2.80,
            "odds_draw":2.45,
            "odds_away":3.50,
            "match_Date":'2025-05-10'
        },
        {
            "HomeTeam": "PEC Zwolle",
            "AwayTeam": "GO Ahead Eagles",
            "comp": "au",
            "odds_home":2.55,
            "odds_draw":2.50,
            "odds_away":3.80,
            "match_Date":'2025-05-11'
        },
        {
            "HomeTeam": "Twente",
            "AwayTeam": "Utrecht",
            "comp": "au",
            "odds_home":2.35,
            "odds_draw":2.70,
            "odds_away":4.20,
            "match_Date":'2025-05-11'
        },
        {
            "HomeTeam": "Ajax",
            "AwayTeam": "NEC Nijmegen",
            "comp": "au",
            "odds_home":2.50,
            "odds_draw":2.40,
            "odds_away":3.90,
            "match_Date":'2025-05-11'
        }
        
    ]
}

# Envoi de la requête POST
response = requests.post(f"{url_base}/predire/pl", json=data)
# Affichage de la réponse
#print(response.text)
response_data=response.json()
formatted_json = json.dumps(response_data, indent=2, ensure_ascii=False)  # Indentation de 2 espaces
print(formatted_json)


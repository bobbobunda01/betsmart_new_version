#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025

@author: bobunda
"""


import json
from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request
from typing import List
import numpy as np
import pandas as pd
import os
from numpy import floating, integer, ndarray
import datetime
import pathlib
from dateutil import parser
import json

##------------------------------- PREDICTION DES EQUIPES WIN LOSS DRAW ------------------------------------------------


# log des prédictions utilisateurs
def log_prediction(prediction):
    log_data = {
        "request_date": datetime.datetime.utcnow().isoformat(),
        #"input": data,
        "prediction": prediction
    }
    print("➡️ Donnée à logger :", prediction)
    os.makedirs("logs", exist_ok=True)
    with open("logs/logs.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
        
    
        
def log_dataframe_features_to_file(features_df, home, away, match_date, output_path="logs/features_log.jsonl"):
    os.makedirs("logs", exist_ok=True)
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "home_team": home,
        "away_team": away,
        "match_date": str(match_date),
        "features": features_df.to_dict(orient="records")[0]
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(log_data) + "\n")


## Nouvelle version input_data_user


def enrich_form_stats_dynamic(df, team, match_date, window=5):
    """
    Calcule les statistiques dynamiques sur les derniers matchs avant match_date.
    """
    recent_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['Date'] < match_date)].sort_values('Date', ascending=False).head(window)
    
    if recent_matches.empty:
        return {"Form": 0.0, "GD": 0.0, "WinRate": 0.0, "DrawRate": 0.0, "GoalsAvg": 0.0}

    points = 0
    goals_diff = 0
    draws = 0
    wins = 0
    total_goals = 0

    for _, row in recent_matches.iterrows():
        if row['HomeTeam'] == team:
            goals_for = row['FTHG']
            goals_against = row['FTAG']
            result = row['FTR']
        else:
            goals_for = row['FTAG']
            goals_against = row['FTHG']
            result = 'H' if row['FTR'] == 'A' else 'A' if row['FTR'] == 'H' else 'D'
        
        if result == 'D':
            draws += 1
            points += 1
        elif (result == 'H' and row['HomeTeam'] == team) or (result == 'A' and row['AwayTeam'] == team):
            wins += 1
            points += 3
        
        goals_diff += (goals_for - goals_against)
        total_goals += goals_for

    matches_played = len(recent_matches)
    return {
        "Form": points / (3 * matches_played),
        "GD": goals_diff / matches_played,
        "WinRate": wins / matches_played,
        "DrawRate": draws / matches_played,
        "GoalsAvg": total_goals / matches_played
    }

def add_ranks_and_importance(df, home_team, away_team, match_date):
    """
    Calcule le classement dynamique + importance du match (binaire) à une date donnée.
    """
    df = df[df['Date'] < match_date].copy()
    df['Points_H'] = df['FTR'].apply(lambda x: 3 if x == 'H' else 1 if x == 'D' else 0)
    df['Points_A'] = df['FTR'].apply(lambda x: 3 if x == 'A' else 1 if x == 'D' else 0)

    team_points = {}

    for _, row in df.iterrows():
        team_points[row['HomeTeam']] = team_points.get(row['HomeTeam'], 0) + row['Points_H']
        team_points[row['AwayTeam']] = team_points.get(row['AwayTeam'], 0) + row['Points_A']

    sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
    ranks = {team: idx + 1 for idx, (team, _) in enumerate(sorted_teams)}

    rank_home = ranks.get(home_team, 10)
    rank_away = ranks.get(away_team, 10)

    match_importance = 1 if abs(rank_home - rank_away) <= 4 and match_date.month >= 4 else 0
    return rank_home, rank_away, match_importance

def prepare_input_features_enriched(home_team, away_team, match_date, b365h, b365a, b365d, season_df):
    """
    Prépare les features enrichies pour la prédiction d'un match avec classement dynamique.
    """
    df = season_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    if home_team not in all_teams or away_team not in all_teams:
        print(f"⚠️ Attention : {home_team} ou {away_team} n'a pas d'historique. Les stats seront neutres.")

    match_date = pd.to_datetime(match_date)
    df_past = df[df['Date'] < match_date]

    def safe_stats(stats_dict):
        for key in ['Form', 'GD', 'WinRate', 'DrawRate', 'GoalsAvg']:
            if stats_dict.get(key) is None:
                stats_dict[key] = 0.0
        return stats_dict

    home_stats = safe_stats(enrich_form_stats_dynamic(df_past, home_team, match_date))
    away_stats = safe_stats(enrich_form_stats_dynamic(df_past, away_team, match_date))

    odds_ratio_ha = b365h / b365a if b365a > 0 else 0
    odds_diff_hd = b365h - b365d
    odds_diff_ad = b365a - b365d
    odds_gap_min_delta = max(b365h, b365a, b365d) - min(b365h, b365a, b365d)
    form_diff = home_stats["Form"] - away_stats["Form"]

    rank_home, rank_away, match_importance = add_ranks_and_importance(df, home_team, away_team, match_date)

    features = pd.DataFrame([{
        'HTHG': 0, 'HTAG': 0, 'HTR': 0,
        'B365H': b365h, 'B365A': b365a, 'B365D': b365d,
        'OddsRatio_HA': odds_ratio_ha,
        'OddsDiff_HD': odds_diff_hd,
        'OddsDiff_AD': odds_diff_ad,
        'OddsGap_MinDelta': odds_gap_min_delta,
        'Year': match_date.year,
        'Month': match_date.month,
        'Weekday': match_date.weekday(),
        'HomeForm': home_stats["Form"], 'AwayForm': away_stats["Form"],
        'HomeGD': home_stats["GD"], 'AwayGD': away_stats["GD"],
        'DrawRate_Home': home_stats["DrawRate"], 'DrawRate_Away': away_stats["DrawRate"],
        'WinRate_Home': home_stats["WinRate"], 'WinRate_Away': away_stats["WinRate"],
        'GoalsAvg_Home': home_stats["GoalsAvg"], 'GoalsAvg_Away': away_stats["GoalsAvg"],
        'Form_Diff': form_diff,
        'Rank_Home': rank_home,
        'Rank_Away': rank_away,
        'MatchImportance': match_importance
    }])
    
    print(features.shape)

    return features


## lecture des variables dynamiques 
RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]
chemin_csv = RACINE_PROJET / "data" /"champ_config.json"

def parametres(league_code):
    # Seuil dynamique par championnat (paramétrable)
    #seuil_base = 0.12
    
    with open(chemin_csv, "r") as f:
        CHAMP_CONFIG = json.load(f)

    # Exemple : lecture du bookmaker_margin pour pl
     # ou fl, bl, lg, sa
    params = CHAMP_CONFIG.get(league_code, {})
    bookmaker_margin = params.get("bookmaker_margin", 0.0711)
    uncertainty_threshold = params.get("uncertainty_threshold", 0.12)
    importance = params.get("importance", 3)
    season_stage = params.get("season_stage", "mid")  # par défaut à "mid"
    return bookmaker_margin, uncertainty_threshold, importance,season_stage

def detect_double_chance(proba_0, proba_1, proba_2, final_prediction, league_code):
    bookmaker_margin, uncertainty_threshold, importance, season_stage = parametres(league_code)
    seuil_incertitude = uncertainty_threshold - 0.02 * (importance / 5)

    probs = np.array([proba_0, proba_1, proba_2])
    sorted_probs = np.sort(probs)
    ecart = sorted_probs[-1] - sorted_probs[-2]

    if ecart <= seuil_incertitude:
        if final_prediction == 0 and proba_0 < 0.60:
            return "1X"
        elif final_prediction == 2 and proba_2 < 0.60:
            return "X2"
    return None

def detect_bias(features_df):
    odds = features_df[['B365H', 'B365A', 'B365D']].values[0]
    max_odds = np.max(odds)
    min_odds = np.min(odds)
    bias_score = abs(max_odds - min_odds) / np.mean(odds)
    return bias_score > 0.6  # Seuil configurable

def is_confidence_low(proba_0, proba_1, proba_2):
    ecart_principal = np.max([proba_0, proba_1, proba_2]) - np.median([proba_0, proba_1, proba_2])
    return ecart_principal < 0.07

def adjust_odds_weight_by_season(odds_gap, season_stage):
    if season_stage == "early":
        return odds_gap * 1.3  # Cotes moins fiables en début de saison
    elif season_stage == "mid":
        return odds_gap
    else:
        return odds_gap * 0.9  # Confiance plus forte sur fin de saison

def predict_match_with_proba(
    features_df: pd.DataFrame,
    model_stage1,
    model_stage2,
    threshold_draw=0.0,
    user_profile="standard",
    league_code="default"
) -> dict:

    # Récupération des paramètres du championnat
    bookmaker_margin, _, _, season_stage = parametres(league_code)

    # Étape 1 : préparation des features pour le modèle de match nul
    features_df_stage1 = features_df.copy()
    for feature in model_stage1.feature_names_in_:
        if feature not in features_df_stage1.columns:
            features_df_stage1[feature] = 0
    features_df_stage1 = features_df_stage1[model_stage1.feature_names_in_]

    # Prédiction proba nul
    proba_draw = model_stage1.predict_proba(features_df_stage1)[0][1]

    # Calcul brut et ajusté de l'écart de cotes
    odds_gap_raw = (
        features_df_stage1[['B365H', 'B365A', 'B365D']].max(axis=1).values[0]
        - features_df_stage1[['B365H', 'B365A', 'B365D']].min(axis=1).values[0]
    )
    odds_gap = adjust_odds_weight_by_season(odds_gap_raw, season_stage)

    # Cas particulier : cotes proches et proba nul ≈ seuil
    draw_margin_band = 0.02
    if threshold_draw - draw_margin_band <= proba_draw <= threshold_draw + draw_margin_band:
        if odds_gap <= bookmaker_margin:
            proba_1 = proba_draw
            proba_0 = proba_2 = (1 - proba_1) / 2
            double_chance = detect_double_chance(proba_0, proba_1, proba_2, 1, league_code)
            return {
                "prediction": 1,
                "proba_0": f"{round(proba_0*100,0)}%",
                "proba_1": f"{round(proba_1*100,0)}%",
                "proba_2": f"{round(proba_2*100,0)}%",
                "rule_applied": "margin_adjusted",
                "explanation": generate_explanation("margin_adjusted", features_df, user_profile),
                "double_chance": double_chance
            }

    # Cas simple : proba nul dépasse le seuil
    if proba_draw >= threshold_draw:
        proba_1 = proba_draw
        proba_0 = proba_2 = (1 - proba_1) / 2
        double_chance = detect_double_chance(proba_0, proba_1, proba_2, 1, league_code)
        return {
            "prediction": 1,
            "proba_0": f"{round(proba_0*100,0)}%",
            "proba_1": f"{round(proba_1*100,0)}%",
            "proba_2": f"{round(proba_2*100,0)}%",
            "rule_applied": "threshold",
            "explanation": generate_explanation("threshold", features_df, user_profile),
            "double_chance": double_chance
        }

    # Étape 2 : prédiction domicile/extérieur
    features_df_stage2 = features_df.copy()
    for feature in model_stage2.feature_names_in_:
        if feature not in features_df_stage2.columns:
            features_df_stage2[feature] = 0
    features_df_stage2 = features_df_stage2[model_stage2.feature_names_in_]

    proba_rf = model_stage2.predict_proba(features_df_stage2)[0]
    prediction_rf = int(model_stage2.predict(features_df_stage2)[0])

    total = proba_rf[0] + proba_draw + proba_rf[1]
    proba_0 = proba_rf[0] / total
    proba_1 = proba_draw / total
    proba_2 = proba_rf[1] / total

    # Ajustement UX si contradiction
    if prediction_rf == 0 and proba_draw >= proba_rf[0]:
        proba_0, proba_1 = proba_1, proba_0
    elif prediction_rf == 2 and proba_draw >= proba_rf[1]:
        proba_2, proba_1 = proba_1, proba_2

    # Calcul double chance
    double_chance = detect_double_chance(proba_0, proba_1, proba_2, prediction_rf, league_code)

    # Détection de biais ou faible confiance
    bias_detected = detect_bias(features_df)
    low_confidence = is_confidence_low(proba_0, proba_1, proba_2)

    if low_confidence or bias_detected:
        # Forcer une double chance si absente
        if double_chance is None:
            if prediction_rf == 0:
                double_chance = "1X"
            elif prediction_rf == 2:
                double_chance = "X2"
            else:
                double_chance = "1X"  # arbitrage conservateur par défaut

        return {
            "prediction": prediction_rf,
            "proba_0": f"{round(proba_0*100,0)}%",
            "proba_1": f"{round(proba_1*100,0)}%",
            "proba_2": f"{round(proba_2*100,0)}%",
            "rule_applied": "filtered_out",
            "explanation": "⚠️ Attention : prédiction peu recommandée en raison d’un biais ou d’une incertitude élevée. Appuyez-vous sur la double chance suggérée.",
            "double_chance": double_chance
        }

    # Résultat final
    return {
        "prediction": prediction_rf,
        "proba_0": f"{round(proba_0*100,0)}%",
        "proba_1": f"{round(proba_1*100,0)}%",
        "proba_2": f"{round(proba_2*100,0)}%",
        "rule_applied": "rf_decision",
        "explanation": generate_explanation("rf_decision", features_df, user_profile),
        "double_chance": double_chance
    }

def generate_explanation(rule_applied, features, user_profile):
    odds_ratio = features.get("OddsRatio_HA", 1)
    form_diff = features.get("Form_Diff", 0)
    match_importance = features.get("MatchImportance", 0)

    # Gestion du type Series si la feature est issue d’un DataFrame à une seule ligne
    if isinstance(match_importance, pd.Series):
        match_importance = match_importance.values[0]

    if user_profile == "débutant":
        if rule_applied == "threshold":
            msg = "L'IA pense qu’il y aura un match nul car la probabilité dépasse le seuil fixé."
        elif rule_applied == "margin_adjusted":
            msg = "Les cotes sont très proches : cela suggère un match équilibré, donc nul."
        else:
            msg = "L’IA prédit une victoire car les chances sont déséquilibrées entre les équipes."
    elif user_profile == "expert":
        if rule_applied == "threshold":
            msg = f"Proba_nul = {features.get('proba_1', 0):.2f}, supérieur au seuil : nul prédit."
        elif rule_applied == "margin_adjusted":
            msg = f"Match ajusté à nul : cotes trop proches (écart ≈ {features.get('OddsGap_MinDelta', 0):.3f})."
        else:
            msg = (
                f"Proba_RF = [{features.get('proba_0', 0):.2f}, {features.get('proba_2', 0):.2f}], "
                f"écart de forme = {form_diff:.2f}"
            )
    else:  # profil standard
        if rule_applied == "threshold":
            msg = "Match nul probable : la probabilité dépasse le seuil."
        elif rule_applied == "margin_adjusted":
            msg = "Les cotes sont serrées, et l’IA anticipe un nul."
        else:
            msg = "Victoire probable : un déséquilibre a été détecté entre les deux équipes."

    # Ajout conditionnel selon importance du match
    if match_importance == 1:
        msg += " Ce match est considéré comme important."

    return msg
# conversion de la date

def get_valid_date(user_input):
    """
    Convertit différentes représentations de date en format 'YYYY-MM-DD'.
    """
    try:
        # Parse intelligent (fonctionne avec des formats très variés)
        date_obj = parser.parse(user_input)
        return date_obj.strftime("%Y-%m-%d")
    except Exception:
        raise ValueError("⛔ Format de date non reconnu. Essayez par exemple : '2025-02-14' ou '14/02/2025'")



##---------------------- NOMBRE DE BUTS MARQUES PAR EQUIPE ------------------------------------------


def entree_utilisateur(home_team, away_team, b365h,b365a,b365d, season_current, season_previous):
    # 🔧 Chargement des arguments
    # ---------------------
    home_team=str(home_team)
    away_team=str(away_team)
    b365h=float(b365h)
    b365a=float(b365a)
    b365d=float(b365d)
    
    #df_curr = pd.read_csv(args.season_current, parse_dates=["Date"])
    df_curr = season_current.copy()
    df_curr['Date']=pd.to_datetime(df_curr['Date'])
    df_curr=df_curr.sort_values(by='Date')
    df_prev = season_previous.copy()
    df_prev['Date']=pd.to_datetime(df_prev['Date'])
    df_prev=df_prev.sort_values(by='Date')
    
    df_prev["goals_1s"] = df_prev["HTHG"] + df_prev["HTAG"]
    df_prev["goals_2n"] = (df_prev["FTHG"] + df_prev["FTAG"]) - df_prev["goals_1s"]

    df_prev["conceded_1s"] = df_prev["goals_1s"]  # pour les moyennes globales, c’est la même chose
    df_prev["conceded_2n"] = df_prev["goals_2n"]

    # Calcul des points (pts) par match selon le résultat
    df_prev["pts"] = df_prev["FTR"].map({"H": 3, "D": 1, "A": 0})
    
    # 📊 Moyennes globales
    # ---------------------
    league_avg = {
        "goals_1st": round(df_prev["goals_1s"].mean(), 2),
        "goals_2nd": round(df_prev["goals_2n"].mean(), 2),
        "conceded_1st": round(df_prev["conceded_1s"].mean(), 2),
        "conceded_2nd": round(df_prev["conceded_2n"].mean(), 2),
        "pts": round(df_prev["pts"].mean(), 2)
    }
    
    def compute_form(team, df, window=5):
        
        df_team = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date", ascending=True)
        if len(df_team) == 0:
            return None
        form = []
        for _, row in df_team.iterrows():
            is_home = row["HomeTeam"] == team
            hthg, fthg = row["HTHG"], row["FTHG"]
            htag, ftag = row["HTAG"], row["FTAG"]

            g1 = hthg if is_home else htag
            g2 = (fthg - hthg) if is_home else (ftag - htag)
            c1 = htag if is_home else hthg
            c2 = (ftag - htag) if is_home else (fthg - hthg)

            if (fthg == ftag): pts = 1
            elif (is_home and fthg > ftag) or (not is_home and ftag > fthg): pts = 3
            else: pts = 0

            form.append((g1, g2, c1, c2, pts))

        if len(form) < 3:
            return None
        last = form[-window:]
        return {
            "goals_1st": np.mean([x[0] for x in last]),
            "goals_2nd": np.mean([x[1] for x in last]),
            "conceded_1st": np.mean([x[2] for x in last]),
            "conceded_2nd": np.mean([x[3] for x in last]),
            "pts": np.mean([x[4] for x in last])
            }
    def get_final_form(team):
        
        # Priorité : saison en cours
        f1 = compute_form(team, df_curr)
        if f1: return f1
        # Sinon, saison précédente
        f2 = compute_form(team, df_prev)
        if f2: return f2
        # Sinon, valeurs moyennes
        return league_avg
    home_stats = get_final_form(home_team)
    away_stats = get_final_form(away_team)
    
    input_features = {
    "total_avg_goals_home": home_stats["goals_1st"] + home_stats["goals_2nd"] + home_stats["conceded_1st"] + home_stats["conceded_2nd"],
    "total_avg_goals_away": away_stats["goals_1st"] + away_stats["goals_2nd"] + away_stats["conceded_1st"] + away_stats["conceded_2nd"],
    "goal_diff_home": (home_stats["goals_1st"] + home_stats["goals_2nd"]) - (home_stats["conceded_1st"] + home_stats["conceded_2nd"]),
    "goal_diff_away": (away_stats["goals_1st"] + away_stats["goals_2nd"]) - (away_stats["conceded_1st"] + away_stats["conceded_2nd"]),
    "pts_recent_home": home_stats["pts"],
    "pts_recent_away": away_stats["pts"],
    "odds_diff": b365h - b365a,
    "odds_draw_gap": b365d - np.mean([b365h,b365a]),
    "odds_mean": np.mean([b365h, b365d, b365a])}
    
    return pd.DataFrame([input_features])

def to_serializable(obj):
    if isinstance(obj, floating):
        return float(obj)
    elif isinstance(obj, integer):
        return int(obj)
    elif isinstance(obj, ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def get_last5_results_pattern(df, team_name, match_date):
    """
    Retourne les 5 derniers résultats ('W', 'L', 'D') d'une équipe donnée avant une date donnée.
    Si aucun match joué avant la date → 'MMMMM'.
    Sinon → complète les matchs manquants avec 'M'.
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    match_date = pd.to_datetime(match_date)

    past_matches = df[
        ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) &
        (df['Date'] < match_date)
    ].sort_values(by='Date', ascending=False).head(5)

    if past_matches.empty:
        return "MMMMM"

    results = []

    for _, row in past_matches.iterrows():
        if row['HomeTeam'] == team_name:
            if row['FTR'] == 'H':
                results.append('W')
            elif row['FTR'] == 'D':
                results.append('D')
            else:
                results.append('L')
        elif row['AwayTeam'] == team_name:
            if row['FTR'] == 'A':
                results.append('W')
            elif row['FTR'] == 'D':
                results.append('D')
            else:
                results.append('L')

    # Compléter avec 'M' si moins de 5 matchs
    while len(results) < 5:
        results.append('M')

    return ''.join(results)

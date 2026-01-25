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
from functools import lru_cache

##------------------------------- PREDICTION DES EQUIPES WIN LOSS DRAW ------------------------------------------------

########
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


###----------- DEBUT DES FONCTIONS DE PREDICTION-----

# -*- coding: utf-8 -*-
import pathlib, json
from functools import lru_cache
import numpy as np
import pandas as pd

RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]
chemin_csv = RACINE_PROJET / "data" / "champ_config.json"

@lru_cache(maxsize=1)
def _load_champ_config():
    with open(chemin_csv, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # double index: str et int
    cfg_by_str = {str(k): v for k, v in cfg.items()}
    cfg_by_int = {}
    for k, v in cfg.items():
        try:
            cfg_by_int[int(k)] = v
        except Exception:
            pass
    return {"by_str": cfg_by_str, "by_int": cfg_by_int}

def _get_params(league_code):
    cfg = _load_champ_config()
    if league_code in cfg["by_int"]:
        return cfg["by_int"][league_code]
    if league_code in cfg["by_str"]:
        return cfg["by_str"][league_code]
    try:
        return cfg["by_int"].get(int(league_code), cfg["by_str"].get(str(league_code), {}))
    except Exception:
        return cfg["by_str"].get(str(league_code), {})

def parametres(league_code):
    """
    Retourne 8 valeurs:
    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight)
    """
    p = _get_params(league_code)

    bookmaker_margin      = float(p.get("bookmaker_margin", 0.0711))
    uncertainty_threshold = float(p.get("uncertainty_threshold", 0.12))
    importance            = int(p.get("importance", 3))
    season_stage          = str(p.get("season_stage", "mid"))

    # ⚠️ pas de virgules finales ici (sinon -> tuples)
    upset_threshold = float(p.get("upset_threshold", 0.55))
    skip_threshold  = float(p.get("skip_threshold", 1.50))
    bogey_weight    = float(p.get("bogey_weight", 0.40))
    gki_weight      = float(p.get("gki_weight", 0.60))

    return (bookmaker_margin, uncertainty_threshold, importance, season_stage,
            upset_threshold, skip_threshold, bogey_weight, gki_weight)

# ---------- AJOUT: hyperparamètres de la porte de forme ----------
def parametres_form_gate(league_code):
    """
    Lit (si dispo) les hyperparamètres de la 'porte forme' depuis champ_config.json :
      - k_market_form  : intensité max de transfert H↔A (0..1)  (défaut 0.45)
      - gate_slope     : pente de la sigmoïde (défaut 14.0)
      - gate_tolerance : tolérance d’écart de forme avant d’agir (défaut 0.036)
    """
    p = _get_params(league_code)
    k     = float(p.get("k_market_form", 0.45))
    slope = float(p.get("gate_slope", 14.0))
    tau   = float(p.get("gate_tolerance", 0.036))
    return k, slope, tau
# ---------------------------------------------------------------

##---------------------- FONCTION DE PREDICTION ------------------------------------------
# -------------------------------------------------------------------
# 🔒 Adaptateurs de types + lecture sûre des paramètres de ligue
# -------------------------------------------------------------------
def _as_float(x, default=0.0):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _safe_parametres(league_code):
    """
    S'adapte à l'ancienne signature (4 valeurs) et la nouvelle (8),
    force les bons types (floats/ints/str) et fournit des valeurs par défaut sûres.
    On suppose que `parametres(league_code)` existe déjà dans ton projet.
    """
    vals = parametres(league_code)  # <-- ta fonction existante

    # Ancienne version: 4 valeurs
    if isinstance(vals, (list, tuple)) and len(vals) == 4:
        bookmaker_margin, uncertainty_threshold, importance, season_stage = vals
        upset_threshold, skip_threshold, bogey_weight, gki_weight = 0.55, 1.50, 0.40, 0.60
    # Nouvelle version: 8 valeurs
    elif isinstance(vals, (list, tuple)) and len(vals) >= 8:
        (bookmaker_margin, uncertainty_threshold, importance, season_stage,
         upset_threshold, skip_threshold, bogey_weight, gki_weight) = vals[:8]
    else:
        # Fallback très sûr
        bookmaker_margin, uncertainty_threshold, importance, season_stage = 0.0711, 0.12, 3, "mid"
        upset_threshold, skip_threshold, bogey_weight, gki_weight = 0.55, 1.50, 0.40, 0.60

    # Coercions
    bookmaker_margin      = _as_float(bookmaker_margin, 0.0711)
    uncertainty_threshold = _as_float(uncertainty_threshold, 0.12)
    importance            = _as_int(importance, 3)
    season_stage          = str(season_stage) if season_stage is not None else "mid"
    upset_threshold       = _as_float(upset_threshold, 0.55)
    skip_threshold        = _as_float(skip_threshold, 1.50)
    bogey_weight          = _as_float(bogey_weight, 0.40)
    gki_weight            = _as_float(gki_weight, 0.60)

    return (bookmaker_margin, uncertainty_threshold, importance, season_stage,
            upset_threshold, skip_threshold, bogey_weight, gki_weight)

def _fav_by_demarged(bh: float, bd: float, ba: float, eps: float = 0.02):
    """
    Détermine le favori via probabilités implicites dé-margées.
    1) Convertit les cotes 1X2 en probs implicites, dé-marge (normalisation),
    2) Replie en 2-voies (home vs away) en ignorant le nul,
    3) Retourne (side, pH2, pA2, gap) où side ∈ {"home","away", None si coin-flip}.
    eps = marge minimale de gap favori (ajuste par ligue si besoin).
    """
    bh = float(bh); bd = float(bd); ba = float(ba)
    if min(bh, bd, ba) <= 1.0 or any(not np.isfinite(x) for x in (bh, bd, ba)):
        return None, np.nan, np.nan, 0.0

    qH, qD, qA = 1.0/bh, 1.0/bd, 1.0/ba
    s = qH + qD + qA
    if s <= 0:
        return None, np.nan, np.nan, 0.0

    # probs 3-voies dé-margées
    pH, pD, pA = qH/s, qD/s, qA/s
    # replie en 2-voies (on ignore D)
    denom = (pH + pA)
    if denom <= 0:
        return None, np.nan, np.nan, 0.0
    pH2, pA2 = pH/denom, pA/denom
    gap = pH2 - pA2

    if   gap >  eps: side = "home"
    elif gap < -eps: side = "away"
    else:            side = None  # marché trop équilibré
    return side, pH2, pA2, gap

# -------------------------------------------------------------------
# ✅ BUGFIX : enrich_form_stats_dynamic (victoires à l’extérieur)
# -------------------------------------------------------------------
def enrich_form_stats_dynamic(df, team, match_date, window=5):
    """
    Calcule les statistiques dynamiques sur les derniers matchs avant match_date.
    Form = points / (3*N), GD = diff de buts moyens, etc.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    match_date = pd.to_datetime(match_date)

    recent_matches = (
        df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['Date'] < match_date)]
        .sort_values('Date', ascending=False)
        .head(window)
    )
    if recent_matches.empty:
        return {"Form": 0.0, "GD": 0.0, "WinRate": 0.0, "DrawRate": 0.0, "GoalsAvg": 0.0}

    points = goals_diff = draws = wins = total_goals = 0

    for _, row in recent_matches.iterrows():
        is_home = (row['HomeTeam'] == team)

        if is_home:
            goals_for, goals_against = row['FTHG'], row['FTAG']
            win = (row['FTR'] == 'H')
        else:
            goals_for, goals_against = row['FTAG'], row['FTHG']
            win = (row['FTR'] == 'A')

        draw = (row['FTR'] == 'D')

        if draw:
            draws += 1; points += 1
        elif win:
            wins += 1; points += 3

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

# -------------------------------------------------------------------
# Classement dynamique + importance binaire (inchangé)
# -------------------------------------------------------------------
def _league_profile(league_code: str | int | None):
    """
    Retourne un profil (region, late_months, late_threshold) par ligue.
    """
    try:
        code = int(league_code) if league_code is not None else None
    except Exception:
        code = None

    EURO = {
        39,61,78,140,135,88,207,94,203,144,197,119,179,180,253,
        2,3,233,62,40,79,136,141
    }
    CAL_Y = {71,98,262,292,128}

    if code in EURO:
        return {"region":"europe","late_months":{4,5,6},"late_threshold":0.70}
    elif code in CAL_Y:
        return {"region":"calendar_year","late_months":{10,11,12},"late_threshold":0.70}
    else:
        return {"region":"unknown","late_months":set(),"late_threshold":0.70}

def _season_progress_by_dates(df_all: pd.DataFrame, asof) -> float:
    if df_all is None or df_all.empty:
        return 0.0
    d = df_all.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    if d.empty:
        return 0.0
    asof = pd.to_datetime(asof)
    dmin, dmax = d["Date"].min(), d["Date"].max()
    total = (dmax - dmin).days
    if total <= 0:
        return 0.0
    prog = (asof - dmin).days / total
    return float(max(0.0, min(1.0, prog)))

def add_ranks_and_importance(df, home_team, away_team, match_date, league_code):
    if df is None or df.empty:
        return 10, 10, 0

    prof = _league_profile(league_code)
    late_months = prof["late_months"]
    late_th = prof["late_threshold"]

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    md = pd.to_datetime(match_date)
    d = d[d["Date"] < md].dropna(subset=["Date"])

    d["Points_H"] = d["FTR"].apply(lambda x: 3 if x == "H" else 1 if x == "D" else 0)
    d["Points_A"] = d["FTR"].apply(lambda x: 3 if x == "A" else 1 if x == "D" else 0)

    team_points = {}
    for _, row in d.iterrows():
        team_points[row["HomeTeam"]] = team_points.get(row["HomeTeam"], 0) + row["Points_H"]
        team_points[row["AwayTeam"]] = team_points.get(row["AwayTeam"], 0) + row["Points_A"]

    if not team_points:
        return 10, 10, 0

    sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
    ranks = {team: idx + 1 for idx, (team, _) in enumerate(sorted_teams)}
    n_teams = len(ranks)

    rank_home = ranks.get(home_team, min(10, n_teams))
    rank_away = ranks.get(away_team, min(10, n_teams))
    rank_diff = abs(rank_home - rank_away)

    season_prog = _season_progress_by_dates(df, md)
    late_season = (season_prog >= late_th) or (md.month in late_months)

    top_k = 5
    close_ranks = (rank_diff <= 4)
    top_clash = (rank_home <= top_k and rank_away <= top_k)

    releg_zone = max(3, int(round(0.12 * n_teams)))
    six_pointer_releg = late_season and ((rank_home > n_teams - releg_zone) or (rank_away > n_teams - releg_zone))
    euro_spot_fight = late_season and ((rank_home <= 7) or (rank_away <= 7)) and (rank_diff <= 6)

    importance = 1 if (top_clash or (late_season and (close_ranks or six_pointer_releg or euro_spot_fight))) else 0
    return rank_home, rank_away, importance

# -------------------------------------------------------------------
# Préparation des features (retour DataFrame — pas de tuple)
# -------------------------------------------------------------------
def prepare_input_features_enriched(home_team, away_team, match_date, b365h, b365a, b365d, season_df, league_code):
    df = season_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    if (home_team not in all_teams) or (away_team not in all_teams):
        print(f"⚠️ Attention : {home_team} ou {away_team} n'a pas d'historique. Les stats seront neutres.")

    match_date = pd.to_datetime(match_date)
    df_past = df[df['Date'] < match_date]

    def safe_stats(d):
        d = dict(d or {})
        for key in ['Form', 'GD', 'WinRate', 'DrawRate', 'GoalsAvg']:
            if d.get(key) is None:
                d[key] = 0.0
        return d

    home_stats = safe_stats(enrich_form_stats_dynamic(df_past, home_team, match_date))
    away_stats = safe_stats(enrich_form_stats_dynamic(df_past, away_team, match_date))

    odds_ratio_ha = b365h / b365a if b365a > 0 else 0
    odds_diff_hd = b365h - b365d
    odds_diff_ad = b365a - b365d
    odds_gap_min_delta = max(b365h, b365a, b365d) - min(b365h, b365a, b365d)
    form_diff = home_stats["Form"] - away_stats["Form"]

    rank_home, rank_away, match_importance = add_ranks_and_importance(df, home_team, away_team, match_date, league_code)

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

    return features

# -------------------------------------------------------------------
# Règles auxiliaires (inchangées / petites sécurités)
# -------------------------------------------------------------------
def detect_double_chance(proba_0, proba_1, proba_2, final_prediction, league_code):
    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(league_code)

    seuil_incertitude = uncertainty_threshold - 0.02 * (importance / 5)

    probs = np.array([proba_0, proba_1, proba_2], dtype=float)
    sorted_probs = np.sort(probs)
    ecart = sorted_probs[-1] - sorted_probs[-2]

    if ecart <= seuil_incertitude:
        if final_prediction == 0 and proba_0 < 0.60:
            return "1X"
        elif final_prediction == 2 and proba_2 < 0.60:
            return "X2"
    return None

def detect_bias(features_df):
    odds = features_df[['B365H', 'B365A', 'B365D']].values[0].astype(float)
    max_odds = np.max(odds)
    min_odds = np.min(odds)
    bias_score = abs(max_odds - min_odds) / np.mean(odds)
    return bias_score > 0.6

def is_confidence_low(proba_0, proba_1, proba_2):
    arr = np.array([proba_0, proba_1, proba_2], dtype=float)
    ecart_principal = np.max(arr) - np.median(arr)
    return ecart_principal < 0.07

def adjust_odds_weight_by_season(odds_gap, season_stage):
    if season_stage == "early":
        return odds_gap * 1.3
    elif season_stage == "mid":
        return odds_gap
    else:
        return odds_gap * 0.9

# ---------- AJOUT: porte "forme récente" ----------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _apply_form_gate(proba_0, proba_1, proba_2, features_df, league_code):
    """
    Ajuste UNIQUEMENT la répartition H/A (proba_0 / proba_2). Le nul (proba_1) est conservé,
    puis renormalisation H/A. Effet piloté par (k, slope, tau) par ligue.
    """
    # Favori marché dé-margé
    try:
        b365h = float(features_df["B365H"].values[0])
        b365d = float(features_df["B365D"].values[0])
        b365a = float(features_df["B365A"].values[0])
        eps = max(0.02, 0.5*_safe_parametres(league_code)[0])
        fav_side, _, _, _ = _fav_by_demarged(b365h, b365d, b365a, eps=eps)
    except Exception:
        fav_side = None

    if fav_side is None:
        return proba_0, proba_1, proba_2, {"form_gate":"skipped_no_clear_fav"}

    home_form = float(features_df["HomeForm"].values[0])
    away_form = float(features_df["AwayForm"].values[0])
    k, slope, tau = parametres_form_gate(league_code)

    # delta > 0 si OUTSIDER a meilleure forme
    if fav_side == "home":
        delta = (away_form - home_form)
        # +1 => transfert home->away
        sign = +1 if delta > 0 else -1
    else:
        delta = (home_form - away_form)
        # +1 => transfert away->home
        sign = +1 if delta > 0 else -1

    gate_strength = k * _sigmoid(slope * (abs(delta) - tau))
    if sign < 0:
        gate_strength *= 0.15  # on évite de renforcer le favori si l’outsider n’est pas meilleur

    h = float(proba_0)
    d = float(proba_1)
    a = float(proba_2)

    mass_HA = max(1e-9, (h + a))
    transfer = gate_strength * mass_HA

    if fav_side == "home":
        h_new = max(0.0, h - transfer)
        a_new = a + transfer
    else:
        a_new = max(0.0, a - transfer)
        h_new = h + transfer

    # renormaliser H/A en conservant d
    scale = (h + a) / max(1e-9, (h_new + a_new))
    h_new *= scale
    a_new *= scale

    meta = {
        "form_gate":"applied",
        "fav_side":fav_side,
        "home_form":round(home_form,3),
        "away_form":round(away_form,3),
        "delta":round(delta,3),
        "k":round(k,3),
        "slope":round(slope,2),
        "tau":round(tau,3),
        "transfer":round(float(transfer),4)
    }
    return h_new, d, a_new, meta
# ---------------------------------------------------

# -------------------------------------------------------------------
# 💡 Ton pipeline de prédiction (utilise _safe_parametres)
# -------------------------------------------------------------------
def predict_match_with_proba(
    features_df: pd.DataFrame,
    model_stage1,
    model_stage2,
    threshold_draw=0.63,
    user_profile="standard",
    league_code="default"
) -> dict:

    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(league_code)

    # Étape 1 : modèle de nul
    features_df_stage1 = features_df.copy()
    for feature in model_stage1.feature_names_in_:
        if feature not in features_df_stage1.columns:
            features_df_stage1[feature] = 0
    features_df_stage1 = features_df_stage1[model_stage1.feature_names_in_]

    proba_draw = float(model_stage1.predict_proba(features_df_stage1)[0][1])

    odds_gap_raw = (
        features_df_stage1[['B365H', 'B365A', 'B365D']].max(axis=1).values[0]
        - features_df_stage1[['B365H', 'B365A', 'B365D']].min(axis=1).values[0]
    )
    odds_gap = adjust_odds_weight_by_season(odds_gap_raw, season_stage)

    # Cas "margin band" autour du seuil de nul
    draw_margin_band = 0.02
    if threshold_draw - draw_margin_band <= proba_draw <= threshold_draw + draw_margin_band:
        if odds_gap <= bookmaker_margin:
            proba_1 = proba_draw
            proba_0 = proba_2 = (1 - proba_1) / 2
            # 🔁 Applique la porte forme sur le split H/A
            proba_0, proba_1, proba_2, _ = _apply_form_gate(proba_0, proba_1, proba_2, features_df, league_code)
            double_chance = detect_double_chance(proba_0, proba_1, proba_2, 1, league_code)
            return {
                "prediction": 1,
                "proba_0": f"{round(proba_0*100,0)}%",
                "proba_1": f"{round(proba_1*100,0)}%",
                "proba_2": f"{round(proba_2*100,0)}%",
                "rule_applied": "margin_adjusted|form_gate",
                "explanation": generate_explanation("margin_adjusted", features_df, user_profile),
                "double_chance": double_chance
            }

    # Cas "nul clair"
    if proba_draw >= threshold_draw:
        proba_1 = proba_draw
        proba_0 = proba_2 = (1 - proba_1) / 2
        proba_0, proba_1, proba_2, _ = _apply_form_gate(proba_0, proba_1, proba_2, features_df, league_code)
        double_chance = detect_double_chance(proba_0, proba_1, proba_2, 1, league_code)
        return {
            "prediction": 1,
            "proba_0": f"{round(proba_0*100,0)}%",
            "proba_1": f"{round(proba_1*100,0)}%",
            "proba_2": f"{round(proba_2*100,0)}%",
            "rule_applied": "threshold|form_gate",
            "explanation": generate_explanation("threshold", features_df, user_profile),
            "double_chance": double_chance
        }

    # Étape 2 : modèle domicile/extérieur
    features_df_stage2 = features_df.copy()
    for feature in model_stage2.feature_names_in_:
        if feature not in features_df_stage2.columns:
            features_df_stage2[feature] = 0
    features_df_stage2 = features_df_stage2[model_stage2.feature_names_in_]

    proba_rf = model_stage2.predict_proba(features_df_stage2)[0]
    proba_rf = np.asarray(proba_rf, dtype=float)  # [proba_home, proba_away]
    prediction_rf = int(model_stage2.predict(features_df_stage2)[0])

    total = proba_rf[0] + proba_draw + proba_rf[1]
    proba_0 = float(proba_rf[0] / total)
    proba_1 = float(proba_draw / total)
    proba_2 = float(proba_rf[1] / total)

    if prediction_rf == 0 and proba_draw >= proba_rf[0]:
        proba_0, proba_1 = proba_1, proba_0
    elif prediction_rf == 2 and proba_draw >= proba_rf[1]:
        proba_2, proba_1 = proba_1, proba_2

    # 🔁 Applique la porte forme ici aussi (décisive pour le split H/A)
    proba_0, proba_1, proba_2, _ = _apply_form_gate(proba_0, proba_1, proba_2, features_df, league_code)

    double_chance = detect_double_chance(proba_0, proba_1, proba_2, prediction_rf, league_code)
    bias_detected = detect_bias(features_df)
    low_confidence = is_confidence_low(proba_0, proba_1, proba_2)

    if low_confidence or bias_detected:
        if double_chance is None:
            if prediction_rf == 0:
                double_chance = "1X"
            elif prediction_rf == 2:
                double_chance = "X2"
            else:
                double_chance = "1X"

        return {
            "prediction": prediction_rf,
            "proba_0": f"{round(proba_0*100,0)}%",
            "proba_1": f"{round(proba_1*100,0)}%",
            "proba_2": f"{round(proba_2*100,0)}%",
            "rule_applied": "filtered_out|form_gate",
            "explanation": "⚠️ Attention : prédiction peu recommandée en raison d’un biais ou d’une incertitude élevée. Appuyez-vous sur la double chance suggérée.",
            "double_chance": double_chance
        }

    return {
        "prediction": prediction_rf,
        "proba_0": f"{round(proba_0*100,0)}%",
        "proba_1": f"{round(proba_1*100,0)}%",
        "proba_2": f"{round(proba_2*100,0)}%",
        "rule_applied": "rf_decision|form_gate",
        "explanation": generate_explanation("rf_decision", features_df, user_profile),
        "double_chance": double_chance
    }

# -------------------------------------------------------------------
# Explication (inchangée – si tu veux, tu pourras y injecter les proba)
# -------------------------------------------------------------------
def generate_explanation(rule_applied, features, user_profile):
    odds_ratio = features.get("OddsRatio_HA", 1)
    form_diff = features.get("Form_Diff", 0)
    match_importance = features.get("MatchImportance", 0)

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
    else:
        if rule_applied == "threshold":
            msg = "Match nul probable : la probabilité dépasse le seuil."
        elif rule_applied == "margin_adjusted":
            msg = "Les cotes sont serrées, et l’IA anticipe un nul."
        else:
            msg = "Victoire probable : un déséquilibre a été détecté entre les deux équipes."

    if match_importance == 1:
        msg += " Ce match est considéré comme important."

    return msg

# -------------------------------------------------------------------
# 🧠 Couche "hors-cadre" — PURE (n'altère jamais prediction/probas)
# -------------------------------------------------------------------

def apply_unexpected_layer(
    base_pred: dict,
    season_current_df: pd.DataFrame,
    season_past_list: list,
    home: str, away: str, match_date: str,
    feats_df: pd.DataFrame,
    league_code: str = "default",
    X_ref_features: pd.DataFrame = None
) -> dict:
    """
    Ajoute (éventuellement) :
      1) une double chance 'anti-upset' (bogey/GKI) SANS modifier prediction/probas
      2) une double chance 'forme vs favori du marché' prioritaire sur (1)
    """

    # --- 0) arbitre DC : coeur vs anti-upset ---------------------------------
    def _combine_double_chance(base_dc, anti_dc, base_rule_applied, upset_score, upset_th):
        # Cas simples
        if anti_dc is None:
            return base_dc, "base_only"
        if base_dc is None:
            return anti_dc, "anti_only"
        if base_dc == anti_dc:
            return base_dc, "agree"
        # Conflit : prioriser l'anti-upset si coeur est déjà en doute,
        # ou si le risque d'upset dépasse largement le seuil.
        if ("filtered_out" in str(base_rule_applied)) or (upset_score >= 1.2 * float(upset_th)):
            return anti_dc, "override_by_anti"
        # Sinon on garde la DC du coeur et on expose l'alternative
        return base_dc, "keep_base_conflict"

    # --- 0bis) paramètres couche "forme vs marché" ---------------------------
    def _get_form_layer_params(league_code):
        """
        Lit des seuils éventuels dans champ_config.json ; sinon défauts sûrs.
        - form_dc_threshold: écart de forme minimal (HomeForm - AwayForm) en valeur absolue pour activer la couche
        - max_fav_gap_for_override: favoritisme du marché (gap 2-voies) au-delà duquel on n'outrepasse pas le marché
        - min_uncertainty_for_form_layer: (optionnel) niveau d'incertitude ligue requis pour activer la couche
        """
        try:
            p = _get_params(league_code)  # déjà présent dans ton module
        except Exception:
            p = {}
        form_dc_threshold = float(p.get("form_dc_threshold", 0.20))         # ~20 pts de forme
        max_fav_gap_for_override = float(p.get("max_fav_gap_for_override", 0.08))  # 8 points de prob. 2-voies
        min_uncertainty_for_form_layer = float(p.get("min_uncertainty_for_form_layer", 0.10))
        return form_dc_threshold, max_fav_gap_for_override, min_uncertainty_for_form_layer

    form_dc_threshold, max_fav_gap_for_override, min_unc_for_form = _get_form_layer_params(league_code)

    # --- 1) Copier et geler les sorties du modèle ----------------------------
    enriched = dict(base_pred)
    for k in ["prediction", "proba_0", "proba_1", "proba_2", "explanation", "double_chance", "rule_applied"]:
        if k in base_pred:
            enriched[k] = base_pred[k]

    # --- helpers internes -----------------------------------------------------
    def _slice_asof(df: pd.DataFrame, asof: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        return d[d["Date"] < pd.to_datetime(asof)]

    def _combine_for_signals(cur: pd.DataFrame, past_list: list, asof: str, w_curr=1.0, w_past=0.6):
        frames = []
        c = _slice_asof(cur, asof)
        if not c.empty:
            c = c.copy(); c["_w"] = w_curr; frames.append(c)
        if past_list:
            plist = []
            for x in past_list:
                if x is not None:
                    sl = _slice_asof(x, asof)
                    if not sl.empty:
                        plist.append(sl)
            if plist:
                p = pd.concat(plist, ignore_index=True)
                p = p.copy(); p["_w"] = w_past; frames.append(p)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "HTHG", "HTAG", "_w"])

    def _points_for_team(row, team: str) -> int:
        if row["FTR"] == "D":
            return 1
        return 3 if ((row["FTR"]=="H" and row["HomeTeam"]==team) or (row["FTR"]=="A" and row["AwayTeam"]==team)) else 0

    def _table_points_asof(df):
        teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]]))
        pts = {t: 0.0 for t in teams}
        for _, r in df.iterrows():
            w = float(r.get("_w", 1.0))
            if r["FTR"]=="H":
                pts[r["HomeTeam"]] += 3*w
            elif r["FTR"]=="A":
                pts[r["AwayTeam"]] += 3*w
            else:
                pts[r["HomeTeam"]] += 1*w
                pts[r["AwayTeam"]] += 1*w
        tab = pd.DataFrame({"Team": list(pts.keys()), "Pts": list(pts.values())})
        return tab.sort_values("Pts", ascending=False).reset_index(drop=True)

    def bogey_index(df_for_signals, team_a, team_b, asof, window=5):
        d = _slice_asof(df_for_signals, asof)
        m = ((d["HomeTeam"]==team_a) & (d["AwayTeam"]==team_b)) | ((d["HomeTeam"]==team_b) & (d["AwayTeam"]==team_a))
        d = d.loc[m].sort_values("Date").tail(window)
        if d.empty:
            return 0.0
        w = np.ones(len(d))
        if "_w" in d.columns:
            w *= d["_w"].to_numpy()
        if not w.sum():
            w = np.ones(len(d))
        ptsA = d.apply(lambda r: _points_for_team(r, team_a), axis=1).to_numpy()
        ptsA_w = np.average(ptsA, weights=w)
        expected = 1.5  # environ neutre
        return float(max(-0.5, min(0.5, (ptsA_w - expected)/3.0)))

    def giant_killer_index(df_for_signals, team, asof, topn=5):
        d = _slice_asof(df_for_signals, asof)
        if d.empty:
            return 0.0
        top = set(_table_points_asof(d)["Team"].head(topn).tolist())
        sel = d[(d["HomeTeam"]==team) | (d["AwayTeam"]==team)]
        vs_top_pts = vs_top_g = vs_rest_pts = vs_rest_g = 0.0
        for _, r in sel.iterrows():
            p = _points_for_team(r, team)
            opp = r["AwayTeam"] if r["HomeTeam"]==team else r["HomeTeam"]
            if opp in top:
                vs_top_pts += p; vs_top_g += 1
            else:
                vs_rest_pts += p; vs_rest_g += 1
        if vs_top_g == 0:
            return 0.0
        s = (vs_top_pts/max(1,vs_top_g)) - (vs_rest_pts/max(1,vs_rest_g))
        return float(max(-1.0, min(1.0, s/3.0)))

    # --- 2) paramètres ligue (typage sûr) ------------------------------------
    (bm, ut, imp, stage, upset_th, skip_th, bw, gw) = _safe_parametres(league_code)
    bw = _as_float(bw, 0.40); gw = _as_float(gw, 0.60)
    upset_th = _as_float(upset_th, 0.55); skip_th = _as_float(skip_th, 1.50)

    # --- 3) historique pondéré ------------------------------------------------
    df_sig = _combine_for_signals(season_current_df, season_past_list, match_date, 1.0, 0.6)

    # --- 4) favori marché (démargé) ------------------------------------------
    try:
        b365h = float(feats_df["B365H"].values[0])
        b365d = float(feats_df["B365D"].values[0])
        b365a = float(feats_df["B365A"].values[0])

        # eps: petite marge dépendante de la marge bookmaker par ligue
        bm_eps, *_ = _safe_parametres(league_code)
        eps = max(0.02, 0.5*float(bm_eps))

        fav_side, pH2, pA2, fav_gap = _fav_by_demarged(b365h, b365d, b365a, eps=eps)
        if fav_side is None:
            home_is_fav = None
            outsider = None
        else:
            home_is_fav = (fav_side == "home")
            outsider = away if home_is_fav else home

        enriched.setdefault("notes", [])
        enriched["notes"].append(
            f"fav_demarged: side={fav_side}, pH2={pH2:.3f}, pA2={pA2:.3f}, gap={fav_gap:.3f}, eps={eps:.3f}"
        )
    except Exception as e:
        home_is_fav = None
        outsider = None
        enriched.setdefault("notes", [])
        enriched["notes"].append(f"fav_demarged: error={type(e).__name__}")

    # --- 5) signaux hors-cadre (anti-upset) ----------------------------------
    bidx_home = bogey_index(df_sig, home, away, match_date)
    gki_home  = giant_killer_index(df_sig, home, match_date)
    gki_away  = giant_killer_index(df_sig, away, match_date)
    gki_outs  = gki_away if home_is_fav else gki_home

    bogey_for_outsider = bidx_home if outsider == home else -bidx_home
    bogey_for_outsider = _as_float(bogey_for_outsider, 0.0)
    gki_outs           = _as_float(gki_outs, 0.0)

    upset_score = max(0.0, bw*max(0.0, bogey_for_outsider) + gw*max(0.0, gki_outs))
    upset_score = float(min(1.0, upset_score))

    base_dc = base_pred.get("double_chance")
    if home_is_fav is None:
        anti_dc = None
    else:
        anti_dc = "X2" if (upset_score >= upset_th and home_is_fav) else \
                  "1X" if (upset_score >= upset_th and not home_is_fav) else None

    dc_after_anti, dc_reason = _combine_double_chance(
        base_dc=base_dc,
        anti_dc=anti_dc,
        base_rule_applied=enriched.get("rule_applied", ""),
        upset_score=upset_score,
        upset_th=upset_th
    )

    enriched["double_chance"] = dc_after_anti
    enriched["dc_reason"] = dc_reason

    if anti_dc is not None:
        ra = enriched.get("rule_applied", "")
        if "unexpected_layer" not in str(ra):
            enriched["rule_applied"] = (ra + "|unexpected_layer") if ra else "unexpected_layer"

    # --- 6) Couche PRIORITAIRE : Forme vs Favori du marché -------------------
    # Règle métier (définie avec toi) :
    #  - Si favori = domicile ET HomeForm < AwayForm (écart significatif)  -> DC = "1X"
    #  - Si favori = extérieur ET HomeForm > AwayForm (écart significatif) -> DC = "12"
    dc_form = None
    try:
        hform = float(feats_df["HomeForm"].values[0]) if "HomeForm" in feats_df.columns else 0.0
        aform = float(feats_df["AwayForm"].values[0]) if "AwayForm" in feats_df.columns else 0.0
        form_diff = hform - aform  # >0 avantage domicile ; <0 avantage extérieur

        # Garde-fou: ne pas renverser si le marché a un favori très net
        market_dominant = (home_is_fav is not None) and (abs(float(fav_gap)) >= max_fav_gap_for_override)

        # (optionnel) gating par incertitude ligue
        league_uncertainty_ok = (float(ut) >= float(min_unc_for_form))

        if (home_is_fav is not None) and (not market_dominant) and league_uncertainty_ok:
            if home_is_fav and (form_diff < -form_dc_threshold):
                dc_form = "1X"   # couvre 1 + nul
            elif (not home_is_fav) and (form_diff >  form_dc_threshold):
                dc_form = "12"   # privilégie domicile, couvre favori extérieur (pas de nul)

        # Journalisation
        enriched.setdefault("notes", [])
        enriched["notes"].append(
            f"form_vs_market: HomeForm={hform:.2f}, AwayForm={aform:.2f}, diff={form_diff:.2f}, "
            f"fav={('home' if home_is_fav else ('away' if home_is_fav is False else 'none'))}, "
            f"fav_gap={float(fav_gap) if home_is_fav is not None else 'nan'}, "
            f"th_form={form_dc_threshold:.2f}, max_fav_gap={max_fav_gap_for_override:.2f}, "
            f"league_unc_ok={league_uncertainty_ok}"
        )
    except Exception as e:
        enriched.setdefault("notes", [])
        enriched["notes"].append(f"form_vs_market: error={type(e).__name__}")

    # Priorité: si dc_form est posée, elle ECRASE la DC précédente
    if dc_form is not None:
        prev_dc = enriched.get("double_chance")
        prev_reason = enriched.get("dc_reason", "")
        enriched["double_chance"] = dc_form
        enriched["dc_reason"] = f"override_by_form({prev_reason or 'none'})"
        ra = enriched.get("rule_applied", "")
        enriched["rule_applied"] = (ra + "|form_layer") if ra else "form_layer"
        enriched.setdefault("notes", []).append(f"form_layer_applied: prev_dc={prev_dc} -> dc_form={dc_form}")

    # --- 7) notes d'audit -----------------------------------------------------
    enriched.setdefault("notes", [])
    enriched["notes"].append(
        f"anti-oc: Upset={upset_score:.2f}, Bogey={bidx_home:.2f}, "
        f"GKI_outsider={(gki_away if home_is_fav else gki_home):.2f}, "
        f"DC_base={base_dc}, DC_anti={anti_dc}, DC_final={enriched.get('double_chance')}, reason={enriched.get('dc_reason')}"
    )
    enriched["_upset_score"] = upset_score
    enriched["_upset_threshold"] = float(upset_th)

    return enriched



##---------------------- FIN FONCTION  ------------------------------------------


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
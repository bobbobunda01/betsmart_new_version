#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025

@author: bobunda
"""

import datetime
import json
from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request
from typing import List
import numpy as np
import pandas as pd
import os
from dateutil.parser import parse
from numpy import floating, integer, ndarray
##------------------------------- PREDICTION DES EQUIPES WIN LOSS DRAW ------------------------------------------------
# forme des équipes
def form(d_plf):

    def get_points(result):
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0
    
    def get_form_points(string):
        sum = 0
        for letter in string:
            sum += get_points(letter)
        return sum

    d_plf['HTFormPtsStr'] = d_plf['HM1'] + d_plf['HM2'] + d_plf['HM3'] + d_plf['HM4'] + d_plf['HM5']
    d_plf['ATFormPtsStr'] = d_plf['AM1'] + d_plf['AM2'] + d_plf['AM3'] + d_plf['AM4'] + d_plf['AM5']

    d_plf['HTFormPts'] = d_plf['HTFormPtsStr'].apply(get_form_points)
    d_plf['ATFormPts'] = d_plf['ATFormPtsStr'].apply(get_form_points)

    # Identify Win/Loss Streaks if any.
    def get_3game_ws(string):
        if string[-3:] == 'WWW':
            return 1
        else:
            return 0

    def get_5game_ws(string):
        if string == 'WWWWW':
            return 1
        else:
            return 0

    def get_3game_ls(string):
        if string[-3:] == 'LLL':
            return 1
        else:
            return 0

    def get_5game_ls(string):
        if string == 'LLLLL':
            return 1
        else:
            return 0

    d_plf['HTWinStreak3'] = d_plf['HTFormPtsStr'].apply(get_3game_ws)
    d_plf['HTWinStreak5'] = d_plf['HTFormPtsStr'].apply(get_5game_ws)
    d_plf['HTLossStreak3'] = d_plf['HTFormPtsStr'].apply(get_3game_ls)
    d_plf['HTLossStreak5'] = d_plf['HTFormPtsStr'].apply(get_5game_ls)

    d_plf['ATWinStreak3'] = d_plf['ATFormPtsStr'].apply(get_3game_ws)
    d_plf['ATWinStreak5'] = d_plf['ATFormPtsStr'].apply(get_5game_ws)
    d_plf['ATLossStreak3'] = d_plf['ATFormPtsStr'].apply(get_3game_ls)
    d_plf['ATLossStreak5'] = d_plf['ATFormPtsStr'].apply(get_5game_ls)
    return d_plf

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
        
# Mise à jour des colonnes de dataset conformément aux variables des modèles
def data_df(df, model):
    
    for feature in model.feature_names_in_:
        if feature not in df.columns:
            df[feature] = False  # or np.nan, depending on your use case
    # Reorder columns to match training data
    df = df[model.feature_names_in_]
    df.replace({True:1, False:0}, inplace=True)
    return df

## fome des équipes 
def df_data(sa_25, home, away):
    cols_home=['HomeTeam','FTHG', 'FTAG', 'HTGS', 'HTGC','HTP','HM1','HM2', 'HM3', 'HM4', 'HM5', 'Date']
    
    cols_away=['AwayTeam','FTHG','FTAG','ATGS', 'ATGC','ATP','AM1','AM2', 'AM3', 'AM4', 'AM5','Date']
    #HOME
    df_home=pd.DataFrame()
    df_away=pd.DataFrame()
    ### Home
    date_hh=sa_25.loc[sa_25['HomeTeam']==home,['Date']].sort_values(by='Date', ascending=False).head(1)
    date_hw=sa_25.loc[sa_25['AwayTeam']==home,['Date']].sort_values(by='Date', ascending=False).head(1)
    if (date_hh['Date'].values>date_hw['Date'].values) or  (date_hw.empty):
        df_home=sa_25.loc[sa_25['HomeTeam']==home,cols_home].sort_values(by='Date', ascending=False).head(1)
        df_home['HTGS']=df_home['HTGS']+df_home['FTHG']
        df_home['HTGC']=df_home['HTGC']+df_home['FTAG']
        #df_home['HHGS']=df_home['HHGS']+df_home['HTHG']
        #df_home['HHGC']=df_home['HHGC']+df_home['HTAG']
        if df_home['FTHG'].values>df_home['FTAG'].values:
            df_home['HTP']=df_home['HTP']+3
            a='W'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
        elif df_home['FTHG'].values==df_home['FTAG'].values:
            df_home['HTP']=df_home['HTP']+1
            a='D'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
        else:
            a='L'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
    else:#
        df_home=sa_25.loc[sa_25['AwayTeam']==home,cols_away].sort_values(by='Date', ascending=False).head(1)
        df_home.columns=cols_home
        #df_home['HTHG']=pd.to_numeric(df['HTHG'], errors='coerce')
        df_home['HTGS']=df_home['HTGS']+df_home['FTAG']
        df_home['HTGC']=df_home['HTGC']+df_home['FTHG']
        #df_home['HHGS']=df_home['HHGS']+df_home['HTAG']
        #df_home['HHGC']=df_home['HHGC']+df_home['HTHG']
        if df_home['FTAG'].values>df_home['FTHG'].values:
            df_home['HTP']=df_home['HTP']+3
            a='W'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
            df_home['FTHG']=df_home['FTAG']
        elif df_home['FTAG'].values==df_home['FTHG'].values:
            df_home['HTP']=df_home['HTP']+1
            a='D'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
            df_home['FTHG']=df_home['FTAG']
        else:
            a='L'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
            df_home['FTHG']=df_home['FTAG']
    #df_home['FTHG']=round((df_home['HTGS']/32),0)
    #df_home=df_home.drop(['FTAG','HTAG'], axis=1)
    #AWAY
    date_hh=sa_25.loc[sa_25['HomeTeam']==away,['Date']].sort_values(by='Date', ascending=False).head(1)
    date_hw=sa_25.loc[sa_25['AwayTeam']==away,['Date']].sort_values(by='Date', ascending=False).head(1)
    if (date_hh['Date'].values>date_hw['Date'].values) or (date_hw.empty):
        df_away=sa_25.loc[sa_25['HomeTeam']==away,cols_home].sort_values(by='Date', ascending=False).head(1)
        df_away.columns=cols_away
        df_away['ATGS']=df_away['ATGS']+df_away['FTHG']
        df_away['ATGC']=df_away['ATGC']+df_away['FTAG']
        #df_away['AHGS']=df_away['AHGS']+df_away['HTHG']
        #df_away['AHGC']=df_away['AHGC']+df_away['HTAG']

        if df_away['FTHG'].values>df_away['FTAG'].values:
            df_away['ATP']=df_away['ATP']+3
            a='W'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
            df_away['FTAG']=df_home['FTHG']
        elif df_away['FTHG'].values==df_away['FTAG'].values:
            df_away['ATP']=df_away['ATP']+1
            a='D'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
            df_away['FTAG']=df_home['FTHG']
        else:
            a='L'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
            df_away['FTAG']=df_home['FTHG']
    else:#
        df_away=sa_25.loc[sa_25['AwayTeam']==away,cols_away].sort_values(by='Date', ascending=False).head(1)
        df_away['ATGS']=df_away['ATGS']+df_away['FTAG']
        df_away['ATGC']=df_away['ATGC']+df_away['FTHG']
        #df_away['AHGS']=df_away['AHGS']+df_away['HTAG']
        #df_away['AHGC']=df_away['AHGC']+df_away['HTHG']

        if df_away['FTAG'].values>df_away['FTHG'].values:
            df_away['ATP']=df_away['ATP']+3
            a='W'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
        elif df_away['FTHG'].values==df_away['FTAG'].values:
            df_away['ATP']=df_away['ATP']+1
            a='D'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
        else:
            a='L'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
    #df_away['FTAG']=round((df_away['ATGS']/32),0)
    #df_away=df_away.drop(['FTHG', 'HTHG'], axis=1)

    df_home=df_home.reset_index()
    df_away=df_away.reset_index()
    df_home_away=pd.concat([df_home, df_away], axis=1)
    df_home_away=df_home_away.drop('index', axis=1)
    #df_home_away
    #df_home_away=form(df_home_away)
    #df_home_away['DiffPts']=df_home_away['HTP']-df_home_away['ATP']
    #df_home_away['DiffFormPts']=df_home_away['HTFormPts']-df_home_away['ATFormPts']

    #d_plf
    # Get Goal Difference
    df_home_away['HTGD'] = df_home_away['HTGS'] - df_home_away['HTGC']
    df_home_away['ATGD'] = df_home_away['ATGS'] - df_home_away['ATGC']
    #df_home_away['HHGD'] = df_home_away['HHGS'] - df_home_away['HHGC']
    #df_home_away['AHGD'] = df_home_away['AHGS'] - df_home_away['AHGC']

    # Diff in points
    df_home_away['DiffPts'] = df_home_away['HTP'] - df_home_away['ATP']
    #df_home_away['DiffFormPts'] = df_home_away['HTFormPts'] - df_home_away['ATFormPts']
    #cols = ['HTGD','ATGD', 'HHGD', 'AHGD','DiffPts','HTP','ATP']
    #for col in cols:
    #    df_home_away[col] = df_home_away[col] /36
    return df_home_away, df_home, df_away

## Nouvelle version input_data_user

### enrichissement des variables
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
## position des équipes et l'importance des matchs
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
## entrées utilisateurs
def prepare_input_features_enriched(home_team, away_team, match_date, b365h, b365a, b365d, season_df):
    """
    Prépare les features enrichies pour la prédiction d'un match avec classement dynamique.
    """
    df = season_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    if home_team not in df['HomeTeam'].values or away_team not in df['AwayTeam'].values:
        raise ValueError("Une des équipes n'existe pas dans l'historique")

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

    return features

## Fonction de prédiction 

def predict_match_with_proba(
    features_df: pd.DataFrame,
    model_stage1,
    model_stage2,
    threshold_draw,
    bookmaker_margin=0.0711,
    user_profile="standard"
) -> dict:
    """
    Prédiction hybride LogReg + RF avec logique métier, probas normalisées,
    et explication intégrée selon le profil utilisateur.
    """

    # Étape 1 : préparation des features pour le modèle LogReg
    features_df_etape_1 = features_df.copy()
    for feature in model_stage1.feature_names_in_:
        if feature not in features_df_etape_1.columns:
            features_df_etape_1[feature] = 0
    features_df_etape_1 = features_df_etape_1[model_stage1.feature_names_in_]

    # Prédiction LogReg
    proba_draw_class = model_stage1.predict_proba(features_df_etape_1)[0]
    proba_draw = proba_draw_class[1]
    odds_gap = features_df_etape_1[['B365H', 'B365A', 'B365D']].max(axis=1).values[0] - \
               features_df_etape_1[['B365H', 'B365A', 'B365D']].min(axis=1).values[0]

    # Règle 1 : Proche du seuil + cotes équilibrées
    draw_margin_band = 0.02
    if threshold_draw - draw_margin_band <= proba_draw <= threshold_draw + draw_margin_band:
        if odds_gap <= bookmaker_margin:
            proba_1 = proba_draw
            proba_0 = proba_2 = (1 - proba_1) / 2
            return {
                "prediction": 1,
                "proba_0": str(round(proba_0*100,0))+'%',
                "proba_1": str(round(proba_1*100,0))+'%',
                "proba_2": str(round(proba_2*100,0))+'%',
                "rule_applied": "margin_adjusted",
                "explanation": generate_explanation("margin_adjusted", features_df, user_profile)
            }

    # Règle 2 : Seuil classique
    if proba_draw >= threshold_draw:
        proba_1 = proba_draw
        proba_0 = proba_2 = (1 - proba_1) / 2
        return {
            "prediction": 1,
            "proba_0": str(round(proba_0*100,0))+'%',
            "proba_1": str(round(proba_1*100,0))+'%',
            "proba_2": str(round(proba_2*100,0))+'%',
            "rule_applied": "threshold",
            "explanation": generate_explanation("threshold", features_df, user_profile)
        }

    # Étape 2 : RandomForest sur les cas non-nuls
    features_df_etape_2 = features_df.copy()
    for feature in model_stage2.feature_names_in_:
        if feature not in features_df_etape_2.columns:
            features_df_etape_2[feature] = 0
    features_df_etape_2 = features_df_etape_2[model_stage2.feature_names_in_]

    proba_rf = model_stage2.predict_proba(features_df_etape_2)[0]
    prediction_rf = int(model_stage2.predict(features_df_etape_2)[0])

    total = proba_rf[0] + proba_draw + proba_rf[1]
    proba_0 = proba_rf[0] / total
    proba_1 = proba_draw / total
    proba_2 = proba_rf[1] / total

    if prediction_rf==0:
        
        if proba_draw>=proba_rf[0]:
            a=proba_0
            proba_0=proba_1
            proba_1=a
    else:
        
        if proba_draw>=proba_rf[1]:
                a=proba_1
                proba_2=proba_1
                proba_1=a
    return {
        "prediction": prediction_rf,
        "proba_0":str(round(proba_0*100,0))+'%',
        "proba_1": str(round(proba_1*100,0))+'%',
        "proba_2": str(round(proba_2*100,0))+'%',
        "rule_applied": "rf_decision",
        "explanation": generate_explanation("rf_decision", features_df, user_profile)
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
    while True:
        user_input = user_input.strip()
        try:
            date_obj = parse(user_input, dayfirst=True)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            print("Format invalide. Réessayez.")


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
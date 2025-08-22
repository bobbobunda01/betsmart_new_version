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
import pathlib
import sys
import logging
from datetime import datetime
from  fichier_py.fonction import  prepare_input_features_enriched, predict_match_with_proba,log_prediction, get_valid_date, entree_utilisateur,get_last5_results_pattern
thread=0
app = Flask(__name__)


def log_dataframe_features_to_file(features_df, home_team, away_team, match_date, log_dir="logs"):
    """
    Enregistre les features dans un fichier log JSON avec horodatage.
    
    Args:
        features_df (pd.DataFrame): Données features en DataFrame (une seule ligne).
        home_team (str): Nom de l'équipe à domicile.
        away_team (str): Nom de l'équipe à l'extérieur.
        match_date (str): Date du match (format: "YYYY-MM-DD").
        log_dir (str): Dossier de destination des fichiers log.
    """
    # Crée le dossier s'il n'existe pas
    os.makedirs(log_dir, exist_ok=True)
    
    # Formatage du nom de fichier (par date de log)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"log_features_{home_team}_vs_{away_team}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    
    # Ajout d’un contexte au log
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "features": features_df.to_dict(orient="records")[0]  # on prend la première ligne (unique)
    }

    # Écriture dans le fichier log
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)


# Modèle Pydantic pour une entrée
class MatchInput(BaseModel):
    HomeTeam: str
    AwayTeam: str
    comp: int
    odds_home:float
    odds_draw:float
    odds_away:float
    match_Date:str
    

# Modèle pour recevoir un tableau d'entrées
class RequestBody(BaseModel):
    matches: List[MatchInput]  # Accepte un tableau de 4 entrées


#RACINE_PROJET = pathlib.Path().resolve().parent.parent
#RACINE_PROJET = pathlib.Path(__file__).resolve().parent.parent

RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]
@app.route('/', methods=["GET"])
def Accueil():
    return jsonify({'Message': 'Bienvenue sur l\'API de prédiction de matchs'})



@app.route('/predire/pl', methods=["POST"])
def prediction():
    if not request.json:
        return jsonify({'Erreur': 'Aucun fichier JSON fourni'}), 400
    
    try:
        # Extraction des 4 entrées
        body = RequestBody(**request.json)
        all_results = []

        for match in body.matches:
            # Traitement pour chaque match
            donnees_df = pd.DataFrame([match.dict()])
            
            home=np.array(donnees_df.HomeTeam.values).item()
            away=np.array(donnees_df.AwayTeam.values).item()
            #comp=np.array(donnees_df.comp.values).item()
            comp=donnees_df["comp"].values[0]
            odds_h = donnees_df["odds_home"].values[0]
            odds_d = donnees_df["odds_draw"].values[0]
            odds_a = donnees_df["odds_away"].values[0]
            match_date=np.array(donnees_df.match_Date.values).item()
            # Premiere league ANGLETERRE
            if comp==39:
                
                # Chargement des données de la Première league
                
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "pl" / "pl_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "pl" / "saison_encours.csv"
                #season_encours=pd.read_csv(s_encours)
                #season_encours['Date']=pd.to_datetime(season_encours['Date'])
                #s_preced=RACINE_PROJET / "data" / "pl" / "pl.csv"
                season_preced=pd.read_csv(chemin_csv)
                #season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            ## SERIE A
            elif comp==135:
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "sa1" / "sa_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "sa1" / "saison_encours.csv"
                #season_encours=pd.read_csv(s_encours)
                #season_encours['Date']=pd.to_datetime(season_encours['Date'])
                #s_preced=RACINE_PROJET / "data" / "sa1" / "sa.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            ### LIGA
            elif comp==140:
                chemin_csv = RACINE_PROJET / "data" / "lg1" / "lg_24_25.csv"
                # Chargement des données historiques
                s_encours=RACINE_PROJET / "data" / "lg1" / "saison_encours.csv"
                #season_encours=pd.read_csv(s_encours)
                #season_encours['Date']=pd.to_datetime(season_encours['Date'])
                #s_preced=RACINE_PROJET / "data" / "lg1" / "lg.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            ## BUNDESLIGA
            elif comp==78:
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "bl1" / "bl_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "bl1" / "saison_encours.csv"
                #season_encours=pd.read_csv(s_encours)
                #season_encours['Date']=pd.to_datetime(season_encours['Date'])
                #s_preced=RACINE_PROJET / "data" / "bl1" / "bl.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
                
            ## PREMIERE LEAGUE FRANCAISE
            elif comp==61:
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "fl" / "fl_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "fl" / "saison_encours.csv"
                #season_encours=pd.read_csv(s_encours)
                #season_encours['Date']=pd.to_datetime(season_encours['Date'])
                #s_preced=RACINE_PROJET / "data" / "fl" / "fl.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            ### NEEDERLANDE
            elif comp==88:
                chemin_csv = RACINE_PROJET / "data" / "N1" / "N_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "N1" / "saison_encours.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            
            ## SUISSE
            elif comp==207:
                chemin_csv = RACINE_PROJET / "data" / "sui" / "suisse_2024_2025.csv"
                s_encours=RACINE_PROJET / "data" / "sui" / "saison_encours.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            ### Portugal
            elif comp==94:
                chemin_csv = RACINE_PROJET / "data" / "port" / "port_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "port" / "saison_encours.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            
                ### Turquie
            elif comp==203:
                chemin_csv = RACINE_PROJET / "data" / "turk" / "turk_24_25.csv"
                s_encours=RACINE_PROJET / "data" / "turk" / "saison_encours.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
            # Japon
            elif comp==98:
                chemin_csv = RACINE_PROJET / "data" / "japon" / "japon_2024.csv"
                s_encours=RACINE_PROJET / "data" / "jaclpon" / "saison_encours.csv"
                season_preced=pd.read_csv(chemin_csv)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(s_encours)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi


            #df=df[['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTGS', 'HTGC','ATGS', 'ATGC', 
            #       'HTP', 'ATP','HM1','AM1','HM2','AM2','HM3','AM3','HM4','AM4','HM5','AM5']]
            #df_home_away, df_home, df_away=df_data(df, home, away)
            #df_home_away = df_home_away.loc[:, ~df_home_away.columns.duplicated(keep='first')]
            
            date_match=get_valid_date(match_date)
           
            features_input=prepare_input_features_enriched(home, away,date_match, odds_h,odds_a,odds_d,df)
            #log_dataframe_features_to_file(features_input,home="West Ham",away="Chelsea",match_date="2025-08-22",)
            #log_prediction(features_input.to_json())
            
            #log_dataframe_features_to_file(features_input,home, away, match_date)
            
            X_inputs=entree_utilisateur(home, away, odds_h,odds_d,odds_a, df, season_preced)
            log_prediction(X_inputs.to_json())
            

           
            #bon modèle ANGLETERRE
            if comp==39:
                chemin_model1 = RACINE_PROJET / "modele" / "pl" / "rf_pl_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "pl" / "rf_pl_stage2.joblib"
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                chemin_but = RACINE_PROJET / "modele" / "pl" / "xgboost_nbre_but_marque_pl.joblib"
                model_but=load(chemin_but)
                thread=0.63
            
            #bon modèle   SERIE A 
            elif comp==135:
                chemin_model1 = RACINE_PROJET / "modele" / "sa1" / "rf_sa1_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "sa1" / "rf_sa1_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "sa1" / "xgboost_nbre_but_marque_sa1.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.63
            ##bon modele LIGA
            elif comp==140:
                chemin_model1 = RACINE_PROJET / "modele" / "lg1" / "lg_bl1_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "lg1" / "rf_bl1_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "lg1" / "xgboost_nbre_but_marque_lg.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.4
            ##Bundesliga
            elif comp==78:
                chemin_model1 = RACINE_PROJET / "modele" / "bl1" / "rf_bl1_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "bl1" / "rf_bl1_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "bl1" / "xgboost_nbre_but_marque_bl1.joblib"
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                model_but=load(chemin_but)
                thread=0.65
            ## Bon modèle France
            elif comp==61:
                chemin_model1 = RACINE_PROJET / "modele" / "fl" / "rf_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "fl" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "fl" / "xgboost_nbre_but_marque_fl.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
            ## Bon modèle NEERDERLAND, PAYS BAS
            elif comp==88:
                chemin_model1 = RACINE_PROJET / "modele" / "N1" / "xg_boost_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "N1" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "N1" / "rf_nbre_but_marque_autre.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
            ## SUISSE
            elif comp==207:
                chemin_model1 = RACINE_PROJET / "modele" / "sui" / "rf_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "sui" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "sui" / "xgboost_nbre_but_marque_sui.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            # Portugal
            elif comp==94:
                chemin_model1 = RACINE_PROJET / "modele" / "port" / "rf_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "port" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "port" / "xgboost_nbre_but_marque_port.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
             # Turquie
            elif comp==203:
                chemin_model1 = RACINE_PROJET / "modele" / "turk" / "rf_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "turk" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "turk" / "xgboost_nbre_but_marque_turk.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
            # japon
            elif comp==98:
                chemin_model1 = RACINE_PROJET / "modele" / "japon" / "rf_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "japon" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "japon" / "xgboost_nbre_but_marque_japon.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
            perf_home=get_last5_results_pattern(df, home, date_match)
            perf_away=get_last5_results_pattern(df, away, date_match)
            
            pred = predict_match_with_proba(features_input,model_stage1=modele1,model_stage2=modele2,threshold_draw=thread, league_code=comp)
          
            pred_but = model_but.predict(X_inputs)[0]
            mess_but="✅ Prédiction :", "Plus de buts en 2ᵉ mi-temps" if pred_but == 1 else "Plus de buts en 1ʳᵉ mi-temps"
            pred['home']=home
            pred['away']=away
            pred['5_dern_perf_home']=np.array(perf_home).item()
            pred['5_dern_perf_away']=np.array(perf_away).item()
            pred['plus_but']=int(pred_but)
            pred['mess_but']=str(mess_but)
            all_results.append(pred)
            # Log l'entrée + les prédictionsÒ
            #log_prediction(all_results)
        
        logging.basicConfig(level=logging.INFO)

        logging.info(f"📊 Résultats all_results : {all_results}")
        return jsonify({'Resultats': all_results})
     
    

    except Exception as e:
        return jsonify({'Erreur': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
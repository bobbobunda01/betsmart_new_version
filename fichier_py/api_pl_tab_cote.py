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
import pathlib
import sys
import logging
from fichier_py.fonction import df_data, prepare_input_features_enriched, predict_match_with_proba,log_prediction, get_valid_date, entree_utilisateur
thread=0
app = Flask(__name__)


# Modèle Pydantic pour une entrée
class MatchInput(BaseModel):
    HomeTeam: str
    AwayTeam: str
    comp: str
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
            comp=np.array(donnees_df.comp.values).item()
            odds_h = donnees_df["odds_home"].values[0]
            odds_d = donnees_df["odds_draw"].values[0]
            odds_a = donnees_df["odds_away"].values[0]
            match_date=np.array(donnees_df.match_Date.values).item()
            
            if comp=='pl':
                
                # Chargement des données de la Première league
                
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "pl" / "pl.csv"
                s_encours=RACINE_PROJET / "data" / "pl" / "saison_encours.csv"
                season_encours=pd.read_csv(s_encours)
                season_encours['Date']=pd.to_datetime(season_encours['Date'])
                s_preced=RACINE_PROJET / "data" / "pl" / "pl.csv"
                season_preced=pd.read_csv(s_preced)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(chemin_csv)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

            elif comp=='sa':
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "sa1" / "sa.csv"
                s_encours=RACINE_PROJET / "data" / "sa1" / "saison_encours.csv"
                season_encours=pd.read_csv(s_encours)
                season_encours['Date']=pd.to_datetime(season_encours['Date'])
                s_preced=RACINE_PROJET / "data" / "sa1" / "sa.csv"
                season_preced=pd.read_csv(s_preced)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(chemin_csv)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
                
            elif comp=='lg':
                chemin_csv = RACINE_PROJET / "data" / "lg1" / "lg.csv"
                # Chargement des données historiques
                s_encours=RACINE_PROJET / "data" / "lg1" / "saison_encours.csv"
                season_encours=pd.read_csv(s_encours)
                season_encours['Date']=pd.to_datetime(season_encours['Date'])
                s_preced=RACINE_PROJET / "data" / "lg1" / "lg.csv"
                season_preced=pd.read_csv(s_preced)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(chemin_csv)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

            elif comp=='bl':
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "bl1" / "bl.csv"
                s_encours=RACINE_PROJET / "data" / "bl1" / "saison_encours.csv"
                season_encours=pd.read_csv(s_encours)
                season_encours['Date']=pd.to_datetime(season_encours['Date'])
                s_preced=RACINE_PROJET / "data" / "bl1" / "bl.csv"
                season_preced=pd.read_csv(s_preced)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                hi=pd.read_csv(chemin_csv)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
                
            
            elif comp=='fl':
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "fl" / "fl.csv"
                #chemin_csv = "../../data/fl/fl_26_04_2025.csv"
                
                s_encours=RACINE_PROJET / "data" / "fl" / "saison_encours.csv"
                season_encours=pd.read_csv(s_encours)
                season_encours['Date']=pd.to_datetime(season_encours['Date'])
                s_preced=RACINE_PROJET / "data" / "fl" / "fl.csv"
                season_preced=pd.read_csv(s_preced)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(chemin_csv)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi
                
            
            elif comp=='au':
                # Chargement des données historiques
                chemin_csv = RACINE_PROJET / "data" / "Autres" / "N_2025.csv"
                #chemin_csv = "../../data/fl/fl_26_04_2025.csv"
                
                s_encours=RACINE_PROJET / "data" / "Autres" / "saison_encours.csv"
                season_encours=pd.read_csv(s_encours)
                season_encours['Date']=pd.to_datetime(season_encours['Date'])
                s_preced=RACINE_PROJET / "data" / "Autres" / "N_2025.csv"
                season_preced=pd.read_csv(s_preced)
                season_preced['Date']=pd.to_datetime(season_preced['Date'])
                
                hi=pd.read_csv(chemin_csv)
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

            df=df[['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTGS', 'HTGC','ATGS', 'ATGC', 
                   'HTP', 'ATP','HM1','AM1','HM2','AM2','HM3','AM3','HM4','AM4','HM5','AM5']]
            df_home_away, df_home, df_away=df_data(df, home, away)
            df_home_away = df_home_away.loc[:, ~df_home_away.columns.duplicated(keep='first')]
            date_match=get_valid_date(match_date)
            features_input=prepare_input_features_enriched(home, away,date_match, odds_h, odds_d, odds_a,df)
            
            X_inputs=entree_utilisateur(home, away, odds_h,odds_d,odds_a, season_encours, season_preced)
            #log_prediction(X_inputs.to_json())
            #new_df = pd.DataFrame([df_test])
            
            #df_home_away = pd.concat([df_home_away, new_df], axis=1)
    
            
            perf_home=df_home['HM1']+df_home['HM2']+df_home['HM3']+df_home['HM4']+df_home['HM5']
            perf_away=df_away['AM1']+df_away['AM2']+df_away['AM3']+df_away['AM4']+df_away['AM5']
            #log_prediction(str(perf_home))
           
            #bon modèle
            if comp=='pl':
                chemin_model1 = RACINE_PROJET / "modele" / "pl" / "rf_pl_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "pl" / "rf_pl_stage2.joblib"
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                chemin_but = RACINE_PROJET / "modele" / "pl" / "xgboost_nbre_but_marque_pl.joblib"
                model_but=load(chemin_but)
                thread=0.63
            
            #bon modèle    
            elif comp=='sa':
                chemin_model1 = RACINE_PROJET / "modele" / "sa1" / "rf_sa1_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "sa1" / "rf_sa1_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "sa1" / "xgboost_nbre_but_marque_sa1.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.63
            ##bon modele
            elif comp=='lg':
                chemin_model1 = RACINE_PROJET / "modele" / "lg1" / "lg_bl1_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "lg1" / "rf_bl1_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "lg1" / "xgboost_nbre_but_marque_lg.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.4
            ##BON MODÈLE
            elif comp=='bl':
                chemin_model1 = RACINE_PROJET / "modele" / "bl1" / "rf_bl1_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "bl1" / "rf_bl1_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "bl1" / "xgboost_nbre_but_marque_bl1.joblib"
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                model_but=load(chemin_but)
                thread=0.65
            ## Bon modèle
            elif comp=='fl':
                chemin_model1 = RACINE_PROJET / "modele" / "fl" / "rf_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "fl" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "fl" / "xgboost_nbre_but_marque_fl.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
            ## Bon modèle
            elif comp=='au':
                chemin_model1 = RACINE_PROJET / "modele" / "Autres" / "xg_boost_stage1.joblib"
                chemin_model2 = RACINE_PROJET / "modele" / "Autres" / "rf_stage2.joblib"
                chemin_but = RACINE_PROJET / "modele" / "Autres" / "rf_nbre_but_marque_autre.joblib"
                model_but=load(chemin_but)
                modele1=load(chemin_model1)
                modele2=load(chemin_model2)
                thread=0.6
            
            pred = predict_match_with_proba(features_input,model_stage1=modele1,model_stage2=modele2,threshold_draw=thread)
            log_prediction(pred)
            pred_but = model_but.predict(X_inputs)[0]
            mess_but="✅ Prédiction :", "Plus de buts en 2ᵉ mi-temps" if pred_but == 1 else "Plus de buts en 1ʳᵉ mi-temps"
            pred['home']=home
            pred['away']=away
            pred['5_dern_perf_home']=np.array(perf_home).item()
            pred['5_dern_perf_away']=np.array(perf_away).item()
            pred['plus_but']=int(pred_but)
            pred['mess_but']=str(mess_but)
            all_results.append(pred)
            # Log l'entrée + les prédictions
            #log_prediction(all_results)
        
        logging.basicConfig(level=logging.INFO)

        logging.info(f"📊 Résultats all_results : {all_results}")
        return jsonify({'Resultats': all_results})
     
    

    except Exception as e:
        return jsonify({'Erreur': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
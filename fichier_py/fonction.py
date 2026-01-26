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
from functools import lru_cache
import requests
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple, List
import datetime as dt

##------------------------------- PREDICTION DES EQUIPES WIN LOSS DRAW ------------------------------------------------


# log des prédictions utilisateurs


REALTIME_API_URL="https://v3.football.api-sports.io"
REALTIME_API_KEY="1ccc14e8da5a40c0575ae0c272645ecf"
DEBUG_REALTIME=1
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _norm_team_name(name: Any) -> str:
    """Normalize team name for safer matching (accent/spacing/case)."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_match_date(match_date: Any) -> Optional[dt.date]:
    """Accepts date, datetime, or common string formats; returns date or None."""
    if match_date is None or match_date == "":
        return None
    if isinstance(match_date, dt.date) and not isinstance(match_date, dt.datetime):
        return match_date
    if isinstance(match_date, dt.datetime):
        return match_date.date()
    s = str(match_date).strip()
    # try ISO first
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:19], fmt).date()
        except Exception:
            continue
    # last resort: try fromisoformat
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None


def _safe_get_first(df_like: Any, col: str) -> Any:
    """Return first value of df_like[col] if possible (supports dict-like and pandas DataFrame)."""
    try:
        # pandas DataFrame
        if hasattr(df_like, "columns") and col in getattr(df_like, "columns"):
            if len(df_like) == 0:
                return None
            return df_like.iloc[0][col]
    except Exception:
        pass
    try:
        # dict-like
        v = df_like.get(col)
        if isinstance(v, list) and v:
            return v[0]
        return v
    except Exception:
        return None


def _resolve_fixture_id_from_df(
    season_df: Any,
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
) -> Optional[int]:
    """
    Try to resolve fixture_id from a season dataframe (if present).
    This is the preferred (offline) method used in apply_unexpected_layer().
    """
    if season_df is None:
        return None
    # candidate columns
    fid_cols = [c for c in ["fixture_id", "FixtureID", "fixture", "Fixture", "id", "ID"] if hasattr(season_df, "columns") and c in season_df.columns]
    if not fid_cols:
        return None

    # team/date column candidates
    home_cols = [c for c in ["HomeTeam", "home", "home_name", "Home", "HomeTeamName"] if hasattr(season_df, "columns") and c in season_df.columns]
    away_cols = [c for c in ["AwayTeam", "away", "away_name", "Away", "AwayTeamName"] if hasattr(season_df, "columns") and c in season_df.columns]
    date_cols = [c for c in ["Date", "date", "match_date", "MatchDate", "fixture_date"] if hasattr(season_df, "columns") and c in season_df.columns]

    if not home_cols or not away_cols or not date_cols:
        return None

    h = _norm_team_name(home_name)
    a = _norm_team_name(away_name)
    d = _parse_match_date(match_date)

    # if no date, try only teams (less precise)
    try:
        df = season_df
        # normalize to strings for compare
        # We keep it safe: any error -> None
        for hc in home_cols:
            for ac in away_cols:
                tmp = df
                try:
                    # filter by teams
                    mask = tmp[hc].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).eq(h) & \
                           tmp[ac].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).eq(a)
                    tmp2 = tmp[mask]
                except Exception:
                    continue

                if d is not None:
                    for dc in date_cols:
                        try:
                            tmp3 = tmp2.copy()
                            # parse date column safely
                            tmp3[dc] = tmp3[dc].astype(str).str.slice(0, 10)
                            maskd = tmp3[dc].apply(lambda x: _parse_match_date(x)).eq(d)
                            tmp4 = tmp2[maskd]
                            if len(tmp4) > 0:
                                fid = tmp4.iloc[0][fid_cols[0]]
                                try:
                                    return int(fid)
                                except Exception:
                                    return None
                        except Exception:
                            continue

                # no date match, but teams match
                if len(tmp2) > 0:
                    fid = tmp2.iloc[0][fid_cols[0]]
                    try:
                        return int(fid)
                    except Exception:
                        return None
    except Exception:
        return None

    return None


def _resolve_fixture_id_by_names(
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
) -> Optional[int]:
    """
    Online fallback fixture resolver (API).
    It is ONLY used when you don't have season_current_df to resolve offline.

    Configure with env vars:
      - REALTIME_API_URL (base, e.g. https://v3.football.api-sports.io)
      - REALTIME_API_KEY
      - REALTIME_API_HOST (optional, for RapidAPI-style hosts)
    """
    
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY",REALTIME_API_KEY)
    if not api_url or not api_key or requests is None:
        return None

    d = _parse_match_date(match_date)
    if d is None:
        return None

    # Endpoint strategy: /fixtures?date=YYYY-MM-DD
    # Then filter by team names (best effort).
    url = f"{api_url}/fixtures"
    headers = {
        "x-apisports-key": api_key,
    }
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    params = {"date": d.isoformat()}
    if league_code:
        # if league_code is numeric, pass as league; otherwise ignore
        try:
            int(league_code)
            params["league"] = league_code
        except Exception:
            pass

    try:
        r = requests.get(url, headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    # api-sports returns {"response":[{"fixture":{"id":...},"teams":{"home":{"name":...},"away":{"name":...}} ...}]}
    home_n = _norm_team_name(home_name)
    away_n = _norm_team_name(away_name)

    try:
        resp = data.get("response", [])
        for item in resp:
            th = _norm_team_name(item.get("teams", {}).get("home", {}).get("name"))
            ta = _norm_team_name(item.get("teams", {}).get("away", {}).get("name"))
            if th == home_n and ta == away_n:
                fid = item.get("fixture", {}).get("id")
                if fid is not None:
                    return int(fid)
    except Exception:
        return None

    return None


def _safe_resolve_fixture_id(
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
    season_df: Any = None,
) -> Optional[int]:
    """Safe wrapper: try offline df, then online resolver."""
    # 1) offline resolution (best)
    fid = _resolve_fixture_id_from_df(season_df, home_name, away_name, match_date, league_code=league_code)
    if fid is not None:
        return fid

    # 2) online fallback
    return _resolve_fixture_id_by_names(home_name, away_name, match_date, league_code=league_code)

class RealtimeFetchError(Exception):
    """Internal exception used to carry http/debug info without breaking the pipeline."""
    def __init__(self, code: str, detail: str = "", status: Optional[int] = None):
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.status = status


def _get_realtime_api_config() -> Tuple[str, str, str]:
    """
    Returns (api_url, api_key, api_host).
    - api_url/api_key first from env
    - then fallback to module-level constants if you defined them (optional)
    """
    # Optional module-level fallbacks if you defined them elsewhere:
    fallback_url = globals().get("REALTIME_API_URL",REALTIME_API_URL)
    fallback_key = globals().get("REALTIME_API_KEY", REALTIME_API_KEY)
    fallback_host = globals().get("REALTIME_API_HOST", "")

    api_url = os.getenv("REALTIME_API_URL", fallback_url).strip().rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY", fallback_key).strip()
    api_host = os.getenv("REALTIME_API_HOST", fallback_host).strip()
    return api_url, api_key, api_host

def _fetch_realtime_context(fixture_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time context from API-Sports:
      GET {REALTIME_API_URL}/fixtures?id=<fixture_id>

    Returns:
      - dict (response[0]) if available
      - raises RealtimeFetchError for any diagnosable error
      - NEVER returns invalid shapes
    """
    api_url, api_key, api_host = _get_realtime_api_config()

    if requests is None:
        raise RealtimeFetchError("requests_not_available", "requests is None")

    if not api_url:
        raise RealtimeFetchError("realtime_api_url_missing", "REALTIME_API_URL not set")
    if not api_key:
        raise RealtimeFetchError("realtime_api_key_missing", "REALTIME_API_KEY not set")

    url = f"{api_url}/fixtures"

    # ✅ API-Sports direct header
    headers = {"x-apisports-key": api_key}

    # ✅ RapidAPI mode (optional)
    # If you use RapidAPI, REALTIME_API_HOST must be set, and the key may need to be x-rapidapi-key.
    # We'll support both safely:
    if api_host:
        headers["x-rapidapi-host"] = api_host
        # If you are *really* on RapidAPI, uncomment next line and/or set REALTIME_RAPIDAPI_MODE=1
        rapid_mode = os.getenv("REALTIME_RAPIDAPI_MODE", "").strip() in ("1", "true", "True", "yes", "YES")
        if rapid_mode:
            headers["x-rapidapi-key"] = api_key

    try:
        r = requests.get(url, headers=headers, params={"id": int(fixture_id)}, timeout=10)

        # Diagnose common HTTP issues explicitly
        status = getattr(r, "status_code", None)

        if status == 401:
            raise RealtimeFetchError("http_401_unauthorized", "Invalid API key or wrong header", status=status)
        if status == 403:
            raise RealtimeFetchError("http_403_forbidden", "Forbidden (plan/host/key mismatch)", status=status)
        if status == 429:
            raise RealtimeFetchError("http_429_rate_limited", "Rate limit reached", status=status)

        r.raise_for_status()

        data = r.json() if r is not None else None
        if not isinstance(data, dict):
            raise RealtimeFetchError("invalid_json", "Response JSON is not a dict", status=status)

        resp = data.get("response", [])
        if not isinstance(resp, list):
            raise RealtimeFetchError("invalid_response_shape", "data['response'] is not a list", status=status)

        return resp[0] if len(resp) > 0 else None

    except RealtimeFetchError:
        raise
    except Exception as e:
        # Any unexpected error becomes diagnosable
        raise RealtimeFetchError("fetch_exception", f"{type(e).__name__}: {e}")
    
def _realtime_risk_score(ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert real-time context to a risk score.
    This function is intentionally conservative: it NEVER throws, and defaults to UNKNOWN.
    You can enrich later with your own signals without breaking the pipeline.
    """
    out = {
        "risk_level": "UNKNOWN",
        "risk_score": 0.0,
        "reasons": [],
    }
    if not ctx:
        out["reasons"].append("realtime_ctx_missing")
        return out

    # Example: if fixture status is not "Not Started" we flag (because prediction might be late)
    try:
        status = (ctx.get("fixture", {}).get("status", {}) or {}).get("short")  # e.g. NS, 1H, HT...
        if status and status != "NS":
            out["risk_level"] = "HIGH"
            out["risk_score"] = 0.9
            out["reasons"].append(f"fixture_status:{status}")
            return out
    except Exception:
        pass

    # If we have any injuries list (provider-specific), flag moderate.
    try:
        injuries = ctx.get("injuries") or ctx.get("players") or None
        if injuries:
            out["risk_level"] = "MEDIUM"
            out["risk_score"] = max(out["risk_score"], 0.4)
            out["reasons"].append("possible_injuries_or_lineup_changes")
    except Exception:
        pass

    return out

def _build_realtime_block(
    features_df: Any,
    league_code: Optional[str] = None,
    home_name: Any = None,
    away_name: Any = None,
    match_date: Any = None,
    season_df: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Single, unambiguous builder used by BOTH predict_match_with_proba() and apply_unexpected_layer().
    Does NOT change prediction. Only enriches realtime_risk + notes.
    """

    # read from df if not provided
    if home_name is None:
        home_name = _safe_get_first(features_df, "home")
    if away_name is None:
        away_name = _safe_get_first(features_df, "away")
    if match_date is None:
        match_date = _safe_get_first(features_df, "match_date")

    use_realtime_val = _safe_get_first(features_df, "_use_realtime")
    use_realtime = bool(use_realtime_val) if use_realtime_val is not None else False

    missing_fields = []
    if not home_name:
        missing_fields.append("home_name_missing")
    if not away_name:
        missing_fields.append("away_name_missing")
    if not match_date:
        missing_fields.append("match_date_missing")

    if not use_realtime:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["realtime_not_enabled_or_unavailable"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
        }
        return block, "realtime: not enabled"

    if missing_fields:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": missing_fields,
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
        }
        return block, f"realtime: skipped_missing_fields={missing_fields}"

    # 1) Resolve fixture_id (offline preferred if season_df is provided)
    try:
        fixture_id = _safe_resolve_fixture_id(
            home_name, away_name, match_date,
            league_code=league_code,
            season_df=season_df
        )
    except Exception as e:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": [f"fixture_resolve_error:{type(e).__name__}"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
        }
        return block, f"realtime: resolve error={type(e).__name__}"

    if not fixture_id:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["fixture_id_not_found"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
        }
        return block, "realtime: fixture not found"

    # 2) Fetch ctx
    try:
        ctx = _fetch_realtime_context(int(fixture_id))
        
        debug_rt = os.getenv("DEBUG_REALTIME", "0") == "1"

        ctx_debug = {}
        if debug_rt and ctx is not None:
            fixture = ctx.get("fixture") or {}
            status = fixture.get("status") or {}
            ctx_debug = {
                "ctx_keys": sorted(list(ctx.keys())),
                "fixture_keys": sorted(list(fixture.keys())) if isinstance(fixture, dict) else [],
                "status_obj": status,
                "status_short": status.get("short") if isinstance(status, dict) else None,
                "goals": ctx.get("goals"),
                "score": ctx.get("score"),
                "has_lineups": bool(ctx.get("lineups")),
                "has_events": bool(ctx.get("events")),
                "has_players": bool(ctx.get("players")),
            }

        # If API responded but empty response => ctx truly not available yet (normal sometimes)
        if ctx is None:
            block = {
                
                "available": False,
                "fixture_id": int(fixture_id),
                "missing": ["realtime_ctx_empty"],
                "reasons": ["realtime_ctx_empty"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
            }
            if debug_rt:
                    block["debug"] = ctx_debug
            return block, f"realtime: ok fixture_id={int(fixture_id)} but ctx empty"

        risk = _realtime_risk_score(ctx) or {}
        block = {
            "available": True,
            "fixture_id": int(fixture_id),
            "missing": [],
            "reasons": risk.get("reasons", []),
            "risk_level": risk.get("risk_level", "UNKNOWN"),
            "risk_score": float(risk.get("risk_score", 0.0) or 0.0),
        }
        return block, f"realtime: ok fixture_id={int(fixture_id)}"

    except RealtimeFetchError as e:
        # ✅ Here you finally see the real cause (401/403/429/url_missing/key_missing/etc.)
        code = e.code
        detail = e.detail
        status = e.status

        miss = [code] if status is None else [f"{code}:{status}"]

        block = {
            "available": False,
            "fixture_id": int(fixture_id),
            "missing": miss,
            "reasons": miss,
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
        }
        # Keep note concise but informative
        if detail:
            return block, f"realtime: ok fixture_id={int(fixture_id)} but fetch failed ({code})"
        return block, f"realtime: ok fixture_id={int(fixture_id)} but fetch failed"

    except Exception as e:
        block = {
            "available": False,
            "fixture_id": int(fixture_id),
            "missing": [f"realtime_error:{type(e).__name__}"],
            "reasons": [f"realtime_error:{type(e).__name__}"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
        }
        return block, f"realtime: ok fixture_id={int(fixture_id)} but error={type(e).__name__}"

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



# -------------------------------------------------------------------
# 🔢 Conventions BetSmart (IMPORTANT: éviter toute confusion 0/1/2)
# 0 = Victoire domicile (Home)
# 1 = Match nul (Draw)
# 2 = Victoire extérieur (Away)
# -------------------------------------------------------------------
LABEL_HOME = 0
LABEL_DRAW = 1
LABEL_AWAY = 2


# =========================
# Utils probas / mapping
# =========================
def _proba_for_class(model, X, cls_label, default=0.0):
    """Récupère une probabilité de classe en utilisant model.classes_ (robuste à l'ordre)."""
    try:
        classes = list(getattr(model, "classes_", []))
        if cls_label not in classes:
            return float(default)
        idx = classes.index(cls_label)
        p = model.predict_proba(X)[0][idx]
        return float(p)
    except Exception:
        return float(default)


def _normalize3(p0, p1, p2):
    p0 = float(p0)
    p1 = float(p1)
    p2 = float(p2)
    s = p0 + p1 + p2
    if not np.isfinite(s) or s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (p0 / s, p1 / s, p2 / s)


def _final_prediction_from_probas(p0, p1, p2):
    arr = np.array([p0, p1, p2], dtype=float)
    if not np.isfinite(arr).all():
        return LABEL_DRAW
    return int([LABEL_HOME, LABEL_DRAW, LABEL_AWAY][int(np.argmax(arr))])


def _format_pct(p):
    try:
        return f"{round(float(p) * 100, 0)}%"
    except Exception:
        return "0%"


# =========================
# Config ligue
# =========================
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

    bookmaker_margin = float(p.get("bookmaker_margin", 0.0711))
    uncertainty_threshold = float(p.get("uncertainty_threshold", 0.12))
    importance = int(p.get("importance", 3))
    season_stage = str(p.get("season_stage", "mid"))

    upset_threshold = float(p.get("upset_threshold", 0.55))
    skip_threshold = float(p.get("skip_threshold", 1.50))
    bogey_weight = float(p.get("bogey_weight", 0.40))
    gki_weight = float(p.get("gki_weight", 0.60))

    return (
        bookmaker_margin,
        uncertainty_threshold,
        importance,
        season_stage,
        upset_threshold,
        skip_threshold,
        bogey_weight,
        gki_weight,
    )


# ---------- AJOUT: hyperparamètres de la porte de forme ----------
def parametres_form_gate(league_code):
    """
    Lit (si dispo) les hyperparamètres de la 'porte forme' depuis champ_config.json :
      - k_market_form  : intensité max de transfert H↔A (0..1)  (défaut 0.45)
      - gate_slope     : pente de la sigmoïde (défaut 14.0)
      - gate_tolerance : tolérance d’écart de forme avant d’agir (défaut 0.036)
    """
    p = _get_params(league_code)
    k = float(p.get("k_market_form", 0.45))
    slope = float(p.get("gate_slope", 14.0))
    tau = float(p.get("gate_tolerance", 0.036))
    return k, slope, tau


# =========================
# Safe adapters
# =========================
def _as_float(x, default=0.0):
    try:
        if x is None:
            return default
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
    S'adapte à l'ancienne signature (4 valeurs) et la nouvelle (8).
    Force les types et fournit des valeurs par défaut sûres.
    """
    vals = parametres(league_code)

    if isinstance(vals, (list, tuple)) and len(vals) == 4:
        bookmaker_margin, uncertainty_threshold, importance, season_stage = vals
        upset_threshold, skip_threshold, bogey_weight, gki_weight = 0.55, 1.50, 0.40, 0.60
    elif isinstance(vals, (list, tuple)) and len(vals) >= 8:
        (
            bookmaker_margin,
            uncertainty_threshold,
            importance,
            season_stage,
            upset_threshold,
            skip_threshold,
            bogey_weight,
            gki_weight,
        ) = vals[:8]
    else:
        bookmaker_margin, uncertainty_threshold, importance, season_stage = 0.0711, 0.12, 3, "mid"
        upset_threshold, skip_threshold, bogey_weight, gki_weight = 0.55, 1.50, 0.40, 0.60

    bookmaker_margin = _as_float(bookmaker_margin, 0.0711)
    uncertainty_threshold = _as_float(uncertainty_threshold, 0.12)
    importance = _as_int(importance, 3)
    season_stage = str(season_stage) if season_stage is not None else "mid"
    upset_threshold = _as_float(upset_threshold, 0.55)
    skip_threshold = _as_float(skip_threshold, 1.50)
    bogey_weight = _as_float(bogey_weight, 0.40)
    gki_weight = _as_float(gki_weight, 0.60)

    return (
        bookmaker_margin,
        uncertainty_threshold,
        importance,
        season_stage,
        upset_threshold,
        skip_threshold,
        bogey_weight,
        gki_weight,
    )


def _fav_by_demarged(bh: float, bd: float, ba: float, eps: float = 0.02):
    """
    Détermine le favori via probabilités implicites dé-margées.
    Retourne (side, pH2, pA2, gap) où side ∈ {"home","away", None}.
    """
    bh = float(bh)
    bd = float(bd)
    ba = float(ba)
    if min(bh, bd, ba) <= 1.0 or any(not np.isfinite(x) for x in (bh, bd, ba)):
        return None, np.nan, np.nan, 0.0

    qH, qD, qA = 1.0 / bh, 1.0 / bd, 1.0 / ba
    s = qH + qD + qA
    if s <= 0:
        return None, np.nan, np.nan, 0.0

    pH, pD, pA = qH / s, qD / s, qA / s
    denom = pH + pA
    if denom <= 0:
        return None, np.nan, np.nan, 0.0
    pH2, pA2 = pH / denom, pA / denom
    gap = pH2 - pA2

    if gap > eps:
        side = "home"
    elif gap < -eps:
        side = "away"
    else:
        side = None
    return side, pH2, pA2, gap


# =========================
# Form stats
# =========================
def enrich_form_stats_dynamic(df, team, match_date, window=5):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    match_date = pd.to_datetime(match_date)

    recent_matches = (
        df[((df["HomeTeam"] == team) | (df["AwayTeam"] == team)) & (df["Date"] < match_date)]
        .sort_values("Date", ascending=False)
        .head(window)
    )
    if recent_matches.empty:
        return {"Form": 0.0, "GD": 0.0, "WinRate": 0.0, "DrawRate": 0.0, "GoalsAvg": 0.0}

    points = 0
    goals_diff = 0
    draws = 0
    wins = 0
    total_goals = 0

    for _, row in recent_matches.iterrows():
        is_home = row["HomeTeam"] == team

        if is_home:
            goals_for, goals_against = row["FTHG"], row["FTAG"]
            win = row["FTR"] == "H"
        else:
            goals_for, goals_against = row["FTAG"], row["FTHG"]
            win = row["FTR"] == "A"

        draw = row["FTR"] == "D"

        if draw:
            draws += 1
            points += 1
        elif win:
            wins += 1
            points += 3

        goals_diff += goals_for - goals_against
        total_goals += goals_for

    matches_played = len(recent_matches)
    return {
        "Form": points / (3 * matches_played),
        "GD": goals_diff / matches_played,
        "WinRate": wins / matches_played,
        "DrawRate": draws / matches_played,
        "GoalsAvg": total_goals / matches_played,
    }


# =========================
# Importance / ranks
# =========================
def _league_profile(league_code: str | int | None):
    try:
        code = int(league_code) if league_code is not None else None
    except Exception:
        code = None

    EURO = {
        39,
        61,
        78,
        140,
        135,
        88,
        207,
        94,
        203,
        144,
        197,
        119,
        179,
        180,
        253,
        2,
        3,
        233,
        62,
        40,
        79,
        136,
        141,
    }
    CAL_Y = {71, 98, 262, 292, 128}

    if code in EURO:
        return {"region": "europe", "late_months": {4, 5, 6}, "late_threshold": 0.70}
    elif code in CAL_Y:
        return {"region": "calendar_year", "late_months": {10, 11, 12}, "late_threshold": 0.70}
    else:
        return {"region": "unknown", "late_months": set(), "late_threshold": 0.70}


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
    close_ranks = rank_diff <= 4
    top_clash = (rank_home <= top_k and rank_away <= top_k)

    releg_zone = max(3, int(round(0.12 * n_teams)))
    six_pointer_releg = late_season and ((rank_home > n_teams - releg_zone) or (rank_away > n_teams - releg_zone))
    euro_spot_fight = late_season and ((rank_home <= 7) or (rank_away <= 7)) and (rank_diff <= 6)

    importance = 1 if (top_clash or (late_season and (close_ranks or six_pointer_releg or euro_spot_fight))) else 0
    return rank_home, rank_away, importance


# =========================
# Features
# =========================
def prepare_input_features_enriched(home_team, away_team, match_date, b365h, b365a, b365d, season_df, league_code):
    df = season_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    all_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()

    if (home_team not in all_teams) or (away_team not in all_teams):
        print(f"⚠️ Attention : {home_team} ou {away_team} n'a pas d'historique. Les stats seront neutres.")

    match_date = pd.to_datetime(match_date)
    df_past = df[df["Date"] < match_date]

    def safe_stats(d):
        d = dict(d or {})
        for key in ["Form", "GD", "WinRate", "DrawRate", "GoalsAvg"]:
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

    features = pd.DataFrame(
        [
            {
                "HTHG": 0,
                "HTAG": 0,
                "HTR": 0,
                "B365H": b365h,
                "B365A": b365a,
                "B365D": b365d,
                "OddsRatio_HA": odds_ratio_ha,
                "OddsDiff_HD": odds_diff_hd,
                "OddsDiff_AD": odds_diff_ad,
                "OddsGap_MinDelta": odds_gap_min_delta,
                "Year": match_date.year,
                "Month": match_date.month,
                "Weekday": match_date.weekday(),
                "HomeForm": home_stats["Form"],
                "AwayForm": away_stats["Form"],
                "HomeGD": home_stats["GD"],
                "AwayGD": away_stats["GD"],
                "DrawRate_Home": home_stats["DrawRate"],
                "DrawRate_Away": away_stats["DrawRate"],
                "WinRate_Home": home_stats["WinRate"],
                "WinRate_Away": away_stats["WinRate"],
                "GoalsAvg_Home": home_stats["GoalsAvg"],
                "GoalsAvg_Away": away_stats["GoalsAvg"],
                "Form_Diff": form_diff,
                "Rank_Home": rank_home,
                "Rank_Away": rank_away,
                "MatchImportance": match_importance,
            }
        ]
    )

    return features


# =========================
# Règles auxiliaires
# =========================
def detect_double_chance(proba_0, proba_1, proba_2, final_prediction, league_code):
    (bookmaker_margin, uncertainty_threshold, importance, season_stage, upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(
        league_code
    )

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
    odds = features_df[["B365H", "B365A", "B365D"]].values[0].astype(float)
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


# =========================
# Porte "forme récente"
# =========================
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))




LABEL_HOME = 0
LABEL_DRAW = 1
LABEL_AWAY = 2


def _apply_form_gate(
    p0, p1, p2,
    features_df: pd.DataFrame,
    league_code="default",
    *,
    # --- règle métier: forme prioritaire si contradiction ---
    form_pick_threshold: float = 0.20,         # seuil contradiction forme
    # --- intensité gate (transfert de masse sur H/A) ---
    k_market_form: float = 0.35,               # intensité max (raisonnable)
    gate_slope: float = 14.0,
    gate_tolerance: float = 0.036,
    # --- sécurité ---
    preserve_draw_mass: bool = True            # on ne touche jamais p1
):
    """
    Gate = ajuste UNIQUEMENT p0/p2 (Home/Away) en fonction de la FORME,
    en restant compatible avec tes outputs (notes: form_vs_market etc.)

    ✅ Règle respectée:
    - La FORME peut déplacer la décision (Home/Away) si contradiction significative.
    - Le draw (p1) est conservé (on ne le gonfle pas), car ton verrou stage2 gère déjà le nul.
    """

    # safe extraction
    try:
        home_form = float(features_df["HomeForm"].values[0])
        away_form = float(features_df["AwayForm"].values[0])
    except Exception:
        return p0, p1, p2, {"form_gate": "skipped_missing_form"}

    # contradiction forme ?
    form_diff = home_form - away_form  # + => home mieux
    if abs(form_diff) < float(form_pick_threshold):
        return p0, p1, p2, {
            "form_gate": "skipped_no_strong_form_signal",
            "home_form": round(home_form, 3),
            "away_form": round(away_form, 3),
            "form_diff": round(form_diff, 3),
            "th": float(form_pick_threshold),
        }

    # gate strength via sigmoid
    gate_strength = float(k_market_form) * _sigmoid(float(gate_slope) * (abs(form_diff) - float(gate_tolerance)))
    gate_strength = float(np.clip(gate_strength, 0.0, 1.0))

    h, d, a = float(p0), float(p1), float(p2)

    # masse HA uniquement
    mass_HA = max(1e-9, (h + a))
    transfer = gate_strength * mass_HA

    # direction: vers l'équipe en forme
    if form_diff > 0:   # home en forme
        # transfère de Away -> Home
        take = min(transfer, a)
        a_new = a - take
        h_new = h + take
    else:               # away en forme
        take = min(transfer, h)
        h_new = h - take
        a_new = a + take

    # renormalise HA pour garder d identique
    if preserve_draw_mass:
        scale = (h + a) / max(1e-9, (h_new + a_new))
        h_new *= scale
        a_new *= scale
        d_new = d
    else:
        # (non utilisé ici)
        h_new, d_new, a_new = _normalize3(h_new, d, a_new)

    return h_new, d_new, a_new, {
        "form_gate": "applied",
        "home_form": round(home_form, 3),
        "away_form": round(away_form, 3),
        "form_diff": round(form_diff, 3),
        "gate_strength": round(gate_strength, 4),
        "transfer": round(float(transfer), 4),
    }


def predict_match_with_proba(
    features_df: pd.DataFrame,
    model_stage1,
    model_stage2,
    threshold_draw=0.63,
    user_profile="standard",
    league_code="default"
) -> dict:
    """
    ✅ LOGIQUE BETSMART (verrouillée) — version stable

    + Ajout REALTIME (fixture/injuries/lineups) SANS IMPACTER la logique de prédiction.
    """

    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(league_code)

    # ---- params ligue / config ----
    try:
        params = _get_params(league_code)
    except Exception:
        params = {}

    form_pick_threshold = float(params.get("form_pick_threshold", 0.20))
    strong_conf_threshold = float(params.get("strong_conf_threshold", 0.70))
    strong_conf_draw_cap = float(params.get("strong_conf_draw_cap", 0.12))
    dc_disable_if_strong_conf = bool(params.get("dc_disable_if_strong_conf", True))

    # ------------------------------------------------------------------
    # ✅ REALTIME helpers (ne modifie pas les proba / décision finale)
    # ------------------------------------------------------------------
    def _pick_first_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                try:
                    v = df[c].values[0]
                    if v is not None and str(v).strip() != "":
                        return v
                except Exception:
                    continue
        return None

    def _as_bool(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in ("1", "true", "yes", "y", "on")

    # ---- util explication ----
    def _explain(rule_tag, p0, p1, p2, extra=None):
        f = features_df.copy()
        f["proba_0"] = float(p0)
        f["proba_1"] = float(p1)
        f["proba_2"] = float(p2)
        if isinstance(extra, dict):
            for k, v in extra.items():
                try:
                    f[k] = v
                except Exception:
                    pass
        return generate_explanation(rule_tag, f, user_profile)

    # ---- clamp draw non-dominant (stage2) ----
    def _clamp_draw_not_dominant(p0, p1, p2, eps=1e-6):
        p0, p1, p2 = float(p0), float(p1), float(p2)
        p0, p1, p2 = _normalize3(p0, p1, p2)
        max_ha = max(p0, p2)
        if p1 >= max_ha:
            target = max(0.0, max_ha - float(eps))
            if p1 > 0:
                scale = target / p1
                p1 = p1 * scale
                rest = 1.0 - p1
                ha_sum = max(1e-9, (p0 + p2))
                p0 = rest * (p0 / ha_sum)
                p2 = rest * (p2 / ha_sum)
        return _normalize3(p0, p1, p2)

    # ---- helper marché dé-margé + DC protection marché ----
    def _market_fav_and_dc():
        try:
            b365h = float(features_df["B365H"].values[0])
            b365d = float(features_df["B365D"].values[0])
            b365a = float(features_df["B365A"].values[0])
            eps_m = max(0.02, 0.5 * float(bookmaker_margin))
            fav_side, pH2, pA2, fav_gap = _fav_by_demarged(b365h, b365d, b365a, eps=eps_m)
            dc_market = None
            if fav_side == "home":
                dc_market = "1X"
            elif fav_side == "away":
                dc_market = "X2"
            return fav_side, float(fav_gap), dc_market
        except Exception:
            return None, 0.0, None

    # ------------------------------------------------------------------
    # STAGE 1 : pDraw
    # ------------------------------------------------------------------
    X1 = features_df.copy()
    for col in model_stage1.feature_names_in_:
        if col not in X1.columns:
            X1[col] = 0
    X1 = X1[model_stage1.feature_names_in_]

    p_draw = _proba_for_class(model_stage1, X1, LABEL_DRAW, default=0.0)
    p_draw = float(np.clip(p_draw, 0.0, 1.0))

    # ------------------------------------------------------------------
    # CAS DRAW DIRECT
    # ------------------------------------------------------------------
    if p_draw >= float(threshold_draw):
        p1 = p_draw
        p0 = p2 = (1.0 - p1) / 2.0

        p0, p1, p2, meta_gate = _apply_form_gate(
            p0, p1, p2, features_df, league_code,
            form_pick_threshold=form_pick_threshold
        )
        p0, p1, p2 = _normalize3(p0, p1, p2)

        pred_final = LABEL_DRAW
        dc = detect_double_chance(p0, p1, p2, pred_final, league_code)

        # ✅ realtime (info only)
        rt_block, rt_note = _build_realtime_block(features_df)
        notes = []
        if rt_note:
            notes.append(rt_note)

        return {
            "prediction": int(pred_final),
            "prediction_model": LABEL_DRAW,
            "proba_0": _format_pct(p0),
            "proba_1": _format_pct(p1),
            "proba_2": _format_pct(p2),
            "rule_applied": "threshold|draw_dominant|form_gate",
            "explanation": _explain("threshold", p0, p1, p2, extra={"form_gate_meta": str(meta_gate)}),
            "double_chance": dc,
            "realtime_risk": rt_block,
            "notes": notes
        }

    # ------------------------------------------------------------------
    # STAGE 2 : Home vs Away (ND)
    # ------------------------------------------------------------------
    X2 = features_df.copy()
    for col in model_stage2.feature_names_in_:
        if col not in X2.columns:
            X2[col] = 0
    X2 = X2[model_stage2.feature_names_in_]

    pH_nd = _proba_for_class(model_stage2, X2, LABEL_HOME, default=0.5)
    pA_nd = _proba_for_class(model_stage2, X2, LABEL_AWAY, default=0.5)
    s = float(pH_nd) + float(pA_nd)
    if not np.isfinite(s) or s <= 0:
        pH_nd, pA_nd = 0.5, 0.5
    else:
        pH_nd, pA_nd = float(pH_nd) / s, float(pA_nd) / s

    prediction_rf = int(model_stage2.predict(X2)[0])

    p1 = p_draw
    pND = max(0.0, 1.0 - p1)
    p0 = pND * pH_nd
    p2 = pND * pA_nd
    p0, p1, p2 = _normalize3(p0, p1, p2)

    p0, p1, p2 = _clamp_draw_not_dominant(p0, p1, p2)

    p0, p1, p2, meta_gate = _apply_form_gate(
        p0, p1, p2, features_df, league_code,
        form_pick_threshold=form_pick_threshold
    )
    p0, p1, p2 = _normalize3(p0, p1, p2)

    p0, p1, p2 = _clamp_draw_not_dominant(p0, p1, p2)

    strong_side = max(float(p0), float(p2))
    strong_conf = (strong_side >= float(strong_conf_threshold))

    if strong_conf:
        cap = float(np.clip(strong_conf_draw_cap, 0.0, 0.30))
        if float(p1) > cap:
            rest = 1.0 - cap
            ha_sum = max(1e-9, (float(p0) + float(p2)))
            p0 = rest * (float(p0) / ha_sum)
            p2 = rest * (float(p2) / ha_sum)
            p1 = cap
            p0, p1, p2 = _normalize3(p0, p1, p2)
        strong_tag = "strong_conf_draw_cut"
    else:
        strong_tag = None

    pred_final = LABEL_HOME if float(p0) >= float(p2) else LABEL_AWAY

    fav_side, fav_gap, dc_market = _market_fav_and_dc()

    try:
        home_form = float(features_df["HomeForm"].values[0])
        away_form = float(features_df["AwayForm"].values[0])
        form_diff = home_form - away_form
    except Exception:
        home_form = away_form = form_diff = 0.0

    override_tag = None
    dc_override = None

    if abs(float(form_diff)) >= float(form_pick_threshold):
        form_side = "home" if form_diff > 0 else "away"
        if fav_side is not None and form_side != fav_side:
            pred_final = LABEL_HOME if form_side == "home" else LABEL_AWAY
            override_tag = "form_over_market_pick_" + ("home" if form_side == "home" else "away")
            dc_override = dc_market

    bias_detected = bool(detect_bias(features_df))
    low_confidence = bool(is_confidence_low(p0, p1, p2))

    dc = detect_double_chance(p0, p1, p2, pred_final, league_code)

    if dc_override is not None:
        dc = dc_override

    if strong_conf and dc_disable_if_strong_conf and (not bias_detected) and (not low_confidence):
        dc = None

    if (bias_detected or low_confidence) and dc is None:
        dc = "1X" if pred_final == LABEL_HOME else "X2"

    rule_parts = ["rf_decision", "stage2_locked_no_draw", "form_gate"]
    if strong_tag:
        rule_parts.append(strong_tag)
    if override_tag:
        rule_parts.append(override_tag)

    rule_applied = "|".join(rule_parts)

    extra = {
        "bias_detected": int(bias_detected),
        "low_confidence": int(low_confidence),
        "form_gate_meta": str(meta_gate),
        "strong_conf": int(bool(strong_conf)),
        "fav_side": str(fav_side),
        "fav_gap": float(fav_gap),
        "form_diff": float(form_diff),
    }

    # ✅ realtime (info only)
    rt_block, rt_note = _build_realtime_block(features_df)
    notes = []
    if rt_note:
        notes.append(rt_note)

    return {
        "prediction": int(pred_final),
        "prediction_model": prediction_rf,
        "proba_0": _format_pct(p0),
        "proba_1": _format_pct(p1),
        "proba_2": _format_pct(p2),
        "rule_applied": rule_applied,
        "explanation": _explain("rf_decision", p0, p1, p2, extra=extra),
        "double_chance": dc,
        "bias_detected": bias_detected,
        "low_confidence": low_confidence,
        "realtime_risk": rt_block,
        "notes": notes
    }

# =========================
# Unexpected / anti-OC layer
# =========================
def apply_unexpected_layer(
    base_pred: dict,
    season_current_df=None,
    season_past_list=None,
    home: str = None,
    away: str = None,
    match_date=None,
    feats_df=None,
    league_code: str = None,
    X_ref_features=None,
    upset_threshold: float = 0.52,
):
    """
    Safe post-layer that can enrich the base prediction with:
      - real-time risk block (without overwriting a valid fixture_id from predict_match_with_proba)
      - conservative 'unexpected' score placeholders

    Constraint respected: does NOT change your existing 1N2 prediction logic;
    it only enriches output fields and notes.
    """
    out = dict(base_pred or {})
    notes = list(out.get("notes", []))

    # ---- feats_df must exist for safe getters ----
    feats_df = feats_df if feats_df is not None else {}

    # ---- propagate home/away (prefer base_pred if already set) ----
    if out.get("home") is None and home is not None:
        out["home"] = str(home)
    if out.get("away") is None and away is not None:
        out["away"] = str(away)

    # Resolve canonical names for realtime calls:
    home_name = str(home) if home is not None else (str(out.get("home")) if out.get("home") is not None else None)
    away_name = str(away) if away is not None else (str(out.get("away")) if out.get("away") is not None else None)

    # match_date: prefer explicit arg, else feats_df["match_date"] if present
    if match_date is None:
        try:
            md = _safe_get_first(feats_df, "match_date")
            match_date = md if md not in ("", None) else None
        except Exception:
            match_date = None

    # _use_realtime: prefer feats_df flag if present else keep existing else False
    try:
        if _safe_get_first(feats_df, "_use_realtime") is not None:
            out["_use_realtime"] = bool(_safe_get_first(feats_df, "_use_realtime"))
        else:
            out["_use_realtime"] = bool(out.get("_use_realtime", False))
    except Exception:
        out["_use_realtime"] = bool(out.get("_use_realtime", False))

    # ------------------------------------------------------------------
    # REALTIME BLOCK (FIXED):
    # Rule: if base already contains a fixture_id, NEVER overwrite it.
    # Even if ctx missing / available False / missing not empty -> keep fixture_id.
    # Only build realtime if fixture_id is absent.
    # ------------------------------------------------------------------
    rt_existing = out.get("realtime_risk")

    existing_fixture_id = None
    if isinstance(rt_existing, dict):
        existing_fixture_id = rt_existing.get("fixture_id", None)

    # If realtime isn't enabled, keep block as-is (or set minimal) and do not try to resolve.
    if not out["_use_realtime"]:
        # keep existing if any, else set minimal
        if not isinstance(rt_existing, dict):
            out["realtime_risk"] = {
                "available": False,
                "fixture_id": None,
                "missing": ["realtime_not_enabled_or_unavailable"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
            }
        notes.append("realtime: not enabled")
    else:
        # realtime enabled
        if existing_fixture_id is not None:
            # ✅ KEEP, do not overwrite (prevents your contradictory logs)
            notes.append(f"realtime: kept_fixture_id={existing_fixture_id}")
            # keep existing block untouched
            out["realtime_risk"] = rt_existing
        else:
            # Only now we attempt to build realtime
            try:
                rt_block, rt_note = _build_realtime_block(
                    feats_df,
                    league_code=league_code,
                    home_name=home_name,
                    away_name=away_name,
                    match_date=match_date,
                    season_df=season_current_df,
                )
                out["realtime_risk"] = rt_block
                notes.append(rt_note)
            except Exception as e:
                out["realtime_risk"] = {
                    "available": False,
                    "fixture_id": None,
                    "missing": [f"realtime_error:{type(e).__name__}"],
                    "reasons": [],
                    "risk_level": "UNKNOWN",
                    "risk_score": 0.0,
                }
                notes.append(f"realtime: error={type(e).__name__}")

    # ------------------------------------------------------------------
    # Unexpected score (unchanged behaviour)
    # ------------------------------------------------------------------
    out.setdefault("_upset_threshold", float(upset_threshold))

    upset_score = out.get("_upset_score")
    if upset_score is None:
        try:
            odd_h = float(_safe_get_first(feats_df, "B365H") or 0.0)
            odd_a = float(_safe_get_first(feats_df, "B365A") or 0.0)
            odd_d = float(_safe_get_first(feats_df, "B365D") or 0.0)

            inv = []
            for o in (odd_h, odd_d, odd_a):
                inv.append(1.0 / o if o and o > 0 else 0.0)
            s = sum(inv) if sum(inv) > 0 else 1.0
            p_h, p_d, p_a = [v / s for v in inv]

            pred = out.get("prediction")
            best = max(p_h, p_a)
            if pred == 0:
                outsider = p_h
            elif pred == 2:
                outsider = p_a
            else:
                outsider = p_d

            upset_score = float(max(0.0, min(1.0, best - outsider)))
        except Exception:
            upset_score = 0.0

    out["_upset_score"] = float(upset_score)

    out["notes"] = notes
    return out

##---------------------- FIN FONCTIONS  ------------------------------------------

def generate_explanation(rule_applied, features, user_profile):
    if isinstance(features, pd.DataFrame):
        row = features.to_dict(orient="records")[0] if not features.empty else {}
    elif isinstance(features, dict):
        row = dict(features)
    else:
        row = {}

    odds_gap = float(row.get("OddsGap_MinDelta", 0.0) or 0.0)
    form_diff = float(row.get("Form_Diff", 0.0) or 0.0)
    match_importance = int(row.get("MatchImportance", 0) or 0)

    p0 = float(row.get("proba_0", 0.0) or 0.0)
    p1 = float(row.get("proba_1", 0.0) or 0.0)
    p2 = float(row.get("proba_2", 0.0) or 0.0)

    bias_detected = bool(int(row.get("bias_detected", 0) or 0))
    low_confidence = bool(int(row.get("low_confidence", 0) or 0))

    if user_profile == "débutant":
        if rule_applied in ("threshold", "margin_adjusted"):
            msg = "Match nul probable : le match paraît équilibré selon les signaux."
        elif rule_applied == "filtered_out":
            msg = "⚠️ Prudence : le match présente une incertitude élevée. Mieux vaut jouer en double chance."
        else:
            msg = "Victoire probable : l'analyse détecte un léger avantage."
    elif user_profile == "expert":
        msg = f"p=[H:{p0:.2f}, D:{p1:.2f}, A:{p2:.2f}] | gap_odds≈{odds_gap:.2f} | form_diff={form_diff:.2f}"
        if rule_applied == "filtered_out":
            msg = "⚠️ " + msg + " | incertitude/biais → privilégier DC"
    else:
        if rule_applied == "filtered_out":
            msg = "⚠️ Attention : incertitude/biais détecté. Appuyez-vous sur la double chance."
        elif rule_applied in ("threshold", "margin_adjusted"):
            msg = "Match nul probable : le match est équilibré."
        else:
            msg = "Victoire probable : un déséquilibre a été détecté entre les équipes."

    if match_importance == 1:
        msg += " Match à enjeu (importance élevée)."

    reasons = []
    if low_confidence:
        reasons.append("confiance faible")
    if bias_detected:
        reasons.append("biais de cotes")
    if reasons and rule_applied != "filtered_out":
        msg += " (" + ", ".join(reasons) + ")"

    return msg



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
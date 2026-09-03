# BetSmart V2.3.7.1.1 — GUARANTEED FINAL STATE
# BetSmart V2.3.7.1 — DECISION STABILITY LAYER
# BetSmart V2.3.7.1 — SINGLE DECISION PIPELINE / ONE pred_final, TWO VIEWS
# BetSmart V2.3.7.1 — VALUE GUARD + EVIDENCE DELTA CAP + INJURY DEDUP
# BetSmart V2.3.7.1 — WEB EVIDENCE RELIABILITY + SINGLE PUBLIC EXPLANATION
# BetSmart V2.3.7.1 — LIGHT STABILIZATION
# BetSmart V2.3.7.1 — DYNAMIC MULTI-SOURCE EXPLANATION
# BetSmart V2.3.7.1 — MULTI-SOURCE PROFESSIONAL DECISION FUSION
# V2.3 FIX — datetime/time corrected for Web Intelligence
# BetSmart V2.3 — REAL-TIME WEB INTELLIGENCE ACTIVE
# BetSmart V2.3 — REAL-TIME INTELLIGENCE + MANDATORY 1X2 DECISION
# BetSmart V2.2.3 — EARLY SEASON HISTORICAL PRIOR + REAL AI ARBITRATION
# BetSmart V2.2.2 — REAL AI ARBITRATION + FORM GUARD + SIGNAL COMPATIBILITY
# BetSmart V2.2.1 — FORM GUARD + AI SIGNAL COMPATIBILITY + JSON FULL
# BetSmart V2.2.1 - form guard + signal compatibility + JSON réduit
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025
BetSmart V2.2: historical + market + AI arbitration

@author: bobunda
"""


BETSMART_FUNCTION_VERSION = "2.2.1-form-guard-full"

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
from typing import Any, Dict, Optional, Tuple, List
import datetime as dt
from datetime import timedelta
from openai import OpenAI
import threading
import time
##------------------------------- PREDICTION DES EQUIPES WIN LOSS DRAW ------------------------------------------------
#### amélioration de temps de réponse

# Session HTTP réutilisée (keep-alive)
_HTTP = requests.Session()

# Cache mémoire simple avec TTL
_RT_CACHE = {}
_RT_CACHE_LOCK = threading.Lock()

# Mode realtime: "off" | "light" | "full"
REALTIME_MODE = os.getenv("REALTIME_MODE", "light").strip().lower()



REALTIME_TIMEOUT_FIXTURE = 10
REALTIME_TIMEOUT_OPTIONAL = 5
REALTIME_TIMEOUT_STANDINGS = 6
REALTIME_LINEUPS_SOON_MINUTES = 90
REALTIME_MODE = "light"   # ou env var

# lineups uniquement si proche du match (minutes)
REALTIME_LINEUPS_SOON_MINUTES = int(os.getenv("REALTIME_LINEUPS_SOON_MINUTES", "90"))

def _cache_get(key):
    with _RT_CACHE_LOCK:
        item = _RT_CACHE.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at is not None and time.time() > expires_at:
            _RT_CACHE.pop(key, None)
            return None
        return value

def _cache_set(key, value, ttl=60):
    with _RT_CACHE_LOCK:
        expires_at = None if ttl is None or ttl <= 0 else (time.time() + ttl)
        _RT_CACHE[key] = (expires_at, value)

def _cache_key(endpoint: str, params: dict) -> tuple:
    try:
        items = tuple(sorted((params or {}).items()))
    except Exception:
        items = tuple()
    return (endpoint, items)

###---------------------------------------------------------

# log des prédictions utilisateurs

LABEL_HOME = 0
LABEL_DRAW = 1
LABEL_AWAY = 2

REALTIME_API_URL="https://v3.football.api-sports.io"

DEBUG_REALTIME=1

USE_LLM_EXPLANATION = True          

OPENAI_EXPLAIN_ENABLED = True    # True/False
OPENAI_EXPLAIN_MODEL = "gpt-4.1"
OPENAI_EXPLAIN_TEMPERATURE = 0.5
OPENAI_EXPLAIN_MAX_TOKENS = 460
OPENAI_EXPLAIN_TIMEOUT = 20

# BetSmart V2.2 - IA d'arbitrage (ne modifie jamais prediction/prediction_model)
BETSMART_AI_ENABLED = os.getenv("BETSMART_AI_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
BETSMART_AI_MODEL = os.getenv("BETSMART_AI_MODEL", OPENAI_EXPLAIN_MODEL)
BETSMART_AI_TEMPERATURE = float(os.getenv("BETSMART_AI_TEMPERATURE", "0.0"))
BETSMART_AI_MAX_TOKENS = int(os.getenv("BETSMART_AI_MAX_TOKENS", "700"))
BETSMART_AI_TIMEOUT = int(os.getenv("BETSMART_AI_TIMEOUT", "20"))

# BetSmart V2.3 - Real-Time Web Intelligence
BETSMART_WEB_RESEARCH_ENABLED = os.getenv(
    "BETSMART_WEB_RESEARCH_ENABLED", "1"
).strip().lower() in ("1", "true", "yes", "on")
BETSMART_WEB_MODEL = os.getenv("BETSMART_WEB_MODEL", "gpt-4.1")
BETSMART_WEB_TIMEOUT = int(os.getenv("BETSMART_WEB_TIMEOUT", "35"))
BETSMART_WEB_CACHE_TTL = int(os.getenv("BETSMART_WEB_CACHE_TTL", "1800"))
BETSMART_WEB_SEARCH_CONTEXT = os.getenv("BETSMART_WEB_SEARCH_CONTEXT", "medium").strip().lower()

_WEB_RESEARCH_CACHE = {}
_WEB_RESEARCH_LOCK = threading.Lock()

LLM_DEBUG = False          

# Mapping officiel BetSmart (API-SPORTS)
LEAGUES = {
    "Premier League": 39,
    "Ligue 1": 61,
    "Bundesliga": 78,
    "La Liga": 140,
    "Serie A": 135,
    "Neerdeland": 88,
    "Suisse": 207,
    "Portugais": 94,
    "Turquie": 203,
    "Belgique": 144,
    "Japon": 98,
    "Grece": 197,
    "bresil": 71,
    "ecosse": 179,
    "ecosse_div_1": 180,
    "coree_sud": 292,
    "Argentine_league_1": 128,
    "League_europa": 3,
    "champions_league": 2,
    "egypte": 233,
    "mexique": 262,
    "france_league_2": 62,
    "bundesliga_2": 79,
    "serie_B": 136,
    "Championship": 40,
    "secunda": 141,
    "can": 6
}

REASON_TRANSLATION_FR = {
    # Suspensions
    "yellow cards": "Suspendu (cartons)",
    "red card": "Suspendu (carton rouge)",

    # Générique blessure
    "injury": "Blessé",

    # Détails blessures
    "thigh injury": "Blessure à la cuisse",
    "muscle injury": "Blessure musculaire",
    "foot injury": "Blessure au pied",
    "knee injury": "Blessure au genou",
    "ankle injury": "Blessure à la cheville",
    "hamstring injury": "Ischio-jambiers",
    "back injury": "Dos",
    "groin injury": "Adducteurs",
    "calf injury": "Blessure au mollet",

    # Autres
    "illness": "Maladie",
}

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

def _safe_prob(x: Any, default: float = 0.0) -> float:
    """
    Convertit x en probabilité float [0..1] si possible.
    Supporte: 0.13, "13%", "13.0%", "0.13"
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            v = float(x)
            # si quelqu'un passe 13 au lieu de 0.13 -> on interprète comme %
            if v > 1.0 and v <= 100.0:
                return max(0.0, min(1.0, v / 100.0))
            return max(0.0, min(1.0, v))
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return float(default)
            if s.endswith("%"):
                s2 = s[:-1].strip()
                v = float(s2)
                return max(0.0, min(1.0, v / 100.0))
            v = float(s)
            if v > 1.0 and v <= 100.0:
                return max(0.0, min(1.0, v / 100.0))
            return max(0.0, min(1.0, v))
        return float(default)
    except Exception:
        return float(default)


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


def _season_from_date(match_date: Any) -> Optional[int]:
    """
    API-FOOTBALL season is the start year of the season.
    For European leagues: Jan–Jun belongs to previous start year (e.g., Jan 2026 -> season 2025).
    """
    # _parse_match_date must be defined in your codebase
    d = _parse_match_date(match_date)  # noqa: F821
    if d is None:
        return None
    try:
        month = int(getattr(d, "month", 0))
    except Exception:
        month = 0
    return int(d.year - 1) if month <= 6 else int(d.year)

def _safe_get_first(obj, key, default=None):
    try:
        # dict
        if isinstance(obj, dict):
            return obj.get(key, default)

        # pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            if key in obj.columns and len(obj) > 0:
                v = obj[key].iloc[0]
                return v if v is not None else default
            return default

        # pandas Series
        if isinstance(obj, pd.Series):
            v = obj.get(key, default)
            # si v est une Series (cas rare), prends le 1er élément
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else default
            return v

        # fallback attribute access
        if hasattr(obj, "get"):
            v = obj.get(key, default)
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else default
            return v

    except Exception:
        return default

    return default

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

def _resolve_fixture_id_by_names________(
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
) -> Optional[int]:
    """
    Online fallback resolver.
    Strategy:
      - call /fixtures with date
      - if league provided, also pass season (critical)
      - try date delta (0, -1, +1)
      - if league filter returns 0, retry without league/season
      - match by normalized names
    """
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")  # noqa: F821
    api_key = os.getenv("REALTIME_API_KEY", REALTIME_API_KEY)  # noqa: F821
    if not api_url or not api_key or requests is None:
        return None

    d = _parse_match_date(match_date)  # noqa: F821
    if d is None:
        return None

    url = f"{api_url}/fixtures"
    headers = {"x-apisports-key": api_key}
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    home_n = _norm_team_name(home_name)  # noqa: F821
    away_n = _norm_team_name(away_name)  # noqa: F821

    # league param (int or label)
    league_param: Optional[int] = None
    if league_code is not None:
        try:
            league_param = int(league_code)
        except Exception:
            try:
                if isinstance(league_code, str) and league_code in LEAGUES:  # noqa: F821
                    league_param = int(LEAGUES[league_code])  # noqa: F821
            except Exception:
                league_param = None

    season_param: Optional[int] = _season_from_date(match_date) if league_param is not None else None  # noqa: F821

    for delta in (0, -1, 1):
        date_str = (d + timedelta(days=delta)).strftime("%Y-%m-%d")
        params = {"date": date_str}

        if league_param is not None:
            params["league"] = int(league_param)
            if season_param is not None:
                params["season"] = int(season_param)

        # 1) attempt
        try:
            r = requests.get(url, headers=headers, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        resp = (data or {}).get("response", []) or []

        # 2) fallback if league filter too strict
        if league_param is not None and len(resp) == 0:
            try:
                params2 = {"date": date_str}
                r2 = requests.get(url, headers=headers, params=params2, timeout=8)
                r2.raise_for_status()
                data = r2.json()
                resp = (data or {}).get("response", []) or []
            except Exception:
                resp = []

        # 3) match
        try:
            for item in resp:
                th = _norm_team_name(item.get("teams", {}).get("home", {}).get("name"))  # noqa: F821
                ta = _norm_team_name(item.get("teams", {}).get("away", {}).get("name"))  # noqa: F821
                if (th == home_n and ta == away_n) or (th == away_n and ta == home_n):
                    fid = item.get("fixture", {}).get("id")
                    if fid is not None:
                        return int(fid)
        except Exception:
            continue

    return None

def normalize_team_name(name: str) -> str:
    import unicodedata
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = " ".join(name.split())
    return name


def resolve_fixture_id_local(fixtures_df, home, away, match_date, league_code=None):
    """
    fixtures_df doit contenir au minimum:
    - fixture_id
    - home
    - away
    - match_date
    - league_code
    """
    h = normalize_team_name(home)
    a = normalize_team_name(away)
    d = str(match_date)[:10]

    df = fixtures_df.copy()

    df["_home_norm"] = df["HomeTeam"].astype(str).map(normalize_team_name)
    df["_away_norm"] = df["AwayTeam"].astype(str).map(normalize_team_name)
    df["_date_norm"] = df["match_Date"].astype(str).str[:10]

    mask = (
        (df["_home_norm"] == h) &
        (df["_away_norm"] == a) &
        (df["_date_norm"] == d)
    )

    if league_code is not None and "league_code" in df.columns:
        mask = mask & (df["league_code"].astype(str) == str(league_code))

    found = df.loc[mask]

    if found.empty:
        return None

    return int(found.iloc[0]["fixture_id"])

def _safe_resolve_fixture_id(
    home_name: Any,
    away_name: Any,
    match_date: Any,
    league_code: Optional[str] = None,
    season_df: Any = None,
    features_df: Any = None,
) -> Optional[int]:
    """
    Safe wrapper:
      0) if fixture_id already present in features_df -> use it directly (best)
      1) offline resolution (season_df)
      2) online resolver by names
    """

    def _scalar(v):
        """Normalize pandas/array-likes to a single scalar (or None)."""
        try:
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else None
        except Exception:
            return None
        return v

    # 0) Direct fixture_id (FAST + RELIABLE) — avoid `or` on Series
    try:
        fid = None

        if features_df is not None:

            # dict
            if isinstance(features_df, dict):
                for k in ("fixture_id", "_fixture_id"):
                    if k in features_df:
                        candidate = _scalar(features_df.get(k))
                        if candidate is not None and str(candidate).strip() != "":
                            fid = candidate
                            break

            # DataFrame
            elif isinstance(features_df, pd.DataFrame):
                if len(features_df) > 0:
                    for k in ("fixture_id", "_fixture_id"):
                        if k in features_df.columns:
                            candidate = _scalar(features_df[k].iloc[0])
                            if candidate is not None and str(candidate).strip() != "":
                                fid = candidate
                                break

            # Series (row)
            elif isinstance(features_df, pd.Series):
                for k in ("fixture_id", "_fixture_id"):
                    if k in features_df.index:
                        candidate = _scalar(features_df.get(k))
                        if candidate is not None and str(candidate).strip() != "":
                            fid = candidate
                            break

            # fallback: dict-like get, but NO boolean ops
            elif hasattr(features_df, "get"):
                for k in ("fixture_id", "_fixture_id"):
                    candidate = _scalar(features_df.get(k, None))
                    if candidate is not None and str(candidate).strip() != "":
                        fid = candidate
                        break

        if fid is not None and str(fid).strip() != "":
            return int(fid)
    except Exception:
        pass

    # 1) offline resolution (best)
    try:
        fid2 = _resolve_fixture_id_from_df(season_df, home_name, away_name, match_date, league_code=league_code)
        if fid2 is not None and str(fid2).strip() != "":
            return int(fid2)
    except Exception:
        pass

    # 2) online fallback
    try:
        return _resolve_fixture_id_by_names(home_name, away_name, match_date, league_code=league_code)
    except Exception:
        return None

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
    fallback_key = globals().get("REALTIME_API_KEY", "REALTIME_API_KEY")
    fallback_host = globals().get("REALTIME_API_HOST", "")

    api_url = os.getenv("REALTIME_API_URL", fallback_url).strip().rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY", fallback_key).strip()
    api_host = os.getenv("REALTIME_API_HOST", fallback_host).strip()
    return api_url, api_key, api_host

def _api_get___________(endpoint: str, params: dict, timeout: int = 10) -> dict:
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY")

    if not api_url or not api_key:
        raise RealtimeFetchError(code="key_missing", detail="Missing API URL/KEY", status=None)

    url = f"{api_url}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": api_key}
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        status = r.status_code

        if status in (401, 403):
            raise RealtimeFetchError(code="unauthorized", detail="API key unauthorized", status=status)
        if status == 429:
            raise RealtimeFetchError(code="rate_limited", detail="Rate limit", status=status)

        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else {"response": js}

    except RealtimeFetchError:
        raise
    except Exception as e:
        raise RealtimeFetchError(code="http_error", detail=str(e), status=None)


def _api_get(endpoint: str, params: dict, timeout: int = 10, cache_ttl: int = 0) -> dict:
    api_url = os.getenv("REALTIME_API_URL", REALTIME_API_URL).rstrip("/")
    api_key = os.getenv("REALTIME_API_KEY")

    if not api_url or not api_key:
        raise RealtimeFetchError(code="key_missing", detail="Missing API URL/KEY", status=None)

    # cache
    key = _cache_key(endpoint, params)
    if cache_ttl and cache_ttl > 0:
        hit = _cache_get(key)
        if hit is not None:
            return hit

    url = f"{api_url}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": api_key}
    host = os.getenv("REALTIME_API_HOST", "").strip()
    if host:
        headers["x-rapidapi-host"] = host

    try:
        r = _HTTP.get(url, headers=headers, params=params, timeout=timeout)
        status = r.status_code

        if status in (401, 403):
            raise RealtimeFetchError(code="unauthorized", detail="API key unauthorized", status=status)
        if status == 429:
            raise RealtimeFetchError(code="rate_limited", detail="Rate limit", status=status)

        r.raise_for_status()
        js = r.json()
        out = js if isinstance(js, dict) else {"response": js}

        if cache_ttl and cache_ttl > 0:
            _cache_set(key, out, ttl=cache_ttl)

        return out

    except RealtimeFetchError:
        raise
    except Exception as e:
        raise RealtimeFetchError(code="http_error", detail=str(e), status=None)



def _fetch_realtime_context(fixture_id: int) -> Optional[dict]:
    """
    Fetch full realtime context for a fixture_id.
    Returns ctx dict or None if fixture not found / empty response.
    """
    fixture_id = int(fixture_id)
    ctx = {"meta": {"missing": [], "fixture_id": fixture_id}}

    # 1) fixture core (mandatory)
    data_fx = _api_get("fixtures", {"id": fixture_id}, timeout=10)
    if not isinstance(data_fx, dict):
        return None

    resp_fx = data_fx.get("response", []) or []
    if len(resp_fx) == 0:
        return None

    fx = resp_fx[0]
    if not isinstance(fx, dict):
        ctx["meta"]["missing"].append(f"fixture_shape_invalid:{type(fx).__name__}")
        return None

    ctx["fixture"] = fx.get("fixture") if isinstance(fx.get("fixture"), dict) else {}
    ctx["league"]  = fx.get("league")  if isinstance(fx.get("league"), dict)  else {}
    ctx["teams"]   = fx.get("teams")   if isinstance(fx.get("teams"), dict)   else {}
    ctx["goals"]   = fx.get("goals")   if isinstance(fx.get("goals"), dict)   else {}
    ctx["score"]   = fx.get("score")   if isinstance(fx.get("score"), dict)   else {}

    def _optional(name: str, endpoint: str, params: dict):
        try:
            d = _api_get(endpoint, params, timeout=10)
            if isinstance(d, dict):
                ctx[name] = d.get("response", []) or []
            else:
                ctx[name] = []
                ctx["meta"]["missing"].append(f"{name}_shape_invalid:{type(d).__name__}")

            if len(ctx[name]) == 0:
                ctx["meta"]["missing"].append(f"{name}_empty")

        except RealtimeFetchError as e:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:{e.code}")
        except Exception:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:unknown")

    # optional endpoints
    _optional("events", "fixtures/events", {"fixture": fixture_id})
    _optional("lineups", "fixtures/lineups", {"fixture": fixture_id})
    _optional("statistics", "fixtures/statistics", {"fixture": fixture_id})
    _optional("players", "fixtures/players", {"fixture": fixture_id})
    _optional("injuries", "injuries", {"fixture": fixture_id})

    #ctx["meta"]["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    ctx["meta"]["fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
   # ctx["meta"]["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    return ctx

def _fetch_realtime_context_(fixture_id: int) -> Optional[dict]:
    """
    Fetch realtime context for a fixture_id.
    Version hybride:
    - fetch principal fixtures tolérant (comme ancienne version)
    - endpoints optionnels optimisés
    - mode light pré-match
    """

    fixture_id = int(fixture_id)

    ctx = {
        "meta": {
            "missing": [],
            "errors": [],
            "skipped": [],
            "fixture_id": fixture_id,
        },
        "fixture": {},
        "league": {},
        "teams": {},
        "goals": {},
        "score": {},
        "events": [],
        "lineups": [],
        "statistics": [],
        "players": [],
        "injuries": [],
    }

    # --------------------------------------------------
    # 1) fixtures core (reprend l'ancienne logique)
    # --------------------------------------------------
    try:
        # ✅ important : timeout fixe 10, pas de cache ici
        data_fx = _api_get("fixtures", {"id": fixture_id}, timeout=10, cache_ttl=0)

        if not isinstance(data_fx, dict):
            return None

        resp_fx = data_fx.get("response", []) or []
        if len(resp_fx) == 0:
            return None

        fx = resp_fx[0]
        if not isinstance(fx, dict):
            ctx["meta"]["missing"].append(f"fixture_shape_invalid:{type(fx).__name__}")
            return None

        ctx["fixture"] = fx.get("fixture") if isinstance(fx.get("fixture"), dict) else {}
        ctx["league"]  = fx.get("league")  if isinstance(fx.get("league"), dict)  else {}
        ctx["teams"]   = fx.get("teams")   if isinstance(fx.get("teams"), dict)   else {}
        ctx["goals"]   = fx.get("goals")   if isinstance(fx.get("goals"), dict)   else {}
        ctx["score"]   = fx.get("score")   if isinstance(fx.get("score"), dict)   else {}

    except RealtimeFetchError as e:
        ctx["meta"]["errors"].append(f"fixtures_error:{e.code}:{e.detail}")
        return ctx
    except Exception as e:
        ctx["meta"]["errors"].append(f"fixtures_error:{type(e).__name__}:{str(e)[:200]}")
        return ctx

    # --------------------------------------------------
    # 2) status / date
    # --------------------------------------------------
    status_short = ""
    fixture_date = None
    try:
        status_short = str(((ctx.get("fixture") or {}).get("status") or {}).get("short") or "")
        fixture_date = (ctx.get("fixture") or {}).get("date")
    except Exception:
        pass

    minutes_to_kickoff = None
    try:
        if isinstance(fixture_date, str) and fixture_date.strip():
            dt = datetime.fromisoformat(fixture_date.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            minutes_to_kickoff = int((dt - now).total_seconds() // 60)
    except Exception:
        minutes_to_kickoff = None

    # --------------------------------------------------
    # 3) helper optional endpoints
    # --------------------------------------------------
    def _optional(name: str, endpoint: str, params: dict, ttl: int = 20):
        try:
            d = _api_get(endpoint, params, timeout=10, cache_ttl=ttl)

            if isinstance(d, dict):
                ctx[name] = d.get("response", []) or []
            else:
                ctx[name] = []
                ctx["meta"]["missing"].append(f"{name}_shape_invalid:{type(d).__name__}")

            if len(ctx[name]) == 0:
                ctx["meta"]["missing"].append(f"{name}_empty")

        except RealtimeFetchError as e:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:{e.code}:{e.detail}")
        except Exception as e:
            ctx[name] = []
            ctx["meta"]["missing"].append(f"{name}_err:{type(e).__name__}")

    # --------------------------------------------------
    # 4) realtime mode logic
    # --------------------------------------------------
   
    if REALTIME_MODE == "off":
        ctx["meta"]["fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        return ctx

    _optional("injuries", "injuries", {"fixture": fixture_id}, ttl=60)

    if status_short == "NS":
        if minutes_to_kickoff is not None and minutes_to_kickoff <= REALTIME_LINEUPS_SOON_MINUTES:
            _optional("lineups", "fixtures/lineups", {"fixture": fixture_id}, ttl=20)
        else:
            ctx["meta"]["missing"].append("lineups_not_due_yet")

        ctx["meta"]["skipped"].extend(["events", "statistics", "players"])

    else:
        _optional("events", "fixtures/events", {"fixture": fixture_id}, ttl=10)
        _optional("lineups", "fixtures/lineups", {"fixture": fixture_id}, ttl=20)
        _optional("statistics", "fixtures/statistics", {"fixture": fixture_id}, ttl=10)

        if REALTIME_MODE == "full":
            _optional("players", "fixtures/players", {"fixture": fixture_id}, ttl=10)
        else:
            ctx["meta"]["skipped"].append("players")

    ctx["meta"]["fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    return ctx

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

def resolve_fixture_id_from_user_input(
    home: Any,
    away: Any,
    match_date: Any,
    league_code: Optional[str] = None,
    season_df: Any = None,
) -> Optional[int]:
    """
    Résout un fixture_id à partir des entrées utilisateur (home, away, date, league).
    - essaie offline d'abord si season_df fourni
    - sinon fallback online via API (/fixtures?date=YYYY-MM-DD + league+season si possible)
    Retourne uniquement fixture_id (int) ou None.
    """
    # offline preferred
    try:
        if season_df is not None:
            fid = _resolve_fixture_id_from_df(  # noqa: F821
                season_df, home, away, match_date, league_code=league_code
            )
            if fid is not None and str(fid).strip() != "":
                return int(fid)
    except Exception:
        pass

    # online fallback
    try:
        fid = _resolve_fixture_id_by_names(  # noqa: F821
            home, away, match_date, league_code=league_code
        )
        if fid is not None and str(fid).strip() != "":
            return int(fid)
    except Exception:
        pass

    return None


def attach_fixture_id_if_missing(features_input: Any, league_code: Optional[str] = None, season_df: Any = None):
    """
    Backward-compatible wrapper:
    resolves fixture_id from home/away/match_date (+league) then injects it.
    """
    existing = _safe_get_first(features_input, "fixture_id")  # noqa: F821
    if isinstance(existing, pd.Series):
        existing = existing.iloc[0] if len(existing) else None
    elif isinstance(existing, (list, tuple, np.ndarray)):
        existing = existing[0] if len(existing) else None

    if existing is not None and str(existing).strip() != "":
        return features_input

    home = _safe_get_first(features_input, "home")  # noqa: F821
    away = _safe_get_first(features_input, "away")  # noqa: F821
    match_date = _safe_get_first(features_input, "match_date")  # noqa: F821

    if home is None or away is None or match_date is None:
        return features_input

    fid = resolve_fixture_id_from_user_input(  # noqa: F821
        home, away, match_date, league_code=league_code, season_df=season_df
    )
    if fid is None:
        return features_input  # ✅ never int(None)

    try:
        if isinstance(features_input, dict):
            features_input["fixture_id"] = int(fid)
        elif isinstance(features_input, pd.Series):
            features_input.loc["fixture_id"] = int(fid)
        elif isinstance(features_input, pd.DataFrame) and len(features_input) > 0:
            features_input.at[features_input.index[0], "fixture_id"] = int(fid)
    except Exception:
        pass

    return features_input

def translate_reason_fr(reason: str) -> str:
    """
    Traduction métier FR des raisons d'absence (API-Sports → BetSmart).
    """
    if not reason:
        return "Indisponible"

    r = reason.strip().lower()

    # ignorer libellés techniques inutiles
    if r in ("missing fixture",):
        return ""

    return REASON_TRANSLATION_FR.get(r, reason)


def format_absences_summary(summary: Dict[str, Any], max_players_per_team: int = 3) -> str:
    """
    Résumé prudent des absences / incertitudes.

    Important :
    - 'Questionable' n'est pas présenté comme une absence confirmée.
    - Les listes API-Sports peuvent mélanger suspension, blessure, inactivité
      et statut incertain : on parle donc de situations recensées.
    """
    missing_meta = summary.get("missing_meta") or []
    injuries_failed = any(str(x).startswith("injuries_err:") for x in missing_meta)

    def _as_list(x):
        return x if isinstance(x, list) else []

    def _clean(s: Any) -> str:
        return "" if s is None else str(s).strip()

    def _fmt_player(it: Dict[str, Any]) -> str:
        name = _clean(it.get("player"))
        reason_raw = _clean(it.get("reason"))
        status_type = _clean(it.get("status_type"))
        reason_fr = translate_reason_fr(reason_raw)

        details = []
        if reason_fr:
            details.append(reason_fr)
        if status_type and status_type.lower() not in {"missing fixture", "missing"}:
            details.append(status_type)

        return f"{name} ({'; '.join(details)})" if details else name

    home = _clean(summary.get("home")) or "Domicile"
    away = _clean(summary.get("away")) or "Extérieur"

    injuries_home = int(summary.get("injuries_home") or 0)
    injuries_away = int(summary.get("injuries_away") or 0)
    injuries_total = int(summary.get("injuries_total") or (injuries_home + injuries_away) or 0)

    home_list = _as_list(summary.get("top_injuries_home"))[:max_players_per_team]
    away_list = _as_list(summary.get("top_injuries_away"))[:max_players_per_team]

    home_players = ", ".join(
        _fmt_player(it) for it in home_list
        if isinstance(it, dict) and _clean(it.get("player"))
    )
    away_players = ", ".join(
        _fmt_player(it) for it in away_list
        if isinstance(it, dict) and _clean(it.get("player"))
    )

    status_short = _clean(summary.get("status_short"))
    lineups_available = bool(summary.get("lineups_available"))
    lineups_expected_soon = bool(summary.get("lineups_expected_soon"))

    if injuries_total <= 0 and not home_players and not away_players:
        if injuries_failed:
            return "Absences : données indisponibles pour le moment (erreur de récupération)."
        if status_short == "NS" and not lineups_available:
            return "Absences : aucune situation recensée à ce stade ; compositions non publiées."
        return "Absences : aucune situation notable recensée."

    lines = [
        f"Disponibilités joueurs recensées (pré-match) — {home}: {injuries_home} | {away}: {injuries_away}"
    ]
    if home_players:
        lines.append(f"• {home}: {home_players}")
    if away_players:
        lines.append(f"• {away}: {away_players}")

    if status_short == "NS" and not lineups_available:
        if lineups_expected_soon:
            lines.append("Compositions attendues bientôt : certains statuts peuvent encore évoluer.")
        else:
            lines.append("Compositions non publiées : distinguer absences confirmées et joueurs incertains.")

    return "\n".join(lines)

###--------- FONCTION SUR LA POSITION AU CLASSEMENT 

def _fetch_league_standings_________________(league_id: int, season: int) -> Optional[dict]:
    """
    API-Sports: GET /standings?league=..&season=..
    Returns raw response dict or None.
    """
    try:
        data = _api_get("standings", {"league": int(league_id), "season": int(season)}, timeout=10)
        resp = (data or {}).get("response", []) or []
        if not resp:
            return None
        return resp[0]  # usually one object with "league" + "standings"
    except Exception:
        return None

def _fetch_league_standings(league_id: int, season: int) -> Optional[dict]:
    """
    API-Sports: GET /standings?league=..&season=..
    Cached because standings are reused across matches.
    """
    try:
        data = _api_get(
            "standings",
            {"league": int(league_id), "season": int(season)},
            timeout=REALTIME_TIMEOUT_STANDINGS,
            cache_ttl=180,   # 3 minutes
        )
        resp = (data or {}).get("response", []) or []
        if not resp:
            return None
        return resp[0]
    except Exception:
        return None

def _extract_team_rank_from_standings(standings_payload: dict, team_id: int) -> Optional[dict]:
    """
    Extract rank/points/played for a team_id from standings payload.
    API often: payload["league"]["standings"] is a list of groups (list of lists).
    """
    try:
        league_obj = standings_payload.get("league") or {}
        groups = league_obj.get("standings") or []

        # groups can be: [[{...},{...}...]] or [{...},{...}]
        if isinstance(groups, list) and len(groups) > 0 and isinstance(groups[0], list):
            rows = [r for g in groups for r in (g or [])]
        else:
            rows = groups if isinstance(groups, list) else []

        for r in rows:
            if not isinstance(r, dict):
                continue
            t = (r.get("team") or {})
            if int(t.get("id") or -1) == int(team_id):
                all_ = r.get("all") or {}
                return {
                    "team_id": int(team_id),
                    "team": t.get("name"),
                    "rank": r.get("rank"),
                    "points": r.get("points"),
                    "played": all_.get("played"),
                    "win": all_.get("win"),
                    "draw": all_.get("draw"),
                    "lose": all_.get("lose"),
                    "goals_for": (all_.get("goals") or {}).get("for"),
                    "goals_against": (all_.get("goals") or {}).get("against"),
                    "form": r.get("form"),  # sometimes present like "WWDLW"
                }
        return None
    except Exception:
        return None


def _build_ranking_block_from_ctx(ctx: dict) -> dict:
    """
    Build ranking block (home/away) from realtime ctx.
    Safe: returns empty dict if unavailable.
    """
    try:
        league = ctx.get("league") or {}
        teams = ctx.get("teams") or {}

        league_id = league.get("id")
        season = league.get("season")

        home = (teams.get("home") or {})
        away = (teams.get("away") or {})

        home_id = home.get("id")
        away_id = away.get("id")

        if not league_id or season is None or not home_id or not away_id:
            return {}

        payload = _fetch_league_standings(int(league_id), int(season))
        if payload is None:
            return {
                "league_id": int(league_id),
                "season": int(season),
                "available": False,
                "missing": ["standings_empty"],
            }

        home_rank = _extract_team_rank_from_standings(payload, int(home_id))
        away_rank = _extract_team_rank_from_standings(payload, int(away_id))

        return {
            "league_id": int(league_id),
            "season": int(season),
            "available": True,
            "home": home_rank or {"team_id": int(home_id), "team": home.get("name"), "missing": ["team_not_in_standings"]},
            "away": away_rank or {"team_id": int(away_id), "team": away.get("name"), "missing": ["team_not_in_standings"]},
        }
    except Exception:
        return {}

def realtime_summary_enriched__________(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Résumé enrichi (pré-match / live / post-match) basé sur ctx API-Sports.
    Spécifique à ta structure injuries: item = {player, team, fixture, league}.
    Ne plante jamais.
    + Ajout: position au classement (ranking) home/away, sans casser les champs existants.
    """

    def _d(x): 
        return x if isinstance(x, dict) else {}

    def _l(x): 
        return x if isinstance(x, list) else []

    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip().lower()

    def _parse_iso_dt(s: Any) -> Optional[datetime]:
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    # -----------------------------
    # Base ctx
    # -----------------------------
    fixture = _d(ctx.get("fixture"))
    status = _d(fixture.get("status"))
    st_short = status.get("short")
    st_long = status.get("long")
    elapsed = status.get("elapsed")

    teams = _d(ctx.get("teams"))
    home_obj = _d(teams.get("home"))
    away_obj = _d(teams.get("away"))
    home_name = home_obj.get("name")
    away_name = away_obj.get("name")
    home_id = home_obj.get("id")
    away_id = away_obj.get("id")

    league = _d(ctx.get("league"))
    league_id = league.get("id")
    season = league.get("season")

    injuries_raw = _l(ctx.get("injuries"))
    injuries: List[Dict[str, Any]] = [it for it in injuries_raw if isinstance(it, dict)]

    # minutes to kickoff
    minutes_to_kickoff = None
    try:
        dt = _parse_iso_dt(fixture.get("date"))
        if dt is not None:
            now = datetime.now(timezone.utc)
            minutes_to_kickoff = int((dt - now).total_seconds() // 60)
    except Exception:
        pass

    # available blocks
    lineups = _l(ctx.get("lineups"))
    events = _l(ctx.get("events"))
    players = _l(ctx.get("players"))
    stats = _l(ctx.get("statistics"))

    # meta missing
    meta = _d(ctx.get("meta"))
    missing_meta = meta.get("missing", []) if isinstance(meta.get("missing"), list) else []

    # -----------------------------
    # Injuries split home/away (counts)
    # -----------------------------
    injuries_home = 0
    injuries_away = 0
    hn = _norm(home_name)
    an = _norm(away_name)

    for it in injuries:
        team_name = _d(it.get("team")).get("name")
        tn = _norm(team_name)
        if tn and hn and tn == hn:
            injuries_home += 1
        elif tn and an and tn == an:
            injuries_away += 1

    # build top injuries (up to 3) - keeps your existing behavior
    top_injuries = []
    for it in injuries[:3]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        top_injuries.append({
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),   # ex: Missing Fixture
            "reason": p.get("reason"),      # ex: Injury
        })
        
    # split detailed injuries lists
    injuries_home_list = []
    injuries_away_list = []

    for it in injuries:
        team_name = _d(it.get("team")).get("name")
        tn = _norm(team_name)
        if tn and hn and tn == hn:
            injuries_home_list.append(it)
        elif tn and an and tn == an:
            injuries_away_list.append(it)

    top_injuries_home = []
    for it in injuries_home_list[:3]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        top_injuries_home.append({
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
        })

    top_injuries_away = []
    for it in injuries_away_list[:3]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        top_injuries_away.append({
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
        })

    # lineups expected soon if match is close and still empty
    lineups_expected_soon = False
    try:
        if st_short == "NS" and minutes_to_kickoff is not None and minutes_to_kickoff <= 120 and len(lineups) == 0:
            lineups_expected_soon = True
    except Exception:
        pass

    # started/finished flags
    finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
    is_finished = bool(st_short in finished_set)
    is_started = bool(st_short not in (None, "NS") and not is_finished)

    # -----------------------------
    # ✅ Ranking block (standings)
    # -----------------------------
    def _fetch_standings_payload(lid: int, seas: int) -> Optional[dict]:
        """
        Calls API-Sports standings endpoint via your existing _api_get.
        Returns resp[0] or None.
        """
        try:
            # _api_get must exist in your module
            if "_api_get" not in globals():
                return None
            #data = _api_get("standings", {"league": int(lid), "season": int(seas)}, timeout=10)
            data = _api_get(
                                "standings",
                                {"league": int(lid), "season": int(seas)},
                                timeout=REALTIME_TIMEOUT_STANDINGS,
                                cache_ttl=180
                            )
            resp = (data or {}).get("response", []) or []
            if not resp:
                return None
            return resp[0]
        except Exception:
            return None

    def _extract_team_rank(payload: dict, team_id_: int) -> Optional[dict]:
        """
        Extract minimal rank info for one team from standings payload.
        Safe across shapes: standings = [[...]] or [...]
        """
        try:
            league_obj = payload.get("league") or {}
            standings = league_obj.get("standings") or []

            # flatten if grouped
            rows: List[dict] = []
            if isinstance(standings, list) and standings and isinstance(standings[0], list):
                for g in standings:
                    if isinstance(g, list):
                        rows.extend([r for r in g if isinstance(r, dict)])
            elif isinstance(standings, list):
                rows = [r for r in standings if isinstance(r, dict)]

            for r in rows:
                t = r.get("team") or {}
                if int(t.get("id") or -1) == int(team_id_):
                    all_ = r.get("all") or {}
                    goals_ = all_.get("goals") or {}
                    return {
                        "team_id": int(team_id_),
                        "team": t.get("name"),
                        "rank": r.get("rank"),
                        "points": r.get("points"),
                        "played": all_.get("played"),
                        "win": all_.get("win"),
                        "draw": all_.get("draw"),
                        "lose": all_.get("lose"),
                        "goals_for": goals_.get("for"),
                        "goals_against": goals_.get("against"),
                        "form": r.get("form"),
                    }
            return None
        except Exception:
            return None

    ranking: Dict[str, Any] = {
        "available": False,
        "league_id": league_id,
        "season": season,
        "home": {"team_id": home_id, "team": home_name},
        "away": {"team_id": away_id, "team": away_name},
        "missing": []
    }

    try:
        # only attempt if we have necessary ids
        if league_id and season is not None and home_id and away_id:
            payload = _fetch_standings_payload(int(league_id), int(season))
            if payload is None:
                ranking["missing"].append("standings_empty_or_unavailable")
            else:
                home_rank = _extract_team_rank(payload, int(home_id))
                away_rank = _extract_team_rank(payload, int(away_id))
                ranking["available"] = True
                ranking["home"] = home_rank or {"team_id": int(home_id), "team": home_name, "missing": ["team_not_in_standings"]}
                ranking["away"] = away_rank or {"team_id": int(away_id), "team": away_name, "missing": ["team_not_in_standings"]}
        else:
            ranking["missing"].append("ranking_ids_missing")
    except Exception:
        ranking["available"] = False
        ranking["missing"].append("ranking_error")

    # -----------------------------
    # Return (unchanged fields + ranking)
    # -----------------------------
    return {
        "status_short": st_short,
        "status_long": st_long,
        "elapsed": elapsed,
        "home": home_name,
        "away": away_name,

        "is_started": is_started,
        "is_finished": is_finished,

        "minutes_to_kickoff": minutes_to_kickoff,
        "lineups_available": len(lineups) > 0,
        "events_available": len(events) > 0,
        "players_available": len(players) > 0,
        "statistics_available": len(stats) > 0,

        "injuries_total": len(injuries),
        "injuries_home": injuries_home,
        "injuries_away": injuries_away,
        "top_injuries": top_injuries,

        "lineups_expected_soon": bool(lineups_expected_soon),
        "missing_meta": missing_meta,

        # ✅ new (safe)
        "ranking": ranking,
        "top_injuries_home": top_injuries_home,
        "top_injuries_away": top_injuries_away,
    }

def realtime_summary_enriched(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Résumé enrichi (pré-match / live / post-match) basé sur ctx API-Sports.
    Compatible avec format_absences_summary(summary):
      - top_injuries_home
      - top_injuries_away
      - top_injuries
      - ranking
    Ne plante jamais.
    """

    def _d(x):
        return x if isinstance(x, dict) else {}

    def _l(x):
        return x if isinstance(x, list) else []

    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip().lower()

    def _parse_iso_dt(s: Any) -> Optional[datetime]:
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    # -----------------------------
    # Base ctx
    # -----------------------------
    fixture = _d(ctx.get("fixture"))
    status = _d(fixture.get("status"))
    st_short = status.get("short")
    st_long = status.get("long")
    elapsed = status.get("elapsed")

    teams = _d(ctx.get("teams"))
    home_obj = _d(teams.get("home"))
    away_obj = _d(teams.get("away"))
    home_name = home_obj.get("name")
    away_name = away_obj.get("name")
    home_id = home_obj.get("id")
    away_id = away_obj.get("id")

    league = _d(ctx.get("league"))
    league_id = league.get("id")
    season = league.get("season")

    injuries_raw = _l(ctx.get("injuries"))
    injuries: List[Dict[str, Any]] = [it for it in injuries_raw if isinstance(it, dict)]

    lineups = _l(ctx.get("lineups"))
    events = _l(ctx.get("events"))
    players = _l(ctx.get("players"))
    stats = _l(ctx.get("statistics"))

    meta = _d(ctx.get("meta"))
    missing_meta = meta.get("missing", []) if isinstance(meta.get("missing"), list) else []
    skipped_meta = meta.get("skipped", []) if isinstance(meta.get("skipped"), list) else []
    errors_meta = meta.get("errors", []) if isinstance(meta.get("errors"), list) else []

    # minutes to kickoff
    minutes_to_kickoff = None
    try:
        dt = _parse_iso_dt(fixture.get("date"))
        if dt is not None:
            now = datetime.now(timezone.utc)
            minutes_to_kickoff = int((dt - now).total_seconds() // 60)
    except Exception:
        pass

    # -----------------------------
    # Injuries split home/away
    # -----------------------------
    hn = _norm(home_name)
    an = _norm(away_name)

    injuries_home_list: List[Dict[str, Any]] = []
    injuries_away_list: List[Dict[str, Any]] = []

    for it in injuries:
        team_name = _d(it.get("team")).get("name")
        tn = _norm(team_name)
        if tn and hn and tn == hn:
            injuries_home_list.append(it)
        elif tn and an and tn == an:
            injuries_away_list.append(it)

    injuries_home = len(injuries_home_list)
    injuries_away = len(injuries_away_list)

    def _inj_to_rec(it: Dict[str, Any]) -> Dict[str, Any]:
        p = _d(it.get("player"))
        t = _d(it.get("team"))
        return {
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
        }

    top_injuries_home = [_inj_to_rec(it) for it in injuries_home_list[:3]]
    top_injuries_away = [_inj_to_rec(it) for it in injuries_away_list[:3]]
    top_injuries = [_inj_to_rec(it) for it in injuries[:3]]

    # lineups expected soon
    lineups_expected_soon = False
    try:
        if st_short == "NS" and minutes_to_kickoff is not None and minutes_to_kickoff <= 120 and len(lineups) == 0:
            lineups_expected_soon = True
    except Exception:
        pass

    # started/finished flags
    finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
    is_finished = bool(st_short in finished_set)
    is_started = bool(st_short not in (None, "NS") and not is_finished)

    # -----------------------------
    # Ranking block
    # -----------------------------
    def _fetch_standings_payload(lid: int, seas: int) -> Optional[dict]:
        try:
            data = _api_get(
                "standings",
                {"league": int(lid), "season": int(seas)},
                timeout=REALTIME_TIMEOUT_STANDINGS,
                cache_ttl=0,   # debug: pas de cache
            )

            if not isinstance(data, dict):
                print("[standings] invalid data type:", type(data), "league=", lid, "season=", seas)
                return None

            resp = data.get("response", []) or []
            print("[standings] league=", lid, "season=", seas, "resp_len=", len(resp))

            if not resp:
                print("[standings] EMPTY response:", data)
                return None

            return resp[0]

        except Exception as e:
            print("[standings] ERROR:", type(e).__name__, str(e), "league=", lid, "season=", seas)
            return None
    
    def _extract_team_rank(payload: dict, team_id_: int) -> Optional[dict]:
        try:
            league_obj = payload.get("league") or {}
            standings = league_obj.get("standings") or []

            rows: List[dict] = []
            if isinstance(standings, list) and standings and isinstance(standings[0], list):
                for g in standings:
                    if isinstance(g, list):
                        rows.extend([r for r in g if isinstance(r, dict)])
            elif isinstance(standings, list):
                rows = [r for r in standings if isinstance(r, dict)]

            for r in rows:
                t = r.get("team") or {}
                if int(t.get("id") or -1) == int(team_id_):
                    all_ = r.get("all") or {}
                    goals_ = all_.get("goals") or {}
                    return {
                        "team_id": int(team_id_),
                        "team": t.get("name"),
                        "rank": r.get("rank"),
                        "points": r.get("points"),
                        "played": all_.get("played"),
                        "win": all_.get("win"),
                        "draw": all_.get("draw"),
                        "lose": all_.get("lose"),
                        "goals_for": goals_.get("for"),
                        "goals_against": goals_.get("against"),
                        "form": r.get("form"),
                    }
            return None
        except Exception:
            return None

    ranking: Dict[str, Any] = {
        "available": False,
        "league_id": league_id,
        "season": season,
        "home": {"team_id": home_id, "team": home_name},
        "away": {"team_id": away_id, "team": away_name},
        "missing": []
    }

    try:
        if league_id and season is not None and home_id and away_id:
            payload = _fetch_standings_payload(int(league_id), int(season))
            if payload is None:
                ranking["missing"].append(f"standings_empty_or_unavailable:league={league_id}:season={season}")
            else:
                home_rank = _extract_team_rank(payload, int(home_id))
                away_rank = _extract_team_rank(payload, int(away_id))
                ranking["available"] = True
                ranking["home"] = home_rank or {
                    "team_id": int(home_id),
                    "team": home_name,
                    "missing": [f"team_not_in_standings:league={league_id}:season={season}"]
                }
                ranking["away"] = away_rank or {
                    "team_id": int(away_id),
                    "team": away_name,
                    "missing": [f"team_not_in_standings:league={league_id}:season={season}"]
                }
        else:
            ranking["missing"].append(f"standings_empty_or_unavailable:league={league_id}:season={season}")
    except Exception:
        ranking["available"] = False
        ranking["missing"].append("ranking_error")

    # -----------------------------
    # Return
    # -----------------------------
    return {
        "status_short": st_short,
        "status_long": st_long,
        "elapsed": elapsed,
        "home": home_name,
        "away": away_name,

        "is_started": is_started,
        "is_finished": is_finished,

        "minutes_to_kickoff": minutes_to_kickoff,
        "lineups_available": len(lineups) > 0,
        "events_available": len(events) > 0,
        "players_available": len(players) > 0,
        "statistics_available": len(stats) > 0,

        "injuries_total": len(injuries),
        "injuries_home": injuries_home,
        "injuries_away": injuries_away,

        # ✅ compatible with format_absences_summary
        "top_injuries_home": top_injuries_home,
        "top_injuries_away": top_injuries_away,
        "top_injuries": top_injuries,

        "lineups_expected_soon": bool(lineups_expected_soon),

        "missing_meta": missing_meta,
        "skipped_meta": skipped_meta,
        "errors_meta": errors_meta,

        "ranking": ranking,
    }

def _build_realtime_block_____________(
    features_df: Any,
    league_code: Optional[str] = None,
    home_name: Any = None,
    away_name: Any = None,
    match_date: Any = None,
    season_df: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Realtime enrichment block (Option B):
    - summary.top_injuries_home (max 3)
    - summary.top_injuries_away (max 3)

    Does NOT change prediction. Only enriches realtime_risk + notes.
    """

    # -----------------------------
    # helpers (safe & local)
    # -----------------------------
    def _as_scalar(v):
        try:
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else None
        except Exception:
            return None
        return v

    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    def _as_list(x):
        return x if isinstance(x, list) else []

    def _safe_len(x):
        try:
            return len(x) if x is not None else 0
        except Exception:
            return 0

    def _parse_iso_dt(s: Any) -> Optional[datetime]:
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _minutes_to_kickoff(ctx: Dict[str, Any]) -> Optional[int]:
        fixture = _as_dict(ctx.get("fixture"))
        dt = _parse_iso_dt(fixture.get("date"))
        if dt is None:
            return None
        now = datetime.now(timezone.utc)
        return int((dt - now).total_seconds() // 60)

    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return str(s).strip().lower()

    def _inj_to_rec(it: Dict[str, Any]) -> Dict[str, Any]:
        p = _as_dict(it.get("player"))
        t = _as_dict(it.get("team"))
        rec = {
            "team": t.get("name"),
            "player": p.get("name"),
            "status_type": p.get("type"),
            "reason": p.get("reason"),
            # optionnel (si tu veux UI + riche):
            # "player_id": p.get("id"),
            # "photo": p.get("photo"),
            # "team_id": t.get("id"),
            # "logo": t.get("logo"),
        }
        return {k: v for k, v in rec.items() if v is not None and str(v).strip() != ""}

    # --- summary (Option B) ---
    
    def _risk_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        st = summary.get("status_short")
        injuries_total = int(summary.get("injuries_total") or 0)
        lineups_ok = bool(summary.get("lineups_available"))

        finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
        if st in finished_set:
            return {"risk_level": "HIGH", "risk_score": 0.9, "reasons": [f"fixture_status:{st}"]}

        # Pre-match: injuries OR no lineups -> medium
        if st == "NS":
            if injuries_total > 0 or (not lineups_ok):
                score = 0.4
                if injuries_total >= 8:
                    score = 0.55
                return {"risk_level": "MEDIUM", "risk_score": score, "reasons": ["possible_injuries_or_lineup_changes"]}
            return {"risk_level": "LOW", "risk_score": 0.1, "reasons": ["pre_match_no_major_signals"]}

        return {"risk_level": "MEDIUM", "risk_score": 0.5, "reasons": [f"fixture_status:{st}"]}

    # -----------------------------
    # read inputs (df or args)
    # -----------------------------
    if home_name is None:
        home_name = _safe_get_first(features_df, "home")
    if away_name is None:
        away_name = _safe_get_first(features_df, "away")
    if match_date is None:
        match_date = _safe_get_first(features_df, "match_date")

    use_realtime_val = _as_scalar(_safe_get_first(features_df, "_use_realtime"))
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
            "summary": {},
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
            "summary": {},
        }
        return block, f"realtime: skipped_missing_fields={missing_fields}"
    
        # -----------------------------
    # use existing fixture_id first
    # -----------------------------
    fixture_id_existing = None
    try:
        if isinstance(features_df, pd.DataFrame) and "fixture_id" in features_df.columns:
            v = _safe_get_first(features_df, "fixture_id")
            if v is not None and str(v).strip() != "":
                fixture_id_existing = int(v)
    except Exception:
        fixture_id_existing = None

    if fixture_id_existing is not None:
        fixture_id_int = fixture_id_existing
    else:
        try:
            fixture_id = _safe_resolve_fixture_id(
                home_name, away_name, match_date,
                league_code=league_code,
                season_df=season_df,
                features_df=features_df
            )
        except Exception as e:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": [f"fixture_resolve_error:{type(e).__name__}"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: resolve error={type(e).__name__}"

        if fixture_id is None or str(fixture_id).strip() == "":
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_not_found"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, "realtime: fixture not found"

        try:
            fixture_id_int = int(fixture_id)
        except Exception:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_invalid"],
                "reasons": ["fixture_id_invalid"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: fixture id invalid={fixture_id}"

    # -----------------------------
    # resolve fixture_id
    # -----------------------------
    try:
        fixture_id = _safe_resolve_fixture_id(
            home_name, away_name, match_date,
            league_code=league_code,
            season_df=season_df,
            features_df=features_df
        )
    except Exception as e:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": [f"fixture_resolve_error:{type(e).__name__}"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: resolve error={type(e).__name__}"

    if fixture_id is None or str(fixture_id).strip() == "":
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["fixture_id_not_found"],
            "reasons": [],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, "realtime: fixture not found"

    try:
        fixture_id_int = int(fixture_id)
    except Exception:
        block = {
            "available": False,
            "fixture_id": None,
            "missing": ["fixture_id_invalid"],
            "reasons": ["fixture_id_invalid"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: fixture id invalid={fixture_id}"

    # -----------------------------
    # fetch ctx & compute
    # -----------------------------
    debug_rt = os.getenv("DEBUG_REALTIME", "0") == "1"

    try:
        ctx = _fetch_realtime_context(fixture_id_int)

        if ctx is not None and not isinstance(ctx, dict):
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "reasons": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx invalid type={type(ctx).__name__}"

        if ctx is None:
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": ["realtime_ctx_empty"],
                "reasons": ["realtime_ctx_empty"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx empty"

        #summary = _realtime_summary_enriched(ctx)
        summary = realtime_summary_enriched(ctx)
        risk_pm = _risk_from_summary(summary)
        summary["absences_text"] = format_absences_summary(summary)

        # optional legacy scorer (guarded)
        risk_raw = {}
        try:
            rr = _realtime_risk_score(ctx)  # if exists
            if isinstance(rr, dict):
                risk_raw = rr
        except Exception:
            risk_raw = {}

        st_short = summary.get("status_short")
        chosen = risk_pm if st_short == "NS" else (risk_raw or risk_pm)

        block = {
            "available": True,
            "fixture_id": fixture_id_int,
            "missing": [],
            "reasons": chosen.get("reasons", []),
            "risk_level": chosen.get("risk_level", "UNKNOWN"),
            "risk_score": float(chosen.get("risk_score", 0.0) or 0.0),
            "summary": summary,
        }

        if debug_rt:
            try:
                fixture = _as_dict(ctx.get("fixture"))
                status = _as_dict(fixture.get("status"))
                block["debug"] = {
                    "ctx_keys": sorted(list(ctx.keys())),
                    "fixture_keys": sorted(list(fixture.keys())) if isinstance(fixture, dict) else [],
                    "status_short": status.get("short"),
                    "injuries_count": _safe_len(ctx.get("injuries")),
                    "injuries_home": summary.get("injuries_home"),
                    "injuries_away": summary.get("injuries_away"),
                    "missing_meta": summary.get("missing_meta"),
                    "risk_pm": risk_pm,
                    "risk_raw": risk_raw,
                    "risk_chosen": chosen,
                }
            except Exception:
                block["debug"] = {"debug_error": "failed_to_build_debug"}

        return block, f"realtime: ok fixture_id={fixture_id_int}"

    except Exception as e:
        block = {
            "available": False,
            "fixture_id": fixture_id_int,
            "missing": [f"realtime_error:{type(e).__name__}"],
            "reasons": [f"realtime_error:{type(e).__name__}"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: ok fixture_id={fixture_id_int} but error={type(e).__name__}"


def _build_realtime_block(
    features_df: Any,
    league_code: Optional[str] = None,
    home_name: Any = None,
    away_name: Any = None,
    match_date: Any = None,
    season_df: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Realtime enrichment block:
    - uses existing fixture_id if already available in features_df
    - fetches realtime context
    - enriches summary via realtime_summary_enriched
    - returns realtime_risk block + note
    """

    def _as_scalar(v):
        try:
            if isinstance(v, pd.Series):
                return v.iloc[0] if len(v) else None
            if isinstance(v, (list, tuple, np.ndarray)):
                return v[0] if len(v) else None
        except Exception:
            return None
        return v

    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    def _safe_len(x):
        try:
            return len(x) if x is not None else 0
        except Exception:
            return 0

    def _risk_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        st = summary.get("status_short")
        injuries_total = int(summary.get("injuries_total") or 0)
        lineups_ok = bool(summary.get("lineups_available"))

        finished_set = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "SUSP", "INT"}
        if st in finished_set:
            return {"risk_level": "HIGH", "risk_score": 0.9, "reasons": [f"fixture_status:{st}"]}

        if st == "NS":
            if injuries_total > 0 or (not lineups_ok):
                score = 0.4
                if injuries_total >= 8:
                    score = 0.55
                return {
                    "risk_level": "MEDIUM",
                    "risk_score": score,
                    "reasons": ["possible_injuries_or_lineup_changes"]
                }
            return {"risk_level": "LOW", "risk_score": 0.1, "reasons": ["pre_match_no_major_signals"]}

        return {"risk_level": "MEDIUM", "risk_score": 0.5, "reasons": [f"fixture_status:{st}"]}

    # -----------------------------
    # read inputs
    # -----------------------------
    if home_name is None:
        home_name = _safe_get_first(features_df, "home")
    if away_name is None:
        away_name = _safe_get_first(features_df, "away")
    if match_date is None:
        match_date = _safe_get_first(features_df, "match_date")

    use_realtime_val = _as_scalar(_safe_get_first(features_df, "_use_realtime"))
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
            "summary": {},
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
            "summary": {},
        }
        return block, f"realtime: skipped_missing_fields={missing_fields}"

    
    # use existing fixture_id first
    # -----------------------------
    fixture_id_int = None
    try:
        if isinstance(features_df, pd.DataFrame) and "fixture_id" in features_df.columns:
            v = _safe_get_first(features_df, "fixture_id")
            if v is not None and str(v).strip() != "":
                fixture_id_int = int(v)
    except Exception:
        fixture_id_int = None
    """
    # fallback resolve only if needed
    if fixture_id_int is None:
        try:
            fixture_id = _safe_resolve_fixture_id(
                home_name, away_name, match_date,
                league_code=league_code,
                season_df=season_df,
                features_df=features_df
            )
        except Exception as e:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": [f"fixture_resolve_error:{type(e).__name__}"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: resolve error={type(e).__name__}"

        if fixture_id is None or str(fixture_id).strip() == "":
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_not_found"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, "realtime: fixture not found"

        try:
            fixture_id_int = int(fixture_id)
        except Exception:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_invalid"],
                "reasons": ["fixture_id_invalid"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: fixture id invalid={fixture_id}"
    """
    # fallback resolve only if needed
    if fixture_id_int is None:
        try:
            fixture_id = _safe_resolve_fixture_id(
                home_name, away_name, match_date,
                league_code=league_code,
                season_df=season_df,
                features_df=features_df
            )
        except Exception as e:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": [f"fixture_resolve_error:{type(e).__name__}"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: resolve error={type(e).__name__}"

        if fixture_id is None or str(fixture_id).strip() == "":
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_not_found"],
                "reasons": [],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, "realtime: fixture not found"

        try:
            fixture_id_int = int(fixture_id)
        except Exception:
            block = {
                "available": False,
                "fixture_id": None,
                "missing": ["fixture_id_invalid"],
                "reasons": ["fixture_id_invalid"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: fixture id invalid={fixture_id}"

    # -----------------------------
    # fetch ctx & compute
    # -----------------------------
    debug_rt = os.getenv("DEBUG_REALTIME", "0") == "1"

    try:
        ctx = _fetch_realtime_context(fixture_id_int)

        if ctx is not None and not isinstance(ctx, dict):
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "reasons": [f"realtime_ctx_invalid:{type(ctx).__name__}"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx invalid type={type(ctx).__name__}"

        if ctx is None:
            block = {
                "available": False,
                "fixture_id": fixture_id_int,
                "missing": ["realtime_ctx_empty"],
                "reasons": ["realtime_ctx_empty"],
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "summary": {},
            }
            return block, f"realtime: ok fixture_id={fixture_id_int} but ctx empty"

        summary = realtime_summary_enriched(ctx)
        risk_pm = _risk_from_summary(summary)
        summary["absences_text"] = format_absences_summary(summary)

        risk_raw = {}
        try:
            rr = _realtime_risk_score(ctx)
            if isinstance(rr, dict):
                risk_raw = rr
        except Exception:
            risk_raw = {}

        st_short = summary.get("status_short")
        chosen = risk_pm if st_short == "NS" else (risk_raw or risk_pm)

        block = {
            "available": True,
            "fixture_id": fixture_id_int,
            "missing": [],
            "reasons": chosen.get("reasons", []),
            "risk_level": chosen.get("risk_level", "UNKNOWN"),
            "risk_score": float(chosen.get("risk_score", 0.0) or 0.0),
            "summary": summary,
        }

        if debug_rt:
            try:
                fixture = _as_dict(ctx.get("fixture"))
                status = _as_dict(fixture.get("status"))
                meta = _as_dict(ctx.get("meta"))
                block["debug"] = {
                    "ctx_keys": sorted(list(ctx.keys())),
                    "fixture_keys": sorted(list(fixture.keys())) if isinstance(fixture, dict) else [],
                    "status_short": status.get("short"),
                    "injuries_count": _safe_len(ctx.get("injuries")),
                    "injuries_home": summary.get("injuries_home"),
                    "injuries_away": summary.get("injuries_away"),
                    "missing_meta": summary.get("missing_meta"),
                    "skipped_meta": summary.get("skipped_meta"),
                    "errors_meta": summary.get("errors_meta"),
                    "ctx_meta_missing": meta.get("missing", []),
                    "ctx_meta_errors": meta.get("errors", []),
                    "ctx_meta_skipped": meta.get("skipped", []),
                    "risk_pm": risk_pm,
                    "risk_raw": risk_raw,
                    "risk_chosen": chosen,
                }
            except Exception:
                block["debug"] = {"debug_error": "failed_to_build_debug"}

        return block, f"realtime: ok fixture_id={fixture_id_int}"

    except Exception as e:
        block = {
            "available": False,
            "fixture_id": fixture_id_int,
            "missing": [f"realtime_error:{type(e).__name__}"],
            "reasons": [f"realtime_error:{type(e).__name__}"],
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "summary": {},
        }
        return block, f"realtime: ok fixture_id={fixture_id_int} but error={type(e).__name__}"

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
REALTIME_API_KEY = os.getenv("REALTIME_API_KEY")


# -------------------------------------------------------------------
# 🔢 Conventions BetSmart (IMPORTANT: éviter toute confusion 0/1/2)
# 0 = Victoire domicile (Home)
# 1 = Match nul (Draw)
# 2 = Victoire extérieur (Away)
# -------------------------------------------------------------------


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

    # V2.2.1 FORM GUARD -------------------------------------------------
    # Nombre de matchs de saison courante réellement joués AVANT le match ciblé.
    # La forme ne doit jamais pouvoir renverser la décision sur un échantillon
    # trop faible (ex. WMMMM / LMMMM en début de saison).
    def _team_played_count(team):
        try:
            dteam = df_past[(df_past["HomeTeam"] == team) | (df_past["AwayTeam"] == team)].copy()
            if dteam.empty:
                return 0
            if "FTHG" in dteam.columns and "FTAG" in dteam.columns:
                dteam = dteam[dteam["FTHG"].notna() & dteam["FTAG"].notna()]
            return int(len(dteam))
        except Exception:
            return 0

    def _team_recent_form_available(team, window=5):
        try:
            dteam = df_past[(df_past["HomeTeam"] == team) | (df_past["AwayTeam"] == team)].copy()
            if dteam.empty:
                return 0
            if "FTHG" in dteam.columns and "FTAG" in dteam.columns:
                dteam = dteam[dteam["FTHG"].notna() & dteam["FTAG"].notna()]
            return int(min(window, len(dteam)))
        except Exception:
            return 0

    home_matches_played = _team_played_count(home_team)
    away_matches_played = _team_played_count(away_team)
    home_form_available = _team_recent_form_available(home_team, window=5)
    away_form_available = _team_recent_form_available(away_team, window=5)
    current_form_reliability = float(min(home_form_available, away_form_available) / 5.0)

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
                # Métadonnées V2.2.1 FORM GUARD (non utilisées par les modèles entraînés)
                "HomeMatchesPlayedCurrent": home_matches_played,
                "AwayMatchesPlayedCurrent": away_matches_played,
                "HomeFormAvailableMatches": home_form_available,
                "AwayFormAvailableMatches": away_form_available,
                "CurrentFormReliability": current_form_reliability,
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



def _entropy(p0, p1, p2, eps=1e-12):
    p = np.array([p0, p1, p2], dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())  # max ~ 1.098

def detect_double_chance_v2(
    p0: float, p1: float, p2: float,
    pred_final: int,
    *,
    league_code: str = "default",
    bias_detected: bool = False,
    low_confidence: bool = False,
    upset_score: float = 0.0,
    upset_threshold: float = 0.52,
    override_tag: Optional[str] = None,
) -> Optional[str]:
    """
    DC renforcée (version pro).
    Retourne "1X", "X2" ou None.
    """

    # 1) seuils ligue
    try:
        params = _get_params(league_code)
    except Exception:
        params = {}

    # seuils par défaut
    base_gap = float(params.get("dc_gap_threshold", 0.12))          # plus strict que avant
    draw_th = float(params.get("dc_draw_threshold", 0.28))          # si nul >= 28% => DC
    ent_th  = float(params.get("dc_entropy_threshold", 1.03))       # proche du max (1.098)
    max_win_no_dc = float(params.get("dc_max_win_no_dc", 0.72))     # si win >= 72% => pas besoin

    # 2) métriques
    probs = np.array([p0, p1, p2], dtype=float)
    probs = probs / max(1e-9, probs.sum())

    top = float(np.max(probs))
    srt = np.sort(probs)
    gap = float(srt[-1] - srt[-2])
    ent = _entropy(probs[0], probs[1], probs[2])

    # 3) règles dures (force)
    force = False
    reasons = []

    if bias_detected:
        force = True; reasons.append("bias")
    if low_confidence:
        force = True; reasons.append("low_conf")
    if upset_score is not None and upset_score >= (upset_threshold * 0.85):
        force = True; reasons.append("upset_near_threshold")
    if override_tag is not None and "form_over_market" in str(override_tag):
        force = True; reasons.append("form_vs_market_conflict")

    # 4) règles probabilistes (force si risque)
    if float(p1) >= draw_th:
        force = True; reasons.append("high_draw")
    if gap <= base_gap:
        force = True; reasons.append("small_gap")
    if ent >= ent_th:
        force = True; reasons.append("high_entropy")

    # 5) si top win trop fort => on annule DC (sauf force métier)
    if (top >= max_win_no_dc) and (not (bias_detected or low_confidence)):
        return None

    if not force:
        return None

    # 6) sortie DC cohérente
    # pred_final: 0=home, 1=draw, 2=away
    if pred_final == 0:
        return "1X"
    if pred_final == 2:
        return "X2"

    # si draw prédit, DC dépend du plus fort entre home/away
    return "1X" if p0 >= p2 else "X2"


def detect_bias(features_df):
    odds = features_df[["B365H", "B365A", "B365D"]].values[0].astype(float)
    max_odds = np.max(odds)
    min_odds = np.min(odds)
    bias_score = abs(max_odds - min_odds) / np.mean(odds)
    return bias_score > 0.6


def is_confidence_low(proba_0, proba_1, proba_2):
    """BetSmart V2.1.1 — confiance probabiliste, sans changer la prédiction."""
    try:
        arr = np.array([proba_0, proba_1, proba_2], dtype=float)
        if not np.isfinite(arr).all() or arr.sum() <= 0:
            return True
        arr = arr / arr.sum()
        ordered = np.sort(arr)
        top = float(ordered[-1])
        second = float(ordered[-2])
        gap = top - second
        entropy = float(-np.sum(arr * np.log(arr + 1e-12)))
        return bool((top < 0.45) or (gap < 0.10) or (entropy >= 1.03))
    except Exception:
        return True


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


# -----------------------------
# Config (ENV friendly)
# -----------------------------

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY manquante. "
            "En local: mets-la dans un fichier .env. "
            "Sur Render: ajoute-la dans Environment Variables."
        )
    return OpenAI(api_key=api_key)

def _ai_action_user_label(action: Any, selection: Any = None) -> str:
    """
    Traduit les codes techniques internes de l'IA en formulation claire
    pour l'utilisateur final. Les codes restent inchangés dans ai_decision.
    """
    a = str(action or "").strip().upper()
    s = str(selection or "").strip().upper()

    if a == "BET":
        if s in ("HOME", "1"):
            return "Pari recommandé : victoire de l'équipe à domicile"
        if s in ("AWAY", "2"):
            return "Pari recommandé : victoire de l'équipe à l'extérieur"
        if s in ("DRAW", "X", "N"):
            return "Pari recommandé : match nul"
        if s in ("1X", "X2"):
            return f"Pari recommandé : double chance {s}"
        return "Pari recommandé"

    if a == "WATCH":
        return "Aucune prise de position recommandée pour le moment"

    if a == "NO_BET":
        return "Pari déconseillé"

    return "Décision de pari non disponible"


def explanation_from_pred_final(pred_final: Dict[str, Any], user_profile: str = "standard") -> Dict[str, Any]:
    """
    Prend le JSON FINAL (après apply_unexpected_layer) et renvoie le même JSON
    avec pred_final["explanation"] remplacé (4 à 8 phrases FR).
    - Fallback robuste offline
    - Optionnel LLM OpenAI si OPENAI_EXPLAIN_ENABLED=1 et clé ok
    - Ajoute des metas: explain_llm_used, explain_llm_model, explain_llm_error, explain_llm_debug
    """

    def _is_nan(x: Any) -> bool:
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def _get(d: Dict[str, Any], key: str, default=None):
        try:
            v = d.get(key, default)
            if _is_nan(v):
                return default
            return v
        except Exception:
            return default

    def _pct_from_any(v: Any) -> float:
        """
        Convertit:
          - 0.52 -> 0.52
          - "52%" -> 0.52
          - 52 -> 0.52 (si >1 on suppose %)
        """
        try:
            if v is None or _is_nan(v):
                return 0.0
            if isinstance(v, str):
                s = v.strip().replace(",", ".")
                if not s:
                    return 0.0
                if s.endswith("%"):
                    x = float(s[:-1].strip()) / 100.0
                    return max(0.0, min(1.0, x))
                x = float(s)
                if x > 1.0:
                    x /= 100.0
                return max(0.0, min(1.0, x))
            x = float(v)
            if x > 1.0:
                x /= 100.0
            return max(0.0, min(1.0, x))
        except Exception:
            return 0.0

    def _fmt_pct(x: float) -> str:
        try:
            return f"{round(float(x)*100,1)}%"
        except Exception:
            return "0.0%"

    # -----------------------------
    # Extract from FINAL JSON
    # -----------------------------
    home = str(_get(pred_final, "home", "") or "")
    away = str(_get(pred_final, "away", "") or "")
    match_date = str(_get(pred_final, "match_date", "") or _get(pred_final, "date", "") or "")

    form_home = str(_get(pred_final, "5_dern_perf_home", "") or "")
    form_away = str(_get(pred_final, "5_dern_perf_away", "") or "")

    # BetSmart V2.1 - contexte historique structure
    historical_context = _get(pred_final, "historical_context", {}) or {}
    if not isinstance(historical_context, dict):
        historical_context = {}
    market_context = _get(pred_final, "market_context", {}) or {}
    if not isinstance(market_context, dict):
        market_context = {}
    ai_decision = _get(pred_final, "ai_decision", {}) or {}
    if not isinstance(ai_decision, dict):
        ai_decision = {}

    bias_detected = bool(_get(pred_final, "bias_detected", False) or False)
    low_confidence = bool(_get(pred_final, "low_confidence", False) or False)
    double_chance = _get(pred_final, "double_chance", None)
    ai_decision = _get(pred_final, "ai_decision", {}) or {}
    if not isinstance(ai_decision, dict):
        ai_decision = {}

    ai_action = str(ai_decision.get("action") or "").strip().upper()
    ai_selection = str(ai_decision.get("selection") or "").strip().upper()
    ai_user_decision = _ai_action_user_label(ai_action, ai_selection)


    # probs: prefer proba_* if present, else p*_raw
    p0 = _pct_from_any(_get(pred_final, "proba_0", None))
    p1 = _pct_from_any(_get(pred_final, "proba_1", None))
    p2 = _pct_from_any(_get(pred_final, "proba_2", None))
    if (p0 + p1 + p2) <= 1e-6:
        p0 = _pct_from_any(_get(pred_final, "p0_raw", 0.0))
        p1 = _pct_from_any(_get(pred_final, "p1_raw", 0.0))
        p2 = _pct_from_any(_get(pred_final, "p2_raw", 0.0))

    # odds if present in pred_final
    odds = {}
    for k in ("B365H", "B365D", "B365A"):
        v = _get(pred_final, k, None)
        try:
            if v is not None and str(v).strip() != "":
                odds[k] = float(str(v).replace(",", "."))
        except Exception:
            pass

    rule_applied = str(_get(pred_final, "rule_applied", "") or "")
    upset_score = float(_get(pred_final, "_upset_score", 0.0) or 0.0)
    upset_threshold = float(_get(pred_final, "_upset_threshold", 0.52) or 0.52)

    # realtime summary
    realtime_risk = _get(pred_final, "realtime_risk", {}) or {}
    summary = {}
    try:
        summary = (realtime_risk or {}).get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
    except Exception:
        summary = {}

    absences_text = str(summary.get("absences_text") or "")
    missing_meta = summary.get("missing_meta") or []
    if not isinstance(missing_meta, list):
        missing_meta = []

    top_injuries = summary.get("top_injuries") or []
    if not isinstance(top_injuries, list):
        top_injuries = []

    ranking = summary.get("ranking") or {}
    if not isinstance(ranking, dict):
        ranking = {}

    rank_home = rank_away = None
    pts_home = pts_away = None
    played_home = played_away = 0
    if ranking.get("available") is True:
        try:
            rh = ranking.get("home") or {}
            ra = ranking.get("away") or {}
            played_home = int(rh.get("played") or 0)
            played_away = int(ra.get("played") or 0)
            # Un classement avant tout match joué n'est pas interprétable.
            if played_home > 0 and played_away > 0:
                rank_home = rh.get("rank")
                rank_away = ra.get("rank")
                pts_home = rh.get("points")
                pts_away = ra.get("points")
        except Exception:
            pass

    status_short = str(summary.get("status_short") or "")
    status_long = str(summary.get("status_long") or "")
    is_finished = bool(summary.get("is_finished") is True)
    is_started = bool(summary.get("is_started") is True)
    elapsed = summary.get("elapsed")

    # -----------------------------
    # OFFLINE fallback (4-8 phrases)
    # -----------------------------
    def _fallback() -> str:
        lines: List[str] = []

        title = f"{home} vs {away}" if home and away else "Match"
        if match_date:
            title += f" ({match_date})"
        lines.append(f"{title}.")

        if (p0 + p1 + p2) > 1e-6:
            lines.append(f"Probabilités (1/N/2) : {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}.")
        else:
            lines.append("Probabilités (1/N/2) : indisponibles.")

        # favorite
        if (p0 + p1 + p2) > 1e-6:
            fav = "home" if p0 >= max(p1, p2) else ("draw" if p1 >= max(p0, p2) else "away")
            if fav == "home":
                lines.append(f"Lecture modèle : avantage {home} (victoire à domicile).")
            elif fav == "away":
                lines.append(f"Lecture modèle : avantage {away} (victoire à l’extérieur).")
            else:
                lines.append("Lecture modèle : match équilibré (nul plausible).")

        if form_home or form_away:
            lines.append(f"Forme (5 derniers) : {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}.")
        if historical_context:
            try:
                h2h = historical_context.get("h2h") or {}
                lines.append(
                    f"Historique V2.1.1 : signal={historical_context.get('historical_signal')} ; "
                    f"confiance={historical_context.get('historical_confidence')} ; "
                    f"fiabilite forme={historical_context.get('current_form_reliability')} ; "
                    f"qualite donnees={historical_context.get('data_quality')}."
                )
                if h2h.get("black_beast_signal") and h2h.get("dominant_team"):
                    lines.append(f"H2H : domination récurrente détectée en faveur de {h2h.get('dominant_team')} (signal bête noire).")
            except Exception:
                pass
        if market_context and market_context.get("available"):
            lines.append(f"Marché : value théorique modèle={market_context.get('best_value')} ; désaccord modèle/marché={market_context.get('model_market_disagreement')}.")
        if ai_decision:
            lines.append(
                f"Arbitrage IA V2.2.2 : action={ai_decision.get('action')} ; sélection={ai_decision.get('selection')} ; "
                f"confiance décision={ai_decision.get('decision_confidence', ai_decision.get('confidence'))} ; "
                f"confiance sélection={ai_decision.get('selection_confidence')} ; "
                f"accord sources={ai_decision.get('source_agreement')}."
            )

        if isinstance(rank_home, int) and isinstance(rank_away, int):
            pts_txt = ""
            if isinstance(pts_home, int) and isinstance(pts_away, int):
                pts_txt = f" ({pts_home} pts vs {pts_away} pts)"
            lines.append(f"Classement : {home} est {rank_home}ᵉ, {away} est {rank_away}ᵉ{pts_txt}.")

        if odds:
            parts = []
            if "B365H" in odds: parts.append(f"H={odds['B365H']}")
            if "B365D" in odds: parts.append(f"N={odds['B365D']}")
            if "B365A" in odds: parts.append(f"A={odds['B365A']}")
            lines.append("Cotes (B365) : " + ", ".join(parts) + ".")

        if double_chance:
            lines.append(f"Double chance : {double_chance} (filet de sécurité).")

        if bias_detected:
            lines.append("Biais de cotes détecté : prudence (effet popularité / surcote possible).")

        if upset_score > 0 and upset_score >= upset_threshold:
            lines.append("Risque de surprise (upset) élevé : éviter les mises agressives.")

        if absences_text:
            lines.append(absences_text.strip())

        # status (live/FT)
        if status_short:
            if is_finished or status_short == "FT":
                lines.append("Note : le match est terminé (infos temps réel post-match).")
            elif is_started:
                lines.append(f"Note : match en cours ({status_short}), minute ≈ {elapsed}.")

        # prudence if missing key live info
        if isinstance(missing_meta, list) and len(missing_meta) > 0:
            lines.append("Certaines données temps réel manquent encore (compos/stats/événements) : prudence avant de valider un pari.")

        if ai_action:
            lines.append(f"Décision BetSmart : {ai_user_decision}.")

        return " ".join(lines[:9]).strip()

    fallback_text = _fallback()
   

    # -----------------------------
    # LLM explanation (uses FINAL JSON only)
    # -----------------------------
    api_key = get_openai_client()
    if not OPENAI_EXPLAIN_ENABLED or not api_key:
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = ""
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": OPENAI_EXPLAIN_ENABLED, "has_key": bool(api_key), "payload": None}
        return pred_final

    # Build "facts" to reduce hallucination
    facts: List[str] = []
    facts.append(f"Match: {home} vs {away} | date={match_date}")
    facts.append(f"Probas 1/N/2: {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}")
    if form_home or form_away:
        facts.append(f"Forme(5): {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}")
    prediction_context = _get(pred_final, "prediction_context", {}) or {}
    if isinstance(prediction_context, dict) and prediction_context:
        try:
            facts.append(
                "Contexte prédiction V2.2.3: "
                f"mode={prediction_context.get('mode')} ; "
                f"fusion={prediction_context.get('fusion_applied')} ; "
                f"poids_historique={prediction_context.get('effective_historical_weight')} ; "
                f"poids_modele={prediction_context.get('model_weight')} ; "
                f"pré={prediction_context.get('pre_adjustment_probabilities')} ; "
                f"historique={prediction_context.get('historical_probabilities')} ; "
                f"post={prediction_context.get('post_adjustment_probabilities')}."
            )
        except Exception:
            pass

    if historical_context:
        try:
            hc = historical_context
            h2h = hc.get("h2h") or {}
            facts.append(
                "Historique V2.1.1: "
                f"signal={hc.get('historical_signal')} confidence={hc.get('historical_confidence')} "
                f"score={hc.get('historical_score')} data_quality={hc.get('data_quality')} "
                f"form_reliability={hc.get('current_form_reliability')}"
            )
            facts.append(
                "H2H V2.1.1: "
                f"matches={h2h.get('matches',0)} home_wins={h2h.get('home_wins',0)} draws={h2h.get('draws',0)} "
                f"away_wins={h2h.get('away_wins',0)} dominance_score={h2h.get('dominance_score')} "
                f"dominant_team={h2h.get('dominant_team')} black_beast={h2h.get('black_beast_signal')}"
            )
            hp=hc.get("historical_home_at_home") or {}; ap=hc.get("historical_away_away") or {}
            facts.append(
                "Profils historiques venue: "
                f"home n={hp.get('matches')} W={hp.get('win_rate')} D={hp.get('draw_rate')} L={hp.get('loss_rate')} ppg={hp.get('points_per_game')} ; "
                f"away n={ap.get('matches')} W={ap.get('win_rate')} D={ap.get('draw_rate')} L={ap.get('loss_rate')} ppg={ap.get('points_per_game')}"
            )
        except Exception:
            pass
    if market_context and market_context.get("available"):
        facts.append("Market V2.1.1 (calcul Python, source de vérité pour value): " + json.dumps(market_context, ensure_ascii=False))
    if ai_decision:
        try:
            facts.append("AI arbitration V2.2 (DECISION SOURCE OF TRUTH): " + json.dumps(ai_decision, ensure_ascii=False))
        except Exception:
            pass
    if isinstance(rank_home, int) and isinstance(rank_away, int):
        facts.append(f"Classement: {home} rank={rank_home} pts={pts_home} | {away} rank={rank_away} pts={pts_away}")
    elif ranking.get("available") is True and (played_home == 0 or played_away == 0):
        facts.append("Classement: NON INTERPRETABLE — au moins une équipe n'a encore joué aucun match de championnat.")
    if odds:
        facts.append(f"Cotes B365: H={odds.get('B365H')} N={odds.get('B365D')} A={odds.get('B365A')}")
    facts.append(f"Flags: bias_detected={bias_detected} double_chance={double_chance} low_confidence={low_confidence}")
    if ai_action:
        facts.append(
            f"Décision IA interne: action={ai_action} selection={ai_selection or 'NONE'} "
            f"| formulation utilisateur obligatoire: {ai_user_decision}"
        )
    if absences_text:
        facts.append(f"Absences: {absences_text}")
    if top_injuries:
        inj_txt = "; ".join([f"{x.get('team','')}:{x.get('player','')}({x.get('reason','')})" for x in top_injuries[:3]])
        facts.append(f"Top injuries: {inj_txt}")
    if status_short:
        facts.append(f"Status: {status_short} ({status_long}) started={is_started} finished={is_finished} elapsed={elapsed}")
    if upset_score:
        facts.append(f"Upset score: {upset_score} (threshold={upset_threshold})")

    payload = {
        "pred_final": pred_final,          # JSON final complet (source of truth)
        "facts": facts,                   # facts verrouillés anti-hallucination
        "user_profile": user_profile,
    }

    try:
       
        #client = OpenAI(api_key=api_key)
        client =get_openai_client()

        sys_msg = """Tu es un analyste professionnel de football ET un parieur expérimenté.
                    Ton style est celui d’un consultant TV + trader de marché des cotes.

                    Mission :
                    Produire 6 à 9 phrases en français, structurées, claires, avec une vraie prise de position.

                    Règles STRICTES :
                    - Utilise uniquement les données du JSON. N’invente jamais.
                    - Si une donnée manque, dis-le explicitement.
                    - Utilise historical_context comme signal descriptif: H2H, profils domicile/exterieur, domination H2H/bête noire et fiabilite de forme.
                    - Si current_form_reliability est faible, ne presente jamais WMMMM/LMMMM comme une forme robuste.
                    - Pour toute analyse de VALUE BET, utilise EXCLUSIVEMENT market_context calculé en Python. Ne recalcule jamais toi-même probabilité implicite, fair odd, edge ou EV.
                    - Si market_context.best_value == "NONE", ne prétends pas qu'il existe une value bet.
                    - historical_signal est INFORMATION_ONLY en V2.1: il ne remplace pas la prediction du modele.
                    - Analyse la cohérence entre probabilités du modèle et cotes bookmakers, mais NE RECALCULE PAS les probabilités, fair odds, edge ou EV.
                    - market_context est la source de vérité pour la value théorique du modèle. Une value théorique N'EST PAS automatiquement une recommandation de pari.
                    - ai_decision V2.2.2 est la source de vérité pour la décision de pari finale (BET/WATCH/NO_BET) UNIQUEMENT si ai_used=true. Ne la contredis jamais.
                    - decision_confidence = confiance de l'IA dans l'action; selection_confidence = confiance dans la sélection sportive. Ne les confonds pas.
                    - Si ai_decision.action=NO_BET ou WATCH, ne recommande aucun pari simple ou double chance comme décision finale.
                    - Mentionne obligatoirement : probabilités 1/N/2, forme 5 matchs, classement (rank + points),
                    absences (absences_text + top_injuries), risk_level/risk_score, double_chance,
                    bias_detected, low_confidence, statut match (NS/1H/HT/FT).
                    - Si match ≠ NS → préciser que c’est du live/post-match.
                    - IMPORTANT PRESENTATION UTILISATEUR :
                      * Les codes techniques internes BET / WATCH / NO_BET ne doivent JAMAIS apparaître tels quels dans le texte final.
                      * Utilise uniquement la formulation utilisateur fournie dans FACTS.
                      * WATCH = "Aucune prise de position recommandée pour le moment".
                      * NO_BET = "Pari déconseillé".
                      * BET = "Pari recommandé", en précisant la sélection si elle est disponible.
                    - La variable ai_decision est la source de vérité pour la DECISION DE PARI.
                    - market_context.value_bet / best_value décrit seulement une value théorique issue du modèle et ne doit jamais être transformée seule en recommandation.

                    Structure obligatoire :

                    1) Résumé du match + favori.
                    2) Lecture du nul (si ≥25% → "nul non négligeable").
                    3) Classement + écart de points + interprétation.
                    4) Forme récente convertie en bilan (ex: 2 Victoire- 2 Null - 1 Défaite).
                    5) Absences majeures et impact potentiel pour les deux équipes.
                    6) Prédiction du modèle vs côte du marché
                    7) Recommandation EXPERTE :
                    - Niveau de confiance (faible / modéré / élevé)
                    - Gestion de mise (prudente / standard / agressive)
                    - Terminer par "Décision BetSmart :" suivi de la formulation utilisateur de ai_decision, sans afficher les codes techniques internes.

                    Style :
                    - Ton professionnel.
                    - Décision claire.
                    - Pas de blabla.
                    - Conclusion ferme comme un expert parieur.
                    """

        user_msg = (
            "FACTS (à respecter strictement):\n"
            + "\n".join([f"- {x}" for x in facts])
            + "\n\npred_final JSON:\n"
            + json.dumps(pred_final, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=OPENAI_EXPLAIN_MODEL,
            temperature=OPENAI_EXPLAIN_TEMPERATURE,
            max_tokens=OPENAI_EXPLAIN_MAX_TOKENS,
            timeout=OPENAI_EXPLAIN_TIMEOUT,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text_out = (resp.choices[0].message.content or "").strip()
        
        def _one_line(s: str) -> str:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = re.sub(r"\n+", " ", s)      # remplace tous les retours ligne par espace
            s = re.sub(r"\s{2,}", " ", s)   # compact espaces multiples
            return s.strip()
        
        text_out = _one_line(text_out)

        if text_out:
            pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else text_out
            pred_final["explain_llm_used"] = 1
            pred_final["explain_llm_model"] = OPENAI_EXPLAIN_MODEL
            pred_final["explain_llm_error"] = ""
            if LLM_DEBUG:
                pred_final["explain_llm_debug"] = payload
            return pred_final

        # empty => fallback
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "empty_response"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final

    except Exception as e:
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = f"{type(e).__name__}: {e}"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final


def explanation_from_pred_final_________(pred_final: Dict[str, Any], user_profile: str = "standard") -> Dict[str, Any]:
    """
    Prend le JSON FINAL (après apply_unexpected_layer) et renvoie le même JSON
    avec pred_final["explanation"] remplacé (4 à 8 phrases FR).
    - Fallback robuste offline
    - Optionnel LLM OpenAI si OPENAI_EXPLAIN_ENABLED=1 et clé ok
    - Ajoute des metas: explain_llm_used, explain_llm_model, explain_llm_error, explain_llm_debug
    """

    def _is_nan(x: Any) -> bool:
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def _get(d: Dict[str, Any], key: str, default=None):
        try:
            v = d.get(key, default)
            if _is_nan(v):
                return default
            return v
        except Exception:
            return default

    def _pct_from_any(v: Any) -> float:
        """
        Convertit:
          - 0.52 -> 0.52
          - "52%" -> 0.52
          - 52 -> 0.52 (si >1 on suppose %)
        """
        try:
            if v is None or _is_nan(v):
                return 0.0
            if isinstance(v, str):
                s = v.strip().replace(",", ".")
                if not s:
                    return 0.0
                if s.endswith("%"):
                    x = float(s[:-1].strip()) / 100.0
                    return max(0.0, min(1.0, x))
                x = float(s)
                if x > 1.0:
                    x /= 100.0
                return max(0.0, min(1.0, x))
            x = float(v)
            if x > 1.0:
                x /= 100.0
            return max(0.0, min(1.0, x))
        except Exception:
            return 0.0

    def _fmt_pct(x: float) -> str:
        try:
            return f"{round(float(x)*100,1)}%"
        except Exception:
            return "0.0%"

    # -----------------------------
    # Extract from FINAL JSON
    # -----------------------------
    home = str(_get(pred_final, "home", "") or "")
    away = str(_get(pred_final, "away", "") or "")
    match_date = str(_get(pred_final, "match_date", "") or _get(pred_final, "date", "") or "")

    form_home = str(_get(pred_final, "5_dern_perf_home", "") or "")
    form_away = str(_get(pred_final, "5_dern_perf_away", "") or "")

    bias_detected = bool(_get(pred_final, "bias_detected", False) or False)
    low_confidence = bool(_get(pred_final, "low_confidence", False) or False)
    double_chance = _get(pred_final, "double_chance", None)

    # probs: prefer proba_* if present, else p*_raw
    p0 = _pct_from_any(_get(pred_final, "proba_0", None))
    p1 = _pct_from_any(_get(pred_final, "proba_1", None))
    p2 = _pct_from_any(_get(pred_final, "proba_2", None))
    if (p0 + p1 + p2) <= 1e-6:
        p0 = _pct_from_any(_get(pred_final, "p0_raw", 0.0))
        p1 = _pct_from_any(_get(pred_final, "p1_raw", 0.0))
        p2 = _pct_from_any(_get(pred_final, "p2_raw", 0.0))

    # odds if present in pred_final
    odds = {}
    for k in ("B365H", "B365D", "B365A"):
        v = _get(pred_final, k, None)
        try:
            if v is not None and str(v).strip() != "":
                odds[k] = float(str(v).replace(",", "."))
        except Exception:
            pass

    rule_applied = str(_get(pred_final, "rule_applied", "") or "")
    upset_score = float(_get(pred_final, "_upset_score", 0.0) or 0.0)
    upset_threshold = float(_get(pred_final, "_upset_threshold", 0.52) or 0.52)

    # realtime summary
    realtime_risk = _get(pred_final, "realtime_risk", {}) or {}
    summary = {}
    try:
        summary = (realtime_risk or {}).get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
    except Exception:
        summary = {}

    absences_text = str(summary.get("absences_text") or "")
    missing_meta = summary.get("missing_meta") or []
    if not isinstance(missing_meta, list):
        missing_meta = []

    top_injuries = summary.get("top_injuries") or []
    if not isinstance(top_injuries, list):
        top_injuries = []

    ranking = summary.get("ranking") or {}
    if not isinstance(ranking, dict):
        ranking = {}

    rank_home = rank_away = None
    pts_home = pts_away = None
    if ranking.get("available") is True:
        try:
            rh = ranking.get("home") or {}
            ra = ranking.get("away") or {}
            rank_home = rh.get("rank")
            rank_away = ra.get("rank")
            pts_home = rh.get("points")
            pts_away = ra.get("points")
        except Exception:
            pass

    status_short = str(summary.get("status_short") or "")
    status_long = str(summary.get("status_long") or "")
    is_finished = bool(summary.get("is_finished") is True)
    is_started = bool(summary.get("is_started") is True)
    elapsed = summary.get("elapsed")

    # -----------------------------
    # OFFLINE fallback (4-8 phrases)
    # -----------------------------
    def _fallback() -> str:
        lines: List[str] = []

        title = f"{home} vs {away}" if home and away else "Match"
        if match_date:
            title += f" ({match_date})"
        lines.append(f"{title}.")

        if (p0 + p1 + p2) > 1e-6:
            lines.append(f"Probabilités (1/N/2) : {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}.")
        else:
            lines.append("Probabilités (1/N/2) : indisponibles.")

        # favorite
        if (p0 + p1 + p2) > 1e-6:
            fav = "home" if p0 >= max(p1, p2) else ("draw" if p1 >= max(p0, p2) else "away")
            if fav == "home":
                lines.append(f"Lecture modèle : avantage {home} (victoire à domicile).")
            elif fav == "away":
                lines.append(f"Lecture modèle : avantage {away} (victoire à l’extérieur).")
            else:
                lines.append("Lecture modèle : match équilibré (nul plausible).")

        if form_home or form_away:
            lines.append(f"Forme (5 derniers) : {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}.")

        if isinstance(rank_home, int) and isinstance(rank_away, int):
            pts_txt = ""
            if isinstance(pts_home, int) and isinstance(pts_away, int):
                pts_txt = f" ({pts_home} pts vs {pts_away} pts)"
            lines.append(f"Classement : {home} est {rank_home}ᵉ, {away} est {rank_away}ᵉ{pts_txt}.")

        if odds:
            parts = []
            if "B365H" in odds: parts.append(f"H={odds['B365H']}")
            if "B365D" in odds: parts.append(f"N={odds['B365D']}")
            if "B365A" in odds: parts.append(f"A={odds['B365A']}")
            lines.append("Cotes (B365) : " + ", ".join(parts) + ".")

        if double_chance:
            lines.append(f"Double chance : {double_chance} (filet de sécurité).")

        if bias_detected:
            lines.append("Biais de cotes détecté : prudence (effet popularité / surcote possible).")

        if upset_score > 0 and upset_score >= upset_threshold:
            lines.append("Risque de surprise (upset) élevé : éviter les mises agressives.")

        if absences_text:
            lines.append(absences_text.strip())

        # status (live/FT)
        if status_short:
            if is_finished or status_short == "FT":
                lines.append("Note : le match est terminé (infos temps réel post-match).")
            elif is_started:
                lines.append(f"Note : match en cours ({status_short}), minute ≈ {elapsed}.")

        # prudence if missing key live info
        if isinstance(missing_meta, list) and len(missing_meta) > 0:
            lines.append("Certaines données temps réel manquent encore (compos/stats/événements) : prudence avant de valider un pari.")

        return " ".join(lines[:8]).strip()

    fallback_text = _fallback()

    def _should_use_llm() -> bool:
        """
        Active le LLM uniquement pour les cas qui méritent une analyse riche.
        """
        try:
            # 1) cas explicites
            if low_confidence:
                return True
            if bias_detected:
                return True

            # 2) match équilibré
            # - nul important
            # - home et away proches
            # - aucune issue très dominante
            max_p = max(p0, p1, p2)
            min_p = min(p0, p1, p2)
            spread = max_p - min_p

            balanced_match = (
                p1 >= 0.28              # nul significatif
                or abs(p0 - p2) <= 0.10 # home/away proches
                or spread <= 0.12       # distribution serrée
            )

            if balanced_match:
                return True

            return False

        except Exception:
            return False

    # -----------------------------
    # LLM explanation (uses FINAL JSON only)
    # -----------------------------
    if not OPENAI_EXPLAIN_ENABLED:
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "disabled"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": False, "payload": None}
        return pred_final

    # ✅ nouveau : fallback direct si le match n'a pas besoin du LLM
    if not _should_use_llm():
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "skipped_not_needed"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": True, "skipped": True, "payload": None}
        return pred_final

    client = get_openai_client()
    if not client:
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "missing_client"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = {"enabled": True, "has_client": False, "payload": None}
        return pred_final

    # Build "facts" to reduce hallucination
    facts: List[str] = []
    facts.append(f"Match: {home} vs {away} | date={match_date}")
    facts.append(f"Probas 1/N/2: {_fmt_pct(p0)}, {_fmt_pct(p1)}, {_fmt_pct(p2)}")
    if form_home or form_away:
        facts.append(f"Forme(5): {home}={form_home or 'n/a'} ; {away}={form_away or 'n/a'}")
    if isinstance(rank_home, int) and isinstance(rank_away, int):
        facts.append(f"Classement: {home} rank={rank_home} pts={pts_home} | {away} rank={rank_away} pts={pts_away}")
    if odds:
        facts.append(f"Cotes B365: H={odds.get('B365H')} N={odds.get('B365D')} A={odds.get('B365A')}")
    facts.append(f"Flags: bias_detected={bias_detected} double_chance={double_chance} low_confidence={low_confidence}")
    if absences_text:
        facts.append(f"Absences: {absences_text}")
    if top_injuries:
        inj_txt = "; ".join([f"{x.get('team','')}:{x.get('player','')}({x.get('reason','')})" for x in top_injuries[:3]])
        facts.append(f"Top injuries: {inj_txt}")
    if status_short:
        facts.append(f"Status: {status_short} ({status_long}) started={is_started} finished={is_finished} elapsed={elapsed}")
    if upset_score:
        facts.append(f"Upset score: {upset_score} (threshold={upset_threshold})")

    payload = {
        "pred_final": pred_final,          # JSON final complet (source of truth)
        "facts": facts,                   # facts verrouillés anti-hallucination
        "user_profile": user_profile,
    }

    try:
       
        #client = OpenAI(api_key=api_key)
        #client =get_openai_client()

        sys_msg = """Tu es un analyste professionnel de football ET un parieur expérimenté.
                    Ton style est celui d’un consultant TV + trader de marché des cotes.

                    Mission :
                    Produire 6 à 9 phrases en français, structurées, claires, avec une vraie prise de position.

                    Règles STRICTES :
                    - Utilise uniquement les données du JSON. N’invente jamais.
                    - Si une donnée manque, dis-le explicitement.
                    - Analyse la cohérence entre probabilités du modèle et cotes bookmakers, mais NE RECALCULE PAS les probabilités, fair odds, edge ou EV.
                    - market_context est la source de vérité pour la value théorique du modèle. Une value théorique N'EST PAS automatiquement une recommandation de pari.
                    - ai_decision V2.2 est la source de vérité pour la décision de pari finale (BET/WATCH/NO_BET). Ne la contredis jamais.
                    - Si ai_decision.action=NO_BET ou WATCH, ne recommande aucun pari simple ou double chance comme décision finale.
                    - Mentionne obligatoirement : probabilités 1/N/2, forme 5 matchs, classement (rank + points),
                    absences (absences_text + top_injuries), risk_level/risk_score, double_chance,
                    bias_detected, low_confidence, statut match (NS/1H/HT/FT).
                    - Si match ≠ NS → préciser que c’est du live/post-match.

                    Structure obligatoire :

                    1) Résumé du match + favori.
                    2) Lecture du nul (si ≥25% → "nul non négligeable").
                    3) Classement + écart de points + interprétation.
                    4) Forme récente convertie en bilan (ex: 2 Victoire- 2 Null - 1 Défaite).
                    5) Absences majeures et impact potentiel pour les deux équipes.
                    6) Prédiction du modèle vs côte du marché
                    7) Recommandation EXPERTE :
                    - Niveau de confiance (faible / modéré / élevé)
                    - Gestion de mise (prudente / standard / agressive)

                    Style :
                    - Ton professionnel.
                    - Décision claire.
                    - Pas de blabla.
                    - Conclusion ferme comme un expert parieur.
                    """

        user_msg = (
            "FACTS (à respecter strictement):\n"
            + "\n".join([f"- {x}" for x in facts])
            + "\n\npred_final JSON:\n"
            + json.dumps(pred_final, ensure_ascii=False)
        )

        resp = client.chat.completions.create(
            model=OPENAI_EXPLAIN_MODEL,
            temperature=OPENAI_EXPLAIN_TEMPERATURE,
            max_tokens=OPENAI_EXPLAIN_MAX_TOKENS,
            timeout=OPENAI_EXPLAIN_TIMEOUT,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text_out = (resp.choices[0].message.content or "").strip()
        
        def _one_line(s: str) -> str:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = re.sub(r"\n+", " ", s)      # remplace tous les retours ligne par espace
            s = re.sub(r"\s{2,}", " ", s)   # compact espaces multiples
            return s.strip()
        
        text_out = _one_line(text_out)

        if text_out:
            pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else text_out
            pred_final["explain_llm_used"] = 1
            pred_final["explain_llm_model"] = OPENAI_EXPLAIN_MODEL
            pred_final["explain_llm_error"] = ""
            if LLM_DEBUG:
                pred_final["explain_llm_debug"] = payload
            return pred_final

        # empty => fallback
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = "empty_response"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final

    except Exception as e:
        pred_final["explanation"] = pred_final.get("explanation") if pred_final.get("_explanation_locked_v233") else fallback_text
        pred_final["explain_llm_used"] = 0
        pred_final["explain_llm_model"] = ""
        pred_final["explain_llm_error"] = f"{type(e).__name__}: {e}"
        if LLM_DEBUG:
            pred_final["explain_llm_debug"] = payload
        return pred_final




_V236_PUBLIC_KEYS = (
    "5_dern_perf_away",
    "5_dern_perf_home",
    "_upset_score",
    "_upset_threshold",
    "_use_realtime",
    "away",
    "bias_detected",
    "double_chance",
    "explanation",
    "home",
    "low_confidence",
    "mess_but",
    "plus_but",
    "p0_raw",
    "p1_raw",
    "p2_raw",
    "prediction",
    "prediction_model",
    "proba_0",
    "proba_1",
    "proba_2",
    "value_quality",
    "rule_applied",
)


def _v236_full_view(pred_final: dict) -> dict:
    """
    Vue FULL d'un pred_final DEJA CALCULE.
    Aucun appel Web, modèle ML ou LLM n'est effectué ici.
    """
    if not isinstance(pred_final, dict):
        raise ValueError("pred_final doit être un dictionnaire")

    excluded = {
        "explain_llm_debug",
        "explain_llm_error",
        "explain_llm_used",
        "explain_llm_model",
    }
    return to_serializable({
        key: value for key, value in pred_final.items()
        if key not in excluded
    })


def _v236_reduced_view(pred_final: dict) -> dict:
    """
    Vue REDUCED du MEME pred_final.
    C'est une projection pure: zéro recalcul, zéro Web, zéro GPT.
    """
    if not isinstance(pred_final, dict):
        raise ValueError("pred_final doit être un dictionnaire")

    return to_serializable({
        key: pred_final.get(key)
        for key in _V236_PUBLIC_KEYS
    })


def build_output_views_v236(pred_final: dict) -> dict:
    """
    Produit FULL + REDUCED dans le MEME appel depuis le MEME objet.
    C'est la fonction de référence pour comparer les deux formats.

    Usage:
        views = build_output_views_v236(pred_final)
        full_json = views["full"]
        reduced_json = views["reduced"]
    """
    full_view = _v236_full_view(pred_final)
    reduced_view = _v236_reduced_view(pred_final)

    # Contrôle de cohérence obligatoire entre les champs publics.
    mismatches = {}
    for key in _V236_PUBLIC_KEYS:
        fv = full_view.get(key)
        rv = reduced_view.get(key)
        if fv != rv:
            mismatches[key] = {"full": fv, "reduced": rv}

    if mismatches:
        raise RuntimeError(
            "V2.3.6 output consistency violation: "
            + json.dumps(mismatches, ensure_ascii=False, default=str)
        )

    return {
        "version": "2.3.7.1",
        "decision_id": _v236_decision_id(pred_final),
        "full": full_view,
        "reduced": reduced_view,
    }


def _v236_decision_id(pred_final: dict) -> str:
    """
    Empreinte stable de la décision finale permettant de prouver que
    FULL et REDUCED proviennent du même pred_final.
    """
    payload = {
        "home": pred_final.get("home"),
        "away": pred_final.get("away"),
        "prediction": pred_final.get("prediction"),
        "p0_raw": pred_final.get("p0_raw"),
        "p1_raw": pred_final.get("p1_raw"),
        "p2_raw": pred_final.get("p2_raw"),
        "proba_0": pred_final.get("proba_0"),
        "proba_1": pred_final.get("proba_1"),
        "proba_2": pred_final.get("proba_2"),
        "double_chance": pred_final.get("double_chance"),
        "rule_applied": pred_final.get("rule_applied"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    import hashlib
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def validate_output_consistency_v236(pred_final: dict) -> dict:
    """
    Validation explicite utilisable dans test_paris.py.
    """
    views = build_output_views_v236(pred_final)
    return {
        "ok": True,
        "version": "2.3.7.1",
        "decision_id": views["decision_id"],
        "prediction": views["full"].get("prediction"),
        "probas": [
            views["full"].get("p0_raw"),
            views["full"].get("p1_raw"),
            views["full"].get("p2_raw"),
        ],
    }



def _v2371_prediction_from_probs(p0, p1, p2):
    return int(np.argmax([float(p0), float(p1), float(p2)]))


def _v2371_double_chance_from_final(prediction: int, p0: float, p1: float, p2: float) -> str:
    """
    Double chance cohérente avec la décision finale BetSmart.
    HOME -> 1X
    AWAY -> X2
    DRAW -> couverture avec le second scénario le plus probable.
    """
    prediction = int(prediction)
    if prediction == 0:
        return "1X"
    if prediction == 2:
        return "X2"
    return "1X" if float(p0) >= float(p2) else "X2"


def _v2371_final_sync(pred_final: dict) -> dict:
    """
    Garantit un état final unique et cohérent pour chaque match.
    """
    out = dict(pred_final or {})

    # Prefer final formatted probabilities when they exist because they may
    # already include early-season fusion / AI stabilization.
    def from_pct(key):
        try:
            return float(_v211_prob_from_any(out.get(key)))
        except Exception:
            return None

    fp = [from_pct("proba_0"), from_pct("proba_1"), from_pct("proba_2")]

    if all(v is not None for v in fp):
        arr = _v223_normalize_probs(fp)
    else:
        vals = []
        for key in ("p0_raw", "p1_raw", "p2_raw"):
            try:
                vals.append(float(out.get(key)))
            except Exception:
                vals.append(1/3)
        arr = _v223_normalize_probs(vals)

    p0, p1, p2 = map(float, arr)
    prediction = _v2371_prediction_from_probs(p0, p1, p2)

    out["p0_raw"] = p0
    out["p1_raw"] = p1
    out["p2_raw"] = p2

    out["proba_0"] = _format_pct(p0)
    out["proba_1"] = _format_pct(p1)
    out["proba_2"] = _format_pct(p2)

    out["prediction"] = prediction
    out["prediction_model"] = prediction
    out["double_chance"] = _v2371_double_chance_from_final(prediction, p0, p1, p2)

    ai = out.get("ai_decision_v23") or out.get("ai_decision") or {}
    if isinstance(ai, dict):
        try:
            conf = float(ai.get("prediction_confidence", 0.0) or 0.0)
            out["low_confidence"] = bool(conf < 0.60)
        except Exception:
            pass

    if not str(out.get("explanation") or "").strip():
        selection_fr = {
            0: f"victoire de {out.get('home')}",
            1: "match nul",
            2: f"victoire de {out.get('away')}",
        }[prediction]
        out["explanation"] = (
            f"Après consolidation de toutes les informations disponibles, "
            f"BetSmart estime les probabilités finales à "
            f"{p0*100:.1f}% pour {out.get('home')}, "
            f"{p1*100:.1f}% pour le nul et "
            f"{p2*100:.1f}% pour {out.get('away')}. "
            f"Décision BetSmart : {selection_fr}."
        )

    ra = str(out.get("rule_applied") or "")
    tag = "v2371_final_state"
    if tag not in ra:
        out["rule_applied"] = f"{ra}|{tag}" if ra else tag

    out["_final_state_guaranteed"] = True
    out["_final_state_version"] = "2.3.7.1"

    assert out["prediction"] == int(np.argmax([out["p0_raw"], out["p1_raw"], out["p2_raw"]]))
    assert out["proba_0"] == _format_pct(out["p0_raw"])
    assert out["proba_1"] == _format_pct(out["p1_raw"])
    assert out["proba_2"] == _format_pct(out["p2_raw"])
    if out["prediction"] == 0:
        assert out["double_chance"] == "1X"
    elif out["prediction"] == 2:
        assert out["double_chance"] == "X2"

    return out


def _v2371_finalize_results_container(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    results = out.get("Resultats")
    if isinstance(results, list):
        out["Resultats"] = [
            _v2371_final_sync(item) if isinstance(item, dict) else item
            for item in results
        ]
    return out


def clean_extract_final_result(pred_final: dict) -> dict:
    """
    V2.3.7.1 REDUCED: projection pure après Guaranteed Final State.
    """
    pred_final = _v2371_final_sync(pred_final)
    return _v236_reduced_view(pred_final)


## DC
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

    # V2.2.1 FORM GUARD -------------------------------------------------
    # Neutralise aussi le déplacement des probabilités par la forme tant que
    # la saison courante n'est pas suffisamment mature.
    try:
        home_played = int(float(features_df.get("HomeMatchesPlayedCurrent", pd.Series([0])).values[0]))
        away_played = int(float(features_df.get("AwayMatchesPlayedCurrent", pd.Series([0])).values[0]))
        form_rel = float(features_df.get("CurrentFormReliability", pd.Series([0.0])).values[0])
    except Exception:
        home_played = away_played = 0
        form_rel = 0.0

    try:
        _params_guard = _get_params(league_code)
    except Exception:
        _params_guard = {}
    min_matches_guard = int(_params_guard.get("form_override_min_matches", 6))
    min_rel_guard = float(_params_guard.get("form_override_min_reliability", 0.80))

    if (home_played < min_matches_guard or away_played < min_matches_guard or form_rel < min_rel_guard):
        return p0, p1, p2, {
            "form_gate": "skipped_insufficient_form_reliability",
            "home_matches_played": home_played,
            "away_matches_played": away_played,
            "form_reliability": round(form_rel, 3),
            "min_matches_required": min_matches_guard,
            "min_reliability_required": min_rel_guard,
        }

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
    league_code="default",
) -> dict:
    """
    ✅ LOGIQUE BETSMART (verrouillée) — version stable
    + Anti-0%: floor+renormalize appliqué en FIN de pipeline (draw ET non-draw)
    + REALTIME (info only) sans impacter la prédiction
    """

    (bookmaker_margin, uncertainty_threshold, importance, season_stage,
     upset_threshold, skip_threshold, bogey_weight, gki_weight) = _safe_parametres(league_code)

    # ---- params ligue / config ----
    try:
        params = _get_params(league_code)
    except Exception:
        params = {}

    form_pick_threshold = float(params.get("form_pick_threshold", 0.20))
    # V2.2.1 FORM GUARD : la forme ne peut renverser la décision qu'après
    # PLUS de 5 matchs joués par CHAQUE équipe (minimum 6 par défaut),
    # et uniquement si sa fiabilité atteint le seuil minimal.
    form_override_min_matches = int(params.get("form_override_min_matches", 6))
    form_override_min_reliability = float(params.get("form_override_min_reliability", 0.80))
    strong_conf_threshold = float(params.get("strong_conf_threshold", 0.70))
    strong_conf_draw_cap = float(params.get("strong_conf_draw_cap", 0.12))
    dc_disable_if_strong_conf = bool(params.get("dc_disable_if_strong_conf", True))

    # ✅ nouveau: floor proba (évite 0% / 100% strict)
    min_prob_floor = float(params.get("min_prob_floor", 0.01))  # 1% par défaut

    # ---- util explication ---
    
    def _explain(rule_tag, p0, p1, p2, extra=None):
        f = features_df.copy()

        # IMPORTANT: expose les probas au format attendu
        f["p0_raw"] = float(p0)
        f["p1_raw"] = float(p1)
        f["p2_raw"] = float(p2)

        
        if isinstance(extra, dict):
            for k, v in extra.items():
                try:
                    # ✅ dict/list => forcer "scalaire" en cellule, pas alignement par index
                    if isinstance(v, (dict, list)):
                        if len(f) == 1:
                            f.at[f.index[0], k] = v
                        else:
                            f[k] = [v] * len(f)
                    else:
                        f[k] = v
                except Exception:
                    pass

        
        
        assert "realtime_risk" in f.columns, "realtime_risk missing at explain-time"
        text = generate_explanation(rule_tag, f, user_profile)
        
        

        # ✅ PATCH: copier les metas LLM dans features_df pour que _notes_llm_debug(features_df) marche
        for col in ["_llm_used", "_llm_mode", "_llm_model", "_llm_error", "_llm_debug"]:
            try:
                if col in f.columns:
                    features_df.loc[:, col] = f[col].values[0]
            except Exception:
                pass

        return text

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

    # ✅ floor + renormalize (anti 0% / 100%)
    def _clip_and_normalize_probs(p0, p1, p2, *, min_prob=0.01):
        p = np.array([float(p0), float(p1), float(p2)], dtype=float)
        if not np.isfinite(p).all():
            p = np.array([1/3, 1/3, 1/3], dtype=float)

        # clip
        min_prob = float(min_prob)
        min_prob = max(0.0, min(min_prob, 0.10))  # sécurité
        p = np.clip(p, min_prob, 1.0 - min_prob)

        # renormalize
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            p = np.array([1/3, 1/3, 1/3], dtype=float)
        else:
            p = p / s
        return float(p[0]), float(p[1]), float(p[2])
    
    def _notes_llm_debug(df):
        try:
            import pandas as pd
            if not isinstance(df, pd.DataFrame) or df.empty:
                return []
            used = df.get("_llm_used")
            err = df.get("_llm_error")
            model = df.get("_llm_model")
            mode = df.get("_llm_mode")

            used_v = used.values[0] if hasattr(used, "values") else used
            err_v = err.values[0] if hasattr(err, "values") else err
            model_v = model.values[0] if hasattr(model, "values") else model
            mode_v = mode.values[0] if hasattr(mode, "values") else mode

            if str(used_v) == "1":
                return [f"explain: llm_used=1 model={model_v}"]
            else:
                if err_v:
                    return [f"explain: llm_used=0 err={str(err_v)[:160]}"]
                return [f"explain: llm_used=0 mode={mode_v}"]
        except Exception:
            return []
    ### nouvels ajouts
    def soften_probs_temperature(p0, p1, p2, T=1.6, eps=1e-12):
        p = np.array([float(p0), float(p1), float(p2)], dtype=float)
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()

        # log-softmax avec température
        logits = np.log(p + eps)
        logits = logits / float(T)
        exp = np.exp(logits - np.max(logits))
        q = exp / exp.sum()
        return float(q[0]), float(q[1]), float(q[2])
    
    def shrink_to_prior(p0, p1, p2, alpha=0.15, prior=(1/3, 1/3, 1/3)):
        # alpha = part de prior (0.10 à 0.30 en pratique)
        p = np.array([p0, p1, p2], dtype=float)
        p = p / p.sum()
        pr = np.array(prior, dtype=float)
        pr = pr / pr.sum()
        q = (1 - alpha) * p + alpha * pr
        q = q / q.sum()
        return float(q[0]), float(q[1]), float(q[2])

    def clip_probs(p0, p1, p2, min_p=0.05, max_p=0.90):
        p = np.array([p0, p1, p2], dtype=float)
        p = p / p.sum()
        p = np.clip(p, float(min_p), float(max_p))
        p = p / p.sum()
        return float(p[0]), float(p[1]), float(p[2])

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

        # ✅ anti-0%
        p0_raw, p1_raw, p2_raw = float(p0), float(p1), float(p2)
        p0, p1, p2 = _clip_and_normalize_probs(p0, p1, p2, min_prob=min_prob_floor)
        # adoucissement
        p0, p1, p2 = soften_probs_temperature(p0, p1, p2, T=1.6)

        # shrink léger
        p0, p1, p2 = shrink_to_prior(p0, p1, p2, alpha=0.12)

        pred_final = LABEL_DRAW
        dc = detect_double_chance(p0, p1, p2, pred_final, league_code)

        
        rt_block, rt_note = _build_realtime_block(features_df, league_code=league_code)

        explanation_text = _explain(
            "threshold",
            p0, p1, p2,
            extra={
                "form_gate_meta": str(meta_gate),
                "double_chance": dc,
                "realtime_risk": rt_block,
            }
        )
        
        debug_payload = None
        try:
            if bool(os.getenv("LLM_DEBUG", "").strip() in ("1","true","True")) and "_llm_debug" in features_df.columns:
                debug_payload = features_df["_llm_debug"].values[0]
        except Exception:
            pass

        notes = []
        if rt_note:
            notes.append(rt_note)
        
        notes += _notes_llm_debug(features_df)
        
        return {
            "prediction": int(pred_final),
            "prediction_model": LABEL_DRAW,
            "proba_0": _format_pct(p0),
            "proba_1": _format_pct(p1),
            "proba_2": _format_pct(p2),
            "p0_raw": p0_raw, "p1_raw": p1_raw, "p2_raw": p2_raw,
            "rule_applied": "threshold|draw_dominant|form_gate",
            #"explanation": _explain("threshold", p0, p1, p2, extra={"form_gate_meta": str(meta_gate)}),
            "explanation": generate_explanation("margin_adjusted", features_df, user_profile),
            "double_chance": dc,
            "realtime_risk": rt_block,
            "notes": notes,
            "llm_debug": debug_payload,
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

    # strong conf draw cap
    strong_side = max(float(p0), float(p2))
    strong_conf = (strong_side >= float(strong_conf_threshold))
    strong_tag = None
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

    # ✅ anti-0% (IMPORTANT: après TOUS les gates/caps)
    p0_raw, p1_raw, p2_raw = float(p0), float(p1), float(p2)
    p0, p1, p2 = _clip_and_normalize_probs(p0, p1, p2, min_prob=min_prob_floor)
    
     # adoucissement
    p0, p1, p2 = soften_probs_temperature(p0, p1, p2, T=1.6)

        # shrink léger
    p0, p1, p2 = shrink_to_prior(p0, p1, p2, alpha=0.12)

    pred_final = LABEL_HOME if float(p0) >= float(p2) else LABEL_AWAY

    # marché / forme override tag (NE change pas les probas, uniquement la décision)
    fav_side, fav_gap, dc_market = _market_fav_and_dc()

    try:
        home_form = float(features_df["HomeForm"].values[0])
        away_form = float(features_df["AwayForm"].values[0])
        form_diff = home_form - away_form
    except Exception:
        home_form = away_form = form_diff = 0.0

    override_tag = None
    dc_override = None

    # V2.2.1 FORM GUARD : conditions cumulatives avant renversement par la forme.
    try:
        home_played = int(float(features_df.get("HomeMatchesPlayedCurrent", pd.Series([0])).values[0]))
        away_played = int(float(features_df.get("AwayMatchesPlayedCurrent", pd.Series([0])).values[0]))
        form_reliability = float(features_df.get("CurrentFormReliability", pd.Series([0.0])).values[0])
    except Exception:
        home_played = away_played = 0
        form_reliability = 0.0

    form_override_allowed = (
        home_played >= form_override_min_matches
        and away_played >= form_override_min_matches
        and form_reliability >= form_override_min_reliability
    )

    if abs(float(form_diff)) >= float(form_pick_threshold):
        form_side = "home" if form_diff > 0 else "away"
        if fav_side is not None and form_side != fav_side:
            if form_override_allowed:
                pred_final = LABEL_HOME if form_side == "home" else LABEL_AWAY
                override_tag = "form_over_market_pick_" + ("home" if form_side == "home" else "away")
                dc_override = "1X" if form_side == "home" else "X2"
            else:
                override_tag = (
                    "form_override_blocked_insufficient_history"
                    f"_h{home_played}_a{away_played}_r{form_reliability:.2f}"
                )
            
            

    bd = detect_bias(features_df)
    if isinstance(bd, pd.Series):
        bias_detected = bool(bd.iloc[0])
    elif isinstance(bd, (list, tuple, np.ndarray)):
        bias_detected = bool(np.any(bd))
    else:
        bias_detected = bool(bd)
        
    low_confidence = bool(is_confidence_low(p0, p1, p2))

    #dc = detect_double_chance(p0, p1, p2, pred_final, league_code)
    dc = detect_double_chance_v2(
        p0, p1, p2, pred_final,
        league_code=league_code,
        bias_detected=bias_detected,
        low_confidence=low_confidence,
        upset_score=float(_safe_get_first(features_df, "_upset_score") or 0.0),
        upset_threshold=float(_safe_get_first(features_df, "_upset_threshold") or 0.52),
        override_tag=override_tag
    )
    if dc_override is not None:
        dc = dc_override

    if strong_conf and dc_disable_if_strong_conf and (not bias_detected) and (not low_confidence):
        dc = None

    if (bias_detected or low_confidence) and dc is None:
        dc = "1X" if pred_final == LABEL_HOME else "X2"
    
    if dc == "1X" and pred_final == LABEL_AWAY:
        dc = "X2"

    if dc == "X2" and pred_final == LABEL_HOME:
        dc = "1X"

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
   
    rt_block, rt_note = _build_realtime_block(features_df, league_code=league_code)

    extra = {
        "bias_detected": int(bias_detected),
        "low_confidence": int(low_confidence),
        "form_gate_meta": str(meta_gate),
        "strong_conf": int(bool(strong_conf)),
        "fav_side": str(fav_side),
        "fav_gap": float(fav_gap),
        "form_diff": float(form_diff),
        "double_chance": dc,

        # ✅ TRÈS IMPORTANT : passer le realtime au générateur d'explication
        "realtime_risk": rt_block,
    }

    # ✅ ensuite explication (maintenant elle voit summary/ranking/absences)
    explanation_text = _explain("rf_decision", p0, p1, p2, extra=extra)
    
    debug_payload = None
    try:
        if bool(os.getenv("LLM_DEBUG", "").strip() in ("1","true","True")) and "_llm_debug" in features_df.columns:
            debug_payload = features_df["_llm_debug"].values[0]
    except Exception:
        pass
    
    
    notes = []
    if rt_note:
        notes.append(rt_note)
    
    notes += _notes_llm_debug(features_df)

    return {
        "prediction": int(pred_final),
        "prediction_model": prediction_rf,
        "proba_0": _format_pct(p0),
        "proba_1": _format_pct(p1),
        "proba_2": _format_pct(p2),
        "p0_raw": p0_raw, "p1_raw": p1_raw, "p2_raw": p2_raw,
        "rule_applied": rule_applied,
        #"explanation": _explain("rf_decision", p0, p1, p2, extra=extra),
        "explanation": explanation_text,
        "double_chance": dc,
        "bias_detected": bias_detected,
        "low_confidence": low_confidence,
        "realtime_risk": rt_block,
        "notes": notes,
        "llm_debug": debug_payload,
    }



# =========================
# BETSMART V2.1 - HISTORICAL INTELLIGENCE (information only)
# =========================
# Cette couche exploite enfin season_past_list sans modifier la prediction 1N2.
# Elle produit un contexte historique structuré destiné au debug, au LLM
# d'explication et aux futurs backtests V2.2/V2.3.

def _v21_pick_col(df, candidates):
    if df is None or not hasattr(df, "columns"):
        return None
    cols = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        hit = cols.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _v21_norm_name(x):
    try:
        return normalize_team_name(str(x))
    except Exception:
        return str(x or "").strip().lower()


def _v21_prepare_matches(df, match_date=None):
    """Retourne une copie normalisee et, si possible, limitee aux matchs anterieurs."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out = df.copy()
    hc = _v21_pick_col(out, ["HomeTeam", "home", "Home", "home_name"])
    ac = _v21_pick_col(out, ["AwayTeam", "away", "Away", "away_name"])
    dc = _v21_pick_col(out, ["Date", "date", "match_Date", "match_date", "MatchDate"])
    if hc is None or ac is None:
        return None
    out["__home_norm"] = out[hc].astype(str).map(_v21_norm_name)
    out["__away_norm"] = out[ac].astype(str).map(_v21_norm_name)
    if dc is not None:
        out["__date_v21"] = pd.to_datetime(out[dc], errors="coerce", dayfirst=True)
        try:
            cutoff = pd.to_datetime(match_date, errors="coerce")
            if pd.notna(cutoff):
                out = out[(out["__date_v21"].isna()) | (out["__date_v21"] < cutoff)]
        except Exception:
            pass
        out = out.sort_values("__date_v21", na_position="first")
    return out


def _v21_result_for_team(row, team_norm, home_col, away_col, result_col, hg_col, ag_col):
    """W/D/L vu du point de vue de team_norm."""
    try:
        h = _v21_norm_name(row[home_col])
        a = _v21_norm_name(row[away_col])
        # Priorite au score, plus robuste si FTR absent
        if hg_col is not None and ag_col is not None:
            hg = float(row[hg_col]); ag = float(row[ag_col])
            if hg == ag:
                return "D"
            home_won = hg > ag
            if team_norm == h:
                return "W" if home_won else "L"
            if team_norm == a:
                return "L" if home_won else "W"
        if result_col is not None:
            r = str(row[result_col]).strip().upper()
            if r in ("D", "DRAW", "N"):
                return "D"
            if r in ("H", "HOME", "1"):
                return "W" if team_norm == h else "L"
            if r in ("A", "AWAY", "2"):
                return "L" if team_norm == h else "W"
    except Exception:
        return None
    return None


def _v21_team_form(df, team, match_date=None, n=5):
    prep = _v21_prepare_matches(df, match_date)
    if prep is None:
        return {"pattern": "M" * n, "available_matches": 0, "reliability": 0.0, "reliability_label": "VERY_LOW"}
    hc = _v21_pick_col(prep, ["HomeTeam", "home", "Home", "home_name"])
    ac = _v21_pick_col(prep, ["AwayTeam", "away", "Away", "away_name"])
    rc = _v21_pick_col(prep, ["FTR", "Result", "result", "FullTimeResult"])
    hgc = _v21_pick_col(prep, ["FTHG", "HomeGoals", "home_goals", "goals_home"])
    agc = _v21_pick_col(prep, ["FTAG", "AwayGoals", "away_goals", "goals_away"])
    tn = _v21_norm_name(team)
    rows = prep[(prep["__home_norm"] == tn) | (prep["__away_norm"] == tn)].tail(n)
    vals = []
    for _, row in rows.iterrows():
        r = _v21_result_for_team(row, tn, hc, ac, rc, hgc, agc)
        if r:
            vals.append(r)
    vals = vals[-n:]
    available = len(vals)
    pattern = "".join(vals) + ("M" * max(0, n - available))
    rel = min(1.0, available / float(n))
    label = "VERY_LOW" if rel <= 0.2 else ("LOW" if rel <= 0.4 else ("MEDIUM" if rel <= 0.6 else ("HIGH" if rel < 1.0 else "FULL")))
    return {"pattern": pattern, "available_matches": available, "reliability": round(rel, 3), "reliability_label": label}


def _v21_team_profile(df, team, match_date=None, venue=None):
    prep = _v21_prepare_matches(df, match_date)
    if prep is None:
        return {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "win_rate": 0.0, "points_per_game": 0.0}
    hc = _v21_pick_col(prep, ["HomeTeam", "home", "Home", "home_name"])
    ac = _v21_pick_col(prep, ["AwayTeam", "away", "Away", "away_name"])
    rc = _v21_pick_col(prep, ["FTR", "Result", "result", "FullTimeResult"])
    hgc = _v21_pick_col(prep, ["FTHG", "HomeGoals", "home_goals", "goals_home"])
    agc = _v21_pick_col(prep, ["FTAG", "AwayGoals", "away_goals", "goals_away"])
    tn = _v21_norm_name(team)
    if venue == "home":
        rows = prep[prep["__home_norm"] == tn]
    elif venue == "away":
        rows = prep[prep["__away_norm"] == tn]
    else:
        rows = prep[(prep["__home_norm"] == tn) | (prep["__away_norm"] == tn)]
    w = d = l = 0
    gf = ga = 0.0
    goals_n = 0
    for _, row in rows.iterrows():
        r = _v21_result_for_team(row, tn, hc, ac, rc, hgc, agc)
        if r == "W": w += 1
        elif r == "D": d += 1
        elif r == "L": l += 1
        try:
            if hgc is not None and agc is not None:
                hg=float(row[hgc]); ag=float(row[agc])
                if _v21_norm_name(row[hc]) == tn:
                    gf += hg; ga += ag
                else:
                    gf += ag; ga += hg
                goals_n += 1
        except Exception:
            pass
    n = w+d+l
    return {
        "matches": int(n), "wins": int(w), "draws": int(d), "losses": int(l),
        "win_rate": round(w/n, 3) if n else 0.0,
        "draw_rate": round(d/n, 3) if n else 0.0,
        "loss_rate": round(l/n, 3) if n else 0.0,
        "points_per_game": round((3*w+d)/n, 3) if n else 0.0,
        "goals_for_avg": round(gf/goals_n, 3) if goals_n else None,
        "goals_against_avg": round(ga/goals_n, 3) if goals_n else None,
    }


def _v21_h2h(dfs, home, away, match_date=None, max_matches=10):
    """Profil H2H récent, avec domination et signal prudent de « bête noire »."""
    hn, an = _v21_norm_name(home), _v21_norm_name(away)
    records = []
    for df in (dfs or []):
        prep = _v21_prepare_matches(df, match_date)
        if prep is None:
            continue
        hc = _v21_pick_col(prep, ["HomeTeam", "home", "Home", "home_name"])
        ac = _v21_pick_col(prep, ["AwayTeam", "away", "Away", "away_name"])
        rc = _v21_pick_col(prep, ["FTR", "Result", "result", "FullTimeResult"])
        hgc = _v21_pick_col(prep, ["FTHG", "HomeGoals", "home_goals", "goals_home"])
        agc = _v21_pick_col(prep, ["FTAG", "AwayGoals", "away_goals", "goals_away"])
        dc = "__date_v21" if "__date_v21" in prep.columns else None
        mask = (((prep["__home_norm"] == hn) & (prep["__away_norm"] == an)) |
                ((prep["__home_norm"] == an) & (prep["__away_norm"] == hn)))
        for _, row in prep[mask].iterrows():
            rh = _v21_result_for_team(row, hn, hc, ac, rc, hgc, agc)
            rec = {"home_team_result": rh}
            if dc:
                try: rec["date"] = str(pd.to_datetime(row[dc]).date()) if pd.notna(row[dc]) else None
                except Exception: rec["date"] = None
            try:
                if hgc is not None and agc is not None:
                    actual_h = _v21_norm_name(row[hc]); hg=float(row[hgc]); ag=float(row[agc])
                    if actual_h == hn:
                        rec["goals_home"] = hg; rec["goals_away"] = ag
                    else:
                        rec["goals_home"] = ag; rec["goals_away"] = hg
            except Exception:
                pass
            records.append(rec)

    seen=set(); unique=[]
    for r in records:
        key=(r.get("date"),r.get("goals_home"),r.get("goals_away"),r.get("home_team_result"))
        if key not in seen:
            seen.add(key); unique.append(r)
    unique=sorted(unique,key=lambda r:r.get("date") or "")[-max_matches:]

    n=len(unique)
    hw=sum(1 for r in unique if r.get("home_team_result")=="W")
    dd=sum(1 for r in unique if r.get("home_team_result")=="D")
    aw=sum(1 for r in unique if r.get("home_team_result")=="L")
    gh=sum(float(r.get("goals_home",0) or 0) for r in unique)
    ga=sum(float(r.get("goals_away",0) or 0) for r in unique)
    home_wr=hw/n if n else 0.0; away_wr=aw/n if n else 0.0; draw_r=dd/n if n else 0.0
    gdpm=(gh-ga)/n if n else 0.0

    rh=ra=rd=0.0
    if n:
        weights=np.arange(1,n+1,dtype=float); weights=weights/weights.sum()
        for w,r in zip(weights,unique):
            if r.get("home_team_result")=="W": rh+=float(w)
            elif r.get("home_team_result")=="L": ra+=float(w)
            else: rd+=float(w)

    sample=min(1.0,n/6.0)
    win_edge=abs(home_wr-away_wr)
    goal_strength=min(1.0,abs(gdpm)/1.5) if n else 0.0
    recency_edge=abs(rh-ra)
    dom_score=float(np.clip(0.45*win_edge+0.20*goal_strength+0.20*sample+0.15*recency_edge,0,1))

    dominant_side=None
    if n>=3 and dom_score>=0.60:
        if home_wr>=0.60 and home_wr>away_wr: dominant_side="HOME"
        elif away_wr>=0.60 and away_wr>home_wr: dominant_side="AWAY"
    dominant_team=home if dominant_side=="HOME" else (away if dominant_side=="AWAY" else None)
    dominated_team=away if dominant_side=="HOME" else (home if dominant_side=="AWAY" else None)
    if n==0: conf="NONE"
    elif n<=2: conf="LOW"
    elif n<=4: conf="MEDIUM"
    elif n<=6: conf="HIGH"
    else: conf="VERY_HIGH"

    return {
        "matches":n,"home_wins":hw,"draws":dd,"away_wins":aw,
        "home_win_rate":round(home_wr,3),"draw_rate":round(draw_r,3),"away_win_rate":round(away_wr,3),
        "home_goals":round(gh,2),"away_goals":round(ga,2),
        "goal_difference_home_view":round(gh-ga,2),"goals_diff_per_match":round(gdpm,3),
        "recency_home_share":round(rh,3),"recency_draw_share":round(rd,3),"recency_away_share":round(ra,3),
        "h2h_confidence":conf,"dominance_score":round(dom_score,3),
        "dominant_side":dominant_side,"dominant_team":dominant_team,"dominated_team":dominated_team,
        "black_beast_signal":bool(dominant_side),
        "recent_results_home_view":[r.get("home_team_result") for r in unique],"recent_matches":unique,
    }


def _v211_historical_signal(home_hist, away_hist, h2h):
    """Signal descriptif HOME/AWAY/1X/X2/DRAW_LEAN/NEUTRAL. INFORMATION_ONLY."""
    hw=float(home_hist.get("win_rate",0) or 0); hd=float(home_hist.get("draw_rate",0) or 0); hl=float(home_hist.get("loss_rate",0) or 0)
    aw=float(away_hist.get("win_rate",0) or 0); ad=float(away_hist.get("draw_rate",0) or 0); al=float(away_hist.get("loss_rate",0) or 0)
    hm=int(home_hist.get("matches",0) or 0); am=int(away_hist.get("matches",0) or 0); hn=int(h2h.get("matches",0) or 0)
    if hm<5 or am<5:
        return "NEUTRAL",0.25
    volume=min(1.0,min(hm,am)/30.0); h2hc=min(1.0,hn/6.0); dom=float(h2h.get("dominance_score",0) or 0)
    conf=0.55*volume+0.25*h2hc+0.20*dom
    side=h2h.get("dominant_side")
    if side=="HOME" and hl<=0.25 and aw<=0.35: return "1X",min(0.95,conf+0.10)
    if side=="AWAY" and al<=0.25 and hw<=0.35: return "X2",min(0.95,conf+0.10)
    if hw>=0.50 and al>=0.50: return "HOME",min(0.90,conf+0.08)
    if aw>=0.45 and hl>=0.45: return "AWAY",min(0.90,conf+0.08)
    if hd>=0.45 and aw<=0.30: return "1X",min(0.88,conf)
    if ad>=0.45 and hw<=0.30: return "X2",min(0.88,conf)
    hp=float(home_hist.get("points_per_game",0) or 0); ap=float(away_hist.get("points_per_game",0) or 0); delta=hp-ap
    if delta>=0.45: return "1X",min(0.85,conf)
    if delta<=-0.45: return "X2",min(0.85,conf)
    if abs(delta)<=0.15 and (hd+ad)/2>=0.32: return "DRAW_LEAN",min(0.75,conf)
    return "NEUTRAL",min(0.70,conf)


def build_historical_profile(home, away, season_current_df=None, season_past_list=None, match_date=None, lookback_seasons=3):
    """BetSmart V2.1.1 : H2H + profils venue + forme actuelle; n'altère jamais prediction."""
    past=[d for d in (season_past_list or []) if isinstance(d,pd.DataFrame) and not d.empty]
    past=past[-int(max(1,lookback_seasons)):]
    current_home_form=_v21_team_form(season_current_df,home,match_date,5)
    current_away_form=_v21_team_form(season_current_df,away,match_date,5)
    weights=list(range(1,len(past)+1))
    def weighted_profile(team,venue):
        profiles=[]
        for w,df in zip(weights,past):
            pr=_v21_team_profile(df,team,match_date,venue=venue)
            if pr.get("matches",0)>0: profiles.append((float(w),pr))
        if not profiles: return {"matches":0,"points_per_game":0.0,"win_rate":0.0,"seasons_used":0}
        tw=sum(w for w,_ in profiles); tm=sum(p["matches"] for _,p in profiles)
        def wa(k): return sum(w*float(p.get(k,0.0) or 0.0) for w,p in profiles)/tw
        return {"matches":int(tm),"seasons_used":len(profiles),"win_rate":round(wa("win_rate"),3),"draw_rate":round(wa("draw_rate"),3),"loss_rate":round(wa("loss_rate"),3),"points_per_game":round(wa("points_per_game"),3),"goals_for_avg":round(wa("goals_for_avg"),3) if all(p.get("goals_for_avg") is not None for _,p in profiles) else None,"goals_against_avg":round(wa("goals_against_avg"),3) if all(p.get("goals_against_avg") is not None for _,p in profiles) else None}
    home_hist=weighted_profile(home,"home"); away_hist=weighted_profile(away,"away")
    h2h=_v21_h2h(past+([season_current_df] if isinstance(season_current_df,pd.DataFrame) else []),home,away,match_date,max_matches=10)
    signal,hconf=_v211_historical_signal(home_hist,away_hist,h2h)
    hs=float(home_hist.get("points_per_game",0) or 0); aas=float(away_hist.get("points_per_game",0) or 0); hn=int(h2h.get("matches",0) or 0)
    hedge=((h2h.get("home_wins",0)-h2h.get("away_wins",0))/hn) if hn else 0.0
    score=(hs-aas)+0.35*hedge
    rel=min(current_home_form["reliability"],current_away_form["reliability"])
    dq="LOW" if rel<=0.4 else ("MEDIUM" if rel<0.8 else "HIGH")
    return {"version":"2.1.1","lookback_seasons":len(past),"current_form":{"home":current_home_form,"away":current_away_form},"h2h":h2h,"historical_home_at_home":home_hist,"historical_away_away":away_hist,"historical_signal":signal,"historical_confidence":round(float(hconf),3),"historical_score":round(score,3),"current_form_reliability":round(rel,3),"data_quality":dq,"decision_impact":"INFORMATION_ONLY"}



# ============================================================
# BETSMART V2.2.3 - EARLY SEASON HISTORICAL PRIOR
# ============================================================

def _v223_normalize_probs(values):
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        return np.array([1/3, 1/3, 1/3], dtype=float)
    return arr / s


def _v223_historical_prior(historical_context: dict) -> dict:
    """
    Construit un prior historique HOME/DRAW/AWAY à partir de :
      1) profil du HOME à domicile ;
      2) profil du AWAY à l'extérieur ;
      3) H2H récent, avec poids plafonné.

    Le prior est ensuite shrinké vers 1/3-1/3-1/3 selon la confiance historique.
    """
    hc = historical_context if isinstance(historical_context, dict) else {}
    hh = hc.get("historical_home_at_home") if isinstance(hc.get("historical_home_at_home"), dict) else {}
    ah = hc.get("historical_away_away") if isinstance(hc.get("historical_away_away"), dict) else {}
    h2h = hc.get("h2h") if isinstance(hc.get("h2h"), dict) else {}

    hm = int(hh.get("matches", 0) or 0)
    am = int(ah.get("matches", 0) or 0)
    hn = int(h2h.get("matches", 0) or 0)

    # HOME venue profile: W/D/L => HOME/DRAW/AWAY
    home_vec = _v223_normalize_probs([
        float(hh.get("win_rate", 0.0) or 0.0),
        float(hh.get("draw_rate", 0.0) or 0.0),
        float(hh.get("loss_rate", 0.0) or 0.0),
    ])

    # AWAY venue profile from away-team point of view:
    # away loss => HOME ; away draw => DRAW ; away win => AWAY
    away_vec = _v223_normalize_probs([
        float(ah.get("loss_rate", 0.0) or 0.0),
        float(ah.get("draw_rate", 0.0) or 0.0),
        float(ah.get("win_rate", 0.0) or 0.0),
    ])

    # Pondération des profils venue par volume (cap pour éviter qu'un énorme dataset écrase tout).
    hw = min(40.0, float(hm))
    aw = min(40.0, float(am))
    if hw + aw > 0:
        venue_prior = (hw * home_vec + aw * away_vec) / (hw + aw)
    else:
        venue_prior = np.array([1/3, 1/3, 1/3], dtype=float)

    # H2H : maximum 30 % du prior historique, progressif avec le nombre de matchs.
    if hn > 0:
        h2h_vec = _v223_normalize_probs([
            float(h2h.get("home_win_rate", 0.0) or 0.0),
            float(h2h.get("draw_rate", 0.0) or 0.0),
            float(h2h.get("away_win_rate", 0.0) or 0.0),
        ])
        h2h_mix = min(0.30, 0.05 * hn)
    else:
        h2h_vec = np.array([1/3, 1/3, 1/3], dtype=float)
        h2h_mix = 0.0

    structural_prior = _v223_normalize_probs(
        (1.0 - h2h_mix) * venue_prior + h2h_mix * h2h_vec
    )

    historical_confidence = float(hc.get("historical_confidence", 0.0) or 0.0)
    historical_confidence = float(np.clip(historical_confidence, 0.0, 1.0))

    # Une faible confiance historique doit ramener le prior vers la neutralité.
    neutral = np.array([1/3, 1/3, 1/3], dtype=float)
    final_prior = _v223_normalize_probs(
        historical_confidence * structural_prior
        + (1.0 - historical_confidence) * neutral
    )

    return {
        "home": float(final_prior[0]),
        "draw": float(final_prior[1]),
        "away": float(final_prior[2]),
        "venue_prior": {
            "home": float(venue_prior[0]),
            "draw": float(venue_prior[1]),
            "away": float(venue_prior[2]),
        },
        "h2h_prior": {
            "home": float(h2h_vec[0]),
            "draw": float(h2h_vec[1]),
            "away": float(h2h_vec[2]),
        },
        "h2h_mix": round(float(h2h_mix), 3),
        "historical_confidence": round(historical_confidence, 3),
        "home_venue_matches": hm,
        "away_venue_matches": am,
        "h2h_matches": hn,
    }


def apply_early_season_historical_fusion(pred_final: dict, historical_context: dict, feats_df=None, league_code=None):
    """
    V2.2.3 : lorsque la saison courante est encore pauvre (<5 matchs exploitables
    pour au moins une des deux équipes), fusionne progressivement les probabilités
    du moteur avec un prior historique multi-saisons.

    IMPORTANT :
    - prediction_model reste l'opinion du modèle avant fusion.
    - p0_raw/p1_raw/p2_raw restent inchangés pour l'audit.
    - prediction / proba_0 / proba_1 / proba_2 deviennent les probabilités
      décisionnelles après fusion.
    - les odds ne reçoivent PAS un poids séparé ici : elles sont déjà présentes
      dans le moteur et seraient sinon comptées deux fois.
    """
    out = dict(pred_final or {})
    hc = historical_context if isinstance(historical_context, dict) else {}
    feats_df = feats_df if feats_df is not None else {}

    cf = hc.get("current_form") if isinstance(hc.get("current_form"), dict) else {}
    hf = cf.get("home") if isinstance(cf.get("home"), dict) else {}
    af = cf.get("away") if isinstance(cf.get("away"), dict) else {}

    home_matches = int(hf.get("available_matches", 0) or 0)
    away_matches = int(af.get("available_matches", 0) or 0)
    current_matches = min(home_matches, away_matches)

    pre = _v223_normalize_probs([
        _v211_prob_from_any(out.get("proba_0")),
        _v211_prob_from_any(out.get("proba_1")),
        _v211_prob_from_any(out.get("proba_2")),
    ])

    # Grille progressive : historique fort au démarrage, décroît jusqu'au 5e match.
    base_history_weight_by_matches = {
        0: 0.70,
        1: 0.60,
        2: 0.50,
        3: 0.40,
        4: 0.30,
    }

    prior = _v223_historical_prior(hc)
    hist_conf = float(prior.get("historical_confidence", 0.0) or 0.0)

    if current_matches >= 5:
        mode = "CURRENT_SEASON"
        base_history_weight = 0.0
        effective_history_weight = 0.0
        post = pre.copy()
        applied = False
    else:
        mode = "EARLY_SEASON"
        base_history_weight = float(base_history_weight_by_matches.get(current_matches, 0.30))

        # La confiance historique module directement le poids réellement appliqué.
        effective_history_weight = float(np.clip(base_history_weight * hist_conf, 0.0, 0.75))
        model_weight = 1.0 - effective_history_weight

        hist_vec = np.array([
            prior["home"],
            prior["draw"],
            prior["away"],
        ], dtype=float)

        post = _v223_normalize_probs(
            model_weight * pre + effective_history_weight * hist_vec
        )
        applied = bool(effective_history_weight >= 0.05)

    model_weight = 1.0 - effective_history_weight

    if applied:
        out["proba_0"] = _format_pct(float(post[0]))
        out["proba_1"] = _format_pct(float(post[1]))
        out["proba_2"] = _format_pct(float(post[2]))

        # Décision finale après fusion. prediction_model reste intact.
        pred_after = int(np.argmax(post))
        out["prediction"] = pred_after

        # Recalcul des flags dépendant des probabilités finales.
        try:
            out["low_confidence"] = bool(is_confidence_low(float(post[0]), float(post[1]), float(post[2])))
        except Exception:
            pass

        try:
            bias_detected = bool(out.get("bias_detected", False))
            upset_score = float(out.get("_upset_score", 0.0) or 0.0)
            upset_threshold = float(out.get("_upset_threshold", 0.52) or 0.52)
            out["double_chance"] = detect_double_chance_v2(
                float(post[0]), float(post[1]), float(post[2]), pred_after,
                league_code=league_code or "default",
                bias_detected=bias_detected,
                low_confidence=bool(out.get("low_confidence", False)),
                upset_score=upset_score,
                upset_threshold=upset_threshold,
                override_tag="early_season_historical_fusion",
            )
        except Exception:
            pass

        ra = str(out.get("rule_applied") or "")
        tag = "early_season_historical_fusion"
        if tag not in ra:
            out["rule_applied"] = f"{ra}|{tag}" if ra else tag

    out["prediction_context"] = {
        "version": "2.2.3",
        "mode": mode,
        "fusion_applied": bool(applied),
        "current_matches_home": home_matches,
        "current_matches_away": away_matches,
        "current_matches_min": current_matches,
        "current_form_reliability": float(hc.get("current_form_reliability", 0.0) or 0.0),
        "base_historical_weight": round(float(base_history_weight), 3),
        "effective_historical_weight": round(float(effective_history_weight), 3),
        "model_weight": round(float(model_weight), 3),
        "historical_confidence": round(float(hist_conf), 3),
        "historical_probabilities": {
            "home": round(float(prior["home"]), 4),
            "draw": round(float(prior["draw"]), 4),
            "away": round(float(prior["away"]), 4),
        },
        "historical_prior_details": {
            "venue_prior": {k: round(float(v), 4) for k, v in prior["venue_prior"].items()},
            "h2h_prior": {k: round(float(v), 4) for k, v in prior["h2h_prior"].items()},
            "h2h_mix": prior["h2h_mix"],
            "home_venue_matches": prior["home_venue_matches"],
            "away_venue_matches": prior["away_venue_matches"],
            "h2h_matches": prior["h2h_matches"],
        },
        "pre_adjustment_probabilities": {
            "home": round(float(pre[0]), 4),
            "draw": round(float(pre[1]), 4),
            "away": round(float(pre[2]), 4),
        },
        "post_adjustment_probabilities": {
            "home": round(float(post[0]), 4),
            "draw": round(float(post[1]), 4),
            "away": round(float(post[2]), 4),
        },
        "decision_impact": "PROBABILITY_FUSION" if applied else "NONE",
    }

    return out

def _v211_prob_from_any(v):
    try:
        if v is None: return 0.0
        if isinstance(v,str):
            x=v.strip().replace(",",".")
            if x.endswith("%"): return max(0.0,min(1.0,float(x[:-1])/100.0))
            x=float(x)
        else: x=float(v)
        if x>1.0: x/=100.0
        return max(0.0,min(1.0,x))
    except Exception: return 0.0


def build_market_context(pred_final, feats_df=None):
    """Calcul Python de fair odds, marché démargé, edge et EV. INFORMATION_ONLY."""
    feats_df=feats_df if feats_df is not None else {}
    pm=np.array([_v211_prob_from_any((pred_final or {}).get("proba_0")),_v211_prob_from_any((pred_final or {}).get("proba_1")),_v211_prob_from_any((pred_final or {}).get("proba_2"))],dtype=float)
    if pm.sum()<=0:
        pm=np.array([_v211_prob_from_any((pred_final or {}).get("p0_raw")),_v211_prob_from_any((pred_final or {}).get("p1_raw")),_v211_prob_from_any((pred_final or {}).get("p2_raw"))],dtype=float)
    if pm.sum()>0: pm=pm/pm.sum()
    odds=[]
    for k in ("B365H","B365D","B365A"):
        try:
            v=(pred_final or {}).get(k)
            if v is None: v=_safe_get_first(feats_df,k)
            fv=float(v); odds.append(fv if fv>1 else np.nan)
        except Exception: odds.append(np.nan)
    odds=np.array(odds,dtype=float)
    if not np.isfinite(odds).all(): return {"available":False,"reason":"odds_missing_or_invalid"}
    raw=1.0/odds; over=float(raw.sum()); market=raw/over if over>0 else np.array([1/3,1/3,1/3])
    labels=["HOME","DRAW","AWAY"]; rows={}
    for i,lab in enumerate(labels):
        p=float(pm[i]) if pm.sum()>0 else 0.0; fair=(1/p) if p>1e-9 else None; edge=p-float(market[i]); ev=p*float(odds[i])-1.0
        rows[lab.lower()]={"bookmaker_odds":round(float(odds[i]),3),"model_probability":round(p,4),"market_probability_demarged":round(float(market[i]),4),"market_probability_raw":round(float(raw[i]),4),"model_fair_odds":round(float(fair),3) if fair else None,"edge":round(float(edge),4),"expected_value":round(float(ev),4),"value_bet":bool(ev>=0.05 and edge>=0.03)}
    evs=[rows[x.lower()]["expected_value"] for x in labels]; bi=int(np.argmax(evs)); best=labels[bi]; bev=float(evs[bi])
    l1=float(np.abs(pm-market).sum()) if pm.sum()>0 else 0.0; disag="LOW" if l1<0.12 else ("MEDIUM" if l1<0.25 else "HIGH")
    return {"available":True,"overround":round(over-1.0,4),"home":rows["home"],"draw":rows["draw"],"away":rows["away"],"best_value":best if rows[best.lower()]["value_bet"] else "NONE","best_expected_value":round(bev,4),"model_market_disagreement":disag,"decision_impact":"INFORMATION_ONLY"}



# ============================================================
# BETSMART V2.2 - IA D'ARBITRAGE
# ============================================================

def _v22_label_from_prediction(v):
    try:
        iv = int(v)
        return {0: "HOME", 1: "DRAW", 2: "AWAY"}.get(iv, "UNKNOWN")
    except Exception:
        s = str(v or "").upper().strip()
        return s if s in {"HOME", "DRAW", "AWAY"} else "UNKNOWN"


def _v22_realtime_compact(realtime_risk):
    """Réduit le realtime à des faits utiles et JSON-safe pour l'arbitre IA."""
    rt = realtime_risk if isinstance(realtime_risk, dict) else {}
    summary = rt.get("summary") if isinstance(rt.get("summary"), dict) else {}
    ranking = summary.get("ranking") if isinstance(summary.get("ranking"), dict) else {}
    home_rank = ranking.get("home") if isinstance(ranking.get("home"), dict) else {}
    away_rank = ranking.get("away") if isinstance(ranking.get("away"), dict) else {}
    try:
        home_played = int(home_rank.get("played") or 0)
    except Exception:
        home_played = 0
    try:
        away_played = int(away_rank.get("played") or 0)
    except Exception:
        away_played = 0

    ranking_interpretable = bool(
        ranking.get("available", False)
        and home_played > 0
        and away_played > 0
    )

    return {
        "available": bool(rt.get("available", False)),
        "risk_level": str(rt.get("risk_level", "UNKNOWN") or "UNKNOWN"),
        "risk_score": float(rt.get("risk_score", 0.0) or 0.0),
        "status_short": summary.get("status_short"),
        "minutes_to_kickoff": summary.get("minutes_to_kickoff"),
        "lineups_available": bool(summary.get("lineups_available", False)),
        "injuries_total": int(summary.get("injuries_total", 0) or 0),
        "injuries_home": int(summary.get("injuries_home", 0) or 0),
        "injuries_away": int(summary.get("injuries_away", 0) or 0),
        "ranking": {
            "available": bool(ranking.get("available", False)),
            "interpretable": ranking_interpretable,
            "reason": None if ranking_interpretable else "NO_MATCH_PLAYED_OR_RANKING_UNAVAILABLE",
            "home": {
                "rank": home_rank.get("rank") if ranking_interpretable else None,
                "points": home_rank.get("points") if ranking_interpretable else None,
                "played": home_played,
            },
            "away": {
                "rank": away_rank.get("rank") if ranking_interpretable else None,
                "points": away_rank.get("points") if ranking_interpretable else None,
                "played": away_played,
            },
        },
        "missing_meta": list(summary.get("missing_meta") or [])[:12],
    }


def build_ai_arbitration_payload(pred_final: dict) -> dict:
    """
    Construit le dossier de match V2.2 remis à l'IA.
    Les calculs numériques restent produits par Python; l'IA ne les recalcule pas.
    """
    pf = pred_final if isinstance(pred_final, dict) else {}
    hc = pf.get("historical_context") if isinstance(pf.get("historical_context"), dict) else {}
    mc = pf.get("market_context") if isinstance(pf.get("market_context"), dict) else {}
    h2h = hc.get("h2h") if isinstance(hc.get("h2h"), dict) else {}
    cf = hc.get("current_form") if isinstance(hc.get("current_form"), dict) else {}
    hf = cf.get("home") if isinstance(cf.get("home"), dict) else {}
    af = cf.get("away") if isinstance(cf.get("away"), dict) else {}

    return {
        "match": {
            "home": pf.get("home"),
            "away": pf.get("away"),
            "match_date": pf.get("match_date") or pf.get("date"),
        },
        "model": {
            "prediction_model": _v22_label_from_prediction(pf.get("prediction_model")),
            "prediction": _v22_label_from_prediction(pf.get("prediction")),
            "p_home": _v211_prob_from_any(pf.get("proba_0")),
            "p_draw": _v211_prob_from_any(pf.get("proba_1")),
            "p_away": _v211_prob_from_any(pf.get("proba_2")),
            "low_confidence": bool(pf.get("low_confidence", False)),
            "bias_detected": bool(pf.get("bias_detected", False)),
            "double_chance": pf.get("double_chance"),
            "rule_applied": pf.get("rule_applied"),
        },
        "prediction_context": (
            pf.get("prediction_context")
            if isinstance(pf.get("prediction_context"), dict)
            else {}
        ),
        "current_form": {
            "home_pattern": hf.get("pattern") or pf.get("5_dern_perf_home"),
            "away_pattern": af.get("pattern") or pf.get("5_dern_perf_away"),
            "home_available_matches": int(hf.get("available_matches", 0) or 0),
            "away_available_matches": int(af.get("available_matches", 0) or 0),
            "reliability": float(hc.get("current_form_reliability", 0.0) or 0.0),
            "data_quality": hc.get("data_quality"),
        },
        "history": {
            "signal": hc.get("historical_signal"),
            "confidence": float(hc.get("historical_confidence", 0.0) or 0.0),
            "home_at_home": hc.get("historical_home_at_home") or {},
            "away_at_away": hc.get("historical_away_away") or {},
            "h2h": {
                "matches": int(h2h.get("matches", 0) or 0),
                "home_wins": int(h2h.get("home_wins", 0) or 0),
                "draws": int(h2h.get("draws", 0) or 0),
                "away_wins": int(h2h.get("away_wins", 0) or 0),
                "home_goals": float(h2h.get("home_goals", 0.0) or 0.0),
                "away_goals": float(h2h.get("away_goals", 0.0) or 0.0),
                "dominance_score": float(h2h.get("dominance_score", 0.0) or 0.0),
                "dominant_side": h2h.get("dominant_side"),
                "dominant_team": h2h.get("dominant_team"),
                "black_beast_signal": bool(h2h.get("black_beast_signal", False)),
                "h2h_confidence": h2h.get("h2h_confidence"),
            },
        },
        "market": {
            "available": bool(mc.get("available", False)),
            "model_based_value_signal": mc.get("best_value") or "NONE",
            "best_expected_value": float(mc.get("best_expected_value", 0.0) or 0.0),
            "model_market_disagreement": mc.get("model_market_disagreement"),
            "home": mc.get("home") or {},
            "draw": mc.get("draw") or {},
            "away": mc.get("away") or {},
        },
        "realtime": _v22_realtime_compact(pf.get("realtime_risk")),
        "upset": {
            "score": float(pf.get("_upset_score", 0.0) or 0.0),
            "threshold": float(pf.get("_upset_threshold", 0.52) or 0.52),
        },
    }


def _v22_ai_fallback(status="NOT_RUN", error=""):
    """
    Aucun faux arbitrage : si l'IA en ligne ne répond pas, la sortie indique
    explicitement que la décision IA n'a pas été produite.
    """
    return {
        "version": "2.2.2",
        "status": status,
        "ai_used": False,
        "decision_origin": "NO_AI_DECISION",
        "action": "WATCH",
        "selection": "NONE",
        "decision_confidence": 0.0,
        "selection_confidence": 0.0,
        "confidence": 0.0,  # alias rétrocompatible
        "source_agreement": "UNKNOWN",
        "risk_level": "UNKNOWN",
        "model_signal": "UNKNOWN",
        "historical_signal": "NEUTRAL",
        "model_based_value_signal": "NONE",
        "reason_codes": ["AI_ARBITRATION_UNAVAILABLE"],
        "rationale_short": (
            "Arbitrage IA indisponible. BetSmart conserve les sorties du moteur "
            "sans produire de recommandation IA."
        ),
        "error": str(error or "")[:300],
        "decision_impact": "ADVISORY_ONLY",
    }



def _signals_compatible(value_signal: str, historical_signal: str) -> bool:
    """
    Compatibilité logique entre une sélection 1N2 issue de la value théorique
    et un signal historique éventuellement en double chance.

    Compatibilités :
      HOME  <-> HOME, 1X
      DRAW  <-> DRAW_LEAN, 1X, X2
      AWAY  <-> AWAY, X2

    NEUTRAL / UNKNOWN ne sont ni conflit ni confirmation forte.
    """
    v = str(value_signal or "NONE").strip().upper()
    h = str(historical_signal or "UNKNOWN").strip().upper()

    compat = {
        "HOME": {"HOME", "1X"},
        "DRAW": {"DRAW_LEAN", "1X", "X2"},
        "AWAY": {"AWAY", "X2"},
    }
    if v not in compat:
        return False
    return h in compat[v]


def _compatibility_label(value_signal: str, historical_signal: str) -> str:
    v = str(value_signal or "NONE").strip().upper()
    h = str(historical_signal or "UNKNOWN").strip().upper()
    if v in {"NONE", "UNKNOWN"} or h in {"NEUTRAL", "UNKNOWN", "NONE"}:
        return "UNDETERMINED"
    return "COMPATIBLE" if _signals_compatible(v, h) else "CONFLICT"



def _v23_web_cache_key(home, away, match_date=None):
    return (
        str(home or "").strip().lower(),
        str(away or "").strip().lower(),
        str(match_date or "").strip(),
    )


def _v23_web_cache_get(key):
    with _WEB_RESEARCH_LOCK:
        item = _WEB_RESEARCH_CACHE.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at and time.time() > expires_at:
            _WEB_RESEARCH_CACHE.pop(key, None)
            return None
        return value


def _v23_web_cache_set(key, value):
    with _WEB_RESEARCH_LOCK:
        _WEB_RESEARCH_CACHE[key] = (
            time.time() + max(60, BETSMART_WEB_CACHE_TTL),
            value,
        )


def _v23_response_to_dict(resp):
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
    except Exception:
        pass
    return {}


def _v23_extract_web_sources(resp):
    """
    Extrait les sources réellement retournées par l'outil Web Search.
    """
    data = _v23_response_to_dict(resp)
    sources = []
    seen = set()

    def walk(obj):
        if isinstance(obj, dict):
            # source directe
            url = obj.get("url")
            title = obj.get("title") or obj.get("name")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                key = url.strip()
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "title": str(title or "").strip()[:220],
                        "url": key,
                    })
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(data)
    return sources[:20]


def _v23_parse_json_text(raw):
    txt = str(raw or "").strip()
    if not txt:
        return {}
    # Tolérance aux fences markdown.
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
        txt = re.sub(r"\s*```$", "", txt)
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # Dernier recours : extraire le premier objet JSON.
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass
    return {}


def research_match_web_context_v23(pred_final: dict) -> dict:
    """
    REAL-TIME WEB INTELLIGENCE.

    Utilise OpenAI Responses API + outil Web Search pour rechercher des
    informations fraîches susceptibles d'affecter le 1X2:
      - pré-saison / derniers matchs amicaux,
      - blessures / suspensions / retours,
      - compositions probables/officielles,
      - mercato et changements majeurs d'effectif,
      - changement d'entraîneur / système,
      - calendrier, fatigue, compétition européenne,
      - actualités importantes du club.

    La sortie est structurée et transmise à l'arbitre V2.3.
    """
    pf = pred_final if isinstance(pred_final, dict) else {}
    home = str(pf.get("home") or "").strip()
    away = str(pf.get("away") or "").strip()

    if not BETSMART_WEB_RESEARCH_ENABLED:
        return {
            "version": "2.3",
            "status": "DISABLED",
            "web_research_used": False,
            "model": BETSMART_WEB_MODEL,
            "home": home,
            "away": away,
            "external_signals": [],
            "sources": [],
            "data_confidence": 0.0,
        }

    if not home or not away:
        return {
            "version": "2.3",
            "status": "INVALID_MATCH",
            "web_research_used": False,
            "model": BETSMART_WEB_MODEL,
            "home": home,
            "away": away,
            "external_signals": [],
            "sources": [],
            "data_confidence": 0.0,
        }

    match_date = (
        pf.get("date_match")
        or pf.get("match_date")
        or pf.get("date")
        or ""
    )
    competition = (
        pf.get("competition")
        or pf.get("league")
        or pf.get("league_code")
        or pf.get("comp")
        or ""
    )

    cache_key = _v23_web_cache_key(home, away, match_date)
    cached = _v23_web_cache_get(cache_key)
    if cached is not None:
        cached = dict(cached)
        cached["cache_hit"] = True
        return cached

    try:
        client = get_openai_client()
    except Exception as e:
        return {
            "version": "2.3",
            "status": "CLIENT_ERROR",
            "web_research_used": False,
            "model": BETSMART_WEB_MODEL,
            "home": home,
            "away": away,
            "external_signals": [],
            "sources": [],
            "data_confidence": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    research_prompt = f"""
Tu es le module Real-Time Web Intelligence de BetSmart.

MATCH À ANALYSER
HOME: {home}
AWAY: {away}
COMPÉTITION: {competition or "non précisée"}
DATE DU MATCH: {match_date or "à déterminer à partir des sources"}
DATE DE RECHERCHE UTC: {now_iso}

TU DOIS EFFECTUER UNE RECHERCHE WEB RÉELLE ET ACTUELLE.

OBJECTIF:
Trouver uniquement les informations récentes et crédibles qui peuvent
modifier la plausibilité du résultat 1X2 de {home} - {away}.

RECHERCHE PRIORITAIRE:
1. blessures, suspensions, retours et disponibilité des joueurs clés;
2. compositions probables ou officielles si disponibles;
3. forme récente pertinente, y compris pré-saison si la saison démarre;
4. mercato récent: arrivées/départs majeurs susceptibles de changer le niveau;
5. nouvel entraîneur, changement tactique ou crise interne importante;
6. fatigue/calendrier: match européen, voyage, récupération courte;
7. actualités officielles d'avant-match et conférences de presse;
8. tout événement récent matériellement pertinent pour HOME/DRAW/AWAY.

QUALITÉ DES SOURCES:
- Priorité aux sites officiels des clubs, ligues et compétitions.
- Puis médias sportifs reconnus.
- Ignore les rumeurs non corroborées.
- Vérifie la date des informations.
- Ne recycle pas une ancienne blessure si le joueur a depuis repris.
- Ne donne aucun pronostic basé sur une information non vérifiable.

IMPORTANT:
Ne calcule pas les probabilités finales BetSmart.
Transforme les faits trouvés en signaux structurés.

Pour chaque signal:
- side = HOME / AWAY / BOTH / NEUTRAL
- category = PRESEASON / INJURY / SUSPENSION / LINEUP / TRANSFER /
  COACH / FATIGUE / CLUB_NEWS / OTHER
- direction = POSITIVE / NEGATIVE / NEUTRAL
- impact = nombre entre 0 et 1
- confidence = nombre entre 0 et 1
- summary = fait concis et daté si possible

Retourne UNIQUEMENT un JSON valide de cette forme:
{{
  "research_summary": "...",
  "data_confidence": 0.0,
  "preseason": {{
    "home": {{"signal":"UNKNOWN","summary":""}},
    "away": {{"signal":"UNKNOWN","summary":""}}
  }},
  "squad_news": {{
    "home": {{"impact":"UNKNOWN","summary":""}},
    "away": {{"impact":"UNKNOWN","summary":""}}
  }},
  "transfers": {{
    "home": {{"impact":"UNKNOWN","summary":""}},
    "away": {{"impact":"UNKNOWN","summary":""}}
  }},
  "coach_changes": {{
    "home": {{"impact":"UNKNOWN","summary":""}},
    "away": {{"impact":"UNKNOWN","summary":""}}
  }},
  "schedule_context": {{
    "home": {{"impact":"UNKNOWN","summary":""}},
    "away": {{"impact":"UNKNOWN","summary":""}}
  }},
  "probable_lineups": {{
    "home": {{"available":false,"summary":""}},
    "away": {{"available":false,"summary":""}}
  }},
  "external_signals": []
}}
""".strip()

    base_kwargs = dict(
        model=BETSMART_WEB_MODEL,
        input=research_prompt,
        include=["web_search_call.action.sources"],
    )

    resp = None
    last_error = None

    # Outil actuel puis fallback preview pour compatibilité SDK/API.
    tool_variants = [
        [{"type": "web_search", "search_context_size": BETSMART_WEB_SEARCH_CONTEXT}],
        [{"type": "web_search_preview", "search_context_size": BETSMART_WEB_SEARCH_CONTEXT}],
    ]

    for tools in tool_variants:
        try:
            resp = client.responses.create(
                **base_kwargs,
                tools=tools,
                tool_choice="required",
            )
            break
        except Exception as e:
            last_error = e
            resp = None

    if resp is None:
        return {
            "version": "2.3",
            "status": "WEB_SEARCH_ERROR",
            "web_research_used": False,
            "model": BETSMART_WEB_MODEL,
            "home": home,
            "away": away,
            "external_signals": [],
            "sources": [],
            "data_confidence": 0.0,
            "error": f"{type(last_error).__name__}: {last_error}" if last_error else "unknown",
        }

    raw_text = getattr(resp, "output_text", "") or ""
    parsed = _v23_parse_json_text(raw_text)
    sources = _v23_extract_web_sources(resp)

    external_signals = parsed.get("external_signals")
    if not isinstance(external_signals, list):
        external_signals = []

    # --------------------------------------------------------------
    # V2.3.4 - FIABILITE WEB BASEE SUR DES SOURCES VERIFIABLES
    # --------------------------------------------------------------
    source_count = len(sources)
    raw_web_confidence = float(max(
        0.0,
        min(1.0, float(parsed.get("data_confidence", 0.5) or 0.5))
    ))

    # Une recherche sans source traçable ne peut pas avoir une forte confiance.
    if source_count <= 0:
        evidence_tier = "NONE"
        evidence_verified = False
        evidence_actionable = False
        confidence_cap = 0.10
    elif source_count == 1:
        evidence_tier = "LOW"
        evidence_verified = True
        evidence_actionable = True
        confidence_cap = 0.35
    elif source_count == 2:
        evidence_tier = "MEDIUM"
        evidence_verified = True
        evidence_actionable = True
        confidence_cap = 0.60
    else:
        evidence_tier = "HIGH"
        evidence_verified = True
        evidence_actionable = True
        confidence_cap = 1.00

    verified_web_confidence = min(raw_web_confidence, confidence_cap)

    result = {
        "version": "2.3.7.1",
        "status": "OK" if evidence_verified else "UNVERIFIED_NO_SOURCES",
        "web_research_used": True,
        "web_evidence_verified": evidence_verified,
        "web_evidence_actionable": evidence_actionable,
        "web_evidence_tier": evidence_tier,
        "model": BETSMART_WEB_MODEL,
        "researched_at": now_iso,
        "home": home,
        "away": away,
        "match_date": match_date or None,
        "competition": competition or None,
        "research_summary": str(parsed.get("research_summary") or "")[:1800],
        "raw_data_confidence": raw_web_confidence,
        "data_confidence": round(float(verified_web_confidence), 3),
        "preseason": parsed.get("preseason") if isinstance(parsed.get("preseason"), dict) else {},
        "squad_news": parsed.get("squad_news") if isinstance(parsed.get("squad_news"), dict) else {},
        "transfers": parsed.get("transfers") if isinstance(parsed.get("transfers"), dict) else {},
        "coach_changes": parsed.get("coach_changes") if isinstance(parsed.get("coach_changes"), dict) else {},
        "schedule_context": parsed.get("schedule_context") if isinstance(parsed.get("schedule_context"), dict) else {},
        "probable_lineups": parsed.get("probable_lineups") if isinstance(parsed.get("probable_lineups"), dict) else {},
        "external_signals": external_signals[:20] if evidence_actionable else [],
        "sources": sources,
        "source_count": source_count,
        "cache_hit": False,
    }

    _v23_web_cache_set(cache_key, result)
    return result



def _v234_web_context_for_ai(web_ctx: dict) -> dict:
    """
    Retourne uniquement le niveau de preuve Web autorisé pour l'arbitre IA.

    - 0 source : aucune affirmation Web détaillée n'est transmise comme preuve.
    - 1 source : contexte LOW, à utiliser faiblement.
    - 2 sources : contexte MEDIUM.
    - 3+ sources : contexte HIGH, sous réserve de data_confidence.
    """
    w = dict(web_ctx or {})
    source_count = int(w.get("source_count", 0) or 0)
    verified = bool(w.get("web_evidence_verified", False))
    actionable = bool(w.get("web_evidence_actionable", False))

    if source_count <= 0 or not verified or not actionable:
        return {
            "version": "2.3.7.1",
            "status": "UNVERIFIED_NO_SOURCES",
            "web_research_used": bool(w.get("web_research_used", False)),
            "web_evidence_verified": False,
            "web_evidence_actionable": False,
            "web_evidence_tier": "NONE",
            "source_count": 0,
            "sources": [],
            "data_confidence": min(float(w.get("data_confidence", 0.0) or 0.0), 0.10),
            "research_summary": (
                "Une recherche Web a été exécutée mais aucune source traçable n'a été "
                "conservée. Les affirmations Web détaillées sont exclues de l'arbitrage."
            ),
            "preseason": {},
            "squad_news": {},
            "transfers": {},
            "coach_changes": {},
            "schedule_context": {},
            "probable_lineups": {},
            "external_signals": [],
        }

    # Avec sources vérifiables, on transmet les faits et leur niveau de preuve.
    return w


def build_realtime_intelligence_context_v23(pred_final: dict) -> dict:
    """
    Fusionne API-Sports/realtime_risk + Real-Time Web Intelligence.
    """
    pf = pred_final if isinstance(pred_final, dict) else {}
    rt = pf.get("realtime_risk") if isinstance(pf.get("realtime_risk"), dict) else {}
    summary = rt.get("summary") if isinstance(rt.get("summary"), dict) else {}

    web_ctx = (
        pf.get("realtime_web_intelligence")
        if isinstance(pf.get("realtime_web_intelligence"), dict)
        else {}
    )

    api_conf = 0.45 if rt.get("available") else 0.0
    web_conf = float(web_ctx.get("data_confidence", 0.0) or 0.0)
    web_verified = bool(web_ctx.get("web_evidence_verified", False))
    web_actionable = bool(web_ctx.get("web_evidence_actionable", False))

    if web_ctx.get("web_research_used") is True and web_verified and web_actionable:
        combined_conf = min(1.0, 0.35 * api_conf + 0.65 * web_conf)
        source_mode = "API_PLUS_VERIFIED_WEB"
    elif web_ctx.get("web_research_used") is True:
        # La recherche a tourné, mais sans preuve Web suffisamment traçable.
        combined_conf = api_conf
        source_mode = "API_PLUS_UNVERIFIED_WEB"
    else:
        combined_conf = api_conf
        source_mode = "API_CONTEXT_ONLY"

    return {
        "version": "2.3",
        "available": bool(rt.get("available", False) or web_ctx.get("web_research_used")),
        "source_mode": source_mode,
        "web_research_available": bool(BETSMART_WEB_RESEARCH_ENABLED),
        "web_research_used": bool(web_ctx.get("web_research_used", False)),
        "web_evidence_verified": web_verified,
        "web_evidence_actionable": web_actionable,
        "web_evidence_tier": web_ctx.get("web_evidence_tier", "NONE"),
        "web_source_count": int(web_ctx.get("source_count", 0) or 0),
        "data_confidence": round(float(combined_conf), 3),
        "risk_level": str(rt.get("risk_level", "UNKNOWN") or "UNKNOWN"),
        "risk_score": float(rt.get("risk_score", 0.0) or 0.0),
        "lineups_available": bool(summary.get("lineups_available", False)),
        "lineups_expected_soon": bool(summary.get("lineups_expected_soon", False)),
        "injuries_home": int(summary.get("injuries_home", 0) or 0),
        "injuries_away": int(summary.get("injuries_away", 0) or 0),
        "injuries_total": int(summary.get("injuries_total", 0) or 0),
        "absences_text": summary.get("absences_text"),
        "status_short": summary.get("status_short"),
        "minutes_to_kickoff": summary.get("minutes_to_kickoff"),
        "web": _v234_web_context_for_ai(web_ctx),
        "external_signals": (
            list(web_ctx.get("external_signals") or [])[:20]
            if web_actionable else []
        ),
        "sources": list(web_ctx.get("sources") or [])[:20],
    }


def _v23_schema():
    return {
        "name": "betsmart_v23_final_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "final_selection": {"type": "string", "enum": ["HOME", "DRAW", "AWAY"]},
                "prediction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "decision_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "bet_quality": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]},
                "source_agreement": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                "home_delta": {"type": "number", "minimum": -0.20, "maximum": 0.20},
                "draw_delta": {"type": "number", "minimum": -0.20, "maximum": 0.20},
                "away_delta": {"type": "number", "minimum": -0.20, "maximum": 0.20},
                "reason_codes": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
                "rationale_short": {"type": "string", "minLength": 1, "maxLength": 900},
                "explanation": {"type": "string", "minLength": 80, "maxLength": 2600}
            },
            "required": [
                "final_selection", "prediction_confidence", "decision_confidence",
                "bet_quality", "risk_level", "source_agreement",
                "home_delta", "draw_delta", "away_delta",
                "reason_codes", "rationale_short", "explanation"
            ]
        }
    }


def _v23_clamp(x, lo=-0.20, hi=0.20):
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return 0.0



def _v235_value_quality(market_context: dict) -> str:
    mc = market_context if isinstance(market_context, dict) else {}
    best = str(mc.get("best_value") or "NONE").upper()
    try:
        ev = float(mc.get("best_expected_value", 0.0) or 0.0)
    except Exception:
        ev = 0.0
    if best == "NONE":
        return "NONE"
    if ev >= 0.20:
        return "HIGH"
    if ev >= 0.08:
        return "MEDIUM"
    if ev > 0.0:
        return "LOW"
    return "NONE"


def _v235_evidence_delta_cap(pred_final: dict, ai_data: dict) -> float:
    pf = pred_final if isinstance(pred_final, dict) else {}
    rt_ctx = pf.get("realtime_intelligence_context")
    if not isinstance(rt_ctx, dict):
        rt_ctx = {}

    web_tier = str(rt_ctx.get("web_evidence_tier") or "NONE").upper()
    web_verified = bool(rt_ctx.get("web_evidence_verified", False))
    source_agreement = str(ai_data.get("source_agreement") or "LOW").upper()

    hist = pf.get("historical_context")
    if not isinstance(hist, dict):
        hist = {}
    form_rel = float(hist.get("current_form_reliability", 0.0) or 0.0)

    cap = 0.08
    if web_verified:
        if web_tier == "HIGH":
            cap = 0.10
        elif web_tier == "MEDIUM":
            cap = 0.08
        elif web_tier == "LOW":
            cap = 0.06

    if source_agreement == "HIGH":
        cap += 0.03
    elif source_agreement == "LOW":
        cap -= 0.02

    if form_rel <= 0.0:
        cap = min(cap, 0.10)
    elif form_rel < 0.6:
        cap = min(cap, 0.12)

    return float(max(0.04, min(0.14, cap)))


def _v235_strip_value_from_probability_reasoning(ai_data: dict) -> dict:
    data = dict(ai_data or {})
    codes = []
    value_codes = []
    for code in list(data.get("reason_codes") or []):
        s = str(code)
        if "value" in s.lower() or "expected_value" in s.lower() or "edge" in s.lower():
            value_codes.append(s)
        else:
            codes.append(s)
    data["reason_codes"] = codes[:10]
    data["value_reason_codes"] = value_codes[:10]
    return data


def _v235_dedupe_injury_summary(summary: dict) -> dict:
    if not isinstance(summary, dict):
        return summary

    out = dict(summary)

    def dedupe(items):
        result = []
        seen = set()
        for item in items or []:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("team") or "").strip().lower(),
                str(item.get("player") or "").strip().lower(),
                str(item.get("reason") or "").strip().lower(),
                str(item.get("status_type") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    all_i = dedupe(out.get("top_injuries") or [])
    home_i = dedupe(out.get("top_injuries_home") or [])
    away_i = dedupe(out.get("top_injuries_away") or [])

    out["top_injuries"] = all_i
    out["top_injuries_home"] = home_i
    out["top_injuries_away"] = away_i
    out["injuries_home"] = len(home_i)
    out["injuries_away"] = len(away_i)
    out["injuries_total"] = len(home_i) + len(away_i)
    return out



def _v237_market_favorite(market_context: dict):
    mc = market_context if isinstance(market_context, dict) else {}
    probs = {
        "HOME": ((mc.get("home") or {}).get("market_probability_demarged")),
        "DRAW": ((mc.get("draw") or {}).get("market_probability_demarged")),
        "AWAY": ((mc.get("away") or {}).get("market_probability_demarged")),
    }
    clean = {}
    for k, v in probs.items():
        try:
            clean[k] = float(v)
        except Exception:
            pass
    if not clean:
        return None, 0.0
    side = max(clean, key=clean.get)
    vals = sorted(clean.values(), reverse=True)
    margin = vals[0] - vals[1] if len(vals) > 1 else 0.0
    return side, float(max(0.0, margin))


def _v237_historical_favorite(pred_final: dict):
    pf = pred_final if isinstance(pred_final, dict) else {}
    pc = pf.get("prediction_context") if isinstance(pf.get("prediction_context"), dict) else {}
    hp = pc.get("historical_probabilities") if isinstance(pc.get("historical_probabilities"), dict) else {}
    mapping = {"HOME": hp.get("home"), "DRAW": hp.get("draw"), "AWAY": hp.get("away")}
    clean = {}
    for k, v in mapping.items():
        try:
            clean[k] = float(v)
        except Exception:
            pass
    if not clean:
        return None, 0.0
    side = max(clean, key=clean.get)
    vals = sorted(clean.values(), reverse=True)
    margin = vals[0] - vals[1] if len(vals) > 1 else 0.0
    return side, float(max(0.0, margin))


def _v237_base_favorite(base_probs):
    labels = ["HOME", "DRAW", "AWAY"]
    arr = [float(x) for x in base_probs]
    i = int(np.argmax(arr))
    vals = sorted(arr, reverse=True)
    margin = vals[0] - vals[1] if len(vals) > 1 else 0.0
    return labels[i], float(max(0.0, margin))


def _v237_form_reliability(pred_final: dict) -> float:
    hc = pred_final.get("historical_context") if isinstance(pred_final, dict) else {}
    if not isinstance(hc, dict):
        return 0.0
    try:
        return float(hc.get("current_form_reliability", 0.0) or 0.0)
    except Exception:
        return 0.0


def _v237_web_signal_strength(pred_final: dict):
    rt = pred_final.get("realtime_intelligence_context") if isinstance(pred_final, dict) else {}
    if not isinstance(rt, dict):
        return 0.0, "NONE"
    tier = str(rt.get("web_evidence_tier") or "NONE").upper()
    verified = bool(rt.get("web_evidence_verified", False))
    try:
        conf = float(rt.get("data_confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    tier_weight = {"NONE": 0.0, "LOW": 0.25, "MEDIUM": 0.55, "HIGH": 0.85}.get(tier, 0.0)
    if not verified:
        tier_weight *= 0.25
    return float(max(0.0, min(1.0, tier_weight * conf))), tier


def _v237_independent_agreement_score(pred_final: dict, base_probs):
    base_side, base_margin = _v237_base_favorite(base_probs)
    hist_side, hist_margin = _v237_historical_favorite(pred_final)
    market_side, market_margin = _v237_market_favorite(pred_final.get("market_context") or {})

    votes = []
    if base_side:
        votes.append(("base", base_side, base_margin))
    if hist_side:
        votes.append(("history", hist_side, hist_margin))
    if market_side:
        votes.append(("market", market_side, market_margin))

    counts = {}
    for _, side, _ in votes:
        counts[side] = counts.get(side, 0) + 1

    if not counts:
        return {"score": 0.0, "dominant_side": None, "votes": [], "agreement_level": "LOW"}

    dominant_side = max(counts, key=counts.get)
    max_count = counts[dominant_side]
    total = len(votes)
    score = max_count / total

    agreeing_margins = [m for _, side, m in votes if side == dominant_side]
    avg_margin = sum(agreeing_margins) / len(agreeing_margins) if agreeing_margins else 0.0
    score = min(1.0, 0.8 * score + 0.2 * min(1.0, avg_margin / 0.12))

    level = "HIGH" if score >= 0.80 else ("MEDIUM" if score >= 0.60 else "LOW")
    return {
        "score": round(float(score), 3),
        "dominant_side": dominant_side,
        "votes": votes,
        "agreement_level": level,
    }


def _v237_draw_support_score(pred_final: dict, base_probs):
    base = [float(x) for x in base_probs]
    draw, home, away = base[1], base[0], base[2]
    support = 0.0

    if draw >= max(home, away):
        support += min(0.30, max(0.0, draw - max(home, away)) + 0.10)

    pc = pred_final.get("prediction_context") if isinstance(pred_final, dict) else {}
    hp = pc.get("historical_probabilities") if isinstance(pc, dict) else {}
    if isinstance(hp, dict):
        try:
            hd = float(hp.get("draw", 0.0) or 0.0)
            hh = float(hp.get("home", 0.0) or 0.0)
            ha = float(hp.get("away", 0.0) or 0.0)
            if hd >= max(hh, ha):
                support += 0.25
            elif hd >= 0.30:
                support += 0.10
        except Exception:
            pass

    hc = pred_final.get("historical_context") if isinstance(pred_final, dict) else {}
    h2h = hc.get("h2h") if isinstance(hc, dict) else {}
    if isinstance(h2h, dict):
        try:
            dr = float(h2h.get("draw_rate", 0.0) or 0.0)
            if dr >= 0.40:
                support += 0.20
            elif dr >= 0.25:
                support += 0.08
        except Exception:
            pass

    mc = pred_final.get("market_context") if isinstance(pred_final, dict) else {}
    if isinstance(mc, dict):
        try:
            md = float((mc.get("draw") or {}).get("market_probability_demarged", 0.0) or 0.0)
            mh = float((mc.get("home") or {}).get("market_probability_demarged", 0.0) or 0.0)
            ma = float((mc.get("away") or {}).get("market_probability_demarged", 0.0) or 0.0)
            if md >= max(mh, ma):
                support += 0.20
            elif md >= 0.30:
                support += 0.08
        except Exception:
            pass

    return float(max(0.0, min(1.0, support)))


def _v237_stability_cap(pred_final: dict, ai_data: dict, base_probs):
    agreement = _v237_independent_agreement_score(pred_final, base_probs)
    form_rel = _v237_form_reliability(pred_final)
    web_strength, web_tier = _v237_web_signal_strength(pred_final)

    cap = 0.10 if agreement["agreement_level"] == "HIGH" else (0.07 if agreement["agreement_level"] == "MEDIUM" else 0.05)

    if web_strength >= 0.60:
        cap += 0.02
    elif web_strength <= 0.15:
        cap -= 0.01

    if form_rel < 0.40:
        cap = min(cap, 0.08)

    if str(ai_data.get("source_agreement") or "LOW").upper() == "LOW":
        cap -= 0.01

    cap = float(max(0.035, min(0.11, cap)))
    return cap, agreement, web_strength, web_tier


def _v237_stabilize_ai_deltas(pred_final: dict, ai_data: dict, base_probs):
    cap, agreement, web_strength, web_tier = _v237_stability_cap(pred_final, ai_data, base_probs)

    raw = np.array([
        _v23_clamp(ai_data.get("home_delta", 0.0)),
        _v23_clamp(ai_data.get("draw_delta", 0.0)),
        _v23_clamp(ai_data.get("away_delta", 0.0)),
    ], dtype=float)

    stabilized = np.clip(raw, -cap, cap)

    final_selection = str(ai_data.get("final_selection") or "").upper()
    draw_support = _v237_draw_support_score(pred_final, base_probs)

    if final_selection == "DRAW":
        if draw_support < 0.30 and stabilized[1] > 0.03:
            stabilized[1] = 0.03
        elif draw_support < 0.45 and stabilized[1] > 0.05:
            stabilized[1] = 0.05

    return stabilized, {
        "delta_cap": round(cap, 4),
        "agreement": agreement,
        "web_strength": round(float(web_strength), 3),
        "web_tier": web_tier,
        "draw_support_score": round(float(draw_support), 3),
        "raw_ai_deltas": {
            "home": round(float(raw[0]), 4),
            "draw": round(float(raw[1]), 4),
            "away": round(float(raw[2]), 4),
        },
    }


def _v237_confidence_adjustment(pred_final: dict, ai_data: dict, stability_meta: dict):
    try:
        pred_conf = float(ai_data.get("prediction_confidence", 0.0) or 0.0)
    except Exception:
        pred_conf = 0.0

    form_rel = _v237_form_reliability(pred_final)
    agreement_level = ((stability_meta.get("agreement") or {}).get("agreement_level") or "LOW")

    if form_rel < 0.40:
        pred_conf = min(pred_conf, 0.62)
    if agreement_level == "LOW":
        pred_conf = min(pred_conf, 0.52)
    elif agreement_level == "MEDIUM":
        pred_conf = min(pred_conf, 0.66)

    return float(max(0.0, min(1.0, pred_conf)))


def apply_v23_final_decision(pred_final: dict, ai_data: dict) -> dict:
    """
    BetSmart V2.3.7.1 — source unique de vérité finale.

    Les différentes briques (ML, forme, historique, H2H, marché, realtime API,
    Web Intelligence) sont des sources de preuve. L'IA arbitre leur ensemble.
    Python applique ensuite la décision et synchronise TOUTES les variables
    publiques avec le même état final.
    """
    out = dict(pred_final or {})

    # V2.3.5: la VALUE n'est pas une preuve sportive indépendante.
    ai_data = _v235_strip_value_from_probability_reasoning(ai_data)

    # Sauvegarde audit du modèle réellement brut avant synchronisation publique.
    out.setdefault("model_raw_probabilities", {
        "home": float(out.get("p0_raw", 0.0) or 0.0),
        "draw": float(out.get("p1_raw", 0.0) or 0.0),
        "away": float(out.get("p2_raw", 0.0) or 0.0),
    })
    out.setdefault("model_raw_prediction", out.get("prediction_model"))

    # Etat pré-IA (après éventuelle fusion EARLY_SEASON).
    base = _v223_normalize_probs([
        _v211_prob_from_any(out.get("proba_0")),
        _v211_prob_from_any(out.get("proba_1")),
        _v211_prob_from_any(out.get("proba_2")),
    ])

    # V2.3.7: Decision Stability Layer.
    delta, stability_meta = _v237_stabilize_ai_deltas(out, ai_data, base)
    delta_cap = float(stability_meta.get("delta_cap", 0.05))

    final_probs = np.clip(base + delta, 0.01, None)
    final_probs = _v223_normalize_probs(final_probs)

    selection = str(ai_data.get("final_selection", "")).upper()
    final_idx = {"HOME": 0, "DRAW": 1, "AWAY": 2}.get(selection)
    if final_idx is None:
        final_idx = int(np.argmax(final_probs))
        selection = {0: "HOME", 1: "DRAW", 2: "AWAY"}[final_idx]

    # La sélection IA doit être cohérente avec la distribution finale.
    if int(np.argmax(final_probs)) != final_idx:
        gap = float(np.max(final_probs) - final_probs[final_idx] + 0.01)
        final_probs[final_idx] += max(0.0, gap)
        final_probs = _v223_normalize_probs(final_probs)

    p_home, p_draw, p_away = map(float, final_probs)

    # --------------------------------------------------------------
    # ETAT FINAL PUBLIC SYNCHRONISE
    # --------------------------------------------------------------
    out["p0_raw"] = p_home
    out["p1_raw"] = p_draw
    out["p2_raw"] = p_away

    out["proba_0"] = _format_pct(p_home)
    out["proba_1"] = _format_pct(p_draw)
    out["proba_2"] = _format_pct(p_away)

    out["prediction"] = int(final_idx)
    out["prediction_model"] = int(final_idx)

    # V2.3.3: double chance dérivée uniquement des probabilités finales.
    if final_idx == 0:
        out["double_chance"] = "1X"
    elif final_idx == 2:
        out["double_chance"] = "X2"
    else:
        out["double_chance"] = "1X" if p_home >= p_away else "X2"

    pred_conf = _v237_confidence_adjustment(out, ai_data, stability_meta)
    dec_conf = float(max(0.0, min(1.0, float(
        ai_data.get("decision_confidence", 0.0) or 0.0
    ))))
    out["low_confidence"] = bool(pred_conf < 0.60)
    out["value_quality"] = _v235_value_quality(out.get("market_context") or {})

    # --------------------------------------------------------------
    # EXPLICATION DYNAMIQUE ISSUE DU MEME ARBITRAGE IA
    # --------------------------------------------------------------
    ai_explanation = str(ai_data.get("explanation") or "").strip()
    if ai_explanation:
        selection_fr = {
            "HOME": f"victoire de {out.get('home')}",
            "DRAW": "match nul",
            "AWAY": f"victoire de {out.get('away')}",
        }.get(selection, selection)

        # Python ajoute uniquement les chiffres finaux exacts après normalisation.
        final_intro = (
            f"Après confrontation de l'ensemble des informations disponibles, "
            f"BetSmart estime les probabilités finales à "
            f"{p_home*100:.1f}% pour {out.get('home')}, "
            f"{p_draw*100:.1f}% pour le nul et "
            f"{p_away*100:.1f}% pour {out.get('away')}. "
        )
        final_tail = (
            f" Décision BetSmart : {selection_fr}. "
            f"Confiance de prédiction : {pred_conf:.2f} ; "
            f"qualité du pari : {str(ai_data.get('bet_quality', 'LOW')).lower()}."
        )
        out["explanation"] = (final_intro + ai_explanation + final_tail).strip()
        out["_explanation_locked_v233"] = True
    else:
        # Fallback très court basé sur le rationale du même arbitrage.
        selection_fr = {
            "HOME": f"victoire de {out.get('home')}",
            "DRAW": "match nul",
            "AWAY": f"victoire de {out.get('away')}",
        }.get(selection, selection)
        out["explanation"] = (
            f"Après analyse multi-sources, BetSmart estime les probabilités finales à "
            f"{p_home*100:.1f}% pour {out.get('home')}, "
            f"{p_draw*100:.1f}% pour le nul et "
            f"{p_away*100:.1f}% pour {out.get('away')}. "
            f"{str(ai_data.get('rationale_short') or '').strip()} "
            f"Décision BetSmart : {selection_fr}."
        ).strip()
        out["_explanation_locked_v233"] = True

    out["ai_decision_v23"] = {
        "version": "2.3.7.1",
        "status": "OK",
        "ai_used": True,
        "decision_origin": ai_data.get("decision_origin", "OPENAI_LLM"),
        "final_selection": selection,
        "prediction_confidence": round(pred_conf, 3),
        "decision_confidence": round(dec_conf, 3),
        "bet_quality": str(ai_data.get("bet_quality", "LOW")),
        "value_quality": out.get("value_quality", "NONE"),
        "risk_level": str(ai_data.get("risk_level", "UNKNOWN")),
        "source_agreement": str(ai_data.get("source_agreement", "LOW")),
        "reason_codes": list(ai_data.get("reason_codes") or [])[:10],
        "rationale_short": str(ai_data.get("rationale_short", ""))[:1200],
        "base_probabilities": {
            "home": round(float(base[0]), 4),
            "draw": round(float(base[1]), 4),
            "away": round(float(base[2]), 4),
        },
        "final_probabilities": {
            "home": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away": round(p_away, 4),
        },
        "applied_deltas": {
            "home": round(float(delta[0]), 4),
            "draw": round(float(delta[1]), 4),
            "away": round(float(delta[2]), 4),
        },
        "delta_cap": round(float(delta_cap), 4),
        "stability": stability_meta,
        "value_reason_codes": list(ai_data.get("value_reason_codes") or [])[:10],
        "public_state_synchronized": True,
        "market_value_separated": True,
        "explanation_source": "TOP_LEVEL_ONLY_SAME_AI_ARBITRATION",
    }
    out["ai_decision"] = out["ai_decision_v23"]

    ra = str(out.get("rule_applied") or "")
    tag = "v2371_final_state"
    if tag not in ra:
        out["rule_applied"] = f"{ra}|{tag}" if ra else tag

    return _v2371_final_sync(out)


def ai_match_arbitrator(pred_final: dict) -> dict:
    """
    V2.3 : arbitre IA final. Il choisit obligatoirement HOME, DRAW ou AWAY.
    """
    pf = pred_final if isinstance(pred_final, dict) else {}

    if not BETSMART_AI_ENABLED:
        return {
            "version": "2.3.7.1", "status": "DISABLED", "ai_used": False,
            "decision_origin": "NO_AI_DECISION",
            "final_selection": _v22_label_from_prediction(pf.get("prediction")),
            "prediction_confidence": 0.0, "decision_confidence": 0.0,
            "bet_quality": "LOW", "risk_level": "UNKNOWN",
            "source_agreement": "UNKNOWN", "reason_codes": ["AI_DISABLED"],
            "rationale_short": "IA V2.3 désactivée."
        }

    try:
        client = get_openai_client()
    except Exception as e:
        return {
            "version": "2.3.7.1", "status": "ERROR", "ai_used": False,
            "decision_origin": "NO_AI_DECISION",
            "final_selection": _v22_label_from_prediction(pf.get("prediction")),
            "prediction_confidence": 0.0, "decision_confidence": 0.0,
            "bet_quality": "LOW", "risk_level": "UNKNOWN",
            "source_agreement": "UNKNOWN", "reason_codes": ["AI_CLIENT_ERROR"],
            "rationale_short": f"Client IA indisponible: {type(e).__name__}"
        }

    dossier = build_ai_arbitration_payload(pf)
    dossier["prediction_context"] = pf.get("prediction_context") or {}

    _raw_web_ctx = pf.get("realtime_web_intelligence") or {}
    dossier["realtime_web_intelligence"] = _v234_web_context_for_ai(_raw_web_ctx)
    dossier["realtime_intelligence_context"] = build_realtime_intelligence_context_v23(pf)

    schema = _v23_schema()

    system_msg = """Tu es le moteur de décision final BetSmart V2.3.7.1.
Tu raisonnes comme un analyste/parieur professionnel discipliné.

Tu dois TOUJOURS choisir une issue finale unique parmi HOME, DRAW ou AWAY.
Tu ne peux jamais répondre NONE, WATCH ou NO_BET comme final_selection.

PRINCIPE CENTRAL:
Aucune source n'est automatiquement la vérité.
Le modèle ML, le marché, l'historique, le H2H, l'API realtime et le Web
sont des SOURCES DE PREUVE que tu dois confronter.

Tu analyses ensemble:
- modèle ML et probabilités;
- forme de saison courante et sa fiabilité;
- historique multi-saisons domicile/extérieur;
- confrontations H2H et éventuelle bête noire;
- odds, marché et value;
- blessures, suspensions, retours et compositions;
- pré-saison;
- transferts et changements d'effectif;
- entraîneur et changements tactiques;
- fatigue, calendrier et contexte du match;
- Real-Time Web Intelligence et qualité de ses sources.

METHODE:
1. Évalue la fraîcheur et la fiabilité de chaque information.
2. Repère les contradictions entre familles de signaux.
3. Évite tout double comptage.
4. Les faits récents confirmés peuvent corriger un historique ancien.
5. En EARLY_SEASON, compense le manque de forme officielle avec historique,
   H2H, pré-saison et contexte récent, sans laisser les odds dominer seules.
6. FIABILITE WEB OBLIGATOIRE:
   - web_evidence_tier=NONE ou web_evidence_actionable=false:
     ignore les affirmations Web détaillées pour final_selection et pour les deltas;
   - tier=LOW (1 source): influence faible uniquement;
   - tier=MEDIUM (2 sources): influence modérée;
   - tier=HIGH (3+ sources): influence possible selon data_confidence.
   Une affirmation Web sans URL/source traçable ne doit jamais devenir un reason_code décisif.
7. Ne confonds jamais composition PROBABLE et composition OFFICIELLE.
8. Sépare STRICTEMENT MARKET SIGNAL et VALUE SIGNAL:
   - MARKET SIGNAL = lecture des probabilités implicites dé-margées du bookmaker;
   - VALUE SIGNAL = conséquence de l'écart entre probabilités BetSmart et marché.
9. Une value, un edge ou un expected_value ne sont JAMAIS une preuve sportive.
   INTERDICTION de les utiliser pour justifier final_selection ou les deltas.
   Pour la décision 1X2, raisonne sur market_probability_demarged mais ignore
   best_value, expected_value, edge et value_bet.
   La value sert uniquement APRÈS la décision sportive à qualifier value_quality/bet_quality.
10. Si le marché dé-margé favorise une autre issue que la value, décris cela comme
    un désaccord modèle/marché et non comme une convergence.
11. Un black_beast_signal est un facteur statistique/psychologique, pas une certitude.
12. Une composition officielle peut peser davantage qu'une tendance ancienne.
13. INCERTITUDE ≠ MATCH NUL.
    Le manque de forme, les absences contradictoires ou des signaux faibles
    doivent d'abord réduire prediction_confidence / decision_confidence.
    Ils ne doivent augmenter DRAW que s'il existe de vrais arguments sportifs
    en faveur du nul (équilibre structurel, historique compatible, H2H, marché compatible).
14. Si plusieurs sources indépendantes convergent vers HOME ou AWAY, ne te réfugie
    pas vers DRAW uniquement parce que le contexte est incertain.
15. Tu dois choisir l'issue la PLUS PLAUSIBLE, pas la plus rentable.
16. bet_quality mesure l'intérêt du pari, indépendamment de la prédiction 1X2.

EXPLICATION DYNAMIQUE:
L'explication est destinée directement à l'utilisateur BetSmart.
Elle doit être le résumé fidèle DU MÊME RAISONNEMENT qui produit final_selection.

Règles pour explanation:
- elle est DYNAMIQUE: ne suis jamais une liste fixe de rubriques;
- mentionne uniquement les familles de sources réellement disponibles et pertinentes;
- si une source est absente ou non fiable, ne l'invente pas et ne lui consacre pas artificiellement une phrase;
- priorise les 3 à 6 facteurs qui ont réellement influencé la décision;
- lorsqu'il existe une contradiction importante (ex: historique HOME mais H2H AWAY), explique-la clairement;
- distingue les faits confirmés des incertitudes;
- distingue la prédiction 1X2 de la qualité du pari;
- distingue le MARKET SIGNAL (probabilités implicites dé-margées) du VALUE SIGNAL (edge/EV);
- ne présente jamais une value, un edge ou un EV comme une confirmation sportive;
- la value décrit uniquement l'attractivité du prix proposé, jamais la probabilité que l'issue se réalise;
- si le marché et BetSmart divergent, explique clairement ce désaccord;
- ne transforme jamais "incertitude" en justification automatique du nul;
- si DRAW est retenu, cite les arguments sportifs spécifiques qui le soutiennent;
- ne parle pas de "probabilités du modèle" pour les probabilités finales BetSmart;
- n'invente pas les probabilités finales exactes: Python les ajoutera après ton arbitrage;
- utilise les noms réels des équipes;
- écris en français naturel, clair, professionnel et concis;
- termine par une justification claire de final_selection, mais SANS phrase standard rigide.

SORTIE OBLIGATOIRE:
- final_selection = HOME / DRAW / AWAY;
- prediction_confidence = confiance dans l'issue sportive;
- decision_confidence = confiance dans l'arbitrage multi-sources;
- bet_quality = LOW / MEDIUM / HIGH;
- home_delta / draw_delta / away_delta entre -0.20 et +0.20;
- reason_codes = principaux facteurs;
- rationale_short = résumé court pour audit;
- explanation = synthèse dynamique multi-sources destinée à l'utilisateur.

N'invente jamais une information absente.
Retourne uniquement le JSON conforme au schéma."""

    user_msg = "DOSSIER BETSMART V2.3.7.1:\n" + json.dumps(dossier, ensure_ascii=False, default=str)

    try:
        resp = client.chat.completions.create(
            model=BETSMART_AI_MODEL,
            temperature=BETSMART_AI_TEMPERATURE,
            max_tokens=BETSMART_AI_MAX_TOKENS,
            timeout=BETSMART_AI_TIMEOUT,
            response_format={"type": "json_schema", "json_schema": schema},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        data = json.loads((resp.choices[0].message.content or "{}").strip())
        if not isinstance(data, dict):
            raise ValueError("structured_output_not_object")
        data.update({
            "version": "2.3.7.1",
            "status": "OK",
            "ai_used": True,
            "decision_origin": "OPENAI_LLM",
            "model": BETSMART_AI_MODEL,
        })
        return data
    except Exception as e:
        return {
            "version": "2.3.7.1", "status": "ERROR", "ai_used": False,
            "decision_origin": "NO_AI_DECISION",
            "final_selection": _v22_label_from_prediction(pf.get("prediction")),
            "prediction_confidence": 0.0, "decision_confidence": 0.0,
            "bet_quality": "LOW", "risk_level": "UNKNOWN",
            "source_agreement": "UNKNOWN",
            "reason_codes": ["AI_ARBITRATION_UNAVAILABLE"],
            "rationale_short": f"Arbitrage IA V2.3 indisponible: {type(e).__name__}",
            "error": str(e)[:300],
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

    V2.2.3: enrichit le résultat et peut ajuster les probabilités finales en EARLY_SEASON via un prior historique traçable.
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

    # ------------------------------------------------------------------
    # BETSMART V2.1 - historique multi-saisons (INFORMATION ONLY)
    # ------------------------------------------------------------------
    try:
        historical_context = build_historical_profile(
            home=home_name,
            away=away_name,
            season_current_df=season_current_df,
            season_past_list=season_past_list,
            match_date=match_date,
            lookback_seasons=3,
        )
        out["historical_context"] = historical_context

        # V2.2.3 : en début de saison, le moteur ne doit pas rester dominé
        # par les features pauvres / odds lorsque la forme actuelle manque.
        out = apply_early_season_historical_fusion(
            out,
            historical_context=historical_context,
            feats_df=feats_df,
            league_code=league_code,
        )

        pc = out.get("prediction_context") if isinstance(out.get("prediction_context"), dict) else {}
        if pc:
            notes.append(
                f"early_season_v2_2_3: mode={pc.get('mode')} "
                f"applied={pc.get('fusion_applied')} "
                f"hW={pc.get('effective_historical_weight')} "
                f"mW={pc.get('model_weight')}"
            )

        # Rend la forme visible au LLM AVANT l'ajout tardif fait dans l'API.
        cf = historical_context.get("current_form", {}) if isinstance(historical_context, dict) else {}
        hf = (cf.get("home") or {}).get("pattern") if isinstance(cf, dict) else None
        af = (cf.get("away") or {}).get("pattern") if isinstance(cf, dict) else None
        if hf and out.get("5_dern_perf_home") in (None, ""):
            out["5_dern_perf_home"] = hf
        if af and out.get("5_dern_perf_away") in (None, ""):
            out["5_dern_perf_away"] = af
        notes.append(
            f"historical_v2_1_1: signal={historical_context.get('historical_signal')} "
            f"quality={historical_context.get('data_quality')} "
            f"form_rel={historical_context.get('current_form_reliability')}"
        )
    except Exception as e:
        out["historical_context"] = {
            "version": "2.1.1", "available": False,
            "error": f"{type(e).__name__}: {str(e)[:160]}",
            "decision_impact": "INFORMATION_ONLY",
        }
        notes.append(f"historical_v2_1_1:error={type(e).__name__}")

    # Copier les cotes dans pred_final afin que l'explication LLM puisse les lire.
    for _k in ("B365H", "B365D", "B365A"):
        try:
            _v = _safe_get_first(feats_df, _k)
            if _v is not None and str(_v).strip() != "":
                out[_k] = float(_v)
        except Exception:
            pass

    # BETSMART V2.1.1 - lecture marché/value calculée en Python (information only)
    try:
        out["market_context"] = build_market_context(out, feats_df=feats_df)
        mc = out.get("market_context") or {}
        notes.append(f"market_v2_1_1: best_value={mc.get('best_value')} disagreement={mc.get('model_market_disagreement')}")
    except Exception as e:
        out["market_context"] = {"available": False, "reason": f"{type(e).__name__}: {str(e)[:120]}"}
        notes.append(f"market_v2_1_1:error={type(e).__name__}")

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

    # ------------------------------------------------------------------
    # BETSMART V2.3 - REAL-TIME INTELLIGENCE + DECISION FINALE 1X2
    # ------------------------------------------------------------------
    try:
        # 1) Recherche Web réelle
        out["realtime_web_intelligence"] = research_match_web_context_v23(out)

        # 2) Fusion API + Web
        # V2.3.5: déduplication blessures/suspensions avant usage IA.
        if isinstance(out.get("realtime_risk"), dict):
            _rt = dict(out.get("realtime_risk") or {})
            if isinstance(_rt.get("summary"), dict):
                _rt["summary"] = _v235_dedupe_injury_summary(_rt.get("summary") or {})
            out["realtime_risk"] = _rt

        out["realtime_intelligence_context"] = build_realtime_intelligence_context_v23(out)

        _web = out.get("realtime_web_intelligence") or {}
        notes.append(
            f"web_intelligence_v2_3_4: status={_web.get('status')} "
            f"used={_web.get('web_research_used')} "
            f"verified={_web.get('web_evidence_verified')} "
            f"tier={_web.get('web_evidence_tier')} "
            f"sources={_web.get('source_count', 0)} "
            f"confidence={_web.get('data_confidence', 0.0)}"
        )

        # 3) Arbitrage IA final
        ai_v23 = ai_match_arbitrator(out)

        if ai_v23.get("ai_used") is True:
            out = apply_v23_final_decision(out, ai_v23)
        else:
            out["ai_decision_v23"] = ai_v23
            out["ai_decision"] = ai_v23

        _aid = out.get("ai_decision_v23") or {}
        notes.append(
            f"ai_v2_3: status={_aid.get('status')} "
            f"selection={_aid.get('final_selection')} "
            f"prediction_confidence={_aid.get('prediction_confidence')} "
            f"bet_quality={_aid.get('bet_quality')}"
        )
    except Exception as e:
        out["ai_decision_v23"] = {
            "version": "2.3.7.1", "status": "ERROR", "ai_used": False,
            "decision_origin": "NO_AI_DECISION",
            "final_selection": _v22_label_from_prediction(out.get("prediction")),
            "prediction_confidence": 0.0, "decision_confidence": 0.0,
            "bet_quality": "LOW", "risk_level": "UNKNOWN",
            "source_agreement": "UNKNOWN",
            "reason_codes": ["AI_APPLICATION_ERROR"],
            "rationale_short": f"Erreur application V2.3: {type(e).__name__}",
        }
        out["ai_decision"] = out["ai_decision_v23"]
        notes.append(f"ai_v2_3:error={type(e).__name__}")

    out["notes"] = notes
    return out


##---------------------- FIN FONCTIONS  ------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _extract_row(features: Any) -> Dict[str, Any]:
    # features peut être DataFrame (1 ligne), dict, etc.
    try:
        import pandas as pd
        if isinstance(features, pd.DataFrame):
            if features.empty:
                return {}
            return features.to_dict(orient="records")[0] or {}
    except Exception:
        pass

    if isinstance(features, dict):
        return dict(features)
    return {}

# ============================================================
# Helpers (assume you already have these in fonction.py)
# - _safe_get_first(df, col)
# - detect_bias(df)
# - _format_pct(x)  # if x in [0..1] => "13.0%" etc
# ============================================================
def _safe_get_first(df: Any, col: str):
    try:
        if isinstance(df, pd.DataFrame) and col in df.columns and len(df) > 0:
            v = df[col].iloc[0]
            # unwrap numpy scalars
            if isinstance(v, (np.generic,)):
                return v.item()
            return v
    except Exception:
        pass
    return None

def _format_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "0.0%"

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
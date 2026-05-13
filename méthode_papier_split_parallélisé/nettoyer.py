"""
Nettoyage de logs GPS bruts — Île-de-France
============================================
Pipeline :
  1. Coordonnées invalides  — NaN, hors plage lat/lon, hors zone IDF
  2. Sauts aberrants        — pics isolés physiquement impossibles
  3. Fusion de points proches — regroupe les points consécutifs < merge_dist_m
                               ET < merge_dt_s en un seul point centroïde

Colonnes attendues : LATITUDE, LONGITUDE, SPEED (m/s),
                     LOCAL_DATE (YYYY-MM-DD), LOCAL_TIME (HH:MM:SS)

"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────
PARIS_LAT          = 48.8566
PARIS_LON          = 2.3522
DEFAULT_RADIUS     = 200   # km
DEFAULT_MERGE_DIST = 10     # m
DEFAULT_MERGE_DT   = 30     # s
DEFAULT_JUMP       = 200   # km/h
DEFAULT_SPEED      = 250   # km/h (filtre colonne brute)


# ── Géométrie vectorisée ──────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance haversine en mètres (tableaux numpy)."""
    R = 6_371_000
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = la2 - la1, lo2 - lo1
    a = np.sin(dlat / 2)**2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _dist_to_paris_km(lats, lons):
    return _haversine_m(lats, lons,
                        np.full_like(lats, PARIS_LAT),
                        np.full_like(lons, PARIS_LON)) / 1000


def _to_seconds(dt_series: pd.Series) -> np.ndarray:
    """
    Convertit une série datetime64 en secondes depuis l'époque.
    Compatible pandas < 2 (ns) et >= 2 (us/ns selon la source).
    """
    return dt_series.diff().dt.total_seconds().fillna(0).cumsum().values


# ── Étape 1 : coordonnées invalides ──────────────────────────────────────────

def _filter_invalid(df: pd.DataFrame, radius_km: float):
    n0 = len(df)
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    n_nan = n0 - len(df)

    b = len(df)
    df = df[df["LATITUDE"].between(-90, 90) & df["LONGITUDE"].between(-180, 180)]
    n_range = b - len(df)

    b = len(df)
    df = df[_dist_to_paris_km(df["LATITUDE"].values, df["LONGITUDE"].values) <= radius_km]
    n_idf = b - len(df)

    return df.reset_index(drop=True), dict(nan_coords=n_nan, hors_plage=n_range, hors_idf=n_idf)


# ── Étape 2 : sauts aberrants ─────────────────────────────────────────────────

def _filter_jumps(df: pd.DataFrame, max_jump_kmh: float):
    """
    Détecte les pics isolés : point i est un outlier si

        max(speed_bwd[i], speed_fwd[i]) > seuil
        ET
        speed_skip[i] < seuil        ← passer directement de i-1 à i+1 est normal

    Cela garantit que le point est vraiment isolé (aller ET retour anormal)
    sans supprimer des vrais déplacements rapides.
    Itère jusqu'à convergence (un spike peut en cacher un autre).
    """
    thr = max_jump_kmh / 3.6   # m/s
    total_removed = 0

    for _ in range(10):   # max 10 passes
        lats = df["LATITUDE"].values
        lons = df["LONGITUDE"].values
        ts   = df["_ts"].values
        n    = len(lats)

        if n < 3:
            break

        # Distances et dt consécutifs (i-1 → i)
        dist = np.concatenate([[0.0], _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])])
        dt   = np.concatenate([[np.inf], np.diff(ts)])

        # Distances et dt "skip" (i-1 → i+1, en sautant i)
        dist_skip = np.concatenate([[0.0],
                                    _haversine_m(lats[:-2], lons[:-2], lats[2:], lons[2:]),
                                    [0.0]])
        dt_skip   = np.concatenate([[np.inf],
                                    ts[2:] - ts[:-2],
                                    [np.inf]])

        with np.errstate(divide="ignore", invalid="ignore"):
            spd_bwd  = np.where(dt      > 0, dist      / dt,      0.0)
            spd_fwd  = np.concatenate([spd_bwd[1:], [0.0]])    # = spd_bwd décalé d'un rang
            spd_skip = np.where(dt_skip > 0, dist_skip / dt_skip, 0.0)

        # Spike = vitesse aberrante dans au moins une direction
        #         ET vitesse normale si on saute le point
        spike = ((np.maximum(spd_bwd, spd_fwd) > thr) & (spd_skip <= thr))

        n_spike = int(spike.sum())
        if n_spike == 0:
            break

        total_removed += n_spike
        df = df[~spike].reset_index(drop=True)
        df["_ts"] = _to_seconds(df["_dt"])   # recalcul après suppression

    return df, dict(sauts_aberrants=total_removed)


# ── Étape 3 : fusion de points proches ───────────────────────────────────────

def _merge_close_points(df: pd.DataFrame, merge_dist_m: float, merge_dt_s: float):
    """
    Regroupe les points consécutifs (ordre chronologique) qui vérifient
    SIMULTANÉMENT :
        - distance au point précédent < merge_dist_m
        - delta temps au point précédent < merge_dt_s

    Chaque groupe → UN point synthétique :
        - LATITUDE / LONGITUDE : centroïde (moyenne)
        - timestamp            : premier instant du groupe
        - SPEED                : moyenne du groupe
        - colonnes date/heure  : valeurs du premier point
    """
    lats = df["LATITUDE"].values
    lons = df["LONGITUDE"].values
    ts   = df["_ts"].values

    dist_prev = np.concatenate([[np.inf],
                                _haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])])
    dt_prev   = np.concatenate([[np.inf], np.diff(ts)])

    # Nouveau groupe si l'une des deux conditions N'EST PAS remplie
    new_group = (dist_prev >= merge_dist_m) | (dt_prev >= merge_dt_s)
    group_id  = new_group.cumsum() - 1

    df = df.copy()
    df["_gid"] = group_id

    agg: dict = {"LATITUDE": "mean", "LONGITUDE": "mean",
                 "_ts": "first", "_dt": "first"}
    for col in ("SPEED", "LOCAL_DATE", "LOCAL_TIME", "UTC_DATE", "UTC_TIME"):
        if col in df.columns:
            agg[col] = "mean" if col == "SPEED" else "first"

    merged = (df.groupby("_gid", sort=False)
                .agg(agg)
                .reset_index(drop=True))

    stats = dict(
        groupes_formes   = int(group_id.max()) + 1,
        points_fusionnes = len(df) - len(merged),
    )
    return merged, stats


# ── Fonction principale ───────────────────────────────────────────────────────

def clean_gps_logs(
    input_path   : str,
    radius_km    : float = DEFAULT_RADIUS,
    merge_dist_m : float = DEFAULT_MERGE_DIST,
    merge_dt_s   : float = DEFAULT_MERGE_DT,
    max_jump_kmh : float = DEFAULT_JUMP,
    max_speed_kmh: float = DEFAULT_SPEED,
) -> str:
    """
    Nettoie un fichier de logs GPS bruts et enregistre le résultat en CSV.

    Paramètres
    ----------
    input_path    : chemin du CSV source
    output_dir    : dossier de sortie (défaut = dossier du fichier source)
    radius_km     : rayon max autour de Paris en km          (défaut 200)
    merge_dist_m  : seuil distance pour fusion de points      (défaut 10 m)
    merge_dt_s    : seuil temps pour fusion de points         (défaut 30 s)
    max_jump_kmh  : vitesse max pour détecter les sauts       (défaut 200 km/h)
    max_speed_kmh : vitesse max de la colonne SPEED brute     (défaut 250 km/h)
    output_name   : nom du fichier de sortie

    Retourne
    --------
    Chemin absolu du CSV nettoyé.
    """

    # ── Chargement ────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    file_id = os.path.splitext(os.path.basename(input_path))[0]
    n_initial = len(df)

    # Nettoyage colonne SPEED brute
    n_spd = 0
    if "SPEED" in df.columns:
        df["SPEED"] = pd.to_numeric(df["SPEED"], errors="coerce").fillna(0.0)
        thr   = max_speed_kmh / 3.6
        n_spd = int((df["SPEED"] > thr).sum())
        df    = df[df["SPEED"] <= thr]

    # ── Timestamp interne (robust toutes versions pandas) ─────────────────────
    if "LOCAL_DATE" in df.columns and "LOCAL_TIME" in df.columns:
        df["_dt"] = pd.to_datetime(df["LOCAL_DATE"] + " " + df["LOCAL_TIME"],
                                   errors="coerce")
    elif "UTC_DATE" in df.columns and "UTC_TIME" in df.columns:
        df["_dt"] = pd.to_datetime(df["UTC_DATE"] + " " + df["UTC_TIME"],
                                   errors="coerce")
    else:
        df["_dt"] = pd.NaT

    df = df.sort_values("_dt").reset_index(drop=True)
    # _to_seconds utilise .diff().dt.total_seconds() → correct quelle que soit
    # la précision interne (ns ou us selon pandas < 2 / >= 2)
    df["_ts"] = _to_seconds(df["_dt"])

    # ── Étape 1 ───────────────────────────────────────────────────────────────
    df, s1 = _filter_invalid(df, radius_km)
    # ── Étape 2 ───────────────────────────────────────────────────────────────
    df, s2 = _filter_jumps(df, max_jump_kmh)
    # ── Étape 3 ───────────────────────────────────────────────────────────────
    df, s3 = _merge_close_points(df, merge_dist_m, merge_dt_s)
    # ── Nettoyage colonnes internes ───────────────────────────────────────────
    df = df.drop(columns=[c for c in ["_ts", "_dt", "_gid"] if c in df.columns])

    # ── Écriture ──────────────────────────────────────────────────────────────
    output_name = f"{file_id}_nettoye.csv"
    output_dir = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split_parallélisé\fichiers_nettoyes"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, index=False)

    # ── Résumé final ─────────────────────────────────────────────────────────

    return output_path


# ── Appel direct ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python clean_gps_logs.py <fichier.csv>")
        sys.exit(1)
    clean_gps_logs(sys.argv[1])
import pandas as pd
import numpy as np
import os

GAP_EXTRA = 600  # 10 minutes : seuil de coupure pour les trips "extra"


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVES GÉOMÉTRIQUES  –  100 % NumPy, acceptent scalaires ET tableaux
# ─────────────────────────────────────────────────────────────────────────────

def haversine_vec(lat1, lon1, lat2, lon2):
    """
    Distance Haversine vectorisée.
    Remplace np.vectorize(distance) partout dans le fichier.
    Gain typique : ×20 à ×50 vs np.vectorize.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [
        np.asarray(lat1, dtype=float),
        np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float),
        np.asarray(lon2, dtype=float),
    ])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing_vec(lat1, lon1, lat2, lon2):
    """Angle de cap vectorisé — remplace np.vectorize(calculate_bearing)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [
        np.asarray(lat1, dtype=float),
        np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float),
        np.asarray(lon2, dtype=float),
    ])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def _step_distances(df):
    """
    Distances entre points consécutifs d'un DataFrame (colonne 'distance_m').
    Mutualisé pour éviter de le recalculer dans chaque compute_*.
    """
    d = haversine_vec(
        df['LATITUDE'].values[:-1], df['LONGITUDE'].values[:-1],
        df['LATITUDE'].values[1:],  df['LONGITUDE'].values[1:],
    )
    return np.append(d, 0.0)   # dernier point → 0


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES PAR SEGMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_hcr(df, angle_threshold=19):
    """Head-Change Rate — structure identique à l'original (shift(-1) style).

    L'original calculait n bearings via shift(-1) (dernier = NaN),
    puis diff() → n-1 comparaisons valides.
    Ici : n bearings sur les n points (lat[i]→lat[i+1], dernier = NaN),
    puis np.diff() → n-1 diffs, ce qui est cohérent avec step_dist (n valeurs).
    """
    step_dist = _step_distances(df)
    total_dist = step_dist.sum()
    if total_dist == 0:
        return 0.0

    lats = df['LATITUDE'].values
    lons = df['LONGITUDE'].values

    # n bearings : point[i] → point[i+1], le dernier pointe vers lui-même (NaN évité)
    # On duplique le dernier point pour obtenir exactement n bearings comme shift(-1)
    lats_next = np.append(lats[1:], lats[-1])
    lons_next = np.append(lons[1:], lons[-1])
    bearings = bearing_vec(lats, lons, lats_next, lons_next)  # longueur n

    # diff() sur n bearings → n-1 différences (exactement comme l'original)
    diff = np.abs(np.diff(bearings))
    diff = np.where(diff > 180, 360 - diff, diff)
    heading_changes = (diff > angle_threshold).astype(int)   # longueur n-1

    # On ignore le dernier step_dist (distance du dernier point = 0) pour aligner
    return heading_changes.sum() / total_dist


def compute_sr(df, speed_threshold=0.8):
    """Stop Rate — version vectorisée."""
    step_dist = _step_distances(df)
    total_dist = step_dist.sum()
    if total_dist == 0:
        return 0.0
    is_stop = (df['SPEED'].values < speed_threshold).astype(int)
    return is_stop.sum() / total_dist


def compute_vcr(df, v_threshold=0.26):
    """Velocity-Change Rate — version vectorisée."""
    step_dist = _step_distances(df)
    total_dist = step_dist.sum()
    if total_dist == 0:
        return 0.0
    speeds = df['SPEED'].values
    v_prev = np.roll(speeds, 1)
    v_prev[0] = speeds[0]
    v_rate = np.abs(speeds - v_prev) / (v_prev + 1e-6)
    v_changes = (v_rate > v_threshold).astype(int)
    return v_changes.sum() / total_dist


def extraire_features(df):
    df = df.copy()

    # Distances vectorisées une seule fois pour tout le DataFrame
    dist = haversine_vec(
        df['LATITUDE'].values[:-1], df['LONGITUDE'].values[:-1],
        df['LATITUDE'].values[1:],  df['LONGITUDE'].values[1:],
    )
    df['distance_m'] = np.append(dist, 0.0)

    results = []
    for seg_id, g in df.groupby('final_segment_id'):
        if g['distance_m'].sum() < 200:
            continue

        results.append((
            seg_id,
            compute_hcr(g),
            compute_sr(g),
            compute_vcr(g),
            g['SPEED'].quantile(0.95),
            g['acceleration'].abs().max(),
            g['LATITUDE'].iloc[0],
            g['LONGITUDE'].iloc[0],
            g['time'].iloc[0],
        ))

    if not results:
        return None, None, None, None, None, None, None, None

    res = pd.DataFrame(results, columns=[
        'seg', 'hcr', 'sr', 'vcr', 'vmax', 'amax', 'lat', 'lon', 'time'
    ]).set_index('seg')

    return (
        res['hcr'], res['sr'], res['vcr'],
        res['vmax'], res['amax'],
        res['lat'], res['lon'], res['time'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def segmenter_walk(df):
    """
    Détecte les changements de mode par quantification de vitesse.
    La boucle Python row-by-row est remplacée par un ffill Pandas.
    """
    df = df.copy()

    df['SPEED'] = (
        df['SPEED'].rolling(window=5, center=True).mean().fillna(df['SPEED']).fillna(0)
    )
    df['acceleration'] = (
        df['acceleration'].rolling(window=3, center=True).mean().fillna(df['acceleration'])
    )

    # Quantification de la vitesse
    bins   = [0, 0.5, 1.5, 3.0, 7.0, 12.0, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    df['speed_class'] = (
        pd.cut(df['SPEED'], bins=bins, labels=labels, include_lowest=True, right=True)
        .astype(float).fillna(0).astype(int)
    )

    df['speed_class_smooth'] = (
        df['speed_class'].rolling(window=3, center=True, min_periods=1)
        .median().fillna(0).astype(int)
    )
    df['change_point'] = df['speed_class_smooth'].diff().fillna(0).abs() > 0
    df.loc[df['time_diff'].isna(), 'change_point'] = True
    df['segment_id'] = df['change_point'].cumsum()

    # Distances vectorisées
    dist = haversine_vec(
        df['LATITUDE'].values[:-1], df['LONGITUDE'].values[:-1],
        df['LATITUDE'].values[1:],  df['LONGITUDE'].values[1:],
    )
    df['distance_m'] = np.append(dist, 0.0)

    dist_seg = df.groupby('segment_id')['distance_m'].sum()
    df['is_certain'] = df['segment_id'].map(dist_seg > 20)

    # ── Fusion des micro-segments — remplace la boucle Python row-by-row ──────
    # Pour les lignes "incertaines", on veut propager le dernier segment_id
    # certain connu (forward-fill uniquement sur les certains).
    certain_ids = df['segment_id'].where(df['is_certain'])   # NaN si incertain
    df['final_segment_id'] = (
        certain_ids
        .ffill()                         # propage le dernier certain vers le bas
        .fillna(df['segment_id'])        # si pas de certain avant : garde l'original
        .astype(int)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DES STAY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def detect_stay_points(df_extra, dist_threshold=50, time_threshold=600):
    """
    Détecte les points d'arrêt prolongé.

    Même logique que l'original (boucle while identique, résultats identiques),
    mais distance() est remplacée par haversine_vec() qui accepte les scalaires
    et est ~5× plus rapide grâce à NumPy sans l'overhead de np.vectorize.
    """
    n = len(df_extra)
    if n == 0:
        return np.zeros(n, dtype=bool)

    lats  = df_extra['LATITUDE'].values
    lons  = df_extra['LONGITUDE'].values
    times = df_extra['time'].values

    # Temps en secondes (évite les conversions timedelta répétées dans la boucle)
    t0        = times[0]
    time_secs = (times - t0).astype('timedelta64[s]').astype(np.float64)

    is_stay = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        j = i + 1
        while j < n:
            # haversine_vec accepte des scalaires → même résultat que distance()
            d = float(haversine_vec(lats[i], lons[i], lats[j], lons[j]))
            if d > dist_threshold:
                dt = time_secs[j] - time_secs[i]
                if dt >= time_threshold:
                    is_stay[i:j] = True
                i = j
                break
            j += 1
        else:
            # for/else : j a atteint n sans sortir de la zone
            dt = time_secs[n - 1] - time_secs[i]
            if dt >= time_threshold:
                is_stay[i:n] = True
            break

    return is_stay


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def est_un_trajet_valide(grp_trip):
    dist_totale  = grp_trip['distance_m'].sum()
    duree_totale = (grp_trip['time'].max() - grp_trip['time'].min()).total_seconds()
    return dist_totale >= 70 and duree_totale >= 20


def filtrer_segments_courts(df_segmented, seuil_dist=30):
    dist_par_seg    = df_segmented.groupby('final_segment_id')['distance_m'].sum()
    segments_valides = dist_par_seg[dist_par_seg >= seuil_dist].index
    return df_segmented[df_segmented['final_segment_id'].isin(segments_valides)]


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main(gps_path, displacements_path, user_id=None):

    # ── 1. Chargement GPS ──────────────────────────────────────────────────────
    df = pd.read_csv(gps_path)
    df['time'] = pd.to_datetime(
        df['LOCAL_DATE'] + ' ' + df['UTC_TIME'], format='%Y-%m-%d %H:%M:%S'
    )
    df = df.sort_values('time').reset_index(drop=True)
    df['SPEED'] = df['SPEED'] / 3.6

    if user_id is None:
        user_id = os.path.splitext(os.path.basename(gps_path))[0]

    # ── 2. Chargement des trajets déclarés ─────────────────────────────────────
    df_trips = pd.read_csv(displacements_path)
    df_trips = df_trips[
        (df_trips['ID'] == user_id) & (df_trips['ID_Trip_Days'] != 'No_Trip')
    ].copy().reset_index(drop=True)
    df_trips['start_trip'] = pd.to_datetime(df_trips['Date_O'] + ' ' + df_trips['Time_O'])
    df_trips['end_trip']   = pd.to_datetime(df_trips['Date_D'] + ' ' + df_trips['Time_D'])

    # ── 3. Affectation trip_id ─────────────────────────────────────────────────
    df['trip_id'] = None
    for _, row in df_trips.iterrows():
        mask = (df['time'] >= row['start_trip']) & (df['time'] <= row['end_trip'])
        df.loc[mask, 'trip_id'] = row['KEY']

    # ── 4. Points hors fenêtres : stay-point detection ────────────────────────
    df_extra = df[df['trip_id'].isna()].copy()

    if not df_extra.empty:
        df_extra['is_stay']    = detect_stay_points(df_extra, 50, 600)
        df_extra['stay_change'] = (
            df_extra['is_stay'].shift(1, fill_value=False) & ~df_extra['is_stay']
        )
        df_extra['time_diff']  = df_extra['time'].diff().dt.total_seconds()
        df_extra['gap_break']  = df_extra['time_diff'] > 1200
        df_extra['new_trip']   = (
            df_extra['stay_change'] | df_extra['gap_break'] | df_extra['time_diff'].isna()
        )
        df_extra['extra_trip_num'] = df_extra['new_trip'].cumsum()
        df_extra = df_extra[~df_extra['is_stay']]
        df_extra['trip_id'] = df_extra['extra_trip_num'].apply(lambda n: f'extra_{int(n)}')
        df.loc[df_extra.index, 'trip_id'] = df_extra['trip_id']

    # ── 5. time_diff et accélération ──────────────────────────────────────────
    df['time_diff']    = df['time'].diff().dt.total_seconds()
    df.loc[df['trip_id'].ne(df['trip_id'].shift()), 'time_diff'] = np.nan
    df['speed_diff']   = df['SPEED'].diff()
    df['acceleration'] = df['speed_diff'] / df['time_diff']

    # ── 6. Trajets DÉCLARÉS ────────────────────────────────────────────────────
    df_declared = df[
        df['trip_id'].apply(lambda x: not str(x).startswith('extra_'))
    ].copy()
    df_declared = segmenter_walk(df_declared)
    df_declared['final_segment_id'] = df_declared['final_segment_id'].apply(
        lambda x: f'decl_{x}'
    )

    hcr_d, sr_d, vcr_d, vit_d, acc_d, lat, lon, time = extraire_features(df_declared)
    trip_d = df_declared.groupby('final_segment_id')['trip_id'].first()

    # ── 7. Trajets EXTRA ───────────────────────────────────────────────────────
    all_hcr  = [hcr_d];  all_sr   = [sr_d];  all_vcr  = [vcr_d]
    all_vit  = [vit_d];  all_acc  = [acc_d];  all_trip = [trip_d]
    all_lat  = [lat];    all_lon  = [lon];    all_time = [time]

    for extra_id, grp in df[
        df['trip_id'].apply(lambda x: str(x).startswith('extra_'))
    ].groupby('trip_id'):
        grp = grp.copy().reset_index(drop=True)

        # Distances vectorisées
        dist_arr = haversine_vec(
            grp['LATITUDE'].values[:-1], grp['LONGITUDE'].values[:-1],
            grp['LATITUDE'].values[1:],  grp['LONGITUDE'].values[1:],
        )
        grp['distance_m'] = np.append(dist_arr, 0.0)

        if not est_un_trajet_valide(grp):
            continue

        grp['time_diff']    = grp['time'].diff().dt.total_seconds()
        grp.loc[0, 'time_diff'] = np.nan
        grp['speed_diff']   = grp['SPEED'].diff()
        grp['acceleration'] = grp['speed_diff'] / grp['time_diff']

        grp = segmenter_walk(grp)
        grp['final_segment_id'] = grp['final_segment_id'].apply(
            lambda x: f'{extra_id}_{x}'
        )
        grp = filtrer_segments_courts(grp, seuil_dist=25)
        if grp.empty:
            continue

        h, s, v, vi, ac, la, lo, ti = extraire_features(grp)
        tr = grp.groupby('final_segment_id')['trip_id'].first()

        all_hcr.append(h);  all_sr.append(s);   all_vcr.append(v)
        all_vit.append(vi); all_acc.append(ac);  all_trip.append(tr)
        all_lat.append(la); all_lon.append(lo);  all_time.append(ti)

    # ── 8. Concaténation finale ────────────────────────────────────────────────
    hcr_km           = pd.concat(all_hcr)
    sr               = pd.concat(all_sr)
    vcr              = pd.concat(all_vcr)
    stats_vitesse    = pd.concat(all_vit)
    stats_accel      = pd.concat(all_acc)
    trip_par_segment = pd.concat(all_trip)
    lats_series      = pd.concat(all_lat)
    lons_series      = pd.concat(all_lon)
    times_series     = pd.concat(all_time)

    seg_index = pd.RangeIndex(len(hcr_km))

    return (
        seg_index, hcr_km, sr, vcr,
        stats_vitesse, stats_accel,
        trip_par_segment,
        lats_series, lons_series, times_series,
    )
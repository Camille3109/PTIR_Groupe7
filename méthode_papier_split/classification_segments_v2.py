import pandas as pd
import numpy as np
import os

GAP_EXTRA = 600  # 10 minutes : seuil de coupure pour les trips "extra"


def distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points GPS"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calcule l'angle entre deux points GPS"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    return (np.degrees(np.atan2(y, x)) + 360) % 360


def extraire_features(df):
    df = df.copy()

    # distance par segment
    df['lat_next'] = df['LATITUDE'].shift(-1)
    df['lon_next'] = df['LONGITUDE'].shift(-1)

    df['distance_m'] = np.vectorize(distance)(
    df['LATITUDE'], df['LONGITUDE'],
    df['lat_next'], df['lon_next']
)

    df['distance_m'] = pd.Series(df['distance_m']).fillna(0).values
    results = []

    for seg_id, g in df.groupby('final_segment_id'):
        g = g.copy()

        dist = g['distance_m'].sum()
        if dist < 200:
            continue

        hcr = compute_hcr(g)
        sr = compute_sr(g)
        vcr = compute_vcr(g)

        # On prend le TOUT PREMIER point du segment
        lat_start = g['LATITUDE'].iloc[0] 
        lon_start = g['LONGITUDE'].iloc[0]
        time_start = g['time'].iloc[0]

        results.append((seg_id, hcr, sr, vcr,
                        g['SPEED'].quantile(0.95),
                        g['acceleration'].abs().max(),   
                        lat_start, lon_start, time_start))
    if not results:
        return None, None, None, None, None, None, None, None

    res = pd.DataFrame(results, columns=[
        'seg', 'hcr', 'sr', 'vcr', 'vmax', 'amax', 'lat', 'lon', 'time'
    ]).set_index('seg')

    return (
        res['hcr'],
        res['sr'],
        res['vcr'],
        res['vmax'],
        res['amax'],
        res['lat'], res['lon'], res['time']
    )




def segmenter_walk(df):
    """
    APPROCHE PAPIER (Zheng et al. 2008) :
    Détecte TOUS les changements de pattern de mouvement, pas seulement walk/non-walk.
    
    Idée : un changement de mode de transport se manifeste par un changement
    simultané de vitesse ET d'accélération. On découpe le trajet en segments
    homogènes, chacun étant plus susceptible d'être mono-mode.
    
    Différence avec la v1 :
    - Ancien : 2 états seulement (walk / non-walk)
    - Nouveau : N états basés sur des seuils de vitesse → segments plus purs
    """
    df = df.copy()

    df['SPEED'] = df['SPEED'].rolling(window=5, center=True).mean().fillna(df['SPEED'])
    df['acceleration'] = df['acceleration'].rolling(window=3, center=True).mean().fillna(df['acceleration'])
    
    # Sécurité anti-NaN avant le découpage
    df['SPEED'] = df['SPEED'].fillna(0)

    # 2. Quantification
    bins   = [0, 0.5, 1.5, 3.0, 7.0, 12.0, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    
    # On utilise include_lowest=True pour que 0 soit inclus dans la catégorie 0
    # On convertit en float puis en int pour gérer proprement les NaN résiduels s'il y en a
    df['speed_class'] = pd.cut(
        df['SPEED'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True, 
        right=True
    )
    
    # Remplissage simple : si une valeur est hors bins, on met 0
    df['speed_class'] = df['speed_class'].astype(float).fillna(0).astype(int)

    # 3. Lissage des classes
    df['speed_class_smooth'] = df['speed_class'].rolling(
        window=3, center=True, min_periods=1
    ).median().fillna(0).astype(int)
    df['change_point'] = df['speed_class_smooth'].diff().fillna(0).abs() > 0
    # Coupure aussi aux frontières de trip
    df.loc[df['time_diff'].isna(), 'change_point'] = True
    df['segment_id'] = df['change_point'].cumsum()

    # 4. Calcul des distances pour filtrage
    df['lat_next'] = df['LATITUDE'].shift(-1)
    df['lon_next'] = df['LONGITUDE'].shift(-1)
    df['distance_m'] = np.vectorize(distance)(
        df['LATITUDE'], df['LONGITUDE'],
        df['lat_next'], df['lon_next']
    ).astype(float)
    df['distance_m'] = df['distance_m'].fillna(0)

    # 5. Fusionner les micro-segments (< 20m) avec le segment précédent
    #    (même logique que l'ancien code)
    dist_seg = df.groupby('segment_id')['distance_m'].sum()
    df['is_certain'] = df['segment_id'].map(dist_seg > 20)
    df['final_segment_id'] = df['segment_id']

    last_certain_id = None
    for idx in df.index:
        if df.loc[idx, 'is_certain']:
            last_certain_id = df.loc[idx, 'segment_id']
        elif last_certain_id is not None:
            df.loc[idx, 'final_segment_id'] = last_certain_id

    return df

def detect_stay_points(df_extra, dist_threshold=50, time_threshold=600):
    """
    Détecte les poits d'arrêt
    """
    # Conversion en tableaux NumPy pour un accès ultra-rapide
    lats = df_extra['LATITUDE'].values
    lons = df_extra['LONGITUDE'].values
    times = df_extra['time'].values
    
    n = len(df_extra)
    is_stay = np.zeros(n, dtype=bool)
    
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            # Calcul de distance via NumPy
            d = distance(lats[i], lons[i], lats[j], lons[j])
            
            if d > dist_threshold:
                # Calcul de la durée en secondes
                dt = (times[j] - times[i]).astype('timedelta64[s]').astype(int)
                if dt >= time_threshold:
                    is_stay[i:j] = True
                i = j
                break
            j += 1
        else:
            # Gérer la fin du trajet
            dt = (times[n-1] - times[i]).astype('timedelta64[s]').astype(int)
            if dt >= time_threshold:
                is_stay[i:n] = True
            break
            
    return is_stay

def est_un_trajet_valide(grp_trip):
    """ Vérifie si le trajet global est significatif """
    dist_totale = grp_trip['distance_m'].sum()
    duree_totale = (grp_trip['time'].max() - grp_trip['time'].min()).total_seconds()
    
    # Un trajet de moins de 70m ou 20 s est souvent du bruit GPS statique
    if dist_totale < 70 or duree_totale < 20:
        return False
    return True

def filtrer_segments_courts(df_segmented, seuil_dist=30):
    """ 
    Au sein d'un trajet, supprime les segments
    qui sont trop courts pour être physiquement réalistes.
    """
    dist_par_seg = df_segmented.groupby('final_segment_id')['distance_m'].sum()
    segments_valides = dist_par_seg[dist_par_seg >= seuil_dist].index
    
    return df_segmented[df_segmented['final_segment_id'].isin(segments_valides)]

def compute_vcr(df, v_threshold=0.26):
    """Calcule le vcr"""
    df = df.copy()

    # vitesse locale
    df['v_prev'] = df['SPEED'].shift(1)
    df['v_rate'] = np.abs(df['SPEED'] - df['v_prev']) / (df['v_prev'] + 1e-6)

    # points de changement de vitesse
    df['v_change'] = (df['v_rate'] > v_threshold).astype(int)

    # distance entre points
    df['lat_next'] = df['LATITUDE'].shift(-1)
    df['lon_next'] = df['LONGITUDE'].shift(-1)

    df['step_dist'] = np.vectorize(distance)(
        df['LATITUDE'], df['LONGITUDE'],
        df['lat_next'], df['lon_next']
    )
    df['step_dist'] = df['step_dist'].fillna(0)

    total_dist = df['step_dist'].sum()

    if total_dist == 0:
        return 0.0

    vcr = df['v_change'].sum() / total_dist
    return vcr

def compute_hcr(df, angle_threshold=19):
    """Calcule le hcr"""
    df = df.copy()

    df['lat_next'] = df['LATITUDE'].shift(-1)
    df['lon_next'] = df['LONGITUDE'].shift(-1)

    df['bearing'] = np.vectorize(calculate_bearing)(
        df['LATITUDE'], df['LONGITUDE'],
        df['lat_next'], df['lon_next']
    )

    df['bearing_diff'] = df['bearing'].diff().abs()
    df.loc[df['bearing_diff'] > 180, 'bearing_diff'] = 360 - df['bearing_diff']

    df['heading_change'] = (df['bearing_diff'] > angle_threshold).astype(int)

    df['step_dist'] = np.vectorize(distance)(
        df['LATITUDE'], df['LONGITUDE'],
        df['lat_next'], df['lon_next']
    )
    df['step_dist'] = pd.Series(df['step_dist']).fillna(0).values

    total_dist = df['step_dist'].sum()
    if total_dist == 0:
        return 0.0

    return df['heading_change'].sum() / total_dist

def compute_sr(df, speed_threshold=0.8):
    """Calcule le sr"""
    df = df.copy()

    df['is_stop'] = (df['SPEED'] < speed_threshold).astype(int)

    df['lat_next'] = df['LATITUDE'].shift(-1)
    df['lon_next'] = df['LONGITUDE'].shift(-1)

    df['step_dist'] = np.vectorize(distance)(
        df['LATITUDE'], df['LONGITUDE'],
        df['lat_next'], df['lon_next']
    )
    df['step_dist'] = pd.Series(df['step_dist']).fillna(0).values

    total_dist = df['step_dist'].sum()
    if total_dist == 0:
        return 0.0

    return df['is_stop'].sum() / total_dist


def main(gps_path, displacements_path, user_id=None):

    # ── 1. Chargement GPS ──────────────────────────────────────────────────────
    df = pd.read_csv(gps_path)
    df['time'] = pd.to_datetime(df['LOCAL_DATE'] + ' ' + df['UTC_TIME'],
                                format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('time').reset_index(drop=True)
    df['SPEED'] = df['SPEED'] / 3.6

    if user_id is None:
        user_id = os.path.splitext(os.path.basename(gps_path))[0]

    # ── 2. Chargement des trajets déclarés ─────────────────────────────────────
    df_trips = pd.read_csv(displacements_path)
    df_trips = df_trips[df_trips['ID'] == user_id].copy()
    df_trips = df_trips[df_trips['ID_Trip_Days'] != 'No_Trip'].copy()
    df_trips = df_trips.reset_index(drop=True)
    df_trips['start_trip'] = pd.to_datetime(df_trips['Date_O'] + ' ' + df_trips['Time_O'])
    df_trips['end_trip']   = pd.to_datetime(df_trips['Date_D'] + ' ' + df_trips['Time_D'])

    # ── 3. Affectation trip_id par KEY pour les points dans une fenêtre déclarée
    df['trip_id'] = None
    for _, row in df_trips.iterrows():
        mask = (df['time'] >= row['start_trip']) & (df['time'] <= row['end_trip'])
        df.loc[mask, 'trip_id'] = row['KEY']

    # ── 4. Points HORS fenêtres : segmentation par STABILITÉ (Stay Points) ──
    df_extra = df[df['trip_id'].isna()].copy()
    
    if not df_extra.empty:
        # A. Détecter les zones où c'est "stable" pendant > 10 min
        df_extra['is_stay'] = detect_stay_points(df_extra, dist_threshold=50, time_threshold=600)
        
        # B. Un nouveau trajet commence après la fin d'une zone stable
        # (Chaque fois qu'on passe de is_stay=True à is_stay=False)
        df_extra['stay_change'] = df_extra['is_stay'].shift(1, fill_value=False) & ~df_extra['is_stay']
        
        # C. On ajoute aussi la coupure par gros gap temporel (sécurité)
        df_extra['time_diff'] = df_extra['time'].diff().dt.total_seconds()
        df_extra['gap_break'] = df_extra['time_diff'] > 1200 # 20 min de silence total
        
        # D. Création des IDs de trajets "extra"
        df_extra['new_trip'] = df_extra['stay_change'] | df_extra['gap_break'] | df_extra['time_diff'].isna()
        df_extra['extra_trip_num'] = df_extra['new_trip'].cumsum()
        
        # E. On retire les points de "stay" du trajet final (optionnel)
        # Souvent, on ne veut analyser que le mouvement, pas le temps passé au canapé
        df_extra = df_extra[df_extra['is_stay'] == False]
        
        df_extra['trip_id'] = df_extra['extra_trip_num'].apply(lambda n: f'extra_{int(n)}')
        df.loc[df_extra.index, 'trip_id'] = df_extra['trip_id']

    # ── 5. Calcul time_diff et acceleration sur l'ensemble (coupure aux frontières)
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df.loc[df['trip_id'].ne(df['trip_id'].shift()), 'time_diff'] = np.nan
    df['speed_diff']   = df['SPEED'].diff()
    df['acceleration'] = df['speed_diff'] / df['time_diff']

    # ── 6. Segmentation walk/non-walk + features pour les trajets DÉCLARÉS ─────
    df_declared = df[df['trip_id'].apply(lambda x: not str(x).startswith('extra_'))].copy()
    df_declared = segmenter_walk(df_declared)
    # On rend les segment_id globalement uniques en préfixant
    df_declared['final_segment_id'] = df_declared['final_segment_id'].apply(
        lambda x: f'decl_{x}')

    hcr_d, sr_d, vcr_d, vit_d, acc_d, lat, lon, time = extraire_features(df_declared)
    trip_d = df_declared.groupby('final_segment_id')['trip_id'].first()

    # ── 7. Segmentation walk/non-walk + features pour les trajets EXTRA ────────
    all_hcr, all_sr, all_vcr, all_vit, all_acc, all_trip, all_lat, all_lon, all_time = \
        [hcr_d], [sr_d], [vcr_d], [vit_d], [acc_d], [trip_d], [lat], [lon], [time] 

    for extra_id, grp in df[df['trip_id'].apply(
            lambda x: str(x).startswith('extra_'))].groupby('trip_id'):
        grp = grp.copy().reset_index(drop=True)

        grp['lat_next'] = grp['LATITUDE'].shift(-1)
        grp['lon_next'] = grp['LONGITUDE'].shift(-1)
        # Calculer la distance pour le trajet entier
        dist_array = np.vectorize(distance)(
            grp['LATITUDE'], grp['LONGITUDE'],
            grp['lat_next'], grp['lon_next']
        ).astype(float)

        # On l'assigne à la colonne, puis on utilise fillna sur la Série Pandas
        grp['distance_m'] = dist_array
        grp['distance_m'] = grp['distance_m'].fillna(0)

        # Filtre de "Trajet Fantôme"
        # Si le trajet global est insignifiant (bruit statique), on l'ignore
        if not est_un_trajet_valide(grp):
            continue

        grp['time_diff'] = grp['time'].diff().dt.total_seconds()
        grp.loc[0, 'time_diff'] = np.nan
        grp['speed_diff']   = grp['SPEED'].diff()
        grp['acceleration'] = grp['speed_diff'] / grp['time_diff']

        grp = segmenter_walk(grp)
        grp['final_segment_id'] = grp['final_segment_id'].apply(
            lambda x: f'{extra_id}_{x}')

        grp = filtrer_segments_courts(grp, seuil_dist=25)

        if grp.empty:
            continue

        h, s, v, vi, ac, la, lo, ti = extraire_features(grp)
        tr = grp.groupby('final_segment_id')['trip_id'].first()
        all_hcr.append(h); all_sr.append(s); all_vcr.append(v)
        all_vit.append(vi); all_acc.append(ac); all_trip.append(tr)
        all_lat.append(la); all_lon.append(lo); all_time.append(ti)


    # ── 8. Concaténation finale ────────────────────────────────────────────────
    hcr_km        = pd.concat(all_hcr)
    sr            = pd.concat(all_sr)
    vcr           = pd.concat(all_vcr)
    stats_vitesse = pd.concat(all_vit)
    stats_accel   = pd.concat(all_acc)
    trip_par_segment = pd.concat(all_trip)
    lats_series = pd.concat(all_lat)
    lons_series = pd.concat(all_lon)
    times_series = pd.concat(all_time)

    # final_segment_id fictif continu pour l'index de df_res
    seg_index = pd.RangeIndex(len(hcr_km))

    return seg_index, hcr_km, sr, vcr, stats_vitesse, stats_accel, trip_par_segment, lats_series, lons_series, times_series
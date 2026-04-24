import pandas as pd
import math
import numpy as np

def distance(lat1, lon1, lat2, lon2):
    # Rayon de la Terre en mètres
    R = 6371000 
    
    # Conversion en radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    # Conversion en radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dLon = lon2 - lon1
    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    
    # Calcul de l'angle avec atan2
    bearing = np.degrees(np.atan2(y, x))
    return (bearing + 360) % 360  # Normalise entre 0 et 360°

def main(path):

    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['LOCAL_DATE']+' '+df['UTC_TIME'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by='time')
    df['time_diff'] = df['time'].diff().dt.total_seconds()

    # découper en trajets (gap > 20 min = nouveau trajet)
    df['trip_id'] = (df['time_diff'] > 1200).cumsum()
    # les accélérations/distances ne sont pas calculées entre la fin d'un trajet et le début du suivant
    df.loc[df['time_diff'] > 1200, 'time_diff'] = np.nan

    df['speed_diff'] = df['SPEED'].diff()
    df['acceleration'] = df['speed_diff'] / df['time_diff']

    Vt = 1.4 # Vitesse max marche
    At = 0.5 # Accélération max marche
    df['is_walk'] = ((df['SPEED'] < Vt) & (df['acceleration'] < At)).astype(int)

    # forcer un changement de segment à chaque nouveau trajet
    df.loc[df['time_diff'].isna(), 'is_walk'] = -1  # valeur impossible = change_point garanti

    df['change_point'] = df['is_walk'].diff().fillna(0).abs()

    # On crée un ID unique pour chaque segment
    df['segment_id'] = df['change_point'].cumsum()

    # Calcul de l'angle entre deux points

    # .shift(-1) récupère la valeur de la ligne SUIVANTE
    df['lat_next'] = df['LATITUDE'].shift(-1)
    df['lon_next'] = df['LONGITUDE'].shift(-1)

    # np.vectorize permet d'appliquer la fonction très rapidement sur tout le tableau
    df['distance_m'] = np.vectorize(distance)(
        df['LATITUDE'], df['LONGITUDE'], 
        df['lat_next'], df['lon_next']
    )
    # on remplace le dernier NaN (le dernier point n'a pas de suivant) par 0
    df['distance_m'] = df['distance_m'].fillna(0)

    distance_totale_segment = df.groupby('segment_id')['distance_m'].sum()

    # segments certains
    df['is_certain'] = df['segment_id'].map(distance_totale_segment > 30)
    
    # Propager le segment_id du voisin certain vers l'arrière
    df['final_segment_id'] = df['segment_id']

    # Pour chaque segment incertain, lui donner l'id du dernier segment certain
    last_certain_id = None
    for idx in df.index:
        if df.loc[idx, 'is_certain']:
            last_certain_id = df.loc[idx, 'segment_id']
        elif last_certain_id is not None:
            df.loc[idx, 'final_segment_id'] = last_certain_id

    distance_totale_final_segment = df.groupby('final_segment_id')['distance_m'].sum()


    # HCR 

    df['angle'] = df.apply(lambda row: calculate_bearing(
        row['LATITUDE'], 
        row['LONGITUDE'], 
        row['lat_next'], 
        row['lon_next']
    ), axis=1)

    # Différence d'angle entre le déplacement actuel et le précédent
    df['diff_angle'] = df['angle'].diff().abs()

    # Correction pour le passage du Nord (si on passe de 359° à 1°, la diff est 2, pas 358)
    df.loc[df['diff_angle'] > 180, 'diff_angle'] = 360 - df['diff_angle']

    # Seuil Hc = 19 degrés
    df['is_PC'] = (df['diff_angle'] > 19).astype(int)

    nb_pc = df.groupby('final_segment_id')['is_PC'].sum()

    hcr_km = (nb_pc / distance_totale_final_segment) * 1000 # nb de virages par km



    # SR 

    Vs = 3.4
    df['is_PS'] = ((df['SPEED'] < Vs) & (df['SPEED'].shift(1) >= 3.4)).astype(int)
    nb_ps = df.groupby('final_segment_id')['is_PS'].sum()
    # SR basé sur le nombre d'arrêts distincts
    sr = (nb_ps/distance_totale_final_segment) * 1000 # nb de stops par km


    # VCR

    df['diff_v'] = df['SPEED'].diff().abs()
    df['v_ratio'] = df['diff_v'] / df['SPEED'].shift(1).replace(0, np.nan) # On évite la division par zéro avec .replace(0, np.nan)

    Vr = 0.26
    df['is_Vc'] = (df['v_ratio'] > Vr).astype(int)

    nb_vr = df.groupby('final_segment_id')['is_Vc'].sum()

    # VCR par kilomètre pour chaque segment (velocity change rate)
    vcr = (nb_vr/distance_totale_final_segment)*1000

    # Vitesse max par segment
    stats_vitesse = df.groupby('final_segment_id')['SPEED'].max()

    # Accélération max par segment
    stats_accel = df.groupby('final_segment_id')['acceleration'].max()

    trip_par_segment = df.groupby('final_segment_id')['trip_id'].first()

    return df['final_segment_id'], hcr_km, sr, vcr, stats_vitesse, stats_accel, trip_par_segment

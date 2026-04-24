import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Geolife Trajectories 1.3\Data"

def distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Rayon Terre en mètres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dy = np.sin(lon2 - lon1) * np.cos(lat2)
    dx = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.atan2(dy, dx))

def extract_features(df):
    if len(df) < 2: return None
    
    # 1. Calculs de base : Distance et Temps
    df = df.sort_values('datetime')
    df['dist'] = distance(df['lat'].shift(), df['lon'].shift(), df['lat'], df['lon'])
    df['dt'] = df['datetime'].diff().dt.total_seconds()
    
    # 2. Vitesse et Accélération
    df['speed'] = df['dist'] / df['dt']
    df['speed'] = df['speed'].replace([np.inf, -np.inf], 0).fillna(0)
    df['accel'] = df['speed'].diff() / df['dt']
    df['accel'] = df['accel'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Orientation (Heading)
    df['bearing'] = calculate_bearing(df['lat'].shift(), df['lon'].shift(), df['lat'], df['lon'])
    df['bearing_diff'] = df['bearing'].diff().abs()
    df.loc[df['bearing_diff'] > 180, 'bearing_diff'] = 360 - df['bearing_diff']
    
    # --- INDICATEURS DU PAPIER ---
    dist_totale_km = df['dist'].sum() / 1000
    if dist_totale_km == 0: return None
    
    # HCR : Heading Change Rate (Seuil 15°)
    hcr = (df['bearing_diff'] > 19).sum() / dist_totale_km
    
    # SR : Stop Rate (Vitesse < 0.5 m/s)
    df['is_stop'] = ((df['speed'] < 3.4) & (df['speed'].shift(1) >= 3.4)).astype(int)
    sr = df['is_stop'].sum() / dist_totale_km
    
    # VCR : Velocity Change Rate
    vcr = (df['speed'].diff().abs() > 0.26 * df['speed'].shift()).sum() / dist_totale_km
    
    return {
        'hcr': hcr, 'sr': sr, 'vcr': vcr,
        'v_max': df['speed'].max(),
        'a_max': df['accel'].max()
    }

# --- BOUCLE PRINCIPALE ---
NB_USERS_VOULUS = 30
all_features = []
users_traites = 0

# On trie pour avoir un ordre cohérent
utilisateurs = sorted(os.listdir(DATA_ROOT))

for user_id in utilisateurs:
    if users_traites >= NB_USERS_VOULUS:
        break
    
    user_path = os.path.join(DATA_ROOT, user_id)
    label_file = os.path.join(user_path, 'labels.txt')
    
    if os.path.exists(label_file):
        print(f"[{users_traites + 1}/{NB_USERS_VOULUS}] Extraction User {user_id}...")
        
        # 1. Charger les labels
        labels = pd.read_csv(label_file, sep='\t', skiprows=1, header=None, names=['start', 'end', 'mode'])
        labels['start'] = pd.to_datetime(labels['start'])
        labels['end'] = pd.to_datetime(labels['end'])
        
        # 2. Charger TOUTES les données GPS de l'utilisateur d'un coup (plus rapide)
        traj_dir = os.path.join(user_path, 'Trajectory')
        all_points = []
        
        # On ne lit que les 20 premiers fichiers .plt pour aller vite
        plt_files = [f for f in os.listdir(traj_dir) if f.endswith('.plt')][:20] 
        
        for plt_file in plt_files:
            plt_path = os.path.join(traj_dir, plt_file)
            df_plt = pd.read_csv(plt_path, skiprows=6, header=None, 
                                 names=['lat', 'lon', 'x', 'alt', 'days', 'date', 'time'])
            df_plt['datetime'] = pd.to_datetime(df_plt['date'] + ' ' + df_plt['time'])
            all_points.append(df_plt[['lat', 'lon', 'datetime']])
        
        if not all_points: continue
        df_user = pd.concat(all_points).sort_values('datetime')

        # 3. Extraire les segments basés sur les labels
        for _, row in labels.iterrows():
            # On cherche les points entre start et end dans le gros DataFrame de l'utilisateur
            mask = (df_user['datetime'] >= row['start']) & (df_user['datetime'] <= row['end'])
            segment = df_user.loc[mask].reset_index(drop=True)
            
            if len(segment) > 10:
                feat = extract_features(segment)
                if feat:
                    feat['label'] = row['mode']
                    all_features.append(feat)
        
        users_traites += 1

# Sauvegarde
pd.DataFrame(all_features).to_csv("geolife_train.csv", index=False)
print(f"\nFini ! {len(all_features)} segments extraits.")
import pandas as pd
import numpy as np
import os
import classification_points_segments # Import de ton module pour calculate_bearing et distance

# --- CONFIGURATION ---
GPS_FOLDER = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
DISPLACEMENTS_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\displacements_dataset.csv"
MAX_USERS = 100  # LIMITE : Max 50 utilisateurs

def generate_netmob_train():
    # 1. Charger les labels réels
    df_trips = pd.read_csv(DISPLACEMENTS_PATH)
    df_trips = df_trips[df_trips['Mode_1'].notna()].copy()
    
    all_features = []
    users_processed = 0
    
    # 2. Lister les fichiers GPS
    gps_files = [f for f in os.listdir(GPS_FOLDER) if f.endswith(".csv")]
    
    for filename in gps_files:
        if users_processed >= MAX_USERS:
            print(f"Quota de {MAX_USERS} utilisateurs atteint.")
            break # ARRÊT DE LA BOUCLE[cite: 8]
        
        user_id = filename.replace(".csv", "")
        gps_path = os.path.join(GPS_FOLDER, filename)
        
        print(f"[{users_processed + 1}/{MAX_USERS}] Extraction User {user_id}...")
        
        try:
            # On appelle ton main qui gère la segmentation et l'extraction[cite: 8]
            # Assure-toi que ton main retourne bien les IDs de trajets associés aux segments
            _, hcr, sr, vcr, vmax, amax, trip_ids, lat, lon, time = classification_points_segments.main(gps_path, DISPLACEMENTS_PATH)
            
            if hcr is None or hcr.empty:
                continue
            
            aligned_trip_ids = trip_ids.loc[hcr.index]

            # 3. Création du DataFrame de features pour cet utilisateur[cite: 7]
            temp_df = pd.DataFrame({
                'hcr': hcr.values,
                'sr': sr.values,
                'vcr': vcr.values, 
                'v_max': vmax.values,
                'a_max': amax.values,
                'trip_key': aligned_trip_ids.values
            })
            
            # 4. Jointure avec le fichier displacements pour avoir le VRAI mode[cite: 8]
            temp_df = temp_df.merge(df_trips[['KEY', 'Mode_1']], left_on='trip_key', right_on='KEY')
            temp_df = temp_df.rename(columns={'Mode_1': 'label'})
            
            if not temp_df.empty:
                all_features.append(temp_df[['hcr', 'sr', 'vcr', 'v_max', 'a_max', 'label']])
                users_processed += 1 # COMPTEUR[cite: 8]
            
        except Exception as e:
            print(f"Erreur sur {user_id} : {e}")

    # 5. Sauvegarde du fichier d'entraînement[cite: 8]
    if all_features:
        final_df = pd.concat(all_features).dropna()
        # Normalisation des labels (tout en majuscule)[cite: 9]
        final_df['label'] = final_df['label'].astype(str).str.strip().str.upper()
        # Mapping de simplification pour stabiliser l'arbre
        mapping = {
            'PRIV_CAR_DRIVER': 'CAR', 'PRIV_CAR_PASSENGER': 'CAR', 'TWO_WHEELER': 'CAR',
            'TRAIN': 'TRAIN', 'SUBWAY': 'SUBWAY', 'TRAMWAY': 'TRAMWAY', 'TRAIN_EXPRESS': 'TRAIN',
            'WALKING': 'WALK',
            'BUS': 'BUS',
            'ELECT_BIKE': 'BIKE', 'BIKE': 'BIKE'
        }

        # Appliquer le regroupement
        final_df['label'] = final_df['label'].map(mapping)
        final_df.to_csv("netmob_train.csv", index=False)
        print(f"\nTerminé ! {len(final_df)} segments extraits pour {users_processed} utilisateurs.")
    else:
        print("Aucune donnée n'a pu être extraite.")

if __name__ == "__main__":
    generate_netmob_train()
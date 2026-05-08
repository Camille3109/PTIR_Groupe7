import pandas as pd
import os
import json
import  classification_segments_v1

GPS_FOLDER = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
DISPLACEMENTS_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\displacements_dataset.csv"
OUTPUT_DIR = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split"

# ─────────────────────────────────────────────
# 1. CHARGER LE SPLIT 
# ─────────────────────────────────────────────
train_file = os.path.join(OUTPUT_DIR, "train_users.json")

if not os.path.exists(train_file):
    print(" ERREUR: Lance d'abord 1_setup_split.py")
    exit()

with open(train_file, 'r') as f:
    train_users = json.load(f)[:50]

print(f"\n Entraînement sur {len(train_users)} users")
print(f"   Depuis: {train_file}")

# ─────────────────────────────────────────────
# 2. EXTRAIRE LES FEATURES DES TRAIN USERS
# ─────────────────────────────────────────────
def generate_netmob_train():
    """Génère le csv permettant d'entrainer l'arbre en associant les vrais modes de transport aux caractéristiques statistiques utilisées """
    df_trips = pd.read_csv(DISPLACEMENTS_PATH)
    df_trips = df_trips[df_trips['Mode_1'].notna()].copy()
    
    all_features = []
    users_processed = 0
    
    # Lister TOUS les fichiers GPS
    all_gps_files = sorted([f.replace('.csv', '') for f in os.listdir(GPS_FOLDER) if f.endswith('.csv')])
    
    print(f"\n Traitement de {len(train_users)} users...\n")
    
    for user_id in all_gps_files:
        # IMPORTANT: Traiter SEULEMENT les train_users
        if user_id not in train_users:
            continue
        
        gps_path = os.path.join(GPS_FOLDER, f"{user_id}.csv")
        users_processed += 1
        
        print(f"[{users_processed}/{len(train_users)}] {user_id}...", end=" ", flush=True)
        
        try:
            result = classification_segments_v1.main(gps_path, DISPLACEMENTS_PATH)

            _, hcr, sr, vcr, vmax, amax, v_max_abs_all, v_p99_all, v_med_all, \
            pct_rapide_all, pct_tres_rap_all, \
            duree_all, longueur_all, trip_ids, lat, lon, time = result
            
            if hcr is None or hcr.empty:
                print(" Pas de features")
                continue
            
            aligned_trip_ids = trip_ids.loc[hcr.index]

            temp_df = pd.DataFrame({
                'hcr':        hcr.values,
                'sr':         sr.values,
                'vcr':        vcr.values,
                'v_max':      vmax.values,
                'a_max':      amax.values,
                'v_max_abs_all' : v_max_abs_all.values,
                'v_p99_all' : v_p99_all.values,
                'v_med_all' : v_med_all.values,
                'pct_rapide_all' : pct_rapide_all.values,
                'pct_tres_rap_all' : pct_tres_rap_all.values,
                'duree_all'  : duree_all.values,
                'longueur_all' : longueur_all.values,
                'trip_key':   aligned_trip_ids.values,
            })
            
            # Merger avec les vrais modes
            temp_df = temp_df.merge(
                df_trips[['KEY', 'Mode_1']], 
                left_on='trip_key', 
                right_on='KEY',
                how='inner'
            )
            temp_df = temp_df.rename(columns={'Mode_1': 'label'})
            
            if not temp_df.empty:
                all_features.append(temp_df[['hcr', 'sr', 'vcr', 'v_max', 'a_max',
                                             'v_max_abs_all', 'v_p99_all', 'v_med_all',
                                             'pct_rapide_all', 'pct_tres_rap_all',
                                            'duree_all', 'longueur_all', 'trip_key', 'label']])
                print(f"Ok! {len(temp_df)} segments")
            else:
                print(" Merge failed")
            
        except Exception as e:
            print(f" Erreur: {str(e)[:50]}")

    # ─────────────────────────────────────────
    # 3. SAUVEGARDER netmob_train.csv
    # ─────────────────────────────────────────
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True).dropna()
        
        # Normalisation des labels
        final_df['label'] = final_df['label'].astype(str).str.strip().str.upper()
        
        # Mapping de modes (à adapter si besoin)
        mapping = {
            'PRIV_CAR_DRIVER': 'CAR', 'PRIV_CAR_PASSENGER': 'CAR', 'TWO_WHEELER': 'CAR',
            'TRAIN': 'TRAIN', 'SUBWAY': 'SUBWAY', 'TRAMWAY': 'TRAMWAY', 'TRAIN_EXPRESS': 'TRAIN',
            'WALKING': 'WALK', 'BUS': 'BUS',
            'ELECT_BIKE': 'CYCLING', 'BIKE': 'CYCLING', 'ELECT_SCOOTER' : 'CAR'
        }
        final_df['label'] = final_df['label'].map(mapping)
        
        # Sauvegarder
        output_file = os.path.join(OUTPUT_DIR, "netmob_train.csv")
        final_df.to_csv(output_file, index=False)
        
        print(f"\n Terminé!")
        print(f"   Segments: {len(final_df)}")
        print(f"   Users: {users_processed}")
        print(f"   Fichier: {output_file}")
        
        # Distribution
        print(f"\n Distribution des modes:")
        for mode, count in final_df['label'].value_counts().items():
            pct = count / len(final_df) * 100
            print(f"   {mode}: {count:6d} ({pct:5.1f}%)")
    else:
        print(" Aucune donnée extraite")

if __name__ == "__main__":
    generate_netmob_train()
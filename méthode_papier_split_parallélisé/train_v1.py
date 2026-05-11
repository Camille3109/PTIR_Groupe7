import pandas as pd
import os
import json
import multiprocessing as mp
import classification_segments_v1

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

GPS_FOLDER         = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
DISPLACEMENTS_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\displacements_dataset.csv"
OUTPUT_DIR         = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split_parallélisé"
N_WORKERS          = max(1, mp.cpu_count() - 1)

MAPPING_MODES = {
    'PRIV_CAR_DRIVER':    'CAR',
    'PRIV_CAR_PASSENGER': 'CAR',
    'TWO_WHEELER':        'CAR',
    'ELECT_SCOOTER':      'CAR',
    'TRAIN':              'TRAIN',
    'TRAIN_EXPRESS':      'TRAIN',
    'SUBWAY':             'SUBWAY',
    'TRAMWAY':            'TRAMWAY',
    'WALKING':            'WALK',
    'BUS':                'BUS',
    'ELECT_BIKE':         'CYCLING',
    'BIKE':               'CYCLING',
}


# ─────────────────────────────────────────────
# INITIALIZER  (appelé une fois par worker)
# ─────────────────────────────────────────────

def _init_worker(df_trips_shared):
    """
    Stocke df_trips dans une variable globale du worker.
    Évite de re-lire le CSV à chaque appel de process_one_user_train.
    """
    global _DF_TRIPS
    _DF_TRIPS = df_trips_shared


# ─────────────────────────────────────────────
# FONCTION WORKER  (top-level → picklable)
# ─────────────────────────────────────────────

def process_one_user_train(args):
    """
    Extrait les features GPS d'un user et les merge avec les vrais modes.
    Retourne un DataFrame ou None en cas d'échec.
    """
    user_id, gps_path = args

    try:

        result = classification_segments_v1.main(gps_path, DISPLACEMENTS_PATH, user_id)

        _, hcr, sr, vcr, vmax, amax, v_max_abs_all, v_p99_all, v_med_all, \
            pct_rapide_all, pct_tres_rap_all, \
            duree_all, longueur_all, trip_ids, lat, lon, time = result

        if hcr is None or hcr.empty:
            return None, user_id, "Pas de features"

        aligned_trip_ids = trip_ids.loc[hcr.index]

        temp_df = pd.DataFrame({
            'hcr':             hcr.values,
            'sr':              sr.values,
            'vcr':             vcr.values,
            'v_max':           vmax.values,
            'a_max':           amax.values,
            'v_max_abs_all':   v_max_abs_all.values,
            'v_p99_all':       v_p99_all.values,
            'v_med_all':       v_med_all.values,
            'pct_rapide_all':  pct_rapide_all.values,
            'pct_tres_rap_all':pct_tres_rap_all.values,
            'duree_all':       duree_all.values,
            'longueur_all':    longueur_all.values,
            'trip_key':        aligned_trip_ids.values,
        })

        # Merge avec df_trips chargé dans le worker via l'initializer
        temp_df = temp_df.merge(
            _DF_TRIPS[['KEY', 'Mode_1']],
            left_on='trip_key',
            right_on='KEY',
            how='inner',
        ).rename(columns={'Mode_1': 'label'})

        if temp_df.empty:
            return None, user_id, "Merge vide"

        temp_df = temp_df[[
            'hcr', 'sr', 'vcr', 'v_max', 'a_max',
            'v_max_abs_all', 'v_p99_all', 'v_med_all',
            'pct_rapide_all', 'pct_tres_rap_all',
            'duree_all', 'longueur_all',
            'trip_key', 'label',
        ]]
        return temp_df, user_id, f"OK ({len(temp_df)} segments)"

    except Exception as e:
        return None, user_id, f"Erreur : {str(e)[:80]}"


# ─────────────────────────────────────────────
# GÉNÉRATION DU CSV D'ENTRAÎNEMENT
# ─────────────────────────────────────────────

def generate_netmob_train():

    # 1. Chargement du split
    train_file = os.path.join(OUTPUT_DIR, "train_users.json")
    if not os.path.exists(train_file):
        print("ERREUR : Lance d'abord 1_setup_split.py")
        return

    with open(train_file, 'r') as f:
        train_users = set(json.load(f)[:100])

    print(f"\n Entraînement sur {len(train_users)} users")
    print(f"   Depuis : {train_file}")

    # 2. Liste des fichiers GPS à traiter (filtrés sur train_users)
    all_gps = sorted(
        f.replace('.csv', '')
        for f in os.listdir(GPS_FOLDER)
        if f.endswith('.csv')
    )
    args_list = [
        (uid, os.path.join(GPS_FOLDER, f"{uid}.csv"))
        for uid in all_gps
        if uid in train_users
    ]

    print(f"\n Traitement de {len(args_list)} users avec {N_WORKERS} workers…\n")

    # 3. Chargement de df_trips UNE SEULE FOIS dans le processus principal
    df_trips = pd.read_csv(DISPLACEMENTS_PATH)
    df_trips = df_trips[df_trips['Mode_1'].notna()].copy()

    # 4. Pool avec initializer : df_trips est copié une fois dans chaque worker
    all_features = []
    users_ok     = 0
    users_failed = 0

    with mp.Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
        initargs=(df_trips,),
    ) as pool:
        for i, (df_user, user_id, msg) in enumerate(
            pool.imap_unordered(process_one_user_train, args_list, chunksize=3),
            start=1,
        ):
            pct = i / len(args_list) * 100
            print(f"  [{i:3d}/{len(args_list)}  {pct:5.1f}%]  {user_id}  →  {msg}")

            if df_user is not None:
                all_features.append(df_user)
                users_ok += 1
            else:
                users_failed += 1

    # 5. Assemblage et sauvegarde
    if not all_features:
        print("\n Aucune donnée extraite.")
        return

    final_df = pd.concat(all_features, ignore_index=True).dropna()

    # Normalisation des labels
    final_df['label'] = (
        final_df['label']
        .astype(str).str.strip().str.upper()
        .map(MAPPING_MODES)
    )
    final_df = final_df.dropna(subset=['label'])  # Retire les modes non mappés

    output_file = os.path.join(OUTPUT_DIR, "netmob_train.csv")
    final_df.to_csv(output_file, index=False)

    print(f"\n Terminé !")
    print(f"   Segments extraits : {len(final_df)}")
    print(f"   Users réussis     : {users_ok} / {len(args_list)}")
    print(f"   Users échoués     : {users_failed}")
    print(f"   Fichier           : {output_file}")

    print(f"\n Distribution des modes :")
    for mode, count in final_df['label'].value_counts().items():
        pct = count / len(final_df) * 100
        print(f"   {mode:12s} : {count:6d}  ({pct:5.1f} %)")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    generate_netmob_train()
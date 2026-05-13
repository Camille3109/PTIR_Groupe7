import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import  classification_segments_v2
import os

TRAIN_DATA_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split_parallélisé\netmob_train.csv"
DISPLACEMENTS_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\displacements_dataset.csv"

def train_global_model(train_path):
    """Entraîne le modèle une seule fois."""
    df_train = pd.read_csv(train_path).dropna()
    df_train['label'] = df_train['label'].str.upper()

    # Sélection des features

    X_train = df_train[['hcr', 'sr', 'vcr', 'v_max', 'a_max', 
                    ]].copy()
    X_train['v_over_a'] = X_train['v_max'] / (X_train['a_max'] + 1e-6)
    X_train['hcr_over_vcr'] = X_train['hcr'] / (X_train['vcr'] + 1e-6)
    X_train['sr_v_product'] = X_train['sr'] * X_train['v_max']
    
    y_train = df_train['label']

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    indices_to_keep = ~X_train.isna().any(axis=1)
    X_train = X_train[indices_to_keep]
    y_train = y_train[indices_to_keep]

    # Entraînement de l'arbre
    clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=20,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1 # Utilise tous tes processeurs pour aller plus vite
    )
    clf.fit(X_train, y_train)
    
    return clf

# --- INITIALISATION UNIQUE ---
# Chargement en mémoire au démarrage du script
GLOBAL_CLF = train_global_model(TRAIN_DATA_PATH)


def arbre(gps_path):
    """
    Prédit le mode de transport pour un nouveau fichier GPS.
    Utilise le modèle déjà entraîné.
    """
    # 1. Extraction des segments et features du fichier de test
    user_id = os.path.splitext(os.path.basename(gps_path))[0]
    
    _, hcr_km, sr, vcr, stats_vit, stats_accel, trip_par_segment, lats, lons, time, time_f = \
        classification_segments_v2.main(gps_path, DISPLACEMENTS_PATH, user_id)

    # 2. Construction du DataFrame de test
    features_df = pd.DataFrame({
        'hcr':        hcr_km.values,
        'sr':         sr.values,
        'vcr':        vcr.values,
        'v_max':      stats_vit.values,
        'a_max':      stats_accel.values,
    }, index=hcr_km.index)

    times_series = time.loc[features_df.index]
    times_fin_series = time_f.loc[features_df.index]

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # On identifie les lignes valides (sans NaN)
    indices_valides = features_df.notna().all(axis=1)
    
    # On ne garde que ce qui est propre
    features_df = features_df[indices_valides]
    trip_par_segment = trip_par_segment.loc[features_df.index]
    lats = lats.loc[features_df.index]
    lons = lons.loc[features_df.index]

    # Features def dans train_global_model
    features_df['v_over_a'] = features_df['v_max'] / (features_df['a_max'] + 1e-6)
    features_df['hcr_over_vcr'] = features_df['hcr'] / (features_df['vcr'] + 1e-6)
    features_df['sr_v_product'] = features_df['sr'] * features_df['v_max']
    
    # Aligner les métadonnées sur les lignes valides
    trip_par_segment = trip_par_segment.loc[features_df.index]
    
    # 3. Prédiction
    mes_predictions = GLOBAL_CLF.predict(features_df)
    mes_probas = GLOBAL_CLF.predict_proba(features_df)

    # 4. Formatage du résultat
    classes = GLOBAL_CLF.classes_
    df_res = pd.DataFrame(mes_probas, columns=classes, index=features_df.index)
    df_res['Mode'] = mes_predictions
    df_res['Confiance'] = df_res[classes].max(axis=1)
    df_res['trip_id'] = trip_par_segment.values
    df_res['LATITUDE'] = lats.loc[features_df.index].values
    df_res['LONGITUDE'] = lons.loc[features_df.index].values
    df_res['TIMESTAMP'] = times_series.loc[features_df.index].values
    df_res['TIMESTAMP_FIN'] = times_fin_series.loc[features_df.index].values
    
    df_train = pd.read_csv(TRAIN_DATA_PATH).dropna()

    return df_train, df_res, DISPLACEMENTS_PATH
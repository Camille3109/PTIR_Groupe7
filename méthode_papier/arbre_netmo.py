import pandas as pd
import classification_points_segments
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def arbre(GPS_PATH):
    DISPLACEMENTS_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\displacements_dataset.csv" 

    _, hcr_km, sr, vcr, stats_vit, stats_accel, trip_par_segment, lats, lons, times = \
    classification_points_segments.main(GPS_PATH, DISPLACEMENTS_PATH)


    features_df = pd.DataFrame({
        'hcr':   hcr_km.values,
        'sr':    sr.values,
        'vcr':   vcr.values,
        'v_max': stats_vit.values,
        'a_max': stats_accel.values,
    }, index=hcr_km.index).fillna(0)
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

    # trip_id aligné sur les lignes de features_df
    trip_par_segment = trip_par_segment.loc[features_df.index]

    # Charger les données d'entrainement
    df_train = pd.read_csv("netmob_train.csv")


    # Nettoyage et préparation
    # On normalise les labels pour éviter les doublons (ex: 'car' vs 'Car')
    df_train['label'] = df_train['label'].str.upper()
    df_train = df_train.dropna()
    X_train = df_train[['hcr', 'sr', 'vcr', 'v_max', 'a_max']].copy()
    X_train.loc[:, 'hcr'] = X_train['hcr'].apply(lambda x: 0.0 if x < 0.02 else x)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(0)
    y_train = df_train['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    # Normaliser aussi les features de test
    X_test = features_df.copy()
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


    #print("Distribution des classes :")
    #print(y_train.value_counts(normalize=True))

    # 3. Entraînement avec poids équilibrés (très important si tu as beaucoup de voitures)
    clf = DecisionTreeClassifier(
        max_depth=10, 
        min_samples_leaf=5, 
        min_samples_split=20, 
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)


    # 4. Visualisation des règles spécifiques à NetMob
    #print("\n--- RÈGLES DE DÉCISION NETMOB ---")
    #print(export_text(clf, feature_names=['hcr', 'sr', 'vcr', 'v_max', 'a_max']))

    mes_predictions = clf.predict(X_test_scaled)
    mes_probas      = clf.predict_proba(X_test_scaled)

    classes = clf.classes_
    df_res  = pd.DataFrame(mes_probas, columns=classes, index=features_df.index)
    df_res['Mode']      = mes_predictions
    df_res['Confiance'] = df_res[classes].max(axis=1)
    df_res['trip_id']   = trip_par_segment.values

    df_res['LATITUDE'] = lats.loc[features_df.index].values  # Aligné sur l'index filtré
    df_res['LONGITUDE'] = lons.loc[features_df.index].values

    return df_train, df_res, DISPLACEMENTS_PATH
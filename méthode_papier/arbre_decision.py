import classification_points_segments
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

final_sgm_id, hcr_km, sr, vcr, stats_vit, stats_accel, trip_par_segment = classification_points_segments.main(r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset\2_1790.csv")

# On crée un DataFrame où chaque ligne est un segment unique
features_df = pd.DataFrame({
    'hcr': hcr_km,
    'sr': sr,
    'vcr': vcr,
    'v_max': stats_vit,
    'a_max': stats_accel
}).fillna(0)

# On remplace les infinis par des NaN et on les supprime
features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

# On charge le dataset de données d'entrainement
df = pd.read_csv("geolife_train.csv")

# On définit les entrées (X) et la sortie (y)
X = df[['hcr', 'sr', 'vcr', 'v_max', 'a_max']]
y = df['label']

# Entraînement de l'arbre
clf = DecisionTreeClassifier(max_depth=5, random_state=42) # On limite la profondeur à 5 pour éviter qu'il n'apprenne "par coeur" (overfitting)
clf.fit(X, y)

'''
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()'''

mes_predictions = clf.predict(features_df)
mes_probas = clf.predict_proba(features_df)

# On récupère le nom des classes (modes)
classes = clf.classes_
# On crée le DataFrame de résultats avec les colonnes de probabilités
df_res = pd.DataFrame(mes_probas, columns=classes)
# On ajoute les colonnes nécessaires pour la fonction de post-processing
df_res['Mode'] = mes_predictions
df_res['Confiance'] = df_res[classes].max(axis=1)
df_res['trip_id'] = features_df.index.map(trip_par_segment)





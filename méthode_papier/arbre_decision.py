import classification_points_segments
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

GPS_PATH           = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset\2_1791.csv"
DISPLACEMENTS_PATH = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\displacements_dataset.csv"

_, hcr_km, sr, vcr, stats_vit, stats_accel, trip_par_segment = \
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

df = pd.read_csv("geolife_train.csv")
X  = df[['hcr', 'sr', 'vcr', 'v_max', 'a_max']]
y  = df['label']

print(df['label'].value_counts(normalize=True))
print(df.groupby('label')[['hcr','sr','vcr','v_max','a_max']].mean())

clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
clf.fit(X, y)


tree_rules = export_text(
    clf,
    feature_names=['hcr', 'sr', 'vcr', 'v_max', 'a_max']
)

print(tree_rules)

print(features_df.describe())


mes_predictions = clf.predict(features_df)
mes_probas      = clf.predict_proba(features_df)

classes = clf.classes_
df_res  = pd.DataFrame(mes_probas, columns=classes, index=features_df.index)
df_res['Mode']      = mes_predictions
df_res['Confiance'] = df_res[classes].max(axis=1)
df_res['trip_id']   = trip_par_segment.values
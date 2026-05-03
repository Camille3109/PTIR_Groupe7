from post_processing import lancement_user
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import folium 

dossier = Path(r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset")


def extraire_points_changement(user_id):
    points = []

    try:
        result, precision = lancement_user(user_id)

        # Si lancement_user retourne plusieurs objets
        if isinstance(result, tuple):
            df_res = result[1]
        else:
            df_res = result

        if df_res is None or len(df_res) < 2:
            return points

        required_cols = {"Mode_Final_Norm", "LATITUDE", "LONGITUDE"}
        if not required_cols.issubset(df_res.columns):
            print(f"Colonnes manquantes pour {user_id}: {required_cols - set(df_res.columns)}")
            return points

        df_res = df_res.reset_index(drop=True)

        for i in range(1, len(df_res)):
            mode_before = df_res.loc[i - 1, "Mode_Final_Norm"]
            mode_after = df_res.loc[i, "Mode_Final_Norm"]

            if mode_before != mode_after:
                points.append({
                    "user_id": user_id,
                    "transition": f"{mode_before}→{mode_after}",
                    "LATITUDE": df_res.loc[i, "LATITUDE"],
                    "LONGITUDE": df_res.loc[i, "LONGITUDE"]
                })

    except Exception as e:
        print(f"Erreur user {user_id}: {e}")

    return points, precision

    

all_points = []
count = 0
precision_tot = 0
for element in dossier.iterdir():
    user_id = Path(element.name).stem
    points, precision = extraire_points_changement(user_id)
    precision_tot += precision
    all_points.extend(points)
    count += 1

    if count >= 300:
        break
df_points = pd.DataFrame(all_points)


print(f"\nNombre total de points de changement : {len(df_points)}")

if df_points.empty:
    print("Aucun point trouvé.")
    exit()

print(f"Précision totale : {precision_tot/300} %")

# Sauvegarde CSV
df_points.to_csv("points_changement.csv", index=False)

centre_lat = df_points["LATITUDE"].mean()
centre_lon = df_points["LONGITUDE"].mean()

m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11)

for _, row in df_points.iterrows():
    popup = (
        f"User: {row['user_id']}<br>"
        f"Transition: {row['transition']}"
    )

    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=3,
        popup=popup,
        fill=True
    ).add_to(m)

output_file = "points_changement_map.html"
m.save(output_file)

print(f"Carte enregistrée : {output_file}")








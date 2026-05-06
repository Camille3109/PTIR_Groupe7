import pandas as pd
import numpy as np
import folium 
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import json
import os
import pandas as pd
from méthode_papier_split.post_processing import lancement_user


SPLIT_DIR = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split"
GPS_FOLDER = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"

# 1. CHARGEMENT DU SPLIT
with open(os.path.join(SPLIT_DIR, "train_users.json"), 'r') as f:
    train_users = json.load(f)[:50] # On limite pour le test
with open(os.path.join(SPLIT_DIR, "test_users.json"), 'r') as f:
    test_users = json.load(f)[:100]

def extraire_points_changement(user_id, df_res=None):
    """
    Extrait les points de transition entre deux modes de transport.
    Peut prendre un df_res déjà chargé ou un user_id.
    """
    points = []
    
    # 1. Si df_res n'est pas fourni, on le génère 
    if df_res is None:
        try:
            result, _ = lancement_user(user_id, None)
            df_res = result[1] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {user_id}: {e}")
            return []

    # 2. Vérifications de sécurité
    if df_res is None or len(df_res) < 2:
        return []

    # Nettoyage des colonnes (Gestion des types et NaN)
    col_mode = "Mode_Final_Norm" if "Mode_Final_Norm" in df_res.columns else "Mode"
    
    df_res = df_res.copy()
    df_res[col_mode] = df_res[col_mode].astype(str).replace('nan', np.nan)
    df_res = df_res.dropna(subset=[col_mode, "LATITUDE", "LONGITUDE"])
    df_res = df_res.reset_index(drop=True)

    # 3. Détection des changements de mode
    # On utilise le décalage (shift) de Pandas pour aller 10x plus vite qu'une boucle for
    mode_suivant = df_res[col_mode].shift(-1)
    
    # Un changement est détecté si le mode actuel est différent du suivant (et pas NaN)
    changements = df_res[(df_res[col_mode] != mode_suivant) & (mode_suivant.notna())]

    for idx, row in changements.iterrows():
        m_before = row[col_mode]
        m_after = mode_suivant[idx]
        
        points.append({
            "user_id": user_id,
            "transition": f"{m_before}→{m_after}",
            "LATITUDE": row["LATITUDE"],
            "LONGITUDE": row["LONGITUDE"],
            "mode_before": m_before,
            "mode_after": m_after
        })

    return points


def generer_palette_transitions(df_points):
    """Génère une palette de couleurs pour chaque transition unique."""
    transitions = df_points['transition'].unique()
    
    # Palette de couleurs vibrantes
    palette = plt.colormaps['tab20'].resampled(max(20, len(transitions)))
    
    couleur_map = {}
    for idx, transition in enumerate(sorted(transitions)):
        couleur_map[transition] = to_hex(palette(idx % 20))
    
    return couleur_map

'''
# --- PHASE 1 : APPRENTISSAGE SPATIAL (Sur Train uniquement) ---
all_train_points = []

for user_id in train_users:
    try:
        # On lance l'arbre SANS correction spatiale pour voir où il se trompe/change
        df_res, _ = lancement_user(user_id, spatial_knowledge=None)
        points = extraire_points_changement(user_id, df_res=df_res)
        all_train_points.extend(points)
        print(f"  [Train] {user_id} : {len(points)} points trouvés")
    except Exception as e:
        continue

df_points_train = pd.DataFrame(all_train_points)
spatial_knowledge = build_spatial_knowledge(df_points_train)'''

# --- PHASE 2 : ÉVALUATION (Sur Test uniquement) ---
resultats_test = []
toutes_les_precisions = []
all_test_points = []
for user_id in test_users:
    try:
        # On applique ici le savoir spatial acquis sur le groupe Train, pour cela spatial_knowledge = spatial_knowledge
        df_res_final, precision_finale = lancement_user(user_id, spatial_knowledge=None) 
        toutes_les_precisions.append(precision_finale)
        resultats_test.append({
            'user_id': user_id,
            'precision': precision_finale
        })
        points = extraire_points_changement(user_id, df_res=df_res_final)
        all_test_points.extend(points)
    except Exception as e:
        print(f"  [Test] {user_id} : Erreur {e}")


# --- SYNTHÈSE FINALE ---
df_res = pd.DataFrame(resultats_test)
df_points = pd.DataFrame(all_test_points)
moyenne_globale = df_res['precision'].mean()

print("\n" + "="*40)
print(f"RÉSULTATS FINAUX SUR LE TEST SET")
print(f"Nombre d'utilisateurs testés : {len(df_res)}")
if len(toutes_les_precisions) > 0:
    moyenne = sum(toutes_les_precisions) / len(toutes_les_precisions)
    print(f"Précision totale : {moyenne:.2f} %")
else:
    print("Aucun résultat à calculer.")

# Sauvegarde CSV
df_points.to_csv("points_changement.csv", index=False)

centre_lat = df_points["LATITUDE"].mean()
centre_lon = df_points["LONGITUDE"].mean()

api_key = 'd6047c3a-3cb7-4a7b-8193-98d381efc90e'
tiles_url = f'https://tiles.stadiamaps.com/tiles/osm_bright/{{z}}/{{x}}/{{y}}.png?api_key={api_key}'
m = folium.Map(location=[48.8566, 2.3522], zoom_start=13, tiles=tiles_url, attr='Stadia Maps')

couleur_map = generer_palette_transitions(df_points)

for _, row in df_points.iterrows():
    popup = (
    f"<b>User:</b> {row['user_id']}<br>"
    f"<b>Transition:</b> {row['transition']}")
 
    couleur = couleur_map[row['transition']]


    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=2.5,  # Points plus petits
        popup=popup,
        color=couleur,
        fill=True,
        fillColor=couleur,
        fillOpacity=0.7,
        weight=1
    ).add_to(m)

legend_html = '''
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 280px; height: auto; 
     background-color: white; border:2px solid grey; z-index:9999; 
     font-size:12px; padding: 10px; border-radius: 5px;
     max-height: 400px; overflow-y: auto;">
     <b style="font-size: 14px;">Transitions de modes</b><br>
'''
for transition in sorted(couleur_map.keys()):
    couleur = couleur_map[transition]
    legend_html += f'''
    <i style="background:{couleur}; width: 15px; height: 15px; 
       display: inline-block; border-radius: 50%; margin-right: 5px; border: 1px solid #666;"></i>
    {transition}<br>
    '''
 
legend_html += '</div>'
 
m.get_root().html.add_child(folium.Element(legend_html))

output_file = "points_changement_map.html"
m.save(output_file)

print(f"Carte enregistrée : {output_file}")


# ─────────────────────────────────────────────
# GRAPHIQUES DE DISTRIBUTION
# ─────────────────────────────────────────────
 
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Analyse des changements de modes de transport', fontsize=16, fontweight='bold')
 
# --- GRAPHIQUE 1 : HISTOGRAMME DES TRANSITIONS ---
ax1 = axes[0]
transition_counts = df_points['transition'].value_counts().sort_values(ascending=False)
colors_bar = [couleur_map[trans] for trans in transition_counts.index]
 
bars = ax1.barh(range(len(transition_counts)), transition_counts.values, color=colors_bar, edgecolor='black', linewidth=0.7)
ax1.set_yticks(range(len(transition_counts)))
ax1.set_yticklabels(transition_counts.index, fontsize=10)
ax1.set_xlabel('Nombre d\'occurrences', fontsize=11, fontweight='bold')
ax1.set_title('Distribution des transitions observées', fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
 
# Ajouter les valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars, transition_counts.values)):
    ax1.text(val + 0.1, i, str(int(val)), va='center', fontsize=9, fontweight='bold')
 
# --- GRAPHIQUE 2 : MATRICE DE TRANSITIONS ---
ax2 = axes[1]
 
# Extraire les modes avant et après
df_points['mode_before'] = df_points['transition'].str.split('→').str[0]
df_points['mode_after'] = df_points['transition'].str.split('→').str[1]

 
# Créer une matrice de transitions
tous_modes = sorted(set(df_points['mode_before'].unique()) | set(df_points['mode_after'].unique()))
matrice = pd.DataFrame(0, index=tous_modes, columns=tous_modes)
 
for _, row in df_points.iterrows():
    matrice.loc[row['mode_before'], row['mode_after']] += 1
 
# Heatmap
im = ax2.imshow(matrice.values, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(tous_modes)))
ax2.set_yticks(range(len(tous_modes)))
ax2.set_xticklabels(tous_modes, rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(tous_modes, fontsize=10)
ax2.set_xlabel('Mode de destination', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mode d\'origine', fontsize=11, fontweight='bold')
ax2.set_title('Matrice des transitions (flux)', fontsize=12, fontweight='bold', pad=10)
 
# Ajouter les valeurs dans la matrice
for i in range(len(tous_modes)):
    for j in range(len(tous_modes)):
        val = matrice.values[i, j]
        if val > 0:
            text = ax2.text(j, i, int(val), ha="center", va="center", 
                          color="black" if val < matrice.values.max()/2 else "white", 
                          fontsize=9, fontweight='bold')
 
cbar = plt.colorbar(im, ax=ax2, label='Nombre de transitions')
 
plt.tight_layout()
output_file_graph = "analyse_transitions.png"
plt.savefig(output_file_graph, dpi=300, bbox_inches='tight')
print(f"✓ Graphiques enregistrés : {output_file_graph}")
plt.show()









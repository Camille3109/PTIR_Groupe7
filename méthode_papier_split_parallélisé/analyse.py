import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import json
import os
import multiprocessing as mp
from functools import partial

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SPLIT_DIR = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split"
GPS_FOLDER = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
N_WORKERS  = max(1, mp.cpu_count() - 1)  # Laisse 1 cœur libre pour l'OS


# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES (top-level → picklables)
# ─────────────────────────────────────────────

def extraire_points_changement(user_id, df_res=None):
    """Extrait les points de transition entre deux modes de transport."""
    from post_processing import lancement_user

    points = []

    if df_res is None:
        try:
            result, _ = lancement_user(user_id, None)
            df_res = result[1] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {user_id}: {e}")
            return []

    if df_res is None or len(df_res) < 2:
        return []

    col_mode = "Mode_Final_Norm" if "Mode_Final_Norm" in df_res.columns else "Mode"

    df_res = df_res.copy()
    df_res[col_mode] = df_res[col_mode].astype(str).replace('nan', np.nan)
    df_res = df_res.dropna(subset=[col_mode, "LATITUDE", "LONGITUDE"])
    df_res = df_res.reset_index(drop=True)

    mode_suivant = df_res[col_mode].shift(-1)
    changements  = df_res[(df_res[col_mode] != mode_suivant) & (mode_suivant.notna())]

    for idx, row in changements.iterrows():
        m_before = row[col_mode]
        m_after  = mode_suivant[idx]
        points.append({
            "user_id":    user_id,
            "transition": f"{m_before}→{m_after}",
            "LATITUDE":   row["LATITUDE"],
            "LONGITUDE":  row["LONGITUDE"],
            "mode_before": m_before,
            "mode_after":  m_after,
        })

    return points


def process_one_user(user_id):
    """
    Traite un seul utilisateur.
    Fonction top-level → picklable par multiprocessing sur Windows.

    Chaque worker importe post_processing (et donc arbre_netmob_v2) une seule
    fois grâce au cache de sys.modules : le modèle est entraîné 1× par worker,
    puis réutilisé pour tous les users assignés à ce worker.
    """
    try:
        # Import local : évite de charger le module dans le processus principal
        from post_processing import lancement_user

        df_res_final, precision_finale, infos = lancement_user(user_id, spatial_knowledge=None)
        points = extraire_points_changement(user_id, df_res=df_res_final)

        return {
            "success":   True,
            "user_id":   user_id,
            "precision": precision_finale,
            "infos":     infos,
            "points":    points,
        }

    except Exception as e:
        print(f"  [ERREUR] {user_id} : {e}")
        return {
            "success":   False,
            "user_id":   user_id,
            "precision": None,
            "infos":     None,
            "points":    [],
        }


def generer_palette_transitions(df_points):
    transitions = df_points["transition"].unique()
    palette     = plt.colormaps["tab20"].resampled(max(20, len(transitions)))
    return {t: to_hex(palette(i % 20)) for i, t in enumerate(sorted(transitions))}


# ─────────────────────────────────────────────
# POINT D'ENTRÉE  (obligatoire sur Windows)
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. CHARGEMENT DU SPLIT
    with open(os.path.join(SPLIT_DIR, "train_users.json"), "r") as f:
        train_users = json.load(f)[:50]
    with open(os.path.join(SPLIT_DIR, "test_users.json"), "r") as f:
        test_users = json.load(f)[:300]

    # ── PHASE 2 : ÉVALUATION PARALLÈLE ───────────────────────────────────────
    print(f"\n Lancement sur {len(test_users)} users avec {N_WORKERS} workers…\n")

    resultats_stats        = []
    resultats_test         = []
    toutes_les_precisions  = []
    all_test_points        = []

    # Pool crée N_WORKERS processus ; chacun charge le modèle une seule fois.
    # imap_unordered retourne les résultats dès qu'ils sont prêts (pas d'attente
    # de fin de batch) → idéal pour afficher la progression.
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(
            pool.imap_unordered(process_one_user, test_users, chunksize=5),
            start=1,
        ):
            pct = i / len(test_users) * 100
            uid = result["user_id"]

            if result["success"]:
                prec = result["precision"]
                print(f"  [{i:3d}/{len(test_users)}  {pct:5.1f}%]  {uid}  →  précision = {prec:.1f} %")

                toutes_les_precisions.append(prec)
                resultats_test.append({"user_id": uid, "precision": prec})
                all_test_points.extend(result["points"])

                if isinstance(result["infos"], dict):
                    resultats_stats.append(result["infos"])
            else:
                print(f"  [{i:3d}/{len(test_users)}  {pct:5.1f}%]  {uid}  →  ÉCHEC")

    # ── SYNTHÈSE ──────────────────────────────────────────────────────────────
    df_res    = pd.DataFrame(resultats_test)
    df_points = pd.DataFrame(all_test_points)
    df_stats  = pd.DataFrame(resultats_stats)

    print("\n" + "=" * 50)
    print(f"RÉSULTATS FINAUX SUR LE TEST SET")
    print(f"Utilisateurs testés : {len(df_res)}")
    if toutes_les_precisions:
        print(f"Précision moyenne   : {sum(toutes_les_precisions)/len(toutes_les_precisions):.2f} %")
    else:
        print("Aucun résultat à calculer.")

    # Sauvegarde CSV
    df_points.to_csv("points_changement.csv", index=False)

    # ── CARTE FOLIUM ──────────────────────────────────────────────────────────
    if not df_points.empty:
        api_key   = "d6047c3a-3cb7-4a7b-8193-98d381efc90e"
        tiles_url = (
            f"https://tiles.stadiamaps.com/tiles/osm_bright/{{z}}/{{x}}/{{y}}.png"
            f"?api_key={api_key}"
        )
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=13,
                       tiles=tiles_url, attr="Stadia Maps")

        couleur_map = generer_palette_transitions(df_points)

        for _, row in df_points.iterrows():
            popup  = f"<b>User:</b> {row['user_id']}<br><b>Transition:</b> {row['transition']}"
            couleur = couleur_map[row["transition"]]
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=2.5,
                popup=popup,
                color=couleur,
                fill=True, fillColor=couleur, fillOpacity=0.7, weight=1,
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:50px;right:50px;width:280px;background:white;
                    border:2px solid grey;z-index:9999;font-size:12px;padding:10px;
                    border-radius:5px;max-height:400px;overflow-y:auto;">
        <b style="font-size:14px;">Transitions de modes</b><br>"""
        for transition in sorted(couleur_map):
            c = couleur_map[transition]
            legend_html += (
                f'<i style="background:{c};width:15px;height:15px;display:inline-block;'
                f'border-radius:50%;margin-right:5px;border:1px solid #666;"></i>'
                f"{transition}<br>"
            )
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))
        m.save("points_changement_map.html")
        print("\n Carte enregistrée : points_changement_map.html")

    # ── GRAPHIQUES TRANSITIONS ────────────────────────────────────────────────
    if not df_points.empty:
        couleur_map = generer_palette_transitions(df_points)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle("Analyse des changements de modes de transport",
                     fontsize=16, fontweight="bold")

        # Histogramme
        ax1 = axes[0]
        transition_counts = df_points["transition"].value_counts().sort_values(ascending=False)
        colors_bar = [couleur_map[t] for t in transition_counts.index]
        bars = ax1.barh(range(len(transition_counts)), transition_counts.values,
                        color=colors_bar, edgecolor="black", linewidth=0.7)
        ax1.set_yticks(range(len(transition_counts)))
        ax1.set_yticklabels(transition_counts.index, fontsize=10)
        ax1.set_xlabel("Nombre d'occurrences", fontsize=11, fontweight="bold")
        ax1.set_title("Distribution des transitions observées",
                      fontsize=12, fontweight="bold", pad=10)
        ax1.grid(axis="x", alpha=0.3, linestyle="--")
        for bar, val in zip(bars, transition_counts.values):
            ax1.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(int(val)), va="center", fontsize=9, fontweight="bold")

        # Matrice
        ax2 = axes[1]
        df_points["mode_before"] = df_points["transition"].str.split("→").str[0]
        df_points["mode_after"]  = df_points["transition"].str.split("→").str[1]
        tous_modes = sorted(
            set(df_points["mode_before"].unique()) | set(df_points["mode_after"].unique())
        )
        matrice = pd.DataFrame(0, index=tous_modes, columns=tous_modes)
        for _, row in df_points.iterrows():
            matrice.loc[row["mode_before"], row["mode_after"]] += 1

        im = ax2.imshow(matrice.values, cmap="YlOrRd", aspect="auto")
        ax2.set_xticks(range(len(tous_modes)))
        ax2.set_yticks(range(len(tous_modes)))
        ax2.set_xticklabels(tous_modes, rotation=45, ha="right", fontsize=10)
        ax2.set_yticklabels(tous_modes, fontsize=10)
        ax2.set_xlabel("Mode de destination", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mode d'origine", fontsize=11, fontweight="bold")
        ax2.set_title("Matrice des transitions (flux)",
                      fontsize=12, fontweight="bold", pad=10)
        vmax = matrice.values.max()
        for i in range(len(tous_modes)):
            for j in range(len(tous_modes)):
                val = matrice.values[i, j]
                if val > 0:
                    ax2.text(j, i, int(val), ha="center", va="center",
                             color="black" if val < vmax / 2 else "white",
                             fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax2, label="Nombre de transitions")

        plt.tight_layout()
        plt.savefig("analyse_transitions.png", dpi=300, bbox_inches="tight")
        print(" Graphiques de transitions enregistrés : analyse_transitions.png")
        plt.show()

    # ── GRAPHIQUES DÉMOGRAPHIQUES ─────────────────────────────────────────────
    if not df_stats.empty and "DUREES_MODES" in df_stats.columns:

        couleurs_modes = {
            "WALKING": "#2ecc71",
            "CAR":     "#e74c3c",
            "BUS":     "#3498db",
            "TRAIN":   "#9b59b6",
            "CYCLING": "#f1c40f",
            "SUBWAY":  "#95a5a6",
            "TRAMWAY": "#e67e22",
        }

        df_durees = pd.json_normalize(df_stats["DUREES_MODES"].tolist()).fillna(0)
        df_durees.index = df_stats.index

        tous_modes = [m for m in df_durees.columns if df_durees[m].sum() > 0]
        df_durees  = df_durees[tous_modes]

        categories    = ["SEX", "AGE_GROUP", "DIPLOMA_GROUP"]
        titres_labels = ["Sexe", "Tranche d'âge", "Diplôme"]

        fig, axes = plt.subplots(len(categories), 1, figsize=(14, 6 * len(categories)))
        fig.suptitle(
            "% de durée de déplacement par mode de transport\n(analyse sociodémographique)",
            fontsize=16, fontweight="bold", y=1.01,
        )

        for i, (cat, titre) in enumerate(zip(categories, titres_labels)):
            ax = axes[i]
            if cat not in df_stats.columns:
                ax.set_visible(False)
                continue

            df_merged      = df_durees.copy()
            df_merged[cat] = df_stats[cat].values
            pivot          = df_merged.groupby(cat)[tous_modes].mean()
            colors         = [couleurs_modes.get(m, "#bdc3c7") for m in pivot.columns]

            pivot.plot(kind="bar", stacked=True, ax=ax, color=colors,
                       edgecolor="white", linewidth=0.5, width=0.65)

            ax.set_title(f"Répartition modale moyenne par {titre}",
                         fontsize=13, fontweight="bold", pad=10)
            ax.set_ylabel("% moyen de durée de déplacement", fontsize=11)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
            ax.legend(title="Mode de transport", bbox_to_anchor=(1.01, 1),
                      loc="upper left", fontsize=9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            for container in ax.containers:
                labels_annot = [f"{v:.0f}%" if v >= 5 else "" for v in container.datavalues]
                ax.bar_label(container, labels=labels_annot,
                             label_type="center", fontsize=8,
                             color="white", fontweight="bold")

        plt.tight_layout()
        plt.savefig("analyse_demographique_transport.png", dpi=300, bbox_inches="tight")
        print(" Graphiques démographiques enregistrés : analyse_demographique_transport.png")
        plt.show()

    else:
        if df_stats.empty:
            print("Attention : df_stats est vide.")
        else:
            print("Attention : colonne 'DUREES_MODES' absente de df_stats.")
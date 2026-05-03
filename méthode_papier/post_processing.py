from arbre_netmo import arbre
import pandas as pd
import numpy as np
from collections import Counter


# ─────────────────────────────────────────────
# 0. NORMALISATION DES MODES
# ─────────────────────────────────────────────

MODE_NORMALISATION = {
    'taxi':'CAR',
    'priv_car_passenger':'CAR',
    'priv_car_driver':'CAR',
    'car':'CAR',
    'walk':     'WALKING',
    'bike':     'CYCLING',
    'bus':      'BUS',
    'subway':   'SUBWAY',
    'train':    'TRAIN',
    'run':      'WALKING',
    'boat':     'BOAT',
    'airplane': 'PLANE',
    'walking':  'WALKING',
    'cycling':  'CYCLING',
    'tramway':  'TRAMWAY',
    'metro':    'SUBWAY',
    'train_express' : 'TRAIN',
    'two_wheeler' : 'CAR',
    'nan' : None
}

def normaliser_mode(mode):
    return MODE_NORMALISATION.get(str(mode).strip().lower(), str(mode).strip().upper())


# ─────────────────────────────────────────────
# 1. POST-PROCESSING
# ─────────────────────────────────────────────

def matrice_transition(df_train):
    modes  = df_train['label'].unique()
    matrix = pd.DataFrame(0.01, index=modes, columns=modes)
    labels = df_train['label'].values
    for i in range(len(labels) - 1):
        matrix.loc[labels[i], labels[i+1]] += 1
    return matrix.div(matrix.sum(axis=1), axis=0)


def graphe_post_processing(df_res, transition_matrix, T1=0.6, T2=0.36):
    final_modes = []
    classes = list(transition_matrix.columns)
    for i in range(len(df_res)):
        p_m_x        = df_res.iloc[i][classes].to_dict()
        max_prob     = max(p_m_x.values())
        current_mode = df_res.iloc[i]['Mode']
        if i > 0 and df_res.iloc[i-1]['Confiance'] > T1:
            prev_mode = final_modes[i-1]
            for m in classes:
                if prev_mode in transition_matrix.index:
                    p_m_x[m] *= transition_matrix.loc[prev_mode, m]
            final_modes.append(max(p_m_x, key=p_m_x.get))
        elif max_prob < T2:
            final_modes.append(current_mode)
        else:
            final_modes.append(current_mode)
    return final_modes

def sliding_majority_vote(predictions, window_size=5):
    """Nettoie les prédictions en regardant les voisins proches."""
    if len(predictions) <= window_size:
        return predictions
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        window = predictions[start:end]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed


def fusion_segments(modes_list):
    modes = list(modes_list)
    for i in range(1, len(modes) - 1):
        if modes[i-1] == modes[i+1] and modes[i] != modes[i-1]:
            modes[i] = modes[i-1]
    return modes


def sequence_modes(modes_series):
    """Retourne la séquence compacte des modes (sans doublons consécutifs)."""
    seq = []
    for m in modes_series:
        if not seq or m != seq[-1]:
            seq.append(m)
    return ' → '.join(seq)


def extraire_modes(row):
    if row['ID_Trip_Days'] == 'No_Trip':
        return []
    
    modes_identifies = []
    # On boucle sur les colonnes Mode_1 à Mode_5
    for i in range(1, 6):
        col = f'Mode_{i}'
        if col in row and pd.notna(row[col]):
            valeur_brute = str(row[col]).strip()
            if valeur_brute != '' and valeur_brute.lower() != 'nan':
                # C'EST ICI : On normalise TAXI -> CAR
                modes_identifies.append(normaliser_mode(valeur_brute))
    
    return modes_identifies


def charger_trajets_declares(displacements_path, user_id):
    df_trips = pd.read_csv(displacements_path)
    df_trips = df_trips[df_trips['ID'] == user_id].copy().reset_index(drop=True)
    df_trips['modes_reels'] = df_trips.apply(extraire_modes, axis=1)
    return df_trips

def generer_resume_declares(df_res, df_trips):
    df_decl = df_res[~df_res['is_extra']]
    rows = []

    for _, trip in df_trips.iterrows():
        if trip['ID_Trip_Days'] == 'No_Trip':
            continue
        key      = trip['KEY']
        seg_trip = df_decl[df_decl['trip_id'] == key]

        modes_predits_str = (sequence_modes(seg_trip['Mode_Final_Norm'])
                             if not seg_trip.empty else 'Pas de données GPS')
        modes_reels_str   = ', '.join(trip['modes_reels']) if trip['modes_reels'] else '—'

        rows.append({
            'KEY':           key,
            'Date':          trip.get('Date_O', ''),
            'Début':         trip.get('Time_O', ''),
            'Fin':           trip.get('Time_D', ''),
            'Modes Prédits': modes_predits_str,
            'Modes Réels':   modes_reels_str,
        })

    return pd.DataFrame(rows)


def generer_resume_extra(df_res):
    df_extra = df_res[df_res['is_extra']]
    rows = []

    for trip_id, grp in df_extra.groupby('trip_id', sort=False):
        rows.append({
            'Trip ID':       trip_id,
            'Modes Prédits': sequence_modes(grp['Mode_Final_Norm']),
            'Nb segments':   len(grp),
        })

    return pd.DataFrame(rows)



def comparer_predictions(df_resume):
    rows = []
    for _, r in df_resume.iterrows():
        if r['Modes Prédits'] == 'Pas de données GPS':
            continue
        if r['Modes Réels'] == '—':
            continue

        predits_set = set(r['Modes Prédits'].split(' → '))
        reels_set   = set(r['Modes Réels'].split(', '))

        intersection = predits_set & reels_set
        precision = len(intersection) / len(predits_set) * 100 if predits_set else 0
        rappel    = len(intersection) / len(reels_set)   * 100 if reels_set   else 0
        f1 = (2 * precision * rappel / (precision + rappel)
              if (precision + rappel) > 0 else 0)

        rows.append({
            'KEY':           r['KEY'],
            'Modes_Prédits': r['Modes Prédits'],
            'Modes_Réels':   r['Modes Réels'],
            'Corrects':      ', '.join(sorted(intersection)) or '—',
            'Précision (%)': round(precision, 1),
            'Rappel (%)':    round(rappel, 1),
            'F1 (%)':        round(f1, 1),
        })

    return pd.DataFrame(rows)


# 2. LISSAGE LOCAL (On traite chaque trajet séparément pour ne pas mélanger les jours)
def process_trip(group):
    # On applique le vote glissant sur les prédictions du graphe
    modes_lisses = sliding_majority_vote(group['Mode_Graph'].tolist(), window_size=5)
    # On applique ta fonction de fusion pour boucher les derniers trous
    modes_finaux = fusion_segments(modes_lisses)
    return pd.Series(modes_finaux, index=group.index)

def lancement_user(USER_ID):
    # 1. Calcul de base (Graphe de transition)
    df_train, df_res, DISPLACEMENTS_PATH = arbre(r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"+ f"\{USER_ID}.csv")
    transition_matrix = matrice_transition(df_train)
    df_res['Mode_Graph'] = graphe_post_processing(df_res, transition_matrix)
    #df_res['Mode_Final']  = fusion_segments(df_res['Mode_Graph'])
    df_res['Mode_Final'] = df_res.groupby('trip_id', group_keys=False).apply(process_trip, include_groups=False)

    # 3. Normalisation
    df_res['Mode_Final_Norm'] = df_res['Mode_Final'].apply(normaliser_mode)
    df_res['is_extra'] = df_res['trip_id'].apply(lambda x: str(x).startswith('extra_'))

    # 4. Génération des rapports
    df_trips = charger_trajets_declares(DISPLACEMENTS_PATH, USER_ID)
    df_resume_decl  = generer_resume_declares(df_res, df_trips)
    df_resume_extra = generer_resume_extra(df_res)
    df_comparaison  = comparer_predictions(df_resume_decl)
    print(df_res['Mode'])

    # Transitions
    print("\n" + "=" * 80)
    print("TRAJETS EXTRA (GPS hors fenêtres CSV, coupure à 10 min)")
    print("=" * 80)
    print(df_resume_extra.to_string(index=False))

    if not df_comparaison.empty:
        print("\n" + "=" * 80)
        print("COMPARAISON PRÉDICTIONS vs MODES RÉELS")
        print("=" * 80)
        print(df_comparaison.to_string(index=False))
        print(f"\n  Précision moyenne : {df_comparaison['Précision (%)'].mean():.1f} %")
        print(f"  Rappel moyen      : {df_comparaison['Rappel (%)'].mean():.1f} %")
        print(f"  F1 moyen          : {df_comparaison['F1 (%)'].mean():.1f} %")

    return df_res, df_comparaison['Précision (%)'].mean()
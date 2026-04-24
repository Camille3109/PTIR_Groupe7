from arbre_decision import df, df_res
import pandas as pd


# calcule la probabilité de passer d'un mode à un autre à partir du fichier geolife_train.csv
def matrice_transition(df_train):
    modes = df_train['label'].unique()
    matrix = pd.DataFrame(0.01, index=modes, columns=modes) # 0.01 pour éviter les probas nulles
    
    # On compte les transitions dans le dataset d'entraînement
    labels = df_train['label'].values
    for i in range(len(labels)-1):
        matrix.loc[labels[i], labels[i+1]] += 1
        
    # Normalisation pour avoir des probabilités
    return matrix.div(matrix.sum(axis=1), axis=0)

def graphe_post_processing(df_res, transition_matrix, T1=0.6, T2=0.36):
    """
    T1: Seuil de confiance élevée
    T2: Seuil de confiance faible 
    """
    final_modes = []
    classes = list(transition_matrix.columns)
    
    for i in range(len(df_res)):
        # Probabilités prédites par l'arbre : P(m|X)
        p_m_x = df_res.iloc[i][classes].to_dict()
        max_prob = max(p_m_x.values())
        current_mode = df_res.iloc[i]['Mode']
        
        # --- LOGIQUE DU PAPIER (Figure 8) ---
        # Transition probability-based enhancement
        if i > 0 and df_res.iloc[i-1]['Confiance'] > T1:
            prev_mode = final_modes[i-1]
            # On recalcule : P'(m) = P(m|X) * P(m_actuel | m_precedent)
            for m in classes:
                p_m_x[m] *= transition_matrix.loc[prev_mode, m]
            
            refined_mode = max(p_m_x, key=p_m_x.get)
            final_modes.append(refined_mode)

        # Prior probability-based enhancement
        elif max_prob < T2:
            # Ici, le papier suggère d'utiliser P(m|E_ij) si le segment est dans le graphe
            # À défaut de graphe spatial complet, on garde le mode actuel ou on lisse
            final_modes.append(current_mode)
            
        else:
            # Normal post-processing
            final_modes.append(current_mode)
            
    return final_modes

def fusion_segments_post_classification(df_res, min_points=10):
    """
    Règle de lissage de l'article (Section 4.2) : 
    On fusionne les segments trop courts ou incertains.
    """
    modes = df_res['Mode_Graph'].tolist()
    
    # Parcours pour supprimer les "sauts" de 1 ou 2 segments
    for i in range(1, len(modes) - 1):
        # Si le segment actuel est une 'vibration' entre deux modes identiques
        # Exemple: subway (100 pts) -> walk (1 pt) -> subway (100 pts)
        if modes[i-1] == modes[i+1] and modes[i] != modes[i-1]:
            modes[i] = modes[i-1]
            
    return modes


transition_matrix = matrice_transition(df)
df_res['Mode_Graph'] = graphe_post_processing(df_res, transition_matrix)
df_res['Mode_Final'] = fusion_segments_post_classification(df_res)
# -------------------------------------------------

def generer_resume_trajet(df_resultats, colonne_mode='Mode_Final'):
    phases = []
    if df_resultats.empty:
        return pd.DataFrame()

    mode_actuel = df_resultats[colonne_mode].iloc[0]
    trip_actuel = df_resultats['trip_id'].iloc[0]
    debut_idx = 0

    for i in range(len(df_resultats)):
        mode_seg = df_resultats[colonne_mode].iloc[i]
        trip_seg = df_resultats['trip_id'].iloc[i]

        # Changement de mode OU de trajet = nouvelle phase
        if mode_seg != mode_actuel or trip_seg != trip_actuel:
            phases.append({
                'Trajet': trip_actuel,
                'Mode': mode_actuel,
                'Segment Début': debut_idx,
                'Segment Fin': i - 1,
                'Nombre de Segments': i - debut_idx
            })
            mode_actuel = mode_seg
            trip_actuel = trip_seg
            debut_idx = i

    phases.append({
        'Trajet': trip_actuel,
        'Mode': mode_actuel,
        'Segment Début': debut_idx,
        'Segment Fin': len(df_resultats) - 1,
        'Nombre de Segments': len(df_resultats) - debut_idx
    })

    return pd.DataFrame(phases)

# --- APPLICATION ---
df_voyage = generer_resume_trajet(df_res)

print("\n" + "="*40)
print("RÉSUMÉ SYNTHÉTIQUE DU VOYAGE")
print("="*40)
print(df_voyage.to_string(index=False))

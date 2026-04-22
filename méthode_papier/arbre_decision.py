import classification_points_segments
import pandas as pd

final_sgm_id, hcr_km, sr, vcr, stats_vit, stats_accel = classification_points_segments.main()

# On crée un DataFrame où chaque ligne est un segment unique
features_df = pd.DataFrame({
    'hcr': hcr_km,
    'sr': sr,
    'vcr': vcr,
    'v_max': stats_vit,
    'a_max': stats_accel
}).fillna(0)




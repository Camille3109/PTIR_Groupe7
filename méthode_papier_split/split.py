"""
Script à lancer UNE FOIS pour faire un split propre train/test
Sauvegarde les listes d'users dans des fichiers JSON
"""
import json
import os
from sklearn.model_selection import train_test_split

GPS_FOLDER = r"C:\Users\Camille\Documents\INSA\3A\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
OUTPUT_DIR = r"C:\Users\Camille\Documents\INSA\3A\PTIR\Code\méthode_papier_split"

# ─────────────────────────────────────────────
# RÉCUPÉRER TOUS LES USERS
# ─────────────────────────────────────────────
gps_files = sorted([
    f.replace('.csv', '') 
    for f in os.listdir(GPS_FOLDER) 
    if f.endswith('.csv')
])


# ─────────────────────────────────────────────
# SPLIT 70/30 ALÉATOIRE
# ─────────────────────────────────────────────
train_users, test_users = train_test_split(
    gps_files,
    test_size=0.3,
    random_state=42 
)

print(f"\n Split effectué:")
print(f"   Train: {len(train_users)} users (70%)")
print(f"   Test:  {len(test_users)} users (30%)")

# ─────────────────────────────────────────────
# 3. SAUVEGARDER LES LISTES
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_file = os.path.join(OUTPUT_DIR, "train_users.json")
test_file = os.path.join(OUTPUT_DIR, "test_users.json")

with open(train_file, 'w') as f:
    json.dump(train_users, f, indent=2)

with open(test_file, 'w') as f:
    json.dump(test_users, f, indent=2)

print(f"\n Fichiers sauvegardés:")
print(f"   {train_file}")
print(f"   {test_file}")


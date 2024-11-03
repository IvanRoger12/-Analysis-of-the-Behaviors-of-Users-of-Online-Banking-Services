
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, entropy

# Simulation des données
np.random.seed(42)

# Création de données fictives pour les transactions et heures de connexion
time_steps = 100
data_train = np.random.normal(loc=5, scale=1.5, size=1000)  # Distribution de référence
data_prod = [np.random.normal(loc=5 + 0.05 * t, scale=1.5, size=1000) for t in range(time_steps)]  # Drift progressif

# Heures de connexion
heures_connexion = np.concatenate([np.random.normal(9, 2, 500), np.random.normal(18, 2, 500)])

# Types de transactions
types_transactions = ["paiements", "virements", "consultations"]
transactions_journalières = pd.DataFrame({
    "date": pd.date_range(start="2023-01-01", periods=365, freq="D"),
    "paiements": np.random.poisson(lam=200, size=365),
    "virements": np.random.poisson(lam=100, size=365),
    "consultations": np.random.poisson(lam=300, size=365)
})

# Calcul de la divergence KL et de la distance de Wasserstein
kl_divergence = [entropy(data_train, data) for data in data_prod]
wasserstein_distances = [wasserstein_distance(data_train, data) for data in data_prod]

# Graphique 1 : Distribution des Comportements d’Utilisation par Type de Transaction
plt.figure(figsize=(10, 6))
for i, transaction in enumerate(types_transactions):
    plt.hist(transactions_journalières[transaction], bins=30, alpha=0.5, label=transaction, density=True)
plt.title("Distribution des Comportements d'Utilisation par Type de Transaction")
plt.xlabel("Nombre de Transactions par Jour")
plt.ylabel("Densité")
plt.legend()
plt.show()

# Graphique 2 : Évolution des Distances KL et Wasserstein au Fil du Temps
plt.figure(figsize=(10, 6))
plt.plot(range(time_steps), kl_divergence, label='KL Divergence', color='red')
plt.plot(range(time_steps), wasserstein_distances, label='Wasserstein Distance', color='blue')
plt.title("Évolution des Distances KL et Wasserstein au Fil du Temps")
plt.xlabel("Étapes temporelles")
plt.ylabel("Valeur de la distance")
plt.legend()
plt.show()

# Graphique 3 : Histogramme des Heures de Connexion
plt.figure(figsize=(10, 6))
plt.hist(heures_connexion, bins=24, alpha=0.7, color="skyblue")
plt.title("Histogramme des Heures de Connexion des Utilisateurs")
plt.xlabel("Heure de Connexion")
plt.ylabel("Nombre d'Utilisateurs")
plt.show()

# Graphique 4 : Tendance Mensuelle des Transactions Bancaires
transactions_journalières.set_index("date", inplace=True)
transactions_mensuelles = transactions_journalières.resample("M").sum()

plt.figure(figsize=(10, 6))
for transaction in types_transactions:
    plt.plot(transactions_mensuelles.index, transactions_mensuelles[transaction], label=transaction)
plt.title("Tendance Mensuelle des Transactions Bancaires")
plt.xlabel("Date")
plt.ylabel("Nombre de Transactions")
plt.legend()
plt.show()

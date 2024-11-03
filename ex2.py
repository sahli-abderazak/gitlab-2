import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/User/PycharmProjects/pythonProject1/voiture.csv',delimiter=',', dtype=str)

print(data.head())
e=(data.shape)
print(e)
data['nombre_ventes'] = pd.to_numeric(data['nombre_ventes'], errors='coerce')
mean = data['nombre_ventes'].mean()
print("moy",f"{mean}")

max = data['nombre_ventes'].max()
print("max",f"{max}")

min = data['nombre_ventes'].min()
print("min",f"{min}")

plt.hist(data['nombre_ventes'], bins=20)
plt.xlabel('Nombre de ventes')
plt.ylabel('Fréquence')
plt.title('Histogramme des données')
plt.show()

data.boxplot(column='nombre_ventes')
plt.title('Boxplot du nombre de ventes')
plt.show()

total_ventes_marque = data.groupby('marque')['nombre_ventes'].sum()
print("\nNombre total de ventes pour chaque marque :")
print(total_ventes_marque)

mois_par_marque = data.loc[data.groupby('marque')['nombre_ventes'].idxmax()]
print(mois_par_marque[['marque', 'mois', 'nombre_ventes']])

max_ventes = data.groupby('marque')['nombre_ventes'].sum().idxmax()
print(f"La marque avec le plus grand nombre de ventes est : {max_ventes}")

ventes_par_modele = data.groupby('modele')['nombre_ventes'].sum()
print(ventes_par_modele)

meilleur_mois_par_modele = data.loc[data.groupby('modele')['nombre_ventes'].idxmax()]
print(meilleur_mois_par_modele[['modele', 'mois', 'nombre_ventes']])

modele_max_ventes = data.groupby('modele')['nombre_ventes'].sum().idxmax()
print(f"Le modèle avec le plus grand nombre de ventes est : {modele_max_ventes}")



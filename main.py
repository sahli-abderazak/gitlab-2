import numpy as np
d= np.loadtxt("C:/Users/User/PycharmProjects/pythonProject1/venv/data.csv",delimiter=',', dtype=str, skiprows=1)
a =d[:,2].astype(float)
moy=a.mean()
var=a.var()
ec=a.std()
print("la moyenne est :",moy)
print("la variance est :",var)
print("l'ecart type est :",ec)
x=d[:,3].astype(float)
y=d[:,2].astype(float)
pm=x*y
print(pm)
cumul = np.cumsum(y)
print("la somme cumulative :",cumul)
print("modification features 1")
print("modification feature 2")


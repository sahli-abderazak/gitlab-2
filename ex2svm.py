import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv(r"C:\Users\User\Desktop\ML\svmdata.csv",delimiter=';')
print(df.head())

X = df.drop('classe', axis=1)
y = df['classe']

kernels = ['poly', 'sigmoid', 'linear', 'rbf']

train_sizes = [0.3, 0.5, 0.7, 0.8, 0.9]

results_matrix = np.zeros((len(kernels), len(train_sizes)))

for i in range(len(kernels)):
    kernel = kernels[i]
    for j in range(len(train_sizes)):
        train_size = train_sizes[j]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

        if len(y_train.unique()) > 1:
            svm_model = SVC(kernel=kernel)
            svm_model.fit(X_train, y_train)

            y_pred = svm_model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)

            results_matrix[i, j] = accuracy
            
            
            
 print("modification feature 3 amal") 
 
 
 print("mod3")          
            
            
            
            
            
            
            

results_df = pd.DataFrame(results_matrix, kernels, train_sizes)
print(results_df)

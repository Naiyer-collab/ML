import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
data = {
        "Patient":['A','B','C'],
        "Age":[35,36,70],
        "BP":[120,118,150],
        "Cough":["yes","yes","no"],
        "Fatigue":["yes","yes","no"]
        }
df = pd.DataFrame(data)
print(df.head(10))


scalar = MinMaxScaler()
df[["Age","BP"]] = scalar.fit_transform(df[["Age","BP"]])

numerical_col = ["Age","BP"]
categorical_col = ["Cough","Fatigue"]

def cosine_similarity(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))


def jaccard_Similarity(x,y):
    x_con = (x=="yes").astype(int)
    y_con = (y=="yes").astype(int)

    intersection = np.sum((x_con == 1) & (y_con ==1))
    union = np.sum((x_con == 1) | (y_con ==1))
    return intersection / union if union !=0 else 0

patient = df.index
result = []

for i in range(len(patient)):
    for j in range(i+1, len(patient)):
        p1 = patient[i]
        p2 = patient[j]

        num1 = df.loc[p1,numerical_col].values
        num2 = df.loc[p2,numerical_col].values
        cos = cosine_similarity(num1,num2)

        cat1 = df.loc[p1,categorical_col].values
        cat2 = df.loc[p2,categorical_col].values
        jac = jaccard_Similarity(cat1,cat2)

        overall = 0.5*cos + 0.5*jac

        result.append([f"{p1}-{p2}",cos,jac,overall])


result_df = pd.DataFrame(result, columns=["Pair", "Cosine", "Jaccard", "overall"])
               
print(result_df)

# Predicts students' exam scores using Linear Regression.
# Handles missing data, scales numeric features, encodes categorical features,
# evaluates with R², MAE, RMSE, and plots Predicted vs True and Residuals.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

#  Read  files
df = pd.read_csv("student_habits_performance.csv")

X = df.drop(columns=["exam_score","student_id"], errors="ignore")
y = df["exam_score"]

#  Split data to Numeric and Categorical 
cat = X.select_dtypes(include="object").columns
num = X.select_dtypes(exclude="object").columns

#  Split data to train/test
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

#  Numeric: impute + scale
num_imp = SimpleImputer(strategy="median")
Xtr_n = num_imp.fit_transform(Xtr[num])
Xte_n = num_imp.transform(Xte[num])

scaler = StandardScaler()
Xtr_n = scaler.fit_transform(Xtr_n)
Xte_n = scaler.transform(Xte_n)

#  Categorical: impute + onehot 
if len(cat)>0:
    cat_imp = SimpleImputer(strategy="most_frequent")
    Xtr_c = cat_imp.fit_transform(Xtr[cat])
    Xte_c = cat_imp.transform(Xte[cat])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Xtr_c = ohe.fit_transform(Xtr_c)
    Xte_c = ohe.transform(Xte_c)

    Xtr_f = np.hstack([Xtr_n, Xtr_c])
    Xte_f = np.hstack([Xte_n, Xte_c])
else:
    Xtr_f, Xte_f = Xtr_n, Xte_n

#  Train 
m = LinearRegression()
m.fit(Xtr_f, ytr)
yp = m.predict(Xte_f)

# Metrics 
r2  = r2_score(yte, yp)
mae = mean_absolute_error(yte, yp)
rmse = np.sqrt(mean_squared_error(yte, yp))

print("R²  =", round(r2,3))
print("MAE =", round(mae,3))
print("RMSE=", round(rmse,3))

#  Plot Pred vs True 
#Shows how close the model’s predictions are to the actual exam scores
# — points near the diagonal line mean good accuracy.
plt.figure(figsize=(5,5))
plt.scatter(yte, yp, alpha=0.6)
lims=[min(yte.min(), yp.min()), max(yte.max(), yp.max())]
plt.plot(lims,lims)
plt.xlabel("True"); plt.ylabel("Predicted")
plt.title("Predicted vs True")
plt.tight_layout(); plt.show()


# Plot Residuals
#Shows the prediction errors (true − predicted) to check if the model has bias or patterns
#— random scatter around zero indicates a well-behaved model.
res = yte - yp
plt.figure(figsize=(5,4))
plt.scatter(yp,res, alpha=0.6)
plt.axhline(0,color='black')
plt.xlabel("Predicted"); plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.tight_layout(); plt.show()
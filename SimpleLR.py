import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Total training samples
N = 1000

# 1) Generate train and test data
X_train, y_train = make_regression(
    n_samples=N,
    n_features=1,
    noise=15,
    random_state=42
)

X_test, y_test = make_regression(
    n_samples=N//10,
    n_features=1,
    noise=15,
    random_state=24
)

percentages = []
mse_scores = []
r2_scores = []

# 2) Training subsets: 10%, 20%, ... 100%
for p in range(10, 101, 10):
    size = int((p / 100) * N)

    X_sub = X_train[:size]
    y_sub = y_train[:size]

    # 3) Create model
    model = LinearRegression()
    model.fit(X_sub, y_sub)

    # 4) Test on same test data
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    percentages.append(p)
    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f"Training Data: {p}% | MSE: {mse:.2f} | R2 Score: {r2:.4f}")

# 5) Plot Error graph
plt.figure(figsize=(8,5))
plt.plot(percentages, mse_scores, marker='o')
plt.xlabel("Training Data Size (%)")
plt.ylabel("Mean Squared Error")
plt.title("Impact of Training Data Size on Model Error")
plt.grid(True)
plt.show()

# Plot Accuracy / R2 graph
plt.figure(figsize=(8,5))
plt.plot(percentages, r2_scores, marker='o')
plt.xlabel("Training Data Size (%)")
plt.ylabel("R2 Score")
plt.title("Impact of Training Data Size on Model Performance")
plt.grid(True)
plt.show()
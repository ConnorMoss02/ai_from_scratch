import numpy as np 

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def loss(y, y_hat): 
    return -(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# "Training data" 
np.random.seed(0)

X = np.random.randn(100, 2)
true_w = np.array([2, -3])
true_b = 0.5
y = (X @ true_w + 0.5 > 0).astype(float)

w = np.random.randn(2)
b = 0.0
lr = 0.1

# Forward Pass 
for epoch in range(1000):
    z = X @ w + b
    y_hat = sigmoid(z)

    # Gradients (learning)
    dw = X.T @ (y_hat - y) / len(y)
    db = np.mean(y_hat - y)

    # Update (apply what we learned)
    w -= lr * dw
    b -= lr * db 

    if epoch % 200 == 0: 
        print(epoch, np.mean(loss(y, y_hat)))

# Evaluation
probs = sigmoid(X @ w + b)
preds = (probs > 0.5).astype(float)
accuracy = np.mean(preds == y)

print("\nFinal learned parameters:")
print("w =", w)
print("b =", b)
print("Accuracy:", accuracy)

print("\nTrue parameters (used to generate labels):")
print("true_w =", true_w)
print("true_b =", true_b)
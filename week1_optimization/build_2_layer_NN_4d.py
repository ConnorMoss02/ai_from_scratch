import numpy as np

# -----------------------
# Setup
# -----------------------

np.random.seed(42)

N = 100      # number of samples
D = 2        # input features
H = 16       # hidden units
C = 3        # number of classes

# Fake data
X = np.random.randn(N, D)
y = np.random.randint(0, C, size=N)

# Weight initialization (small random values)
W1 = np.random.randn(D, H) * 0.01
W2 = np.random.randn(H, C) * 0.01


# -----------------------
# Activation Functions
# -----------------------

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # Numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(x_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    N = y.shape[0]
    correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-9)
    return np.mean(correct_log_probs)


# -----------------------
# Forward Pass
# -----------------------

# Layer 1
z1 = X @ W1
a1 = relu(z1)

# Layer 2
z2 = a1 @ W2
probs = softmax(z2)

loss = cross_entropy_loss(probs, y)

# -----------------------
# Sanity Checks
# -----------------------

print("probs shape:", probs.shape)
print("Row sums (should be ~1):", np.sum(probs, axis=1)[:5])
print("Loss:", loss)
print("Hidden activation stats:")
print("  Mean:", np.mean(a1))
print("  % zeros:", np.mean(a1 == 0) * 100)
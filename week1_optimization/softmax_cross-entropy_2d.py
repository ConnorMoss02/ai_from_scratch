import numpy as np 

def softmax(z): 
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)

def cross_entropy(pred_probs, y):
    return -np.log(pred_probs[y] + 1e-12)

# Test
z = np.array([2.0, 1.0, 0.1])
p = softmax(z)

print("logits:", z)
print("probabilities:", p)
print("loss if class 0:", cross_entropy(p, 0))
print("loss if class 2:", cross_entropy(p, 2))

print("\n--- confidence scaling ---")
for scale in [0.5,1,2,5]:
    z_scaled = z * scale
    p = softmax(z_scaled)
    print(f"scale={scale:1f}   probs={p}   loss={cross_entropy(p,0):.4f}")
import numpy as np
import matplotlib.pyplot as plt

# 1) Simple loss function
# "How wrong is my model?"
# Here it's tiny: f(x) = x^2. Minimum is at x = 0
def loss(x: float) -> float:
    return x ** 2

# 2) Define the gradient (derivative) of the loss
# If I increase x a little, does loss go up or down and by how much?
def grad_loss(x: float) -> float:
    return 2 * x

def run():
    # 3) Initialize a parameter (like a model weight)
    x = 10.0

    # 4) Pick a learning rate (step size)
    lr = 0.1

    x_history = []
    loss_history = []

    # 5) Training loop (This is the main event)
    # Every real model training loop reduces to params -= lr * gradient
    steps = 50
    for step in range(steps):
        x_history.append(x)
        current_loss = loss(x)
        loss_history.append(current_loss)

        g = grad_loss(x)  # Compute gradient at current x
        x = x - lr * g    # Gradient descent update

        if step % 10 == 0 or step == steps - 1:
            print(f"step={step:02d}  x={x: .6f}  loss={current_loss:.6f}  grad={g: .6f}")

    # 6) Plot the loss curve and the path we took on the loss surface
    xs = np.linspace(-10, 10, 400)
    ys = loss(xs)

    plt.figure()
    plt.plot(xs, ys)
    plt.scatter(x_history, [loss(v) for v in x_history])
    plt.title("Gradient Descent on f(x)=x^2")
    plt.xlabel("x (parameter)")
    plt.ylabel("loss(x)")
    plt.show()

    plt.figure()
    plt.plot(range(steps), loss_history)
    plt.title("Loss over training steps")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()

if __name__ == "__main__":
    run()



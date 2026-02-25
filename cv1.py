import random
import visualization

def target_line(x):
    return 3 * x + 2

def perceptron(points, function, max_epoch = 1000, learning_rate = 0.01):
    w1 = 0.0
    w2 = 0.0
    b = 0.0

    labels = [1 if p[1] > function(p[0]) else -1 for p in points]

    for epoch in range(max_epoch):
        epoch_errors = 0

        for i in range(len(points)):
            x, y = points[i]
            label = labels[i]

            z = w1 * x + w2 * y + b

            prediction = 1 if z > 0 else -1
            error = label - prediction

            if error != 0:
                w1 += learning_rate * error * x
                w2 += learning_rate * error * y
                b += learning_rate * error
                epoch_errors += 1
        if epoch_errors == 0:
            print(f"Converged at epoch {epoch}")
            break
    return w1, w2, b

points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(100)]
w1, w2, b = perceptron(points, target_line) # nevyjde to přesně, pokud zvýším počet bodů, tak to je lepší

print("-" * 30)
print("Nalezené hodnoty:")
print(f"Váha 1 (pro x): {w1:.4f}")
print(f"Váha 2 (pro y): {w2:.4f}")
print(f"Bias: {b:.4f}")

visualization.plot_results(points, w1, w2, b, target_line)
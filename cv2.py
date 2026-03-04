import math
import random
import visualization


# --- Aktivační funkce ---

def sigmoid(x):
    # Oříznutí aby nedošlo k přetečení
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output):
    """Derivace sigmoidu z už vypočítaného výstupu."""
    return output * (1 - output)


# --- Vícevrstvý perceptron (MLP) ---

class MLP:
    """
    Vícevrstvý perceptron se skrytou vrstvou.
    Architektura: 2 vstupy → hidden_size neuronů → 1 výstup
    """

    def __init__(self, hidden_size=2, learning_rate=0.5):
        self.lr = learning_rate

        # Váhy: skrytá vrstva (2 vstupy → hidden_size neuronů)
        self.w_hidden = [
            [random.uniform(-1, 1) for _ in range(2)]
            for _ in range(hidden_size)
        ]
        self.b_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]

        # Váhy: výstupní vrstva (hidden_size → 1 neuron)
        self.w_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b_output = random.uniform(-1, 1)

    def forward(self, x, y):
        """Dopředný průchod sítí."""
        inputs = [x, y]

        # Skrytá vrstva
        self.hidden_outputs = []
        for i in range(len(self.w_hidden)):
            z = sum(w * inp for w, inp in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(sigmoid(z))

        # Výstupní vrstva
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = sigmoid(z_out)

        return self.output

    def train(self, x, y, target):
        """Jeden krok backpropagation."""
        inputs = [x, y]

        # --- Dopředný průchod ---
        prediction = self.forward(x, y)

        # --- Zpětný průchod (backpropagation) ---

        # Chyba na výstupu
        output_error = target - prediction
        # delta říká, o kolik a jakým směrem upravit váhy skryté vrstvy
        output_delta = output_error * sigmoid_derivative(self.output)

        # Chyba na skryté vrstvě
        hidden_deltas = [] 
        for i in range(len(self.w_hidden)):
            error = self.w_output[i] * output_delta
            delta = error * sigmoid_derivative(self.hidden_outputs[i])
            hidden_deltas.append(delta)

        # --- Aktualizace vah ---

        # aktualizace vah a biasu výstupní vrstvy
        for i in range(len(self.w_output)):
            self.w_output[i] += self.lr * output_delta * self.hidden_outputs[i]
        self.b_output += self.lr * output_delta

        # aktualizace vah a biasu skryté vrstvy
        for i in range(len(self.w_hidden)):
            for j in range(len(inputs)):
                self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
            self.b_hidden[i] += self.lr * hidden_deltas[i]

        return output_error ** 2  # MSE pro jeden vzorek

    def fit(self, data, epochs=10000, log_interval=2000):
        """Tréninková smyčka přes všechna data."""
        self.losses = []

        for epoch in range(epochs):
            total_loss = 0
            for x, y, target in data:
                loss = self.train(x, y, target)
                total_loss += loss

            avg_loss = total_loss / len(data)
            self.losses.append(avg_loss)

            if log_interval and epoch % log_interval == 0:
                print(f"Epocha {epoch:5d} | Průměrná chyba: {avg_loss:.6f}")

        return self.losses

    def predict(self, x, y):
        """Predikce s prahem 0.5."""
        return 1 if self.forward(x, y) > 0.5 else 0

    def evaluate(self, data):
        """Vyhodnocení a výpis výsledků."""
        print("\n" + "=" * 40)
        print("Výsledky XOR sítě:")
        print("=" * 40)

        for x, y, target in data:
            output = self.forward(x, y)
            prediction = self.predict(x, y)
            status = "✓" if prediction == target else "✗"
            print(f"  {x} XOR {y} = {output:.4f}  →  {prediction}  (cíl: {target})  {status}")


# --- XOR data ---

xor_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

# --- Trénink, vyhodnocení, vizualizace ---

mlp = MLP(hidden_size=4, learning_rate=2.0)
losses = mlp.fit(xor_data, epochs=10000)
mlp.evaluate(xor_data)
visualization.plot_xor(mlp, xor_data, losses)

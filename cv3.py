import numpy as np
from visualization import plot_hopfield_all

class HopfieldNetwork:
    """Hopfieldova síť s Hebbovským učením."""

    def __init__(self, size: int):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns: list[np.ndarray]):
        """Uloží vzory do váhové matice (Hebbovo pravidlo)."""
        self.weights = np.zeros((self.size, self.size))
        for p in patterns:
            flat = p.flatten()
            self.weights += np.outer(flat, flat)
        # Nulová diagonála – neuron neovlivňuje sám sebe
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def energy(self, state: np.ndarray) -> float:
        """Vypočítá energii aktuálního stavu."""
        flat = state.flatten()
        return -0.5 * flat @ self.weights @ flat

    def recover_sync(self, state: np.ndarray, max_steps: int = 20):
        """Synchronní obnova – všechny neurony se aktualizují najednou."""
        current = state.flatten().copy()
        energies = [self.energy(current)]
        steps = 0
        for _ in range(max_steps):
            new_state = np.sign(self.weights @ current)
            new_state[new_state == 0] = 1  # Ošetření nulového výstupu
            steps += 1
            energies.append(self.energy(new_state))
            if np.array_equal(new_state, current):
                break
            current = new_state
        return current, steps, energies

    def recover_async(self, state: np.ndarray, max_steps: int = 1000):
        """Asynchronní obnova – neurony se aktualizují postupně v náhodném pořadí."""
        current = state.flatten().copy()
        energies = [self.energy(current)]
        steps = 0
        for _ in range(max_steps):
            changed = False
            order = np.random.permutation(self.size)
            for i in order:
                h = self.weights[i] @ current
                new_val = 1 if h >= 0 else -1
                if new_val != current[i]:
                    current[i] = new_val
                    changed = True
            steps += 1
            energies.append(self.energy(current))
            if not changed:
                break
        return current, steps, energies

# Vzor 1: Písmeno T
pattern_T = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
])

# Vzor 2: Písmeno X
pattern_X = np.array([
    [ 1, -1, -1, -1, -1, -1, -1, -1, -1,  1],
    [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
    [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
    [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1],
    [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1],
    [-1,  1, -1, -1, -1, -1, -1, -1,  1, -1],
    [ 1, -1, -1, -1, -1, -1, -1, -1, -1,  1],
])

# Vzor 3: Šipka nahoru
pattern_arrow = np.array([
    [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
    [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
])

patterns = [pattern_T, pattern_X, pattern_arrow]
pattern_names = ['T', 'X', 'Šipka ↑']


def add_noise(pattern: np.ndarray, noise_ratio: float = 0.2) -> np.ndarray:
    """Převrátí náhodně vybrané pixely ve vzoru."""
    noisy = pattern.copy()
    flat = noisy.flatten()
    n_flip = int(len(flat) * noise_ratio)
    indices = np.random.choice(len(flat), n_flip, replace=False)
    flat[indices] *= -1
    return flat.reshape(pattern.shape)

if __name__ == '__main__':
    np.random.seed(42)

    N = patterns[0].size  # 100 neuronů (10×10)
    net = HopfieldNetwork(N)
    net.train(patterns)

    recoveries = []

    # Obnova s 25% šumem
    for pat, name in zip(patterns, pattern_names):
        corrupted = add_noise(pat, 0.25)
        rec_sync, s_steps, e_sync = net.recover_sync(corrupted)
        rec_async, a_steps, e_async = net.recover_async(corrupted)
        shape = pat.shape
        recoveries.append(dict(
            original=pat, corrupted=corrupted,
            recovered_sync=rec_sync.reshape(shape),
            recovered_async=rec_async.reshape(shape),
            name=f'{name} (šum 25 %)', sync_steps=s_steps, async_steps=a_steps,
            energies_sync=e_sync, energies_async=e_async,
        ))

    # Obnova s 40% šumem
    for pat, name in zip(patterns, pattern_names):
        corrupted = add_noise(pat, 0.40)
        rec_sync, s_steps, e_sync = net.recover_sync(corrupted)
        rec_async, a_steps, e_async = net.recover_async(corrupted)
        shape = pat.shape
        match_sync = np.array_equal(rec_sync.reshape(shape), pat)
        match_async = np.array_equal(rec_async.reshape(shape), pat)
        print(f'  {name}: sync={"OK" if match_sync else "FAIL"} ({s_steps} kroků), '
              f'async={"OK" if match_async else "FAIL"} ({a_steps} kroků)')
        recoveries.append(dict(
            original=pat, corrupted=corrupted,
            recovered_sync=rec_sync.reshape(shape),
            recovered_async=rec_async.reshape(shape),
            name=f'{name} (šum 40 %)', sync_steps=s_steps, async_steps=a_steps,
            energies_sync=e_sync, energies_async=e_async,
        ))

    plot_hopfield_all(patterns, pattern_names, recoveries)
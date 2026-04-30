import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from visualization import plot_logistic_map


def logistic_map(a: float, x: float) -> float:
    # Logistická mapa: x_{n+1} = a * x_n * (1 - x_n)
    # Parametr 'a' (0–4) řídí dynamiku: fixní bod → cyklus → chaos
    return a * x * (1 - x)


def generate_bifurcation(a_values: np.ndarray, n_warmup: int = 500,
                         n_samples: int = 200, x0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Vrátí (a_list, x_list) pro bifurkační diagram.
    """
    a_list, x_list = [], []
    for a in a_values:
        x = x0
        # Warmup: necháme systém ustálit na atraktoru před sběrem dat.
        # Každá počáteční podmínka prochází přechodným (transientním) režimem,
        # než se trajektorie "usadí" na svém dlouhodobém chování (atraktor).
        # Bez warmupu bychom zaznamenávali přechodné body, ne skutečné bifurkace.
        for _ in range(n_warmup):
            x = logistic_map(a, x)
        # Sběr bodů z ustáleného režimu — tyto tvoří bifurkační diagram
        for _ in range(n_samples):
            x = logistic_map(a, x)
            a_list.append(a)
            x_list.append(x)
    return np.array(a_list), np.array(x_list)


def generate_training_pairs(a_values: np.ndarray, n_warmup: int = 200,
                             n_pairs: int = 100, x0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Vrátí trénovací páry (a, x_n) → x_{n+1}.
    """
    X, y = [], []
    for a in a_values:
        x = x0
        # Warmup: přeskočíme přechodný režim, stejně jako v generate_bifurcation
        for _ in range(n_warmup):
            x = logistic_map(a, x)
        # Sběr párů (vstup: [a, x_n], výstup: x_{n+1}) pro trénink sítě
        for _ in range(n_pairs):
            x_next = logistic_map(a, x)
            X.append([a, x])
            y.append(x_next)
            x = x_next
    return np.array(X), np.array(y)


def predict_bifurcation(model: MLPRegressor, scaler: StandardScaler,
                        a_values: np.ndarray, n_warmup: int = 300,
                        n_samples: int = 100, x0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Generuje bifurkační diagram pomocí natrénované sítě.
    """
    a_list, x_list = [], []
    for a in a_values:
        x = np.clip(x0, 0.0, 1.0)
        # Warmup přes neuronovou síť: stejný princip jako u skutečné mapy —
        # necháme síť iterovat, dokud se trajektorie neustálí na atraktoru modelu.
        # clip [0,1] zabraňuje numerické nestabilitě při opakovaných predikcích.
        for _ in range(n_warmup):
            x = float(model.predict(scaler.transform([[a, x]]))[0])
            x = np.clip(x, 0.0, 1.0)
        # Sběr bodů predikovaného bifurkačního diagramu
        for _ in range(n_samples):
            x = float(model.predict(scaler.transform([[a, x]]))[0])
            x = np.clip(x, 0.0, 1.0)
            a_list.append(a)
            x_list.append(x)
    return np.array(a_list), np.array(x_list)


# Skutečný bifurkační diagram (referenční)
# 1000 hodnot parametru 'a' od 0 do 4; pro každou se zaznamenají ustálené hodnoty x
a_all = np.linspace(0, 4.0, 1000)
print("Generuji bifurkační diagram...")
a_bif, x_bif = generate_bifurcation(a_all)

# Trénovací data
# Řidší mřížka (300 hodnot) snižuje čas tréninku; každá hodnota 'a' přispěje 80 páry
a_train_vals = np.linspace(0, 4.0, 300)
print("Generuji trénovací páry...")
X_train, y_train = generate_training_pairs(a_train_vals, n_warmup=200, n_pairs=80)

# Neuronová síť
# StandardScaler normalizuje vstup (a, x) na nulový průměr a jednotkový rozptyl —
# MLP konverguje rychleji a stabilněji s normalizovanými vstupy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

print(f"Trénuji síť na {len(X_train)} vzorcích...")
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # tři skryté vrstvy, klesající šířka
    activation="tanh",                 # tanh zvládá záporné hodnoty lépe než relu pro tuto dynamiku
    max_iter=600,
    random_state=42,
    verbose=False,
)
model.fit(X_scaled, y_train)
print("Trénink dokončen.")

# Predikce bifurkačního diagramu neuronovou sítí
# Méně hodnot 'a' (400) a vzorků (80) než u referenčního diagramu — inference je pomalejší
a_pred_vals = np.linspace(0, 4.0, 400)
print("Generuji predikovaný bifurkační diagram...")
a_pred, x_pred = predict_bifurcation(model, scaler, a_pred_vals, n_warmup=300, n_samples=80)
print("Hotovo.")

# Porovnání
plot_logistic_map(a_bif, x_bif, a_pred, x_pred)

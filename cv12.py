import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# Parametry modelu lesního požáru
P = 0.05        # Pravděpodobnost vzniku nového stromu na prázdném místě
F = 0.001       # Pravděpodobnost spontánního vzniku požáru
DENSITY = 0.5   # Počáteční hustota lesa
SIZE = 100      # Velikost mřížky (SIZE × SIZE)

# Stavy buněk
EMPTY = 0       # Prázdná/spálená buňka
TREE = 1        # Živý strom
BURNING = 2     # Hořící strom

# Barevná mapa: prázdno=tmavě hnědá, strom=zelená, hoří=oranžová
CMAP = ListedColormap(["#3b1a08", "#228b22", "#ff6600"])


def init_grid(size, density):
    """Vytvoří mřížku s náhodným rozmístěním stromů dle hustoty."""
    return np.random.choice(
        [EMPTY, TREE],
        size=(size, size),
        p=[1 - density, density]
    )


def step(grid):
    """Provede jeden krok simulace lesního požáru (von Neumannovo sousedství).

    Pravidla:
    1. Prázdná/spálená buňka → strom s pravděpodobností p
    2. Strom se vznítí, pokud sousedí s hořícím stromem
    3. Strom se vznítí spontánně s pravděpodobností f
    4. Hořící strom → spálená buňka
    """
    new_grid = grid.copy()
    random = np.random.random(grid.shape)

    # Von Neumannovo sousedství: hoří alespoň jeden ze čtyř sousedů?
    neighbor_burning = (
        (np.roll(grid, 1, axis=0) == BURNING) |
        (np.roll(grid, -1, axis=0) == BURNING) |
        (np.roll(grid, 1, axis=1) == BURNING) |
        (np.roll(grid, -1, axis=1) == BURNING)
    )

    empty = grid == EMPTY
    tree = grid == TREE
    burning = grid == BURNING

    # Prázdná/spálená → strom s pravděpodobností p
    new_grid[empty] = np.where(random[empty] < P, TREE, EMPTY)

    # Živý strom → hoří pokud soused hoří nebo s pravděpodobností f
    new_grid[tree & (neighbor_burning | (random < F))] = BURNING

    # Hořící strom → spálený
    new_grid[burning] = EMPTY

    return new_grid


if __name__ == "__main__":
    grid = init_grid(SIZE, DENSITY)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title("Forest Fire")
    ax.set_title("Forest Fire")

    im = ax.imshow(
        grid,
        cmap=CMAP,
        vmin=0,
        vmax=2,
        interpolation="nearest",
        origin="lower",
    )

    def update(frame):
        global grid
        grid = step(grid)
        im.set_data(grid)
        return (im,)

    ani = FuncAnimation(fig, update, interval=100, blit=True)
    plt.tight_layout()
    plt.show()

import random

from visualization import plot_ifs


# Affinní transformace: [x', y', z'] = M * [x, y, z] + t
# Každý řádek: (a, b, c, d, e, f, g, h, i, j, k, l)

FIRST_MODEL = [
    (0.00,  0.00,  0.01,  0.00,  0.26,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.00),
    (0.20, -0.26, -0.01,  0.23,  0.22, -0.07,  0.07,  0.00,  0.24,  0.00,  0.80,  0.00),
    (-0.25, 0.28,  0.01,  0.26,  0.24, -0.07,  0.07,  0.00,  0.24,  0.00,  0.22,  0.00),
    (0.85,  0.04, -0.01, -0.04,  0.85,  0.09,  0.00,  0.08,  0.84,  0.00,  0.80,  0.00),
]

SECOND_MODEL = [
    (0.05,  0.00,  0.00,  0.00,  0.60,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.00),
    (0.45, -0.22,  0.22,  0.22,  0.45,  0.22, -0.22,  0.22, -0.45,  0.00,  1.00,  0.00),
    (-0.45, 0.22, -0.22,  0.22,  0.45,  0.22,  0.22, -0.22,  0.45,  0.00,  1.25,  0.00),
    (0.49, -0.08,  0.08,  0.08,  0.49,  0.08,  0.08, -0.08,  0.49,  0.00,  2.00,  0.00),
]


def apply_transform(transform: tuple, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Aplikuje jednu affinní transformaci na bod (x, y, z)."""
    a, b, c, d, e, f, g, h, i, j, k, l = transform
    x_new = a * x + b * y + c * z + j
    y_new = d * x + e * y + f * z + k
    z_new = g * x + h * y + i * z + l
    return x_new, y_new, z_new


def generate_ifs(transforms: list, iterations: int = 50_000) -> tuple[list, list, list]:
    """Generuje IFS fraktál iterativní aplikací náhodně vybraných transformací.

    Každá transformace je vybrána s pravděpodobností p = 0.25.
    Vrací seznamy souřadnic xs, ys, zs.
    """
    x, y, z = 0.0, 0.0, 0.0
    xs, ys, zs = [], [], []

    for _ in range(iterations):
        # Náhodný výběr transformace (každá s pravděpodobností 0.25)
        transform = random.choice(transforms)
        x, y, z = apply_transform(transform, x, y, z)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs, ys, zs


if __name__ == '__main__':
    # První model
    xs1, ys1, zs1 = generate_ifs(FIRST_MODEL)
    plot_ifs(xs1, ys1, zs1, 'First model')

    # Druhý model
    xs2, ys2, zs2 = generate_ifs(SECOND_MODEL)
    plot_ifs(xs2, ys2, zs2, 'Second model')

import matplotlib.pyplot as plt
import numpy as np

def plot_results(points, w1, w2, b, line_func):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Nastavení os doprostřed (jako na tvém obrázku)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Zjištění barev bodů podle skutečné funkce
    x_coords, y_coords, colors = [], [], []
    epsilon = 0.01

    for x, y in points:
        x_coords.append(x)
        y_coords.append(y)
        y_true = line_func(x)
        
        if y > y_true + epsilon:
            colors.append('red')
        elif y < y_true - epsilon:
            colors.append('blue')
        else:
            colors.append('green')

    # Vykreslení bodů
    ax.scatter(x_coords, y_coords, c=colors, s=30, zorder=3)

    # Vykreslení SKUTEČNÉ přímky (černá, plná čára)
    x_vals = np.linspace(-12, 12, 100)
    y_true_line = line_func(x_vals)
    ax.plot(x_vals, y_true_line, color='black', linewidth=1.5, label='Skutečná přímka', zorder=2)

    # Vykreslení NAUČENÉ přímky z perceptronu (zelená, čárkovaná)
    # Z rovnice z = w1*x + w2*y + b jsme vyjádřili y:
    if w2 != 0:
        y_pred_line = (-w1 * x_vals - b) / w2
        ax.plot(x_vals, y_pred_line, color='limegreen', linestyle='--', linewidth=2.5, label='Naučená přímka', zorder=4)

    # Finální úpravy grafu
    plt.title("Perceptron vs Skutečnost", pad=20)
    plt.xlim(-12, 12)
    plt.ylim(-20, 30)
    plt.legend(loc='upper left')
    plt.show()
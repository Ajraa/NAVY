import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np

pio.renderers.default = 'browser'

def plot_results(points, w1, w2, b, line_func):
    fig = go.Figure()

    # Zjištění barev bodů podle skutečné funkce
    epsilon = 0.01
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []

    for x, y in points:
        y_true = line_func(x)
        if y > y_true + epsilon:
            red_x.append(x); red_y.append(y)
        elif y < y_true - epsilon:
            blue_x.append(x); blue_y.append(y)
        else:
            green_x.append(x); green_y.append(y)

    # Vykreslení bodů
    fig.add_trace(go.Scatter(x=red_x, y=red_y, mode='markers',
                             marker=dict(color='red', size=8), name='Nad přímkou'))
    fig.add_trace(go.Scatter(x=blue_x, y=blue_y, mode='markers',
                             marker=dict(color='blue', size=8), name='Pod přímkou'))
    if green_x:
        fig.add_trace(go.Scatter(x=green_x, y=green_y, mode='markers',
                                 marker=dict(color='green', size=8), name='Na přímce'))

    # Skutečná přímka
    x_vals = np.linspace(-12, 12, 100)
    y_true_line = line_func(x_vals)
    fig.add_trace(go.Scatter(x=x_vals, y=y_true_line, mode='lines',
                             line=dict(color='black', width=2), name='Skutečná přímka'))

    # Naučená přímka z perceptronu
    if w2 != 0:
        y_pred_line = (-w1 * x_vals - b) / w2
        fig.add_trace(go.Scatter(x=x_vals, y=y_pred_line, mode='lines',
                                 line=dict(color='limegreen', width=3, dash='dash'),
                                 name='Naučená přímka'))

    fig.update_layout(
        title='Perceptron vs Skutečnost',
        xaxis=dict(range=[-12, 12], zeroline=True, zerolinewidth=2),
        yaxis=dict(range=[-20, 30], zeroline=True, zerolinewidth=2),
        width=800, height=600
    )
    fig.show()


def plot_xor(mlp, xor_data, losses):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Rozhodovací plocha MLP pro XOR',
                                        'Průběh chyby při tréninku'),
                        horizontal_spacing=0.15)

    # 1) Rozhodovací plocha
    resolution = 100
    xs = [i / resolution for i in range(resolution + 1)]
    ys = [i / resolution for i in range(resolution + 1)]

    grid = []
    for yi in ys:
        row = []
        for xi in xs:
            row.append(mlp.forward(xi, yi))
        grid.append(row)

    grid_arr = np.array(grid)

    fig.add_trace(go.Heatmap(
        z=grid_arr, x=xs, y=ys,
        colorscale='RdYlBu_r', zmin=0, zmax=1,
        colorbar=dict(title='Výstup sítě', x=0.42),
        showscale=True
    ), row=1, col=1)

    # XOR body
    for x, y, target in xor_data:
        color = 'white' if target == 1 else 'black'
        border = 'black' if target == 1 else 'white'
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(color=color, size=16, line=dict(color=border, width=2)),
            text=[f'{x} XOR {y} = {target}'],
            textposition='top right',
            textfont=dict(size=11),
            showlegend=False
        ), row=1, col=1)

    # 2) Průběh chyby
    fig.add_trace(go.Scatter(
        y=losses, mode='lines',
        line=dict(color='crimson', width=1.5),
        showlegend=False
    ), row=1, col=2)

    fig.update_xaxes(title_text='x', range=[-0.1, 1.1], row=1, col=1)
    fig.update_yaxes(title_text='y', range=[-0.1, 1.1], row=1, col=1)
    fig.update_xaxes(title_text='Epocha', row=1, col=2)
    fig.update_yaxes(title_text='Průměrná chyba (MSE)', type='log', row=1, col=2)

    fig.update_layout(width=1200, height=500, showlegend=False)
    fig.show()


def plot_hopfield_all(patterns, pattern_names, recoveries):
    """Vykreslí vše do jedné Plotly figury – čistá mřížka heatmap.

    recoveries: list of dict s klíči:
        original, corrupted, recovered_sync, recovered_async,
        name, sync_steps, async_steps, energies_sync, energies_async
    """
    n_pat = len(patterns)
    n_rec = len(recoveries)
    cols = 4
    n_rows = 1 + n_rec  # 1 řádek vzorů + 1 řádek na test

    # Titulky: řádek 1 = vzory, další = obnovy
    subplot_titles = list(pattern_names)  # řádek 1 (3 vzory, 4. slot je None – bez titulku)
    for rec in recoveries:
        e_start = int(rec['energies_sync'][0])
        e_end_s = int(rec['energies_sync'][-1])
        e_end_a = int(rec['energies_async'][-1])
        subplot_titles.extend([
            f'Originál',
            f'Poškozený – {rec["name"]}',
            f'Sync ({rec["sync_steps"]} kr.) E: {e_start}→{e_end_s}',
            f'Async ({rec["async_steps"]} kr.) E: {e_start}→{e_end_a}',
        ])

    # Specs: řádek 1 má jen n_pat sloupců
    specs = [[{}] * n_pat + [None] * (cols - n_pat)]
    for _ in range(n_rec):
        specs.append([{}, {}, {}, {}])

    fig = make_subplots(
        rows=n_rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.03,
    )

    bw = [[0, 'white'], [1, 'black']]

    def add_heatmap(matrix, row, col):
        fig.add_trace(go.Heatmap(
            z=matrix[::-1], colorscale=bw, zmin=-1, zmax=1, showscale=False
        ), row=row, col=col)
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    # Řádek 1: uložené vzory
    for i, pat in enumerate(patterns):
        add_heatmap(pat, 1, i + 1)

    # Řádky obnov
    for idx, rec in enumerate(recoveries):
        row = 2 + idx
        for c, mat in enumerate([rec['original'], rec['corrupted'],
                                  rec['recovered_sync'], rec['recovered_async']]):
            add_heatmap(mat, row, c + 1)

    fig.update_layout(
        title='Hopfieldova síť – uložené vzory a obnova',
        height=250 * n_rows,
        width=1100,
        showlegend=False,
    )
    fig.show()
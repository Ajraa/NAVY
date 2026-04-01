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


def plot_find_cheese(rows, cols, start, cheese, holes, path, rewards):
    """Vykreslí průběh učení a animaci finální cesty agenta."""

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Hrací plocha a finální greedy cesta', 'Průběh odměny během tréninku'),
        horizontal_spacing=0.12,
    )

    grid = np.zeros((rows, cols), dtype=float)
    for r, c in holes:
        grid[r, c] = -1.0
    cr, cc = cheese
    grid[cr, cc] = 1.0

    # Heatmap je kvůli orientaci převrácená v ose y.
    heatmap = go.Heatmap(
        z=np.flipud(grid),
        x=list(range(cols)),
        y=list(range(rows)),
        zmin=-1,
        zmax=1,
        colorscale=[
            [0.0, '#1f2937'],
            [0.499, '#1f2937'],
            [0.5, '#f3f4f6'],
            [0.999, '#f3f4f6'],
            [1.0, '#fbbf24'],
        ],
        showscale=False,
    )
    fig.add_trace(heatmap, row=1, col=1)

    hole_x = []
    hole_y = []
    for hr, hc in holes:
        hole_x.append(hc)
        hole_y.append(rows - 1 - hr)

    fig.add_trace(
        go.Scatter(
            x=hole_x,
            y=hole_y,
            mode='markers',
            marker=dict(size=22, color='#111827', symbol='square'),
            showlegend=False,
            hoverinfo='skip',
        ),
        row=1,
        col=1,
    )

    sr, sc = start
    fig.add_trace(
        go.Scatter(
            x=[sc],
            y=[rows - 1 - sr],
            mode='markers+text',
            marker=dict(size=14, color='#2563eb', symbol='square'),
            text=['START'],
            textposition='bottom center',
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[cc],
            y=[rows - 1 - cr],
            mode='markers+text',
            marker=dict(size=16, color='#f59e0b', symbol='star'),
            text=['CHEESE'],
            textposition='bottom center',
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    first_r, first_c = path[0]
    fig.add_trace(
        go.Scatter(
            x=[first_c],
            y=[rows - 1 - first_r],
            mode='lines',
            line=dict(color='#ef4444', width=2),
            opacity=0.45,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    trail_trace_idx = len(fig.data) - 1

    walker_trace = go.Scatter(
        x=[first_c],
        y=[rows - 1 - first_r],
        mode='markers',
        marker=dict(size=18, color='#dc2626', line=dict(color='white', width=1.5)),
        name='Walker',
        showlegend=False,
    )
    fig.add_trace(walker_trace, row=1, col=1)
    walker_trace_idx = len(fig.data) - 1

    moving_rewards = []
    window = 50
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        moving_rewards.append(float(np.mean(rewards[start_idx:i + 1])))

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(rewards) + 1)),
            y=rewards,
            mode='lines',
            line=dict(color='#94a3b8', width=1),
            name='Odměna epizody',
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    rewards_trace_idx = len(fig.data) - 1

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(moving_rewards) + 1)),
            y=moving_rewards,
            mode='lines',
            line=dict(color='#0f766e', width=2.5),
            name='Klouzavý průměr (50)',
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    moving_rewards_trace_idx = len(fig.data) - 1

    trail_x = []
    trail_y = []
    frames = []
    for i, (r, c) in enumerate(path):
        trail_x.append(c)
        trail_y.append(rows - 1 - r)
        frames.append(
            go.Frame(
                name=f'krok-{i}',
                data=[
                    go.Scatter(
                        x=trail_x.copy(),
                        y=trail_y.copy(),
                        mode='lines',
                        line=dict(color='#ef4444', width=2),
                        opacity=0.45,
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=[c],
                        y=[rows - 1 - r],
                        mode='markers',
                        marker=dict(size=18, color='#dc2626', line=dict(color='white', width=1.5)),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=list(range(1, len(rewards) + 1)),
                        y=rewards,
                        mode='lines',
                        line=dict(color='#94a3b8', width=1),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=list(range(1, len(moving_rewards) + 1)),
                        y=moving_rewards,
                        mode='lines',
                        line=dict(color='#0f766e', width=2.5),
                        showlegend=False,
                    )
                ],
                traces=[
                    trail_trace_idx,
                    walker_trace_idx,
                    rewards_trace_idx,
                    moving_rewards_trace_idx,
                ],
            )
        )

    fig.frames = frames

    sliders = [
        {
            'active': 0,
            'currentvalue': {'prefix': 'Krok: '},
            'y': -0.22,
            'steps': [
                {
                    'method': 'animate',
                    'label': str(i),
                    'args': [[f'krok-{i}'], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}}],
                }
                for i in range(len(path))
            ],
        }
    ]

    fig.update_layout(
        width=1300,
        height=550,
        title='Find the cheese - Q-learning vizualizace',
        margin=dict(l=60, r=40, t=160, b=180),
        uirevision='find-cheese-static-layout',
        updatemenus=[
            {
                'type': 'buttons',
                'direction': 'right',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [
                            None,
                            {
                                'frame': {'duration': 350, 'redraw': False},
                                'transition': {'duration': 180, 'easing': 'linear'},
                                'fromcurrent': True,
                                'mode': 'immediate',
                            },
                        ],
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}}],
                    },
                ],
                'x': 0.98,
                'y': 1.30,
                'xanchor': 'right',
                'yanchor': 'top',
                'pad': {'r': 0, 't': 0},
                'showactive': False,
            }
        ],
        sliders=sliders,
    )

    fig.update_xaxes(
        row=1,
        col=1,
        title_text='Sloupec',
        dtick=1,
        range=[-0.5, cols - 0.5],
        constrain='domain',
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title_text='Řádek',
        dtick=1,
        range=[-0.5, rows - 0.5],
        scaleanchor='x',
        scaleratio=1,
    )

    fig.update_xaxes(row=1, col=2, title_text='Epizoda')
    fig.update_yaxes(row=1, col=2, title_text='Odměna')

    fig.show()


def plot_cartpole(rewards: list, demo_states: list):
    """Animace DQN agenta na CartPole-v1 – vozík s tyčí v každém kroku demo epizody.

    Stav: (cart_pos, cart_vel, pole_angle, pole_vel)
    Vozík se pohybuje na kolejnici v rozsahu ±2.4, tyč selhává při ±12° (±0.2095 rad).
    """

    # Vizuální konstanty (v souřadnicích grafu)
    CART_W = 0.40
    CART_H = 0.14
    WHEEL_R = 0.07
    POLE_LEN = 0.85   # plná délka tyče (CartPole half-length = 0.5 m)
    CART_Y_BOT = WHEEL_R * 2
    PIVOT_Y = CART_Y_BOT + CART_H

    def make_traces(cart_x: float, pole_angle: float):
        """Vrátí 4 Scatter trace pro vozík, kola, tyč a kuličku na konci tyče."""
        cx_l = cart_x - CART_W / 2
        cx_r = cart_x + CART_W / 2

        tip_x = cart_x + POLE_LEN * np.sin(pole_angle)
        tip_y = PIVOT_Y + POLE_LEN * np.cos(pole_angle)

        cart = go.Scatter(
            x=[cx_l, cx_r, cx_r, cx_l, cx_l],
            y=[CART_Y_BOT, CART_Y_BOT, PIVOT_Y, PIVOT_Y, CART_Y_BOT],
            fill='toself', fillcolor='#2563eb',
            line=dict(color='#1e40af', width=2),
            mode='lines', hoverinfo='skip', showlegend=False,
        )
        wheels = go.Scatter(
            x=[cx_l + 0.08, cx_r - 0.08],
            y=[WHEEL_R, WHEEL_R],
            mode='markers',
            marker=dict(size=16, color='#1f2937', symbol='circle'),
            hoverinfo='skip', showlegend=False,
        )
        pole = go.Scatter(
            x=[cart_x, tip_x],
            y=[PIVOT_Y, tip_y],
            mode='lines',
            line=dict(color='#92400e', width=10),
            hoverinfo='skip', showlegend=False,
        )
        tip_ball = go.Scatter(
            x=[tip_x], y=[tip_y],
            mode='markers',
            marker=dict(size=14, color='#dc2626', symbol='circle'),
            hoverinfo='skip', showlegend=False,
        )
        return cart, wheels, pole, tip_ball

    # Statické pozadí: koleje, hranice selhání
    track = go.Scatter(
        x=[-2.6, 2.6], y=[0, 0],
        mode='lines', line=dict(color='#374151', width=4),
        hoverinfo='skip', showlegend=False,
    )
    left_wall = go.Scatter(
        x=[-2.4, -2.4], y=[0, PIVOT_Y + POLE_LEN + 0.1],
        mode='lines', line=dict(color='#ef4444', width=2, dash='dash'),
        hoverinfo='skip', showlegend=False,
    )
    right_wall = go.Scatter(
        x=[2.4, 2.4], y=[0, PIVOT_Y + POLE_LEN + 0.1],
        mode='lines', line=dict(color='#ef4444', width=2, dash='dash'),
        hoverinfo='skip', showlegend=False,
    )

    s0 = demo_states[0]
    cart0, wheels0, pole0, ball0 = make_traces(s0[0], s0[2])

    fig = go.Figure(data=[track, left_wall, right_wall, cart0, wheels0, pole0, ball0])

    # Indexy animovaných stop (za statickými 3)
    CART_IDX, WHEELS_IDX, POLE_IDX, BALL_IDX = 3, 4, 5, 6

    frames = []
    for i, state in enumerate(demo_states):
        cart_x, _, pole_angle, _ = state
        cart, wheels, pole, ball = make_traces(cart_x, pole_angle)
        frames.append(go.Frame(
            name=str(i),
            data=[cart, wheels, pole, ball],
            traces=[CART_IDX, WHEELS_IDX, POLE_IDX, BALL_IDX],
        ))
    fig.frames = frames

    n_steps = len(demo_states)
    slider_step = max(1, n_steps // 100)   # max ~100 značek na slideru

    sliders = [{
        'active': 0,
        'currentvalue': {'prefix': 'Krok: ', 'font': {'size': 13}},
        'y': -0.08,
        'steps': [
            {
                'method': 'animate',
                'label': str(i) if i % (slider_step * 5) == 0 else '',
                'args': [[str(i)], {'mode': 'immediate',
                                    'frame': {'duration': 0, 'redraw': True}}],
            }
            for i in range(n_steps)
        ],
    }]

    fig.update_layout(
        title=f'CartPole DQN – demo epizoda ({n_steps} kroků)',
        xaxis=dict(
            range=[-2.75, 2.75], zeroline=False,
            title='Pozice vozíku [m]', showgrid=True, gridcolor='#e2e8f0',
        ),
        yaxis=dict(
            range=[-0.18, PIVOT_Y + POLE_LEN + 0.15],
            zeroline=False, showticklabels=False, showgrid=False,
        ),
        width=1000,
        height=480,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#ffffff',
        margin=dict(l=60, r=40, t=60, b=100),
        updatemenus=[{
            'type': 'buttons',
            'direction': 'right',
            'buttons': [
                {
                    'label': '▶  Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 40, 'redraw': True},
                        'transition': {'duration': 0},
                        'fromcurrent': True,
                        'mode': 'immediate',
                    }],
                },
                {
                    'label': '⏸  Pauza',
                    'method': 'animate',
                    'args': [[None], {'mode': 'immediate',
                                      'frame': {'duration': 0, 'redraw': False}}],
                },
            ],
            'x': 0.5, 'y': 1.13,
            'xanchor': 'center', 'yanchor': 'top',
            'showactive': False,
            'font': {'size': 13},
        }],
        sliders=sliders,
    )

    fig.show()
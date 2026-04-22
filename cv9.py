import tkinter as tk
import random

WIDTH = 900
HEIGHT = 600

TERRAINS = [
    {"label": "Tráva",  "color": "#228B22", "base_y": HEIGHT * 0.55, "roughness": 0.6},
    {"label": "Skála",  "color": "#2F2F2F", "base_y": HEIGHT * 0.72, "roughness": 0.4},
    {"label": "Zemina", "color": "#8B4513", "base_y": HEIGHT * 0.86, "roughness": 0.3},
]

SKY_COLOR = "#87CEEB"


def midpoint_displacement(x0: float, y0: float, x1: float, y1: float,
                           iterations: int, roughness: float, scale: float) -> list[tuple[float, float]]:
    # Začínáme s jedním úsekem mezi krajními body (x0,y0) a (x1,y1)
    points = [(x0, y0), (x1, y1)]
    # Počáteční maximální výchylka středového bodu závisí na měřítku a drsnosti terénu
    displacement = scale * roughness

    for _ in range(iterations):
        new_points = [points[0]]
        for i in range(len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            # Vypočítáme střed úseku po ose X
            mx = (ax + bx) / 2
            # Střed po ose Y náhodně posuneme o hodnotu z intervalu [-displacement, +displacement]
            my = (ay + by) / 2 + random.uniform(-displacement, displacement)
            new_points.append((mx, my))
            new_points.append((bx, by))
        points = new_points
        # Každou iterací snížíme výchylku na polovinu — kratší úseky dostávají menší posuny
        displacement *= 0.5

    return points


def generate_terrain(base_y: float, roughness: float, iterations: int) -> list[tuple[float, float]]:
    scale = HEIGHT * 0.15
    return midpoint_displacement(0, base_y, WIDTH, base_y, iterations, roughness, scale)


def build_polygon(points: list[tuple[float, float]]) -> list[float]:
    flat = []
    for x, y in points:
        flat.extend([x, y])
    flat.extend([WIDTH, HEIGHT, 0, HEIGHT])
    return flat


class FractalTerrain:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Fraktální terén 2D — cv9")
        self.root.resizable(False, False)

        self._build_ui()
        self._draw()
        self.root.mainloop()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        tk.Label(frame, text="Iterace:").pack(anchor="w")
        self.iter_var = tk.IntVar(value=8)
        tk.Spinbox(frame, from_=1, to=12, textvariable=self.iter_var, width=6).pack(anchor="w")

        tk.Label(frame, text="Seed (0 = náhodný):").pack(anchor="w", pady=(8, 0))
        self.seed_var = tk.IntVar(value=0)
        tk.Spinbox(frame, from_=0, to=9999, textvariable=self.seed_var, width=6).pack(anchor="w")

        tk.Button(frame, text="Generovat", command=self._draw,
                  bg="#4CAF50", fg="white", width=10).pack(pady=(12, 4))
        tk.Button(frame, text="Nový seed", command=self._random_seed,
                  width=10).pack(pady=2)

        tk.Label(frame, text="Vrstvy:").pack(anchor="w", pady=(12, 0))
        for t in TERRAINS:
            row = tk.Frame(frame)
            row.pack(anchor="w", pady=1)
            tk.Label(row, bg=t["color"], width=2).pack(side=tk.LEFT)
            tk.Label(row, text=t["label"]).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg=SKY_COLOR)
        self.canvas.pack(side=tk.LEFT)

    def _random_seed(self) -> None:
        self.seed_var.set(random.randint(1, 9999))
        self._draw()

    def _draw(self) -> None:
        seed = self.seed_var.get()
        if seed == 0:
            random.seed()
        else:
            random.seed(seed)

        iterations = self.iter_var.get()
        self.canvas.delete("all")
        self.canvas.configure(bg=SKY_COLOR)

        for terrain in TERRAINS:
            pts = generate_terrain(terrain["base_y"], terrain["roughness"], iterations)
            poly = build_polygon(pts)
            self.canvas.create_polygon(poly, fill=terrain["color"], outline="")


if __name__ == "__main__":
    FractalTerrain()

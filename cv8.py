import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

# Parametry výpočtu
MAX_ITER = 256
WIDTH = 900
HEIGHT = 600

# Výchozí rozsahy
MANDELBROT_BOUNDS = (-2.0, 1.0, -1.0, 1.0)   # x_min, x_max, y_min, y_max
JULIA_BOUNDS = (-1.5, 1.5, -1.5, 1.5)
JULIA_C = complex(-0.7, 0.27015)

# Barevná paleta (256 RGB trojic): červená → oranžová → žlutá → zelená → černá
_palette_stops = [
    (0.00, (255,   0,   0)),
    (0.05, (255,  80,   0)),
    (0.13, (255, 160,   0)),
    (0.23, (255, 220,   0)),
    (0.37, (180, 255,   0)),
    (0.52, (  0, 255,  80)),
    (0.66, (  0, 200, 180)),
    (0.79, (  0,  80, 200)),
    (0.90, ( 30,   0, 120)),
    (0.96, (  5,   0,  30)),
    (1.00, (  0,   0,   0)),
]


def _build_palette(n: int = 256) -> np.ndarray:
    """Sestaví paletu n RGB barev interpolací zarážek pomocí np.interp (vektorizovaně)."""
    positions = np.array([s[0] for s in _palette_stops], dtype=np.float64)
    colors = np.array([s[1] for s in _palette_stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    palette = np.empty((n, 3), dtype=np.uint8)
    for ch in range(3):
        palette[:, ch] = np.interp(t, positions, colors[:, ch]).astype(np.uint8)
    return palette


PALETTE = _build_palette(256)


def compute_mandelbrot(x_min: float, x_max: float, y_min: float, y_max: float,
                       width: int, height: int, max_iter: int) -> np.ndarray:
    """Vrátí matici (height × width) s počtem iterací; body v množině = max_iter."""
    # Krok 1: Vytvoření mřížky komplexních čísel c
    # Každý pixel odpovídá jednomu komplexnímu číslu c = x + iy
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_max, y_min, height)   # y_max první => osa jde shora dolů
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]   # mřížka (height × width)

    # Krok 2: Počáteční podmínky
    # Mandelbrot vždy začíná z z₀ = 0
    Z = np.zeros_like(C)
    # Výchozí hodnota: každý pixel "přežil" všechny iterace (= je uvnitř množiny)
    iterations = np.full((height, width), max_iter, dtype=np.int32)
    # Maska aktivních pixelů — ty, které ještě neunikly
    mask = np.ones((height, width), dtype=bool)
    # Pre-alokovaný buffer pro |z|² — plníme jen aktivní pixely, zbytek ignorujeme
    abs2 = np.empty((height, width), dtype=np.float64)

    # Krok 3: Iterační smyčka z = z² + c
    for i in range(max_iter):
        # Iterujeme pouze pixely, které ještě neunikly
        Z[mask] = Z[mask] ** 2 + C[mask]
        # Test úniku: |z|² > 4 (bez sqrt)
        # Počítáme jen na aktivních pixelech, ne na celém poli
        abs2[mask] = Z.real[mask] ** 2 + Z.imag[mask] ** 2
        escaped = mask & (abs2 > 4.0)
        # Zapamatuj číslo iterace, ve které pixel unikl
        iterations[escaped] = i
        # Vyřaď uniknuvší pixely z dalších výpočtů
        mask &= ~escaped
        # Předčasný konec — všechny pixely unikly
        if not mask.any():
            break

    # Pixely stále v masce (neunikly) mají hodnotu max_iter => uvnitř množiny
    return iterations


def compute_julia(x_min: float, x_max: float, y_min: float, y_max: float,
                  c: complex, width: int, height: int, max_iter: int) -> np.ndarray:
    """Vrátí matici (height × width) s počtem iterací; body v množině = max_iter."""
    # Krok 1: Vytvoření mřížky počátečních hodnot z₀
    # U Juliovy množiny je mřížka počáteční hodnota z, ne parametr c
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_max, y_min, height)   # y_max první => osa jde shora dolů
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]   # mřížka (height × width)

    # Krok 2: Počáteční podmínky
    # Výchozí hodnota: každý pixel "přežil" všechny iterace (= je uvnitř množiny)
    iterations = np.full((height, width), max_iter, dtype=np.int32)
    # Maska aktivních pixelů — ty, které ještě neunikly
    mask = np.ones((height, width), dtype=bool)
    # Pre-alokovaný buffer pro |z|² — plníme jen aktivní pixely, zbytek ignorujeme
    abs2 = np.empty((height, width), dtype=np.float64)

    # Krok 3: Iterační smyčka z = z² + c
    # Parametr c je fixní pro celý obrázek (na rozdíl od Mandelbrota)
    for i in range(max_iter):
        # Iterujeme pouze pixely, které ještě neunikly
        Z[mask] = Z[mask] ** 2 + c
        # Test úniku: |z|² > 4 (bez sqrt)
        # Počítáme jen na aktivních pixelech, ne na celém poli
        abs2[mask] = Z.real[mask] ** 2 + Z.imag[mask] ** 2
        escaped = mask & (abs2 > 4.0)
        # Zapamatuj číslo iterace, ve které pixel unikl
        iterations[escaped] = i
        # Vyřaď uniknuvší pixely z dalších výpočtů
        mask &= ~escaped
        # Předčasný konec — všechny pixely unikly
        if not mask.any():
            break

    # Pixely stále v masce (neunikly) mají hodnotu max_iter => uvnitř množiny
    return iterations


def iterations_to_image(grid: np.ndarray, max_iter: int) -> Image.Image:
    """Převede matici iterací na RGB obrázek pomocí palety."""
    # Normalizace: počet iterací (0–max_iter) => index do palety (0–255)
    indices = (grid.astype(np.float32) / max_iter * 255).clip(0, 255).astype(np.uint8)
    # Každý index nahraď RGB barvou z palety
    rgb = PALETTE[indices]
    return Image.fromarray(rgb)


class FractalViewer:
    """
    Interaktivní prohlížeč fraktálu — každý zoom přepočítá oblast.

    Ovládání:
      Scroll nahoru      — zoom dovnitř (centrovaný na kurzor)
      Scroll dolů        — zoom ven
      Levé tl. + táhni   — pan; přepočet nastane po uvolnění tlačítka
      Pravé tlačítko     — reset na výchozí pohled
    """

    ZOOM_FACTOR = 0.6

    def __init__(self, mode: str = 'mandelbrot') -> None:
        self.mode = mode
        self._default = MANDELBROT_BOUNDS if mode == 'mandelbrot' else JULIA_BOUNDS
        self.x_min, self.x_max, self.y_min, self.y_max = self._default

        self._pan_start: tuple | None = None  # (x0_min, x0_max, y0_min, y0_max, px, py)
        self._photo: ImageTk.PhotoImage | None = None  # reference kvůli GC

        # --- tkinter okno ---
        name = 'Mandelbrotova množina' if mode == 'mandelbrot' else 'Juliova množina'
        self.root = tk.Toplevel() if tk._default_root else tk.Tk()
        self.root.title(name)
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT,
                                cursor='crosshair', bg='black')
        self.canvas.pack()

        self._status = tk.StringVar()
        tk.Label(self.root, textvariable=self._status, anchor='w',
                 font=('Courier', 9)).pack(fill='x', padx=4)

        self.canvas.bind('<MouseWheel>', self._on_scroll)          # Windows
        self.canvas.bind('<Button-4>', self._on_scroll)            # Linux scroll up
        self.canvas.bind('<Button-5>', self._on_scroll)            # Linux scroll down
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<ButtonPress-3>', self._on_right_click)

        self._render()

    # ------------------------------------------------------------------
    # Výpočet a kreslení
    # ------------------------------------------------------------------

    def _compute(self) -> np.ndarray:
        if self.mode == 'mandelbrot':
            return compute_mandelbrot(self.x_min, self.x_max,
                                      self.y_min, self.y_max,
                                      WIDTH, HEIGHT, MAX_ITER)
        return compute_julia(self.x_min, self.x_max,
                             self.y_min, self.y_max,
                             JULIA_C, WIDTH, HEIGHT, MAX_ITER)

    def _render(self) -> None:
        """Přepočítá fraktál a zobrazí ho na canvas."""
        self._update_status('Počítám...')
        self.root.update_idletasks()

        grid = self._compute()
        img = iterations_to_image(grid, MAX_ITER)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

        xr = self.x_max - self.x_min
        self._update_status(
            f'Re [{self.x_min:.6f}, {self.x_max:.6f}]  '
            f'Im [{self.y_min:.6f}, {self.y_max:.6f}]  '
            f'šířka {xr:.2e}  |  Scroll=zoom  Táhni=pan  Pravé=reset'
        )

    def _pan_preview(self) -> None:
        """Posune zobrazený obrázek bez přepočtu (plynulý pan)."""
        if self._photo is None:
            return
        # Vypočítáme pixel offset z původního pohledu
        orig_x_min, orig_x_max, orig_y_min, orig_y_max, _, _ = self._pan_start  # type: ignore[misc]
        orig_x_range = orig_x_max - orig_x_min
        orig_y_range = orig_y_max - orig_y_min

        # O kolik datových jednotek jsme posunuli
        dx = self.x_min - orig_x_min
        dy = self.y_min - orig_y_min

        # Převod na pixely
        px_offset = int(-dx / orig_x_range * WIDTH)
        py_offset = int(dy / orig_y_range * HEIGHT)   # y je převrácené

        self.canvas.delete('all')
        self.canvas.create_image(px_offset, py_offset, anchor='nw', image=self._photo)

    def _update_status(self, text: str) -> None:
        self._status.set(text)

    # ------------------------------------------------------------------
    # Pixel ↔ datové souřadnice
    # ------------------------------------------------------------------

    def _pixel_to_data(self, px: int, py: int) -> tuple[float, float]:
        xd = self.x_min + px / WIDTH * (self.x_max - self.x_min)
        yd = self.y_max - py / HEIGHT * (self.y_max - self.y_min)
        return xd, yd

    # ------------------------------------------------------------------
    # Obsluha událostí
    # ------------------------------------------------------------------

    def _on_scroll(self, event) -> None:
        # delta > 0 = scroll nahoru = zoom dovnitř
        if hasattr(event, 'delta'):
            zoom_in = event.delta > 0
        else:
            zoom_in = event.num == 4

        factor = self.ZOOM_FACTOR if zoom_in else 1.0 / self.ZOOM_FACTOR

        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min

        xd, yd = self._pixel_to_data(event.x, event.y)
        xr = (xd - self.x_min) / x_range
        yr = (self.y_max - yd) / y_range  # relativní pozice od horního okraje

        nx = x_range * factor
        ny = y_range * factor

        self.x_min = xd - xr * nx
        self.x_max = xd + (1.0 - xr) * nx
        self.y_max = yd + yr * ny
        self.y_min = yd - (1.0 - yr) * ny

        self._render()

    def _on_press(self, event) -> None:
        self._pan_start = (
            self.x_min, self.x_max, self.y_min, self.y_max,
            event.x, event.y,
        )

    def _on_motion(self, event) -> None:
        if self._pan_start is None:
            return
        x0_min, x0_max, y0_min, y0_max, px0, py0 = self._pan_start

        dpx = event.x - px0
        dpy = event.y - py0

        x_range = x0_max - x0_min
        y_range = y0_max - y0_min

        dx = dpx / WIDTH * x_range
        dy = dpy / HEIGHT * y_range

        self.x_min = x0_min - dx
        self.x_max = x0_max - dx
        self.y_min = y0_min + dy   # py roste dolů, y_min roste nahoru → opačné znaménko
        self.y_max = y0_max + dy

        self._pan_preview()

    def _on_release(self, _) -> None:
        if self._pan_start is not None:
            self._pan_start = None
            self._render()

    def _on_right_click(self, _) -> None:
        self.x_min, self.x_max, self.y_min, self.y_max = self._default
        self._render()


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # Skryjeme prázdné hlavní okno

    FractalViewer('mandelbrot')
    FractalViewer('julia')

    root.mainloop()

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk

class CalculationCancelled(Exception):
    """Výjimka signalizující uživatelské zrušení výpočtu."""

# parse_rule -> expand_lsystem -> build_lsystem_segments

def expand_lsystem(axiom: str, rules: dict[str, str], nesting: int,
                   cancel_event: threading.Event | None = None) -> str:
    """Rozbalí L-systém aplikací pravidel po zadaný počet iterací."""
    # Krok 1: Začínáme s axiomem jako počáteční sekvencí
    result = axiom
    for _ in range(nesting):
        if cancel_event is not None and cancel_event.is_set():
            raise CalculationCancelled()
        # Krok 2: Pro každou iteraci projdeme každý symbol aktuální sekvence
        new_result = []
        for ch in result:
            if cancel_event is not None and cancel_event.is_set():
                raise CalculationCancelled()
            # Krok 3: Symbol nahradíme jeho pravidlem; neznámé symboly ponecháme beze změny
            new_result.append(rules.get(ch, ch))
        # Krok 4: Nové symboly spojíme do řetězce a použijeme jako vstup příští iterace
        result = ''.join(new_result)
    return result


def parse_rule(rule_str: str) -> tuple[str, str]:
    """Parsuje pravidlo ve tvaru 'F -> F+F-F' nebo 'F->F+F-F'.

    Vrací (symbol, náhrada). Pokud formát neobsahuje '->',
    předpokládá se pravidlo pro 'F'.
    """
    rule_str = rule_str.strip()
    if '->' in rule_str:
        parts = rule_str.split('->', 1)
        return parts[0].strip(), parts[1].strip()
    return 'F', rule_str


def draw_lsystem(canvas: tk.Canvas, sequence: str, start_x: float, start_y: float,
                 start_angle_rad: float, line_length: float, angle_delta: float) -> None:
    """Vykreslí L-systém na canvas podle sekvence symbolů.

    Souřadnicový systém:
    - 0 rad = vpravo (kladná osa x)
    - kladný úhel = po směru hodinových ručiček (osa y roste dolů)
    """
    segments = build_lsystem_segments(
        sequence, start_x, start_y, start_angle_rad, line_length, angle_delta
    )
    for x1, y1, x2, y2 in segments:
        canvas.create_line(x1, y1, x2, y2, fill='black', width=1)


def build_lsystem_segments(sequence: str, start_x: float, start_y: float,
                           start_angle_rad: float, line_length: float,
                           angle_delta: float,
                           cancel_event: threading.Event | None = None) -> list[tuple[float, float, float, float]]:
    """Převede sekvenci L-systému na seznam úseček (x1, y1, x2, y2)."""
    # Krok 1: Inicializace a zásobníku pro větvení
    x, y = start_x, start_y
    angle = start_angle_rad
    stack: list[tuple[float, float, float]] = []
    segments: list[tuple[float, float, float, float]] = []

    # Krok 2: Sekvenční interpretace každého symbolu L-systému
    for ch in sequence:
        if cancel_event is not None and cancel_event.is_set():
            raise CalculationCancelled()
        if ch == 'F':
            # Krok 3a: Posun vpřed s kreslením čáry (posune se a zanechá stopu)
            new_x = x + line_length * math.cos(angle)
            new_y = y + line_length * math.sin(angle)
            segments.append((x, y, new_x, new_y))
            x, y = new_x, new_y
        elif ch == 'b':
            # Krok 3b: Posun vpřed bez kreslení (skok bez stopy)
            x += line_length * math.cos(angle)
            y += line_length * math.sin(angle)
        elif ch == '+':
            # Krok 3c: Otočení doprava o úhel angle_delta
            angle += angle_delta
        elif ch == '-':
            # Krok 3d: Otočení doleva o úhel angle_delta
            angle -= angle_delta
        elif ch == '[':
            # Krok 3e: Uložení aktuálního stavu (pozice + úhel) na zásobník — začátek větve
            stack.append((x, y, angle))
        elif ch == ']':
            # Krok 3f: Obnovení stavu ze zásobníku — návrat na místo začátku větve
            if stack:
                x, y, angle = stack.pop()

    return segments


class LSystemApp:
    """Tkinter aplikace pro vykreslování L-systémů."""

    PRESETS = [
        {
            'axiom': 'F+F+F+F',
            'rule': 'F -> F+F-F-FF+F+F-F',
            'angle_deg': 90.0,
        },
        {
            'axiom': 'F++F++F',
            'rule': 'F -> F+F--F+F',
            'angle_deg': 60.0,
        },
        {
            'axiom': 'F',
            'rule': 'F -> F[+F]F[-F]F',
            'angle_deg': math.degrees(math.pi / 7),
        },
        {
            'axiom': 'F',
            'rule': 'F -> FF+[+F-F-F]-[-F+F+F]',
            'angle_deg': math.degrees(math.pi / 8),
        },
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('L-systems drawing')
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.is_rendering = False
        self.render_chunk_size = 1200
        self.action_buttons: list[ttk.Button] = []
        self.cancel_button: ttk.Button | None = None
        self.status_var = tk.StringVar(value='Ready')
        self._build_ui()

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(self.root, bg='white', width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = ttk.Frame(self.root, padding=8)
        panel.pack(side=tk.RIGHT, fill=tk.Y)

        def labeled_entry(label: str, default: str) -> tk.StringVar:
            ttk.Label(panel, text=label).pack(anchor='w')
            var = tk.StringVar(value=default)
            ttk.Entry(panel, textvariable=var, width=18).pack(fill='x')
            return var

        self.var_x = labeled_entry('Starting X position (int)', '400')
        self.var_y = labeled_entry('Starting Y position (int)', '300')
        self.var_angle_deg = labeled_entry('Starting angle (degree)', '0')
        self.var_angle_rad = labeled_entry('Starting angle (radians)', '')
        self.var_nesting = labeled_entry('The number of nesting (int)', '3')
        self.var_line_size = labeled_entry('Size of the line (int)', '5')

        btn = ttk.Button(panel, text='Draw first',
                 command=lambda: self._draw_preset(0))
        btn.pack(fill='x', pady=1)
        self.action_buttons.append(btn)
        btn = ttk.Button(panel, text='Draw second',
                 command=lambda: self._draw_preset(1))
        btn.pack(fill='x', pady=1)
        self.action_buttons.append(btn)
        btn = ttk.Button(panel, text='Draw third',
                 command=lambda: self._draw_preset(2))
        btn.pack(fill='x', pady=1)
        self.action_buttons.append(btn)
        btn = ttk.Button(panel, text='Draw fourth',
                 command=lambda: self._draw_preset(3))
        btn.pack(fill='x', pady=1)
        self.action_buttons.append(btn)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(panel, text='Custom').pack(anchor='w')

        self.var_custom_axiom = labeled_entry('Axiom (F,+,-,[,])', '')
        self.var_custom_rule = labeled_entry('Rule (F,+,-,[,])', '')
        self.var_custom_angle_deg = labeled_entry('Angle (degree)', '')
        self.var_custom_angle_rad = labeled_entry('Angle (radians)', '')

        btn = ttk.Button(panel, text='Draw custom',
                         command=self._draw_custom)
        btn.pack(fill='x', pady=1)
        self.action_buttons.append(btn)

        btn = ttk.Button(panel, text='Clear canvas',
                         command=self._clear_canvas)
        btn.pack(fill='x', pady=1)
        self.action_buttons.append(btn)

        self.cancel_button = ttk.Button(
            panel,
            text='Cancel calculation',
            command=self._cancel_calculation,
            state='disabled',
        )
        self.cancel_button.pack(fill='x', pady=1)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(panel, textvariable=self.status_var, wraplength=180).pack(anchor='w')

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = 'normal' if enabled else 'disabled'
        for button in self.action_buttons:
            button.configure(state=state)

    def _clear_canvas(self) -> None:
        self.canvas.delete('all')
        self.status_var.set('Canvas cleared')

    def _set_cancel_enabled(self, enabled: bool) -> None:
        if self.cancel_button is None:
            return
        self.cancel_button.configure(state='normal' if enabled else 'disabled')

    def _cancel_calculation(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_event.set()
            self.status_var.set('Ruším výpočet...')

    def _start_background_draw(self, axiom: str, rules: dict[str, str], nesting: int,
                               x: float, y: float, start_angle_rad: float,
                               line_size: float, angle_delta: float) -> None:
        if (self.worker_thread and self.worker_thread.is_alive()) or self.is_rendering:
            self.status_var.set('Počkejte na dokončení aktuálního výpočtu')
            return

        self.result_queue = queue.Queue()
        self.cancel_event.clear()
        self._set_controls_enabled(False)
        self._set_cancel_enabled(True)
        self.status_var.set('Počítám L-systém na pozadí...')

        def worker() -> None:
            try:
                sequence = expand_lsystem(axiom, rules, nesting, self.cancel_event)
                segments = build_lsystem_segments(
                    sequence, x, y, start_angle_rad, line_size, angle_delta, self.cancel_event
                )
                self.result_queue.put(('ok', segments))
            except CalculationCancelled:
                self.result_queue.put(('cancelled', None))
            except Exception as exc:
                self.result_queue.put(('error', str(exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self._poll_worker_result()

    def _poll_worker_result(self) -> None:
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(30, self._poll_worker_result)
            return

        self.worker_thread = None
        self._set_cancel_enabled(False)
        if status == 'cancelled':
            self.status_var.set('Výpočet byl zrušen')
            self._set_controls_enabled(True)
            return
        if status == 'error':
            self.status_var.set(f'Chyba při výpočtu: {payload}')
            self._set_controls_enabled(True)
            return

        segments = payload
        self.canvas.delete('all')
        self.is_rendering = True
        self.status_var.set(f'Kreslím {len(segments)} úseček...')
        self._draw_segments_chunked(segments, 0)

    def _draw_segments_chunked(self, segments: list[tuple[float, float, float, float]],
                               start_index: int) -> None:
        end_index = min(start_index + self.render_chunk_size, len(segments))
        for x1, y1, x2, y2 in segments[start_index:end_index]:
            self.canvas.create_line(x1, y1, x2, y2, fill='black', width=1)

        if end_index < len(segments):
            self.root.after(1, self._draw_segments_chunked, segments, end_index)
            return

        self.is_rendering = False
        self.status_var.set(f'Hotovo: {len(segments)} úseček')
        self._set_controls_enabled(True)

    def _common_params(self) -> tuple[float, float, float, int, float]:
        """Vrátí (start_x, start_y, start_angle_rad, nesting, line_size)."""
        x = float(self.var_x.get() or 400)
        y = float(self.var_y.get() or 300)
        nesting = int(self.var_nesting.get() or 3)
        line_size = float(self.var_line_size.get() or 5)

        rad_text = self.var_angle_rad.get().strip()
        if rad_text:
            angle_rad = float(rad_text)
        else:
            angle_rad = math.radians(float(self.var_angle_deg.get() or 0))

        return x, y, angle_rad, nesting, line_size

    def _draw_preset(self, index: int) -> None:
        preset = self.PRESETS[index]
        # Krok 1: Načtení společných parametrů z UI (pozice, úhel, hloubka, délka čáry)
        x, y, start_angle_rad, nesting, line_size = self._common_params()
        # Krok 2: Parsování přepisovacího pravidla předvolby
        symbol, replacement = parse_rule(preset['rule'])
        # Krok 3: Spuštění výpočtu i přípravy geometrie na pozadí
        self._start_background_draw(
            preset['axiom'],
            {symbol: replacement},
            nesting,
            x,
            y,
            start_angle_rad,
            line_size,
            math.radians(preset['angle_deg']),
        )

    def _draw_custom(self) -> None:
        # Krok 1: Načtení uživatelem zadaného axiomu a pravidla
        axiom = self.var_custom_axiom.get().strip()
        rule_str = self.var_custom_rule.get().strip()
        if not axiom or not rule_str:
            return

        # Krok 2: Určení úhlu otočení (radián má přednost před stupni)
        rad_text = self.var_custom_angle_rad.get().strip()
        if rad_text:
            angle_delta = float(rad_text)
        else:
            angle_delta = math.radians(float(self.var_custom_angle_deg.get() or 90))

        # Krok 3: Načtení společných parametrů a parsování pravidla
        x, y, start_angle_rad, nesting, line_size = self._common_params()
        symbol, replacement = parse_rule(rule_str)
        # Krok 4: Expanze i příprava úseček běží mimo UI vlákno
        self._start_background_draw(
            axiom,
            {symbol: replacement},
            nesting,
            x,
            y,
            start_angle_rad,
            line_size,
            angle_delta,
        )


if __name__ == '__main__':
    root = tk.Tk()
    LSystemApp(root)
    root.mainloop()

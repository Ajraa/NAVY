import math
import tkinter as tk
from tkinter import ttk


def expand_lsystem(axiom: str, rules: dict[str, str], nesting: int) -> str:
    """Rozbalí L-systém aplikací pravidel po zadaný počet iterací."""
    result = axiom
    for _ in range(nesting):
        new_result = []
        for ch in result:
            new_result.append(rules.get(ch, ch))
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
    x, y = start_x, start_y
    angle = start_angle_rad
    stack: list[tuple[float, float, float]] = []

    for ch in sequence:
        if ch == 'F':
            new_x = x + line_length * math.cos(angle)
            new_y = y + line_length * math.sin(angle)
            canvas.create_line(x, y, new_x, new_y, fill='black', width=1)
            x, y = new_x, new_y
        elif ch == 'b':
            x += line_length * math.cos(angle)
            y += line_length * math.sin(angle)
        elif ch == '+':
            angle += angle_delta
        elif ch == '-':
            angle -= angle_delta
        elif ch == '[':
            stack.append((x, y, angle))
        elif ch == ']':
            if stack:
                x, y, angle = stack.pop()


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

        ttk.Button(panel, text='Draw first',
                   command=lambda: self._draw_preset(0)).pack(fill='x', pady=1)
        ttk.Button(panel, text='Draw second',
                   command=lambda: self._draw_preset(1)).pack(fill='x', pady=1)
        ttk.Button(panel, text='Draw third',
                   command=lambda: self._draw_preset(2)).pack(fill='x', pady=1)
        ttk.Button(panel, text='Draw fourth',
                   command=lambda: self._draw_preset(3)).pack(fill='x', pady=1)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(panel, text='Custom').pack(anchor='w')

        self.var_custom_axiom = labeled_entry('Axiom (F,+,-,[,])', '')
        self.var_custom_rule = labeled_entry('Rule (F,+,-,[,])', '')
        self.var_custom_angle_deg = labeled_entry('Angle (degree)', '')
        self.var_custom_angle_rad = labeled_entry('Angle (radians)', '')

        ttk.Button(panel, text='Draw custom',
                   command=self._draw_custom).pack(fill='x', pady=1)
        ttk.Button(panel, text='Clear canvas',
                   command=lambda: self.canvas.delete('all')).pack(fill='x', pady=1)

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
        x, y, start_angle_rad, nesting, line_size = self._common_params()
        symbol, replacement = parse_rule(preset['rule'])
        sequence = expand_lsystem(preset['axiom'], {symbol: replacement}, nesting)
        draw_lsystem(self.canvas, sequence, x, y, start_angle_rad,
                     line_size, math.radians(preset['angle_deg']))

    def _draw_custom(self) -> None:
        axiom = self.var_custom_axiom.get().strip()
        rule_str = self.var_custom_rule.get().strip()
        if not axiom or not rule_str:
            return

        rad_text = self.var_custom_angle_rad.get().strip()
        if rad_text:
            angle_delta = float(rad_text)
        else:
            angle_delta = math.radians(float(self.var_custom_angle_deg.get() or 90))

        x, y, start_angle_rad, nesting, line_size = self._common_params()
        symbol, replacement = parse_rule(rule_str)
        sequence = expand_lsystem(axiom, {symbol: replacement}, nesting)
        draw_lsystem(self.canvas, sequence, x, y, start_angle_rad, line_size, angle_delta)


if __name__ == '__main__':
    root = tk.Tk()
    LSystemApp(root)
    root.mainloop()

import random
from collections import deque
import numpy as np

from visualization import plot_find_cheese


class FindCheeseEnv:
	"""Jednoduché grid prostředí pro úlohu Find the cheese."""

	def __init__(self, rows, cols, start, cheese, holes):
		self.rows = rows
		self.cols = cols
		self.start = start
		self.cheese = cheese
		self.holes = set(holes)
		self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
		self.reset()

	def reset(self):
		self.position = self.start
		return self.position

	def _inside(self, r, c):
		return 0 <= r < self.rows and 0 <= c < self.cols

	def step(self, action_idx):
		dr, dc = self.actions[action_idx]
		r, c = self.position
		nr, nc = r + dr, c + dc

		if not self._inside(nr, nc):
			# Penalizace za pokus mimo mapu.
			return self.position, -3.0, False

		self.position = (nr, nc)

		if self.position in self.holes:
			return self.position, -100.0, True
		if self.position == self.cheese:
			return self.position, 100.0, True

		return self.position, -1.0, False


def epsilon_greedy(q_table, state, epsilon):
	# S pravděpodobností epsilon zkoušíme náhodnou akci (exploration),
	# jinak bereme nejlepší známou akci (exploitation).
	if random.random() < epsilon:
		return random.randint(0, q_table.shape[2] - 1)

	r, c = state
	return int(np.argmax(q_table[r, c]))


def greedy_path(env, q_table, max_steps=100):
	# Vyhodnocení naučené politiky bez náhody: vždy volíme argmax akci.
	state = env.reset()
	path = [state]

	for _ in range(max_steps):
		r, c = state
		action = int(np.argmax(q_table[r, c])) # Postupně volíme nejlepší výsledek a zapisujeme cestu
		next_state, _, done = env.step(action)
		state = next_state
		path.append(state)

		if done:
			break

	success = len(path) > 1 and path[-1] == env.cheese
	return path, success


def q_learning_find_cheese(
	env,
	episodes=5000,
	alpha=0.15,
	gamma=0.95,
	epsilon_start=1.0,
	epsilon_decay=0.998,
	epsilon_min=0.05,
	max_steps=100,
):
	# Q-tabulka má tvar [řádek, sloupec, akce].
	# Každá hodnota odhaduje, jak výhodné je provést akci v daném stavu.
	q_table = np.zeros((env.rows, env.cols, len(env.actions))) # 3D pole, pro každý stav (r, c) a akci (N, S, W, E) máme odhad Q-hodnoty.
	rewards = []
	successes = []

	epsilon = epsilon_start

	for _ in range(episodes):
		# Každá epizoda začíná ve startu a končí dosažením cíle/jámy nebo limitem kroků.
		state = env.reset()
		total_reward = 0.0
		done = False

		for _ in range(max_steps):
			action = epsilon_greedy(q_table, state, epsilon) # Výběr směru
			next_state, reward, done = env.step(action)

			r, c = state
			nr, nc = next_state

			q_old = q_table[r, c, action]
			q_best_next = np.max(q_table[nr, nc])
			# Bellmanův update:
			# Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
			q_table[r, c, action] = q_old + alpha * (reward + gamma * q_best_next - q_old)

			state = next_state
			total_reward += reward

			if done:
				break

		rewards.append(total_reward)
		successes.append(1 if state == env.cheese else 0)
		# Postupně snižujeme exploration, aby ke konci převládlo využití naučených znalostí.
		epsilon = max(epsilon_min, epsilon * epsilon_decay)

	return q_table, rewards, successes


def find_learning_iteration(successes, window=100, threshold=0.95):
	"""Najde první epizodu, kde je úspěšnost v klouzavém okně dostatečně vysoká."""
	if len(successes) < window:
		return None

	arr = np.array(successes, dtype=float)
	kernel = np.ones(window) / window
	moving_avg = np.convolve(arr, kernel, mode='valid')

	for idx, value in enumerate(moving_avg):
		if value >= threshold:
			return idx + window

	return None


def has_path(rows, cols, start, cheese, holes):
	"""Ověří BFS, že existuje cesta ze startu k sýru mimo bariéry."""
	queue = deque([start])
	visited = {start}
	actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	while queue:
		r, c = queue.popleft()
		if (r, c) == cheese:
			return True

		for dr, dc in actions:
			nr, nc = r + dr, c + dc
			if not (0 <= nr < rows and 0 <= nc < cols):
				continue
			next_pos = (nr, nc)
			if next_pos in holes or next_pos in visited:
				continue
			visited.add(next_pos)
			queue.append(next_pos)

	return False


def generate_random_start_and_cheese(rows, cols):
	"""Náhodně vybere odlišné pozice myši a sýru."""
	all_cells = [(r, c) for r in range(rows) for c in range(cols)]
	if len(all_cells) < 2:
		raise ValueError('Mapa musí mít alespoň 2 buňky pro různé pozice myši a sýru.')

	start, cheese = random.sample(all_cells, 2)
	return start, cheese


def generate_random_holes(rows, cols, start, cheese, hole_count, max_attempts=1000):
	"""Náhodně vygeneruje bariéry tak, aby mapa zůstala řešitelná."""
	all_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in {start, cheese}]
	max_holes = len(all_cells)
	hole_count = max(0, min(hole_count, max_holes))

	for _ in range(max_attempts):
		holes = set(random.sample(all_cells, hole_count))
		if has_path(rows, cols, start, cheese, holes):
			return holes

	raise ValueError('Nepodařilo se vygenerovat průchozí mapu s daným počtem bariér.')


if __name__ == '__main__':
	rows = 10
	cols = 10
	start, cheese = generate_random_start_and_cheese(rows, cols)
	hole_count = 20
	holes = generate_random_holes(rows, cols, start, cheese, hole_count)

	env = FindCheeseEnv(
		rows=rows,
		cols=cols,
		start=start,
		cheese=cheese,
		holes=holes,
	)

	q_table, rewards, successes = q_learning_find_cheese(env)

	learned_at = find_learning_iteration(successes, window=100, threshold=0.95)
	final_path, final_success = greedy_path(env, q_table, max_steps=50)

	print(f'Náhodná pozice myši (start): {env.start}')
	print(f'Náhodná pozice sýru: {env.cheese}')
	print(f'Náhodně vygenerované bariéry: {sorted(env.holes)}')
	if learned_at is None:
		print('Stabilní naučení v nastaveném počtu epizod nenastalo.')
	else:
		print(f'Počet iterací potřebných k naučení: {learned_at}')

	print(f'Greedy průchod po tréninku: {"úspěch" if final_success else "neúspěch"}')
	print(f'Délka finální cesty: {len(final_path) - 1} kroků')

	# odměna epizody je z rewards, klouzavý průměr úspěšnosti je průměr za posledních 50 epizod
	plot_find_cheese(
		rows=env.rows,
		cols=env.cols,
		start=env.start,
		cheese=env.cheese,
		holes=env.holes,
		path=final_path,
		rewards=rewards,
	)

	start, cheese = generate_random_start_and_cheese(rows, cols)
	env.start = start
	final_path, final_success = greedy_path(env, q_table, max_steps=50)
	plot_find_cheese(
		rows=env.rows,
		cols=env.cols,
		start=env.start,
		cheese=env.cheese,
		holes=env.holes,
		path=final_path,
		rewards=rewards,
	)

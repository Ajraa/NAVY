import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from visualization import plot_cartpole


class DQN(nn.Module):
    """Hluboká Q-síť (Deep Q-Network) pro CartPole.

    Architektura: vstupní vrstva (4 neurony) → skrytá vrstva → skrytá vrstva → výstupní vrstva (2 neurony).
    Počet akcí v CartPole je 2: tlak vlevo (0) nebo vpravo (1).
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        # Plně propojená síť s ReLU aktivacemi
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer pro DQN.

    Ukládá přechody (stav, akce, odměna, nový stav, konec epizody)
    a umožňuje náhodné vzorkování dávek pro trénink.
    """

    def __init__(self, capacity: int = 10_000):
        # Deque s maximální kapacitou – starší záznamy se automaticky odstraňují
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        """Uloží jeden přechod do bufferu."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Vrátí náhodně vybranou dávku přechodů."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def _greedy_eval(net: DQN, env, seed: int) -> float:
    """Spustí jednu greedy epizodu (epsilon=0) a vrátí celkovou odměnu.

    Používá se uvnitř tréninku pro sledování skutečného výkonu sítě
    bez vlivu náhodného průzkumu (epsilon-greedy).
    """
    state, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    with torch.no_grad():
        while not done:
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = int(net(t).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total


def train_dqn(
    episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_capacity: int = 10_000,
    target_update_freq: int = 10,
    hidden: int = 64,
    seed: int = 42,
    eval_freq: int = 10,
    cart_penalty: float = 0.3,
):
    """Natrénuje DQN agenta na prostředí CartPole-v1.

    Algoritmus (Double DQN + reward shaping):
    1. Policy síť vybírá akce (epsilon-greedy).
    2. Přechody se ukládají do replay bufferu s upravenou odměnou:
       shaped_reward = 1.0 − cart_penalty · (cart_pos / 2.4)²
       Penalizace nutí agenta zůstávat uprostřed kolejnice.
    3. Každý krok se vzorkuje dávka a provede gradient krok Huber loss.
       TD cíl počítá Double DQN: policy_net volí akci, target_net ji hodnotí.
    4. Gradienty jsou ořezány (clip_grad_norm) pro stabilitu.
    5. Target síť se periodicky synchronizuje s policy sítí.
    6. Každých eval_freq epizod se spustí greedy evaluace;
       nejlepší greedy výkon určuje, která váha se vrátí jako výsledek.

    Vrací tuple (policy_net, rewards_list), kde policy_net nese váhy
    nejlepší greedy epizody (= demo vždy odpovídá skutečnému vrcholu).
    """
    # Nastavení seedů pro reprodukovatelnost
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prostředí CartPole – jedno pro trénink, druhé pro greedy evaluaci
    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')

    # Dvě sítě: policy (učí se) a target (stabilní cílová reference)
    policy_net = DQN(hidden=hidden)
    target_net = DQN(hidden=hidden)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target síť se pouze čte, neprovádí se gradient

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Huber loss (SmoothL1) – robustnější než MSE, méně citlivá na odlehlé TD chyby
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    epsilon = epsilon_start
    rewards_list = []
    best_greedy_reward = -float('inf')
    best_weights = None  # váhy modelu s nejlepším greedy výkonem

    for episode in range(1, episodes + 1):
        # Krok 1: Reset prostředí a inicializace epizody
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = False

        while not done:
            # Krok 2: Epsilon-greedy výběr akce
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = int(q_values.argmax(dim=1).item()) # Nejlepší akce podle policy sítě

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Reward shaping: penalizace za vzdálení vozíku od středu kolejnice.
            # Bez penalizace agent balancuje tyč, ale ignoruje drift vozíku
            # a epizoda vždy končí vyjetím z mezí ±2.4.
            if not done:
                cart_pos = next_state[0]
                reward -= cart_penalty * (cart_pos / 2.4) ** 2

            # Krok 3: Uložení přechodu do replay bufferu
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Krok 4: Trénink policy sítě, pokud je buffer dostatečně naplněn
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)

                states_t = torch.tensor(states_b)
                actions_t = torch.tensor(actions_b).unsqueeze(1)
                rewards_t = torch.tensor(rewards_b)
                next_states_t = torch.tensor(next_states_b)
                dones_t = torch.tensor(dones_b)

                # Q-hodnoty aktuálního stavu pro zvolené akce
                current_q = policy_net(states_t).gather(1, actions_t).squeeze(1)

                # Double DQN TD cíl:
                #   akce vybírá policy_net (redukuje overestimation bias),
                #   hodnotu hodnotí stabilní target_net
                with torch.no_grad():
                    next_actions = policy_net(next_states_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(next_states_t).gather(1, next_actions).squeeze(1)
                    target_q = rewards_t + gamma * next_q * (1.0 - dones_t)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping – zabraňuje explozi gradientů
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

        rewards_list.append(total_reward)

        # Krok 5: Greedy evaluace každých eval_freq epizod – měří skutečný výkon bez epsilon.
        # Na základě greedy výkonu (ne epsilon-greedy odměny) ukládáme nejlepší váhy.
        if episode % eval_freq == 0:
            greedy_r = _greedy_eval(policy_net, eval_env, seed=seed + episode)
            if greedy_r > best_greedy_reward:
                best_greedy_reward = greedy_r
                best_weights = {k: v.clone() for k, v in policy_net.state_dict().items()}

        # Krok 6: Snižování průzkumné epsilon po každé epizodě (epsilon decay)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Krok 7: Synchronizace target sítě s policy sítí každých target_update_freq epizod
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Výpis průběhu každých 50 epizod
        if episode % 50 == 0:
            avg = np.mean(rewards_list[-50:])
            print(f'Epizoda {episode:4d} | průměrná odměna (50 ep.): {avg:6.1f} | epsilon: {epsilon:.3f} | nejlepší greedy: {best_greedy_reward:.0f}')

    env.close()
    eval_env.close()
    # Před vrácením načteme váhy nejlepší greedy epizody
    if best_weights is not None:
        policy_net.load_state_dict(best_weights)
    return policy_net, rewards_list


def collect_demo_episode(policy_net: DQN, seed: int = 0):
    """Spustí jednu greedy epizodu a vrátí záznamy stavů a celkovou odměnu.

    Greedy chování = epsilon=0 (vždy vybírá nejlepší akci dle Q-sítě).

    Vrací (states_list, total_reward), kde states_list je list čtveřic
    (cart_pos, cart_vel, pole_angle, pole_vel).
    """
    env = gym.make('CartPole-v1')
    state, _ = env.reset(seed=seed)
    states = []
    total_reward = 0.0
    done = False

    policy_net.eval()
    with torch.no_grad():
        while not done:
            states.append(tuple(float(x) for x in state))
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = int(policy_net(state_tensor).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

    env.close()
    return states, total_reward


def find_convergence_episode(rewards: list, window: int = 50, threshold: float = 195.0):
    """Najde první epizodu, kde klouzavý průměr přesáhl zadaný práh.

    Vrací číslo epizody (1-based) nebo None, pokud k překročení nedošlo.
    """
    for i in range(window - 1, len(rewards)):
        moving_avg = np.mean(rewards[i - window + 1 : i + 1])
        if moving_avg >= threshold:
            return i + 1  # 1-based index epizody
    return None


if __name__ == '__main__':
    policy_net, rewards = train_dqn(episodes=500)

    convergence = find_convergence_episode(rewards)
    if convergence:
        print(f'Agent konvergoval v epizodě {convergence}')
    else:
        print('Agent nekonvergoval v nastaveném počtu epizod')

    print(f'Průměrná odměna (posledních 50 epizod): {np.mean(rewards[-50:]):.1f}')

    demo_states, demo_reward = collect_demo_episode(policy_net)
    print(f'Demo epizoda (nejlepší greedy model): {demo_reward:.0f} kroků')

    plot_cartpole(rewards, demo_states)

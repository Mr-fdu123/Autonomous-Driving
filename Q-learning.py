import random


class GridWorld:
    def __init__(
        self,
        size=4,
        reward=10,
        penalty_cells=None,
        reward_cells=None,
        wall_cells=None,
        trap_cells=None,
    ):
        self.n_states = size * size
        self.n_actions = 4  # up 0 down 1 left 2 right 3
        self.size = size
        self.goal_reward = reward
        self.penalty_cells = {} if penalty_cells is None else penalty_cells
        self.reward_cells = {} if reward_cells is None else reward_cells
        self.wall_cells = set() if wall_cells is None else set(wall_cells)
        self.trap_cells = {} if trap_cells is None else trap_cells

        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.start_state = self.pos_to_state(self.start_pos)
        self.goal_state = self.pos_to_state(self.goal_pos)

        self.pos = self.start_pos
        self.state = self.start_state
        self.reward = 0
        self.steps = 0
        self.end = False
        self.collected_reward_cells = set()

    def pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def state_to_pos(self, state):
        return divmod(state, self.size)

    def reset(self):
        self.pos = self.start_pos
        self.state = self.start_state
        self.reward = 0
        self.steps = 0
        self.end = False
        self.collected_reward_cells = set()
        return self.state

    def cell_mark(self, pos):
        if pos == self.goal_pos:
            return "G"
        if pos == self.start_pos:
            return "S"
        if pos in self.wall_cells:
            return "W"
        if pos in self.trap_cells:
            return "T"
        if pos in self.reward_cells:
            return "R"
        if pos in self.penalty_cells:
            return "P"
        return None

    def step(self, action):
        self.steps += 1

        row, col = self.pos
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            row = min(row + 1, self.size - 1)
        elif action == 2:
            col = max(col - 1, 0)
        elif action == 3:
            col = min(col + 1, self.size - 1)
        else:
            raise ValueError("action is wrong")

        next_pos = (row, col)
        if next_pos in self.wall_cells:
            next_pos = self.pos

        self.pos = next_pos
        self.state = self.pos_to_state(self.pos)

        cur_reward = -1
        cur_reward += self.penalty_cells.get(self.pos, 0)
        cur_reward += self.trap_cells.get(self.pos, 0)

        if self.pos in self.reward_cells and self.pos not in self.collected_reward_cells:
            cur_reward += self.reward_cells[self.pos]
            self.collected_reward_cells.add(self.pos)

        if self.pos in self.trap_cells:
            self.end = True
        if self.pos == self.goal_pos:
            cur_reward += self.goal_reward
            self.end = True

        self.reward += cur_reward
        return cur_reward


class QlearningAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1, lr=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.q_table = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def greedy_action(self, state):
        q_values = self.q_table[state]
        max_q = max(q_values)
        actions = [a for a, q in enumerate(q_values) if max_q == q]
        return random.choice(actions)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])
        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += td_error * self.lr


def qlearning_train(
    env: GridWorld,
    agent: QlearningAgent,
    episodes: int = 100,
    max_steps_per_episodes: int = 100,
):
    rewards = []

    for episode in range(episodes):
        state = env.reset()

        for _ in range(max_steps_per_episodes):
            action = agent.choose_action(state=state)
            reward = env.step(action)
            next_state = env.state
            agent.update(state, action, reward, next_state, done=env.end)
            state = next_state

            if env.end:
                break

        rewards.append(env.reward)
        if (episode + 1) % 20 == 0:
            avg = sum(rewards[-20:]) / 20
            print(f"episode: {episode + 1}/{episodes}, avg_reward: {avg:.2f}")

    return rewards


def print_policy(env: GridWorld, agent: QlearningAgent) -> None:
    symbols = {
        0: "^",
        1: "v",
        2: "<",
        3: ">",
    }

    for state in range(env.n_states):
        pos = env.state_to_pos(state)
        mark = env.cell_mark(pos)
        if mark is not None:
            print(mark, end=" ")
        else:
            action = agent.greedy_action(state)
            print(symbols[action], end=" ")

        if (state + 1) % env.size == 0:
            print()


def main():
    env = GridWorld(
        size=4,
        reward=10,
        penalty_cells={(1, 1): -5},
        reward_cells={(0, 2): 3},
        wall_cells={(2, 1)},
        trap_cells={(1, 3): -10},
    )
    agent = QlearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        epsilon=0.1,
        lr=0.1,
        gamma=0.99,
    )
    qlearning_train(env, agent, episodes=400, max_steps_per_episodes=100)
    print_policy(env, agent)


if __name__ == "__main__":
    main()

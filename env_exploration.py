import gym
from gym.envs.box2d.lunar_lander import demo_heuristic_lander
from matplotlib import pyplot as plt

from playing import play_games
from utils import save_random_agent_gif, render_random_agent_episode

env = gym.make('LunarLander-v2', render_mode='human')

# Revisamos los wrappers el entorno:
print(env)

# Imprimimos la documentación del entorno base:
print(env.unwrapped.__doc__)

# Analizamos el entorno:
print(f"El valor del umbral de recompensa definido en el entorno es {env.spec.reward_threshold}.")
print(f"El número máximo de pasos por episodio es {env.spec.max_episode_steps}.")
print(f"Rango de las recompensas: {env.reward_range}")

# Acciones:
print(f"El espacio de acciones contiene {env.action_space.n} posibles acciones.")
"""
There are four discrete actions available: 
do nothing, fire left orientation engine, fire main engine, fire right orientation engine."""

# Observaciones:
print(f"Las dimensiones del espacio de observaciones son {env.observation_space.shape}.")
"""
The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle,
 its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
"""
print(f"El mínimo de los valores de espacio de observaciones es {list(env.observation_space.low)}")
print(f"El máximo de los valores de espacio de observaciones es {list(env.observation_space.high)}")

# Visualizamos el entorno
RANDOM_SEED = 666

# Agente aleatorio vs. heurístico
n_random_exploration_episodes = 3
for episode in range(n_random_exploration_episodes):
    demo_env = gym.make('LunarLander-v2', render_mode='human')
    demo_env.reset(seed=RANDOM_SEED + episode)
    demo_env.action_space.seed(RANDOM_SEED + episode)
    render_random_agent_episode(demo_env)
    demo_env.reset(seed=RANDOM_SEED + episode)
    demo_heuristic_lander(demo_env, render=True)
# Cerramos la visualización:
demo_env.close()

# Y guardamos una ejecución aleatoria:
rgb_env = gym.make('LunarLander-v2', render_mode='rgb_array')
rgb_env.reset(seed=RANDOM_SEED)
rgb_env.observation_space.seed(RANDOM_SEED)
save_random_agent_gif(
    env=rgb_env
)

# Y recompensa media y media de pasos de un agente aleatorio:
tr, ts = play_games(
    environment=rgb_env,
    policy=lambda observation: rgb_env.action_space.sample(),
    n_games=100
)
# Histograma de recompensas:
plt.hist(tr)
plt.title("Histogram of reward per episode")
plt.show()
# Histograma de pasos por episodio:
plt.hist(ts)
plt.title("Histogram of steps per episode")
plt.show()

import gym
import pygame

from utils import save_random_agent_gif, render_random_agent_episode

env = gym.make('LunarLander-v2', render_mode='human')

print(help(env))

print(f"El valor del umbral de recompensa definido en el entorno es {env.spec.reward_threshold}.")
print(f"El número máximo de pasos por episodio es {env.spec.max_episode_steps}.")
print(f"El espacio de acciones contiene {env.action_space.n} posibles acciones.")
"""
There are four discrete actions available: 
do nothing, fire left orientation engine, fire main engine, fire right orientation engine."""

print(f"Las dimensiones del espacio de observaciones son {env.observation_space.shape}.")
"""
The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle,
 its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.
"""
print(f"El mínimo de los valores de espacio de observaciones es {list(env.observation_space.low)}")
print(f"El máximo de los valores de espacio de observaciones es {list(env.observation_space.high)}")
print(f"Rango de las recompensas: {env.reward_range}")

# Visualizamos el entorno
n_random_exploration_episodes = 0
for episode in range(n_random_exploration_episodes):
    render_random_agent_episode(env)
# Cerramos la visualización:
env.close()
if env.screen:
    pygame.display.quit()
    pygame.quit()

# Y guardamos una ejecución aleatoria:

actions_dict = {
    0: 'None',
    1: 'Left engine',
    2: 'Main engine',
    3: 'Right engine'
}

save_random_agent_gif(
    env=gym.make('LunarLander-v2', render_mode='rgb_array'),
    actions_dict=actions_dict
)

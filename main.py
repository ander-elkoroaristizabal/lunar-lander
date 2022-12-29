import gym
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("El entorno utiliza: ", DEVICE)

env = gym.make('LunarLander-v2', render_mode='rgb_array')

print('Rango de las recompensas: ' + str(env.reward_range))

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

# Visualizamos el entorno
for i_episode in range(15):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _, info = env.step(action)
        if done:
            print("Episode finished after {} time steps".format(t + 1))
            break
env.close()

# Preprocessing:

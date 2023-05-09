import gymnasium as gym
from gymnasium.utils.play import play, PlayPlot


def callback(obs_t, obs_tp1, action, rew, termintated, truncated, info):
    return [rew]


def main():
  plotter = PlayPlot(callback, 30 * 5, ["reward"])
  env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
  play(env, callback=plotter.callback, zoom=4)


if __name__ == "__main__":
   main() 
import numpy as np

from rllib.basis import FourierBasis
from rllib.policies import Linear_Softmax
from rllib.agents import ActorCritic, Sarsa, Agent

from rllib.environments import Cartpole

from rllib.utils.memory import TrajectoryMemory

import matplotlib.pyplot as plt

def run_agent(env, agent: Agent, num_episodes: int, memory: TrajectoryMemory, save_dir: str, save_name: str):
    returns = []  # holds the undiscounted returns from each episode.

    for i in range(num_episodes):
        traj = memory.new()
        env.reset()  # reset environment for initial episode
        s = env.state  # gets the first state features
        done = False  # flag to tell when an episode ends
        G = 0.0  # Used to calculate the undiscounted return
        while not done:
            a, logp = agent.get_action(s)  # gets the action and the log probability of that action for the current state
            snext, reward, done = env.step(a)  # gets the next state, reward, and whether or note the episode is over.
            agent.update(s, a, logp, reward, snext, done)  # runs the agent update
            memory.add(traj, s, a, reward, logp, snext, done)  # saves the transition into a buffer (not an experience replay style buffer)
            s = snext  # updates the current state to be the next state
            G += reward  # update return for the current episode
        returns.append(G)  # add return from this episode to the list
        memory.save_memory(save_dir, save_name)  # save trajectories at save_dir/save_name
    return returns


def main():
    env = Cartpole()

    ranges = np.array([env.observation_space.low, env.observation_space.high]).T
    # ranges should be a (n,2) numpy array where each row contains the min and max values for that state variable
    # the ranges are necessary for the fourier basis because it works on inputs in the range [0,1].
    print("feature ranges: \n",ranges)

    dorder = 6  # this is the max order to combine different state variables in the fourier basis,
                # e.g., one feature would be cos(3x[0] + 4x[1]) if there were two state variables and dorder >= 4
                # all coefficients are used. The number of basis functions for this is pow(dorder+1, ranges.shape[0]).
    iorder = 7  # this is the max order for each each state variable applied independently of other variables.
                # e.g., order three would have the features cos(1x[0]), cos(2x[0]), cos(3x[0]) for each state variable in x
                # The number of basis functions for this component is ioder*ranges.shape[0].
                # This term is ignored if dorder >= iorder.
    both = False  # If true then both sine and cosine are used to create features
    basis = FourierBasis(ranges, dorder=dorder, iorder=iorder, both=both)


    num_actions = env.action_space.n  # assumes actions space is discrete. Continuous actions can also be handled by policy-critic

    epsilon = 0.015  # epsilon greedy parameter
    lam = 0.7  # eligibility trace decay parameter.
    gamma = 0.99  # reward discount factor. Usually ok to set to 1.0

    agent1 = Sarsa(basis, num_actions, epsilon=epsilon, lambda_=lam, gamma=gamma)

    policy = Linear_Softmax(basis, num_actions)
    # sets the policy learning rate based on a bound to L1 norm of d/dtheta ln pi(s,a,theta), i.e., 1 / (max_s ||d/d theta ln pi(s,a,theta)||_1)
    # this assumes linear function approximation using fourier basis and softmax policy.
    if both:
        alpha = 1.0 / ((np.sqrt(2) / 2) * basis.getNumFeatures())
    else:
        alpha = 1.0 / basis.getNumFeatures()
    beta = 0.05  # time scale parameter for moving average of the magnitude of returns.
                 # Moving average of return magnitudes is used to further scale the learning rate for the policy.
    Gmag = 9.0  # minimum return magnitude for this environment. If not not known use minimum number of steps. Setting this > 1 is important as it will scale down the learning rate.
                # Cartpole can fall over in 9 steps so we set this to 9.0.
    agent2 = ActorCritic(basis, policy, alpha=alpha, lambda_=lam, gamma=gamma, beta=beta, Gmag=Gmag)

    T = 1000  # max length of one episode
    N = 200  # maximum number of trajectories to store
    act_dim = 1  # only one discrete action to chose. If there are multiple action outputs this should be bigger.
    obs_dim = ranges.shape[0]  # number of state features
    memory = TrajectoryMemory(obs_dim=obs_dim, act_dim=act_dim, max_len=T, max_trajs=N)  # Create a memory object to store the observed trajectories
    srets = run_agent(env, agent1, num_episodes=100, memory=memory, save_dir="./", save_name="sarsa_experience.pkl")
    arets = run_agent(env, agent2, num_episodes=100, memory=memory, save_dir="./", save_name="ac_experience.pkl")
    # Note that the second time run_agent is called the same memory object is used. This means the trajectories saved will be from both sarsa and policy-critic
    # one can load previous saved trajectory buffer and reuse it. Or sets of trajectories can be saved individually and reloaded together.


    plt.plot(srets)
    plt.plot(arets)
    plt.legend(["Sarsa", "Actor-Critic"])
    plt.xlabel("Episodes")
    plt.ylabel("Returns (no discounting)")
    plt.show()

    # NOTE: Sarsa "struggles" on Cartpole, because it cannot adequately represent the q function.
    # Sarsa cannot represent the q function because the state features do not include time and thus
    # making the q have to predict a value between without knowing if the episode will terminate in a few steps or hundreds.
    # there are two fixes to this. The first is to include time as a feature. The second is to set gamma < 1
    # and change the algorithm so that when the episode terminates due to time, the prediction is still made for future value.



if __name__ == "__main__":
    main()
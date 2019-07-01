import numpy as np

from rllib.utils.utils import compute_cumulative_returns
from gym.spaces import Box

def collect_traj(env, agent, train=True, render=False):
    obs = []
    acts = []
    rews = []
    errs = []
    upd = 0
    ob = env.reset()
    done = False
    stochastic = True
    iteration = 0

    while not done:
        if render:
            env.render()
        iteration += 1
        action, logp = agent.get_action(ob.reshape(1, -1), stochastic=stochastic)

        action = action
        if isinstance(env.action_space, Box):
            caction = np.clip(action, env.action_space.low, env.action_space.high)
            caction[np.where(np.isnan(action))] = 0.
        else:
            caction = int(action)
        next_ob, rew, done, _ = env.step(caction)
        obs.append(ob.flatten())
        acts.append(action)
        rews.append(rew)
        # if iteration > 2000:
        #     done=True
        if train:
            updated, err = agent.update(ob, action, logp, rew, next_ob, done)
            errs.append(err)
            if updated:
                upd += 1

        ob = next_ob

    if len(errs) == 0:
        errs = [[0, 0]]
    if not train:
        return rews
    return obs, acts, rews, upd, errs

def train_agent(env, agent, tnum, gamma, num_episodes, plot=False, endeps=0):
    rewlist = []
    retlist = []
    lenlist = []
    vllist = []
    nuplist = []
    endretlist = []

    for episode in range(num_episodes):
        agent.new_episode()
        obs, acts, rews, upd, errs = collect_traj(env, agent, train=True)

        rewards = np.sum(rews)
        returns = compute_cumulative_returns(rews, 0, gamma)[0]
        vloss = np.mean(errs, axis=0)
        episode_lengths = len(obs)

        rewlist.append(rewards)
        retlist.append(returns)
        lenlist.append(episode_lengths)
        vllist.append(vloss)
        nuplist.append(upd)
        if plot:
            print('Trial:{0:03d} Episode: {1:04d} Return: {2:04.2f} Sum Rewards: {3:04.2f} Length: {4:04d}'.format(tnum, episode, returns, rewards, episode_lengths))

    res = rewlist
    if endeps > 0:
        for ep in range(endeps):
            agent.new_episode()
            rews = collect_traj(env, agent, train=False, render=False)
            endretlist.append(np.sum(rews))
    return res, endretlist
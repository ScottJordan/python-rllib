import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .agent import Agent

from rllib.critics import NetCritic
from rllib.policies import NetPolicy
from rllib.utils.netutils import to_numpy, to_tensor
class PPO(Agent):
    '''
    actor critic meant for batch learning with NN
    '''
    def __init__(self, vf:NetCritic, policy:NetPolicy, alpha=1e-4, steps_batch:int=64, batch_size:int=16, epochs:int=10, clip:float=0.2, ent_coef:float=0.0, gamma:float=0.999, lam:float=0.99):
        self.vf = vf
        self.policy = policy
        self.alpha = alpha
        self.step_batch = steps_batch
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip = clip
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.lam = lam

        self.opt = optim.Adam(list(policy.parameters())+list(vf.parameters()), lr=alpha)
        self.states = []
        self.actions = []
        self.blogps = []
        self.rewards = []
        self.terminals = []
        self.vals = []

    def cuda(self):
        self.vf.cuda()
        self.policy.cuda()

    def act(self, env, stochastic=True, train=True):

        state = env.state.astype(np.float32)
        act, blogp = self.policy.get_action(state, stochastic)
        env.step(act)
        reward = env.reward
        terminal = env.done

        self.states.append(to_tensor(state.astype(np.float32)))  # store state before action
        self.rewards.append(to_tensor(np.array([reward], dtype=np.float32))) # store reward for taking action in state
        self.actions.append(to_tensor(np.array([act], dtype=np.float32)))  # store action taken
        self.blogps.append(to_tensor(np.array([blogp], dtype=np.float32))) # store log probability of the action taken
        self.terminals.append(to_tensor(np.array([float(terminal)], dtype=np.float32)))  # store whether or not the action resulted in a terminal state
        v = self.vf.predict(state).detach()
        self.vals.append(v)  # store the value function estimate at the state after taking the action

        if len(self.states) >= self.step_batch:  # if buffer size if greater than the T update the policy and value function
            self.vals.append(torch.zeros_like(v))
            err = self.run_update()  # run update function

            # clear buffer
            self.states = []
            self.rewards = []
            self.actions = []
            self.blogps = []
            self.terminals = []
            self.vals = []
        else:
            err = [0., 0.]

        return err

    def get_action(self, state, stochastic=True):
        '''
        sample action from policy in given state
        :param state: state to sample from. A pytorch tensor batch
        :param stochastic: if false it uses the MLE action
        :return: tuple of (action, log probability of action)
        '''
        a, logp= self.policy.get_action(state, stochastic)
        return a, logp

    def update(self, state, act, blogp, reward, next_state, terminal):
        '''
        update algorithm for both value function and policy
        :param state: tensor of current states (BatchsizeXstate_dims)
        :param act: tensor for action taken (BatchsizeXActdim)
        :param blogp:  of log probabilities (BatchsizeX1)
        :param reward: reward at time t (BatchsizeX1)
        :param next_state: tensor of next states (BatchsizeXstate_dims)
        :param terminal: bool for end of episode
        :return:
        '''

        self.states.append(to_tensor(state.astype(np.float32)))
        self.rewards.append(to_tensor(np.array([reward], dtype=np.float32)))
        self.actions.append(to_tensor(np.array([act], dtype=np.float32)))
        self.blogps.append(to_tensor(np.array([blogp], dtype=np.float32)))
        self.terminals.append(to_tensor(np.array([float(terminal)], dtype=np.float32)))
        v = self.vf.predict(state).detach()
        self.vals.append(v)

        if len(self.states) >= self.step_batch:
            self.vals.append(torch.zeros_like(v))
            err = self.run_update()
            self.states = []
            self.rewards = []
            self.actions = []
            self.blogps = []
            self.terminals = []
            self.vals = []
        else:
            err = [0., 0.]

        return err



    def run_update(self):
        states = torch.stack(self.states, dim=0)
        actions = torch.stack(self.actions, dim=0)
        blogps = torch.stack(self.blogps, dim=0)
        gae = torch.zeros_like(blogps)
        lamret = torch.zeros_like(blogps)

        prev_gae = 0

        # compute advantage estimates and value function targets using lambda returns / generalized advantage estimation
        for t in reversed(range(len(self.rewards))):
            r = self.rewards[t]
            vt = self.vals[t]
            vtp1 = self.vals[t+1]
            terminal = self.terminals[t]
            delta = r + self.gamma * vtp1 * (1-terminal) - vt
            gae[t] = delta + self.gamma * self.lam * (1-terminal) * prev_gae
            lamret[t] = gae[t] + vt
            prev_gae = gae[t]

        gae.detach_()
        lamret.detach_()

        dataset = PPODataset(states, actions, blogps, gae, lamret)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        ploss = []
        eloss = []
        vloss = []
        for epoch in range(self.epochs):
            for i, batch in enumerate(loader):
                state = batch['state']
                action = batch['action']
                blogp = batch['blogp']
                gae = batch['gae']
                ret = batch['lamret']

                self.opt.zero_grad()
                vl, pl, ent = self.ppoloss(state, action, blogp, gae, ret)
                ploss.append(pl.item())
                vloss.append(vl.item())
                eloss.append(ent.item())
                loss = vl - (pl + self.ent_coef*ent) # policy loss needs to be negative to do a maximization
                loss.backward()
                self.opt.step()

        return np.array([np.mean(vloss), np.mean(ploss)])

    def ppoloss(self, state, action, blogp, gae, ret):
        # Loss function for policy and value function
        dist = self.policy(state)
        logp = dist.log_prob(action.squeeze()).view(-1, 1)
        v = self.vf.predict(state)

        vloss = torch.pow(ret - v, 2).mean()

        ratio = torch.exp(logp-blogp)
        ploss1 = ratio*gae
        ploss2 = torch.clamp(ratio, 1.-self.clip, 1.+self.clip) * gae
        ploss = torch.min(ploss1, ploss2).mean()
        if self.ent_coef > 0.:
            ent = dist.entropy().mean()
        else:
            ent = torch.zeros(1)

        return vloss, ploss, ent



    def new_episode(self):
        pass


class PPODataset(Dataset):
    def __init__(self, states, actions, blogp, gae, lamret):
        self.states = states
        self.actions = actions
        self.blogp = blogp
        self.gae = gae
        self.lamret = lamret

    def __len__(self):
        return self.lamret.shape[0]

    def __getitem__(self, index):
        return {'state':self.states[index],
                'action': self.actions[index],
                'blogp': self.blogp[index],
                'gae': self.gae[index],
                'lamret': self.lamret[index]
                }

import numpy as np
from rllib.utils.utils import discount
import cloudpickle
import os

class Trajectory(object):
    def __init__(self, obs_dim, act_dim, max_len):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.iteration = 0

        self.obs = np.zeros((max_len + 1, obs_dim), dtype=np.float32)
        self.acts = np.zeros((max_len, act_dim), dtype=np.float32)
        self.rews = np.zeros((max_len, 1), dtype=np.float32)
        self.logps = np.zeros((max_len, 1), dtype=np.float32)
        self.total_reward = 0
        self.ret = None
        self.done = False

    def add(self, obs, act, rew, logp, next_obs, done):
        iteration = self.iteration
        self.obs[iteration, :] = obs
        self.obs[iteration + 1, :] = next_obs
        self.acts[iteration, :] = act
        self.rews[iteration, 0] = rew
        self.logps[iteration, 0] = logp

        self.iteration += 1

        if done:
            self._finish()
            self.done = True

    def _finish(self):
        nobs = np.copy(self.obs[:self.iteration + 1, :])
        nacts = np.copy(self.acts[:self.iteration, :])
        nrews = np.copy(self.rews[:self.iteration, :])
        nlogps = np.copy(self.logps[:self.iteration, :])

        del self.obs
        del self.acts
        del self.rews
        del self.logps

        self.obs = nobs
        self.acts = nacts
        self.rews = nrews
        self.logps = nlogps
        self.total_reward = np.sum(nrews)



class TrajectoryMemory(object):
    def __init__(self, obs_dim, act_dim, max_len, max_trajs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_len = max_len
        self.max_trajs = max_trajs

        self.trajs = {}
        self.num_traj = 0
        self.finished = []
        self.total_steps = 0

    def save_memory(self, dir, fname):
        file_path = os.path.join(dir, fname)
        with open(file_path, "wb") as f:
            cloudpickle.dump(self, f, protocol=-1)

    def new(self):
        new_traj = Trajectory(self.obs_dim, self.act_dim, self.max_len)
        self.num_traj += 1
        self.trajs[self.num_traj] = new_traj

        if self.num_traj > self.max_trajs:
            to_remove = self.num_traj - self.max_trajs
            self.total_steps -= self.trajs[to_remove].iteration
            self.trajs.pop(to_remove)
            self.finished.remove(to_remove)

        return self.num_traj

    def add(self, num, obs, act, rew, logp, next_obs, done):
        self.total_steps += 1
        traj = self.trajs[num]
        if traj.iteration + 1 >= self.max_len:
            done = True

        traj.add(obs, act, rew, logp, next_obs, done)

        if traj.done:
            self.finished.append(num)
            return True
        return False

    def sample_one(self):
        if len(self.finished) <= 0:
            return None
        idx = np.random.choice(self.finished, size=1, replace=False)[0]
        traj = self.trajs[idx]
        rets = dict(obs=traj.obs, acts=traj.acts, rews=traj.rews)
        return idx, rets

    def sample_trajs(self, num):
        num = min(num, len(self.finished))
        if num <= 0:
            return None

        idx = np.random.choice(self.finished, size=num, replace=False)
        rets = []
        for i in idx:
            traj = self.trajs[i]
            rets.append(dict(obs=traj.obs, acts=traj.acts, rews=traj.rews, logps=traj.logps, length=traj.iteration))

        return idx, rets

    def get_last_trajs(self, num, include_active=False):
        if include_active:
            idx = np.sort(list(self.trajs.keys()))[-num:]
        else:
            idx = np.sort(self.finished)[-num:]

        rets = []
        for i in idx:
            traj = self.trajs[i]
            itrs = traj.iteration
            if traj.done or itrs > 0:
                rets.append(dict(obs=traj.obs[:itrs+1], acts=traj.acts[:itrs], rews=traj.rews[:itrs], logps=traj.logps[:itrs], length=itrs, done=traj.done))


        return idx, rets

    def sample_nsteps(self, num, gamma, nstep=5, include_active=True):
        num = min(num, self.total_steps)

        if include_active:
            idx = np.random.choice(list(self.trajs.keys()), size=num, replace=True)
        else:
            idx = np.random.choice(self.finished, size=num, replace=True)

        s1 = []
        a1 = []
        r = []
        s2 = []
        logp = []
        terminals = []
        for i in idx:
            traj = self.trajs[i]
            itrs = traj.iteration
            if traj.done or itrs > 0:
                t = np.random.choice(range(0, itrs), size=1, replace=False)[0]
                done = int((t == itrs))
                steps = max(min(nstep, itrs-t+1),1)
                s1.append(traj.obs[t])
                a1.append(traj.acts[t])
                r.append(discount(traj.rews[t:(t+steps)], gamma)[0])
                s2.append(traj.obs[t+steps-1])
                logp.append(np.sum(traj.logps[t:(t+steps)]))
                terminals.append(done)

        return dict(s1=np.stack(s1, axis=0), a1=np.stack(a1,axis=0), r=np.stack(r,axis=0), s2=np.stack(s2, axis=0), logp=np.stack(logp, axis=0), terminal=np.stack(terminals,axis=0).reshape(-1, 1))

    def get_rew_ret(self, gamma):
        rews = []
        rets = []
        for tnum in self.finished:
            traj = self.trajs[tnum]
            rews.append(traj.total_reward)
            if not traj.ret:
                traj.ret = discount(traj.rews, gamma)[0]
            rets.append(traj.ret)

        return rews, rets


def load_memory(dir, fname):
    file_path = os.path.join(dir, fname)
    try:
        with open(file_path, "rb") as f:
            return cloudpickle.load(f)
    except EOFError:
        pass


def collate_trajs(trajs, pad_val=np.nan):
    lens = [traj['length'] for traj in trajs]

    pads = [(0, 0) for _ in trajs[0]['obs'].shape[1:]]
    pads = [[(0, max(lens) - l)] + pads for l in lens]

    obs = [traj['obs'] for traj in trajs]
    acts = [traj['acts'] for traj in trajs]
    rews = [traj['rews'] for traj in trajs]
    logps = [traj['logps'] for traj in trajs]

    obs = np.stack([(np.pad(b, p, 'constant')) for b, p in zip(obs, pads)], pad_val)
    acts = np.stack([(np.pad(b, p, 'constant')) for b, p in zip(acts, pads)], pad_val)
    rews = np.stack([(np.pad(b, p, 'constant')) for b, p in zip(rews, pads)], pad_val)
    logps = np.stack([(np.pad(b, p, 'constant')) for b, p in zip(logps, pads)], pad_val)

    return dict(obs=obs, acts=acts, rews=rews, logps=logps, lengths=lens)

def collate_sequence(seqs, lens, pad_val=np.nan):
    pads = [(0, 0) for _ in seqs[0].shape[1:]]
    pads = [[(0, max(lens) - l)] + pads for l in lens]

    seqs = np.stack([(np.pad(s, p, 'constant')) for s, p in zip(seqs, pads)], pad_val)
    return seqs

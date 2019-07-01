
import numpy as np

from gym import spaces
import copy

class Gridworldpy(object):
    def __init__(self, size=5):
        self.size = int(size)
        self.x = int(0)
        self.y = int(0)
        self.count = 0
        nums = self.size **2
        self.nums = nums
        self.numa = 4
        self.observation_space = spaces.Box(low=np.zeros(nums), high=np.ones(nums), dtype=np.float32)
        self.action_space = spaces.Discrete(self.numa)
        self._P = None
        self._R = None

    def reset(self):
        self.x = 0
        self.y = 0
        self.count = 0
        return self.get_state()

    def step(self, action):
        a = int(action)
        if a == 0:
            self.y -= 1
        elif a ==1:
            self.y += 1
        elif a ==2:
            self.x -= 1
        elif a==3:
            self.x += 1
        else:
            raise Exception("Action out of range! Must be in [0,3]: " + a)
        self.x = int(np.clip(self.x, 0, self.size-1))
        self.y = int(np.clip(self.y, 0, self.size-1))
        self.count += 1
        reward = -1.0

        return self.get_state(), reward, self.is_terminal(), None

    def get_state(self):
        x = np.zeros(self.nums, dtype=np.float32)
        x[self.x*self.size + self.y] = 1
        return x

    def is_terminal(self):
        return (self.x == self.size-1 and self.y == self.size-1) or (self.count > 500)

    @property
    def P(self):
        if not (self._P is not None):
            self._P = np.zeros((self.nums, self.numa, self.nums))
            for x in range(self.size):
                for y in range(self.size):
                    s1 = x*self.size + y
                    for a in range(self.numa):
                        x2 = x + 0
                        y2 = y + 0
                        if a == 0:
                            y2 = y - 1
                        elif a == 1:
                            y2 = y + 1
                        elif a == 2:
                            x2 = x - 1
                        elif a == 3:
                            x2 = x + 1
                        x2 = int(np.clip(x2, 0, self.size - 1))
                        y2 = int(np.clip(y2, 0, self.size - 1))
                        s2 = x2*self.size + y2
                        self._P[s1, a, s2] = 1.0
            self._P[-1, :, :] = 0
            self._P[-1, :, -1] = 1
        return self._P

    @property
    def R(self):
        if not (self._R is not None):
            self._R = np.ones((self.nums, self.numa)) * -1.
            self._R[-1, :] = 0.
        return self._R

def policy_evaluationv(env, policy, gamma):
    Psa = env.P
    R = env.R
    pi = np.zeros((env.nums, env.numa))
    Ps = np.zeros((env.nums, env.nums))
    for s in range(env.nums):
        x = np.zeros(env.nums)
        x[s] = 1
        if np.ndim(x) ==1:
            x = x[np.newaxis, :]
        if policy.basis:
            x = policy.basis.basify(x)
        pi[s, :] = policy.np_get_p(x)
        Ps[s, :] += np.sum([pi[s, a] * Psa[s,a] for a in range(env.numa)], axis=0)
    b = np.sum(pi*R, axis=1)
    # Ps2 = np.array([np.sum(pi[s].reshape(-1, 1) * Psa[s], axis=0) for s in range(env.nums)])
    # assert Ps == Ps2, "Ps did not match Ps2"
    I = np.eye(env.nums)
    if gamma == 1:
        gamma = 1.-1e-8
    v = np.linalg.solve(I-gamma*Ps,b)

    return v

def policy_evaluationq(env, policy, gamma):
    Psas = env.P
    pi = np.zeros((env.nums, env.numa))
    v = policy_evaluationv(env, policy, gamma)
    q = np.zeros((env.nums, env.numa))
    for s in range(env.nums):
        for a in range(env.numa):
            q[s,a] = np.sum(Psas[s,a, :] * v)
    #     x = np.zeros(env.nums)
    #     x[s] = 1
    #     if np.ndim(x) ==1:
    #         x = x[np.newaxis, :]
    #     if policy.basis:
    #         x = policy.basis.basify(x)
    #     pi[s, :] = policy.np_get_p(x)
    #
    # for s in range(env.nums):
    #     for a in range(env.numa):
    #         for s2 in range(env.nums):
    #             Psa[s*env.numa+a, s2:(s2+env.numa)] += np.array([Psas[s,a,s2] * pi[s2, a2] for a2 in range(env.numa)])
    # b = (pi*R).reshape(-1)
    #
    # I = np.eye(env.nums*env.numa)
    # if gamma == 1:
    #     gamma = 1.-1e-6
    # q = np.linalg.solve(I-gamma*Psa,b)

    return q


def policy_iteration(mdp, gamma=1, iters=5, plot=True):
    '''
    Performs policy iteration on an mdp and returns the value function and policy
    :param mdp: mdp class (GridWorld_MDP)
    :param gam: discount parameter should be in (0, 1]
    :param iters: number of iterations to run policy iteration for
    :return: two numpy arrays of length |S|. U, pi where U is the value function and pi is the policy
    '''
    pi = np.zeros(mdp.num_states, dtype=np.int)
    b = R = np.array([mdp.R(s) for s in range(mdp.num_states)])
    Ptensor = build_p_tensor(mdp)
    I = np.eye(mdp.num_states)
    U = np.zeros(mdp.num_states)
    Ustart = []
    for i in range(iters):
        # policy evaluation - solve AU = b with (A^TA)^{-1}A^Tb
        P = build_pi_p_matrix(mdp,pi)
        Ainv = np.linalg.pinv(I - gamma*P)
        U = np.dot(Ainv,b)
        print(U)
        Ustart += [U[mdp.loc2state[mdp.start]]]
        # policy improvement
        for s in range(mdp.num_states):
            MEU = np.dot(Ptensor[s], U)
            pi[s] = np.argmax(MEU)

    if plot:
        fig = plt.figure()
        plt.title("Policy Iteration with $\gamma={0}$".format(gamma))
        plt.xlabel("Iteration (k)")
        plt.ylabel("Utility of Start")
        plt.ylim(-2, 1)
        plt.plot(range(1,len(Ustart)+1),Ustart)

        pp = PdfPages('./plots/piplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    return U, pi


def build_p_tensor(mdp):
    '''
    Returns an s x a x s' tensor
    '''
    P = np.zeros((mdp.num_states, mdp.num_actions, mdp.num_states))
    for s in range(mdp.num_states):
        for a in range(mdp.num_actions):
            for s2, p in mdp.P_snexts(s, a).items():
                if not mdp.is_absorbing(s2):
                    P[s, a, s2] = p
    return P


def build_pi_p_matrix(mdp,pi):
    '''
    Returns an s x s' matrix
    '''
    P = np.zeros((mdp.num_states, mdp.num_states))
    for s in range(mdp.num_states):
        a = pi[s]
        for s2, p in mdp.P_snexts(s, a).items():
            if not mdp.is_absorbing(s2):
                P[s, s2] = p
    return P


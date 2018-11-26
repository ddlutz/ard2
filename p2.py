"""
majority taken from: http://aima.cs.berkeley.edu/python/mdp.html
"""

from utils import *
import numpy as np
import random as rand
from qlearner import *

rand.seed(0)
np.random.seed(0)

import time

current_time_ms = lambda: int(round(time.time() * 1000))

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""
    def __init__(self, grid, terminals, init=(0, 0), gamma=.999, actlist=orientations):
        grid.reverse() ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)
    
    def go_rew(self, state, direction):
        "Return the state that results from going in this direction and its reward"
        state1 = vector_add(state, direction)
        state1 = if_(state1 in self.states, state1, state)
        return (state1, self.R(state1))

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


def value_iteration(mdp, epsilon=0.001, gammaNew=None):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    if gammaNew is not None:
        gamma = gammaNew
    iters = 0
    while True:
        iters+=1
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
             return (U, iters)

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_iteration(mdp, k):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    iters = 0
    while True:
        iters+=1
        U = policy_evaluation(pi, U, mdp, k)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return (pi, iters)

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
    return U


def print_grid(grid):
    for row in grid:
        print row

# insipired by, but different than the FrozenLake-v0 on openai.

frozenLake = GridMDP([[-0.01, -0.01, -0.01, -0.01, +1],
                     [-0.01, None,  -0.01, -0.01, 0],
                     [-0.01, -0.01, -0.01, -0.01, -0.01],
                     [None, -0.01, None, -0.01, -0.01],
                     [-0.01, -0.01, -0.01, -0.01, -0.01]],
                    terminals=[(4, 4), (4, 3)])

beforeVI = current_time_ms()
U, viIters = value_iteration(frozenLake)
VIMillis = current_time_ms() - beforeVI
print 'Value Utility:'
print U
print "Value iteration num iterations: %s" % (viIters)
viPolicy = best_policy(frozenLake, U)
print 'Value iteration policy:'
print_grid(frozenLake.to_arrows(viPolicy))
print_grid(frozenLake.to_grid(U))
print 'Value Iteration milliseconds: %s' % (VIMillis)

def ValueIterationTest(mdp, gamma):
    totalTime = 0.0
    totalIters = 0.0

    
    beforeVI = current_time_ms()
    U, viIters = value_iteration(mdp, gammaNew=gamma)
    VIMillis = current_time_ms() - beforeVI

    totalTime += VIMillis
    totalIters += viIters

    print '\n'
    print 'Value Iteration for Gamma = %s' % (gamma)
    print 'Avg num iters: %s' % (totalIters)
    print 'Avg num ms: %s' % (totalTime)
    print 'Value iteration policy:'
    viPolicy = best_policy(mdp, U)
    print_grid(mdp.to_arrows(viPolicy))



def PolicyIterationTest(mdp, k, iters = 20):
    totalTime = 0.0
    totalIters = 0.0
    piPolicy = None
    
    for i in range(iters):
        beforePI = current_time_ms()
        piPolicy, piIters = policy_iteration(mdp, k)
        PIMillis = current_time_ms() - beforePI
        totalIters += piIters
        totalTime += PIMillis

    print 'Performing policy iteration with k=%s' % (k)
    print 'Policy iteration policy:'
    print print_grid(mdp.to_arrows(piPolicy))
    print 'Policy iteration avg num iters: %s' % (totalIters / iters)
    print 'Policy iteration avg milliseconds: %s' % (totalTime / iters)
    print '\n'


def isBestPolicy(mdp, policy):
    #print policy
    state = (0,0)
    pi_to_dir = {'^': (0,+1), 'v': (0,-1), '<': (-1,0), '>': (+1,0)}
    for i in range(100): #put upper limit to make sure we're not stuck in a cycle
        # find action to take from policy
        action = policy[state[0], state[1]]
        # apply
        # get rewards

        #print 'state: ' +  str(state)
        #print 'action: ' +  str(action)
        newpos, rew = mdp.go_rew(state, action)
        #print 'new pos: ' + str(newpos)
        #print 'rew: %s' % (rew)
        if rew == 0:
            return False
        if rew == 1:
            return True
        state = newpos

    return False

def state_to_gs(state):
    return (state/5, state%5)

def gs_to_state(pos):
    return pos[0]*5 + pos[1]

def q_to_pi(Q, emptyVal):
    #convert q function from q-learner to a policy
    actions = {0 : (-1,0), 1: (1,0), 2: (0, +1), 3: (0, -1)}
    pi = {}
    for i in range(len(Q)):
        gs = state_to_gs(i)
        a = np.argmax(Q[i])
        #print (i, a)
        direction = actions[a]
        if (Q[i][a] == emptyVal):
            direction = None
        pi[gs]=direction

    return pi


def qlearn(mdp, rar, radr, alpha, gamma, init):
    q = QLearner(num_states=25, num_actions=4, rar=rar, radr=radr, alpha=alpha, gamma=gamma, init=init)

    startpos = 0
    actions = {0 : (-1,0), 1: (1,0), 2: (0, +1), 3: (0, -1)}

    prevQ = np.copy(q.getQ())

    max_iters = 1000

    for iteration in range(1,max_iters):
        state = startpos
        action = q.querysetstate(state)
        while (state not in [24,23]):
            direction = actions[action]
            gs = state_to_gs(state)

            newpos, rew = mdp.go_rew(gs, direction)
            #print (newpos, rew)
            state = gs_to_state(newpos)

            action = q.query(state, rew)
    
        currentQ = np.copy(q.getQ())
        diff = abs((currentQ-prevQ).sum())
        if (diff < 0.001 and iteration > 10): #set min # of iterations to prevent local minimum in beginning
            return q, iteration
        prevQ = currentQ
    
    return q, max_iters


def run_qlearn_test(mdp, name, rar, radr, alpha, gamma, init):
    bestPolicyCount = 0
    actualIters = 0
    
    totalTime = 0

    for i in range(100):
        before = current_time_ms()
        Q, qIters = qlearn(mdp, rar=rar, radr=radr, alpha=alpha, gamma=gamma, init=init)
        delta = current_time_ms() - before
        totalTime += delta
        actualIters += qIters
        Qpi = q_to_pi(Q.getQ(), init)
        if isBestPolicy(mdp, Qpi):
            bestPolicyCount+=1

    print 'Name: %s' % (name)
    print 'Best policy found %s times' % (bestPolicyCount)
    print 'Average number of iterations: %s' % (actualIters / 100.0)
    print 'Average number of milliseconds: %s' % (totalTime / 100.0)
    print '\n'


Q, qIters = qlearn(frozenLake, 0.7, 0.99, 0.1, 0.6, 1)
print Q.getQ()
Qpi = q_to_pi(Q.getQ(), 1.)
isBestPolicy(frozenLake, Qpi)
print 'QLearning policy:'
print_grid(frozenLake.to_arrows(Qpi))
print 'QLearning num iters: %s' % (qIters)

ValueIterationTest(frozenLake, 0.999)
ValueIterationTest(frozenLake, 0.99)
ValueIterationTest(frozenLake, 0.8)
ValueIterationTest(frozenLake, 0.5)
ValueIterationTest(frozenLake, 0.2)

PolicyIterationTest(frozenLake, 1)
PolicyIterationTest(frozenLake, 5)
PolicyIterationTest(frozenLake, 10)
PolicyIterationTest(frozenLake, 15)
PolicyIterationTest(frozenLake, 20)
PolicyIterationTest(frozenLake, 50)
PolicyIterationTest(frozenLake, 100)

run_qlearn_test(frozenLake, "First", 0.7, 0.99, 0.1, 0.6, 1)
run_qlearn_test(frozenLake, "Second", 0.7, 0.99, 0.2, 0.6, 1)
run_qlearn_test(frozenLake, "Third", 0.7, 0.99, 0.7, 0.6, 1)
run_qlearn_test(frozenLake, "Fourth", 0.1, 1.0, 0.1, 0.6, 1)
run_qlearn_test(frozenLake, "Fifth", 0.7, 0.99, 0.1, 0.6, 0)
run_qlearn_test(frozenLake, "Sixth", 0.7, 0.99, 0.1, 0.9, 1)
run_qlearn_test(frozenLake, "Best", 0.5, 0.99, 0.1, 0.9, 0)
run_qlearn_test(frozenLake, "Best2", 0.5, 0.99, 0.2, 0.9, 0)
import numpy as np
import random as rand
class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        init = 0, \
        verbose = False):

        """
        num_states integer, the number of states to consider
        num_actions integer, the number of actions available.
        alpha float, the learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
        gamma float, the discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
        rar float, random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
        radr float, random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        dyna integer, conduct this number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
        verbose boolean, if True, your class is allowed to print debugging statements, if False, all printing is prohibited.
        """
        
        self.verbose = verbose
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.s = 0
        self.a = 0
        self.Q = np.ones([num_states, num_actions]) * init
        self.num_actions = num_actions

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        return np.argmax(self.Q[self.s])

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The (reward, i'm guessing)
        @returns: The selected action

        is the core method of the Q-Learner. It should keep track of the last state s and the last action a, 
        then use the new information s_prime and r to update the Q table. 
        The learning instance, or experience tuple is <s, a, s_prime, r>. query() should return an integer, 
        which is the next action to take. 
        Note that it should choose a random action with probability rar, 
        and that it should update rar according to the decay rate radr at each step.
        """

        # 1. Get a'
        a_prime = None
        random_num = np.random.rand(1)[0]
        if rand.uniform(0,1) < self.rar:
            a_prime = np.random.randint(0, self.num_actions)
        else:
            a_prime = np.argmax(self.Q[s_prime])
        
        if self.verbose:
            print "s =", s_prime,"a =",action,"r =",r

        #if r == 0:
        #    print 'Found negative reward. Updating s=%s a=%s from %s' % (self.s, self.a, self.Q[self.s, self.a])
        #if r == +1:
        #    print 'Found positive reward. Updating s=%s a=%s from %s' % (self.s, self.a, self.Q[self.s, self.a])
        # 2. Update via core q-learning algorithm
        self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + (self.alpha * (r + self.gamma * self.Q[s_prime, a_prime]))
        #if r == 1 or r == 0:
        #    print 'New value: %s' % (self.Q[self.s, self.a])

        # 3. Keep track of what is now the previous state and action.
        self.s = s_prime
        self.a = a_prime
        
        # 4. Update rar.
        self.rar = self.rar * self.radr

        return self.a

    def getQ(self):
           return self.Q

import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0, epsilon_decay=False):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration
        epsilon_decay -- perform decaying epsilon algorithm

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        rewards -- accumulated rewards of all steps for each episod
        '''
		
        Q = initialQ
        next_state = 0
        rewards = []
        # Keep track of visiting each state
        n_counts = np.zeros([self.mdp.nActions,self.mdp.nStates])
        
        for episod in range(nEpisodes):

            current_state = s0
            #sum the rewards of all steps for each episod that the agent gets from the environment
            reward_episod = 0            

            for step in range(nSteps):
                
                action = 0
                r = 0
                
                    
                # we sample a float from a uniform distribution over 0 and 1
        		# if the sampled flaot is less than the exploration proba
        		#     the agent selects arandom action
        		# else
        		#     the agent choose action based on the Boltzmann equation probabilities
                exploration_prob = np.random.rand(1)[0]	
                if epsilon_decay:
                    epsilon *= 1/(1+ step)
                if exploration_prob < epsilon:
                	# select random action 
                    action = np.random.randint(self.mdp.nActions)
                    # get reward and next state of the action
                    r,  next_state = self.sampleRewardAndNextState(current_state, action)
                
                else: 
                    if temperature == 0:
                        action = Q[:, current_state].argmax()
                    else:
                    	# sum quality of state-actions from current state 
                        Q_all_actions = Q[:, current_state]/(temperature + 0.5)
                        # probability of selecting each action based on Boltzmann equation
                        probabilities = np.exp(Q_all_actions) / np.sum(np.exp(Q_all_actions))
                        # select Q given the Boltzmann probabilities
                        selected_Q = np.random.choice(Q_all_actions, 1, p=probabilities)[0]
                        # select action based on the taken Q
                        action = np.where(Q_all_actions==selected_Q)[0][0]
                    # get reward and next state of the action
                    r,  next_state  = self.sampleRewardAndNextState(current_state, action)

                n_counts[action, current_state] += 1
                lr = 1/(n_counts[action, current_state])
                
                # Updating Q-Values Q_new = (1-lr)*Q_old + lr*(R + gamma * max(Q_next))
                Q[action, current_state] = (1-lr) * Q[action, current_state] \
                                            + lr * (r + (self.mdp.discount * Q[:, next_state].max()))

                current_state = next_state
                # adding step reward to episode reward
                reward_episod += (self.mdp.discount ** step) * r


			# keep track of episode rewards
            rewards.append(reward_episod)
		
		# Extract policy based on updated Q
        policy = Q.argmax(axis=0)
        return [Q, policy, rewards]    

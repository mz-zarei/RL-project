import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # Initialize variables
        V = initialV
        iterId = 0
        epsilon = np.inf
        V_max = 0

        # start value iteration algorithm loop
        while (iterId < nIterations and epsilon >= tolerance):
            # Compute the value function
            V_max = (self.R + ((self.discount * self.T) @ V).squeeze()).max(axis=0)
            # Compute new epsilon
            epsilon = np.abs(V - V_max).max()
            # Update V
            V = V_max
            # Update iteration number
            iterId += 1
        
        return [V_max,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # Compute policy (which action in each state maximize R^a + gamma T^a V)
        policy = (self.R + ((self.discount * self.T) @ V).squeeze()).argmax(axis=0)

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi
        V^pi = (I- Gamma*T)^(-1) * R

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # Compute transition matrix for the given policy
        T_pi = self.T[policy].diagonal().T
        # Compute reward matrix for the given policy
        R_pi = self.R[policy].diagonal()
        # Compute inverse: (I- Gamma*T)^(-1)
        inverse = np.linalg.inv(np.identity(self.nStates) - (self.discount * T_pi))
        # Compute V
        V = np.dot(inverse, R_pi)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        
        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0

        while (iterId < nIterations):
            # Step 1: Policy evaluation
            V = self.evaluatePolicy(policy)
            # Step 2: Policy improvement
            new_policy = self.extractPolicy(V)
            # Check if old policy is different with new policy
            if (policy == new_policy).all():
                break
            else:
                policy = new_policy
            # Update iteration number
            iterId += 1



        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = np.inf

        while (iterId < nIterations and epsilon >= tolerance):
            # Compute transition matrix for the given policy
            T_pi = self.T[policy].diagonal().T
            # Compute reward matrix for the given policy
            R_pi = self.R[policy].diagonal()
            # Compute the value function
            V = R_pi + (self.discount * T_pi) @ initialV
            # Compute new epsilon
            epsilon = np.abs(V - initialV).max()
            # Update V
            initialV = V
            # Update iteration number
            iterId += 1

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy
        V = initialV
        iterId = 0
        epsilon = np.inf

        while (iterId < nIterations) and (epsilon > tolerance):
            # Step 1: Policy evaluation partially
            new_V = self.evaluatePolicyPartially(policy,V,nIterations=nEvalIterations,tolerance=-np.inf)[0]
            # Step 2: Policy improvement
            new_policy = self.extractPolicy(new_V)
            # Compute new epsilon
            epsilon = np.abs(new_V - V).max()
            # Update V
            V = (self.R +  (self.discount * np.dot(self.T, new_V).squeeze())).max(axis=0)
            # Update policy
            policy = new_policy
            # Update iteration number
            iterId += 1

        return [policy,V,iterId,epsilon]
        
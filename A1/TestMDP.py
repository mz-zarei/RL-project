from MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)


'''Test each procedure'''
print("\n======== Value Iteration ========")
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print("V: {}, nIterations: {}, epsilon: {}".format(V, nIterations, epsilon))

print("\n======== Policy Extraction ========")
policy = mdp.extractPolicy(V)
print(f"policy: {policy}")

print("\n======== Policy Evaluation ========")
p1 = np.array([1,0,1,0])
p2 = np.array([0,1,1,1])
V1 = mdp.evaluatePolicy(p1)
V2 = mdp.evaluatePolicy(p2)
print(f"policy {p1} is evaluated to have value {V1}")
print(f"policy {p2} is evaluated to have value {V2}")

print("\n======== Policy Iteration ========")
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("V: {}, nIterations: {}, policy: {}".format(V, iterId, policy))

print("\n======== Partially Policy Evaluation ========")
p1 = np.array([1,0,1,0])
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(p1,np.array([0,10,0,13]))
print(f"policy {p1} is partially evaluated to have value {V}. nIterations: {iterId}, epsilon: {epsilon}")

p2 = np.array([0,1,1,1])
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(p2,np.array([0,0,0,0]))
print(f"policy {p2} is partially evaluated to have value {V}. nIterations: {iterId}, epsilon: {epsilon}")

print("\n======== Modified Policy Iteration ========")
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("optimal Value is: {}\n\
optimal Policy is: {} nIterations: {} epsilon: {}".format(V, policy, iterId, tolerance))


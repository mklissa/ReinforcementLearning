#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the value iteration algorithm, which can be seen an iterative
case of the Bellman optimality equation.
"""

import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])


        
    while True:
        delta=0
        for state in range(env.nS):
            v = V[state]
            actionValue =  np.zeros(env.nA)
            


            for action in range(env.nA):
                for prob, next_state, reward, _ in env.P[state][action]:
                        actionValue[action] += prob * (reward + discount_factor * V[next_state])
              
            V[state] = np.max(actionValue)
        
            # Here we update the policy at the same time as we update the state-value function.
            best_action = np.argmax(actionValue)
            policy[state] = np.eye(env.nA)[best_action]


            delta = max(delta, np.abs(v - V[state]))
        if delta < theta:
            return policy, V

    # Implement!
    
    
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
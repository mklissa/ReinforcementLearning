#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this exercise I implement the policy evalution algorithm. At the end, we should have
a good estimate to the policy's state-value function.
Complixity: O(mn^3) per iteration,
where m is the number of actions and n is the number of states.
"""

import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv


env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
   
    while True:
        delta=0
        # TODO: Implement!
        for state in range(env.nS):
            v=0
            for action,action_prob in enumerate(policy[state]):
                for  prob, next_state, reward, done in env.P[state][action]:   
                    v += action_prob*prob*(reward + discount_factor*V[next_state])
                    # Formula according to p.63 from Sutton's book Reinforcement
                    # Learning: An Introduction (2nd edition)
                
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
                
        if delta < theta:
            break
    return np.array(V)
    

    
    
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)



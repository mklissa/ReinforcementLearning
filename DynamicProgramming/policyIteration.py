#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this exercise I implement the policy iteration algorithm. We alternate between
policy evaluation and policy improvement to obtain utlimately an optimal policy V*.
Complixity: O(mn^3) per iteration,
where m is the number of actions and n is the number of states.
"""

import numpy as np
import pprint
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
    

    
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        
        V = policy_eval_fn(policy, env, discount_factor)
        
        
        # policy_stable will define the end of the algorithm
        policy_stable = True
        
        
        for state in  range(env.nS):
            #The action we would take under the current policy
            probs = policy[state]
            policy_action = np.random.choice(np.arange(len(probs)), p=probs)
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += prob * (reward + discount_factor * V[next_state])
            best_action = np.argmax(action_values)
            
            if policy_action != best_action:
                policy_stable = False
                
            policy[state] = np.eye(env.nA)[best_action]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V
            
    
policy, v = policy_improvement(env)
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
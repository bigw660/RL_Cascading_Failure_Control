from actor import ActorNetwork
from ps_env import PowerSystem
import numpy as np
import tensorflow as tf

sess = tf.Session()



state_dim = 11

action_dim = 2
action_bound = np.array([[1.5, 0.375], [1.8, 0.45]])
ACTOR_LEARNING_RATE = 0.001
TAU = 0.001

actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)
# List = np.zeros((10, 2))
# for i in range(10):
#     sess.run(tf.global_variables_initializer())
#     env = PowerSystem()
#     s = env.reset()
#     s = np.reshape(s, (1, 11))
#     action = actor.predict(s)
#
#     List[i] = (action[0])
#     print(action[0])
#
# print(List)

action_bound = np.array([[0.375, 1.5], [0.45, 1.8]])
# action_bound = np.array([[1.5, 0.375], [1.8, 0.45]])

# s1 = [0.36554113, 0.53080961, 0.41921139, 0.12469007, 0.35247505, 0.22216045, 0.2319377,
#       0.29089014, 0.37623618, 0.09325221, 0.]
# s1 = np.reshape(s1, (1, 11))
# sess.run(tf.global_variables_initializer())
# action_1 = actor.predict_test(s1)
#
#
# print(action_bound[:, 0])
# print(action_bound[:, 1])
# print(action_bound[:, 1] - action_bound[:, 0])
# print((action_bound[:, 1] - action_bound[:, 0]) / 2)
# print((action_bound[:, 1] - action_bound[:, 0]) / 2 + action_bound[:, 0])
# print(action_1)
# print(action_1[0])
# print((action_1[0] + 1) * (action_bound[:, 1] - action_bound[:, 0]) / 2 + action_bound[:, 0], '\n')


def action_scale_out(action_bound, out):
    # Creating a mapping from [a, b] to [c, d]
    # y = (x-a) * (d-c) / (b-a) + c
    # The action will be the output for each generator
    for i in range(len(out[0])):
        out[:, i] = (out[:, i] - (-1)) * (action_bound[i][1] - action_bound[i][0]) / (1 - (-1)) + action_bound[i][0]
        # out[:, 1] = (out[:, 1] - (-1)) * (action_bound[1][1] - action_bound[1][0]) / (1 - (-1)) + action_bound[1][0]
    return out


List = np.zeros((5, 2))

s1 = [0.36554113, 0.53080961, 0.41921139, 0.12469007, 0.35247505, 0.22216045, 0.2319377,
      0.29089014, 0.37623618, 0.09325221, 0.]
s1 = np.reshape(s1, (1, 11))
sess.run(tf.global_variables_initializer())
action_1 = actor.predict_test(s1)
List[0] = action_1[0]

s2 = [0.12986894, 0.2457366, 0.23717131, 0.16874459, 0.62851912, 0.27170114, 0.08628019,
      0.43185089, 0.50797986, 0.093647, 0.]
s2 = np.reshape(s2, (1, 11))
# sess.run(tf.global_variables_initializer())
action_2 = actor.predict_test(s2)
List[1] = action_2[0]

s3 = [0.07838938, 0.26361114, 0.23260744, 0.36119444, 0.57980241, 0.23952, 0.1172384,
      0.527728, 0.67116893, 0.108227, 0.]
s3 = np.reshape(s3, (1, 11))
# sess.run(tf.global_variables_initializer())
action_3 = actor.predict_test(s3)
List[2] = action_3[0]

s4 = [0.12308454, 0.24415589, 0.22902893, 0.32386429, 0.6120362, 0.24990979, 0.08521434,
      0.51756533, 0.64078632, 0.10500136, 0.]
s4 = np.reshape(s4, (1, 11))
# sess.run(tf.global_variables_initializer())
action_4 = actor.predict_test(s4)
List[3] = action_4[0]

s5 = [0.11171232, 0.35887438, 0.28063688, 0.23465881, 0.48252349, 0.22824538, 0.0446117,
      0.42360043, 0.56680426, 0.09871637, 0.]
s5 = np.reshape(s5, (1, 11))
# sess.run(tf.global_variables_initializer())
action_5 = actor.predict_test(s5)
List[4] = action_5[0]

print('List:\n', List, '\n')
print('List[:, 0]:\n', List[:, 0], '\n')
# print('List[:, 1]:\n', List[:, 1], '\n')
print('List[;, 0] + 1 :\n', List[:, 0] + 1, '\n')
print('(action_bound[0][1] - action_bound[0][0]) / (1 - (-1)) + action_bound[0][0]:\n',
      (action_bound[0][1] - action_bound[0][0]) / (1 - (-1)) + action_bound[0][0])
print('Scale out for first column: \n', (List[:, 0] + 1) *
      (action_bound[0][1] - action_bound[0][0]) / (1 - (-1)) + action_bound[0][0])
# print('List[;, 0] + 1 :\n', List[:, 0] + 1, '\n')
# print(List[:, 1], '\n')
# print(List[:, 0], '\n')
# print(List[:, 1] - List[:, 0], '\n')
# print(2 + List[:, 0], '\n')
# print((List[:, 1] - List[:, 0]) / 2 + List[:, 0], '\n')
# print(List + 1)
# print((List + 1) * ((List[:, 1] - List[:, 0]) / 2 + List[:, 0]), '\n')
print('Scale out:\n', action_scale_out(action_bound, List), '\n')






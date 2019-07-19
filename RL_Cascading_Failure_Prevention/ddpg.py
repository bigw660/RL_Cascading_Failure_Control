# ===========================
#   DDPG Model
# ===========================
import tensorflow as tf
import numpy as np
from ps_env import PowerSystem
from replay_buffer import ReplayBuffer
from noise import Noise
from actor import ActorNetwork
from critic import CriticNetwork
import matplotlib.pyplot as plt
import time


# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 100000  # 5000
# Max episode length
MAX_EP_STEPS = 20  # 20
# Episodes with noise
NOISE_MAX_EP = 2500
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5  # The rate of change (time)
SIGMA = 0.5  # Volatility of the stochastic processes
OU_A = 3.  # The rate of mean reversion
OU_MU = 0.  # The long run average interest rate
# Reward parameters
# REWARD_FACTOR = 0.1  # Total episode reward factor
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.0001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# Size of replay buffer
BUFFER_SIZE = 20480
MINIBATCH_SIZE = 64
# Random seed
RANDOM_SEED = 23
# path for saving the model
model_path = r"C:\Users\Mariana Kamel\Documents\PyCharm\mariana\RL_" \
             r"Cascading_Failure_Prevention\saved_models\model_1.ckpt"

# ===========================
#   Agent Training
# ===========================


def action_scale_out(action_bound, out):
    # Creating a mapping from [a, b] to [c, d]
    # y = (x-a) * (d-c) / (b-a) + c
    # The action will be the output for each generator
    for i in range(len(out[0])):
        out[:, i] = (out[:, i] - (-1)) * (action_bound[i][1] - action_bound[i][0]) / (1 - (-1)) + action_bound[i][0]
    return out


def train(sess, env, actor, critic, noise, action_bound):

    # plotting
    plot_step = []
    plot_ep_reward = []
    plot_loss = []

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Record total steps
    total_step = 0

    for i in range(MAX_EPISODES):
        print('########################################')
        print('########################################')
        print('########################################')
        print('Episode: ', i+1)
        random = np.random.randint(1, 5001)
        s = env.reset_offline(random)

        # Initialize episode reward
        ep_reward = 0

        # Initialize step counter
        step_counter = 0

        # Initialize critic loss
        critic_loss = 0

        for j in range(MAX_EP_STEPS):
            total_step += 1
            # Obtain the action from actor network
            # a_without_scale = actor.predict(np.reshape(s, (1, actor.s_dim)))
            # a = action_scale_out(action_bound, a_without_scale)
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))

            # Add exploration noise
            if i < NOISE_MAX_EP:
                noise = 0.9995 * np.random.normal(0, 0.1)
                a = a + noise

            # Set action for continuous action spaces
            action = a[0]
            # print("The actions is: ", action)

            # Obtain next state, reward, and terminal from the environment
            s2, r, terminal, early_stop = env.step(action)

            # Adding s, a, r, terminal, s2 into buffer
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal,
                              np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples
            if total_step > BUFFER_SIZE+1:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                critic_loss, _, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            # Update state, step counter, critic loss, episode reward
            step_r_penalty = 0.3 * (j+1)
            r = r - step_r_penalty  # set penalty for each step (try to use less steps)
            s = s2
            step_counter += 1
            critic_loss += critic_loss
            ep_reward += r

            print("******** Step Reward & Critic Loss ********")
            print('Step Reward:', r, '\t', 'Critic Loss:', critic_loss, '\n')

            if early_stop:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Failed ! Generator exploded! @@@@@@@@@@@@@@@'
                      '@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                break

            if j == MAX_EP_STEPS-1 and not terminal:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Failed! Run out ot Steps! @@@@@@@@@@@@@@@@@@@@'
                      '@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                break

            if terminal:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@;@@@@@@@@@@@@@ Succeeded! Used ', j+1,
                      " steps to make all lines safe! @@@@@@@@@@@@")
                break
        plot_step.append(step_counter)
        plot_ep_reward.append(ep_reward)
        plot_loss.append(critic_loss)

    # plotting the graphs
    plt.figure('Episode Step')
    plt.plot(range(1, MAX_EPISODES + 1), plot_step, 'b-')
    plt.figure('Episode Reward')
    plt.plot(range(1, MAX_EPISODES + 1), plot_ep_reward, 'b-')
    plt.figure('Episode Critic Loss')
    plt.plot(range(1, MAX_EPISODES + 1), plot_loss, 'b-')
    plt.show()


def test(env, actor):
    testing_size = 5000
    # plotting
    plot_step = []
    plot_ep_reward = []

    # Record total steps
    total_step = 0

    for i in range(testing_size):
        print('########################################')
        print('########################################')
        print('########################################')
        print('Testing Case: ', i+1)
        s = env.reset_offline_test(i+1)

        # Initialize episode reward
        ep_reward = 0

        # Initialize step counter
        step_counter = 0

        for j in range(MAX_EP_STEPS):
            total_step += 1
            # Obtain the action from actor network
            # a_without_scale = actor.predict(np.reshape(s, (1, actor.s_dim)))
            # a = action_scale_out(action_bound, a_without_scale)
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))

            # Set action for continuous action spaces
            action = a[0]
            # print("The actions is: ", action)

            # Obtain next state, reward, and terminal from the environment
            s2, r, terminal, early_stop = env.step(action)

            # Update state, step counter, critic loss, episode reward
            step_r_penalty = 0.3 * (j+1)
            r = r - step_r_penalty  # set penalty for each step (try to use less steps)
            s = s2
            step_counter += 1
            ep_reward += r

            print("******************************* Step Reward *******************************")
            print('Step Reward:', r,  '\n')

            if early_stop:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Failed ! Generator exploded! @@@@@@@@@@@@@@@'
                      '@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                break

            if j == MAX_EP_STEPS-1 and not terminal:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Failed! Run out ot Steps! @@@@@@@@@@@@@@@@@@@@'
                      '@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                break

            if terminal:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@;@@@@@@@@@@@@@ Succeeded! Used ', j+1,
                      " steps to make all lines safe@@@@@@@@@@@@!")
                break
        plot_step.append(step_counter)
        plot_ep_reward.append(ep_reward)

    # plotting the graphs

    plt.figure('Episode Step')
    plt.xlim((1, testing_size + 1))
    plt.ylim((1, 11))
    plt.plot(range(1, testing_size + 1), plot_step, 'b-')

    plt.figure('Episode Reward')
    plt.xlim((1, testing_size + 1))
    plt.ylim((-11, 11))
    plt.plot(range(1, testing_size + 1), plot_ep_reward, 'r-')
    plt.show()


def main(_):
    # Training the model
    with tf.Session() as sess:

        env = PowerSystem()
        # System Info
        state_dim = 11  # We only consider the Current of all line as state at this moment
        action_dim = 2  # The number of generators
        action_bound = np.array([[-1, 1], [-0.675, 0.675]])

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU,
                               actor.get_num_trainable_vars())

        saver = tf.train.Saver()

        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)

        # Training the model
        train(sess, env, actor, critic, noise, action_bound)

        # # save the variables
        save_path = saver.save(sess, model_path)
        # print("[+] Model saved in file: %s" % save_path)

    # # Testing the model
    # with tf.Session() as sess:
    #
    #     env = PowerSystem()
    #     # System Info
    #     state_dim = 11  # We only consider the Current of all line as state at this moment
    #     action_dim = 2  # The number of generators
    #     action_bound = np.array([[-1, 1], [-0.675, 0.675]])
    #
    #     actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)
    #     saver = tf.train.Saver()
    #     load_path = saver.restore(sess, model_path)
    #     test(env, actor)


if __name__ == '__main__':
    t1 = time.time()
    tf.app.run()
    print('Running time: ', time.time() - t1)
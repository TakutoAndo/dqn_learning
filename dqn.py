import os
import gym
import sys

from keras.backend_config import epsilon
from numpy.core.fromnumeric import resize
import random
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


#変数
ENV_NAME = 'Breakout-v0'
NUM_EPISODES = 12000    #episode数
NO_OP_STEPS = 30    #何もしない最大ステップ数
INITIAL_EPSILON = 1.0 #ε-greedy法のε初期値
FINAL_EPSILON = 0.1 #εの終値
EXPLORATION_STEPS = 1000000 #εの減少フレーム数
STATE_LENGTH = 4    # 状態を構成するフレーム数
FRAME_WIDTH = 84    # リサイズ後のフレーム幅数
FRAME_HEIGHT = 84   #　リサイズ後のフレーム高さ
LEARNING_RATE = 0.00025     #RMSPropで使われる学習率
MOMENTUM = 0.95     #RMSPropで使われるモメンタム
MIN_GRAD = 0.01     #RMSPropで使われる0で割るのを防ぐための値
ACTION_INTERVAL = 4     #フレームスキップ数  Atariでは1秒間に60回画面が更新 全部の更新を見るのは効率が悪いため4フレームに1回画面をみて行動選択
NUM_REPLAY_MEMORY = 400000  #Replay Memory数
INITIAL_REPLAY_SIZE = 20000 #学習前に事前に確保するReplay Memory数
TRAIN_INTERVAL = 4  #学習を行う間隔
TARGET_UPDATE_INTERVAL = 10000  #Target Networkを更新する間隔
BATCH_SIZE = 32     #バッチサイズ
GAMMA = 0.99    #割引率
TRAIN = True #学習させるかテストさせるか
LOAD_NETWORK = False
SAVE_INTERVAL = 300000  #ネットワークを保存する頻度
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST  = 30  #テスト時にプレイするエピソード数

KERAS_BACKEND = 'tensorflow'

#Agentクラス アルゴリズムが書かれてるクラス
class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.time = 0
        self.repeated_action = 0 #フレームスキップ時にリピートする行動を保持するための変数

        #Replay Memoryをデクで実装・初期化
        self.replay_memory = deque()


        """データ解析で使う値"""
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        #Q-Networkの構築
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        #Target-Networkの構築
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        #定期的なTarget-Networkの更新処理
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        #誤差関数、最適化のための処理の構築
        self.action, self.training_data, self.loss, self.grad_update = self.error_clip(q_network_weights)

        #FIXME: Sessionの構築　理解すること
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        #ネットワークの保存場所がなければ作成
        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        #Q-Networkの初期化
        self.sess.run(tf.initialize_all_variables())

        #Networkのロード
        if LOAD_NETWORK:
            print("Network is loading ...")

        #TargetNetworkの初期化
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])
        q_values = model(s)

        return s, q_values, model

    #エラー(target - Q(s, a; θ))を-1~1でクリップする
    def error_clip(self, q_network_wights):
        action = tf.placeholder(tf.int64, [None])
        training_data = tf.placeholder(tf.float32, [None])

        action_onehot = tf.one_hot(action, self.num_actions, 0.0, 1.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, action_onehot), reduction_indices=1)

        error = tf.abs(training_data - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        liner_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + liner_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=q_network_wights)

        return action, training_data, loss, grad_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT), mode="constant") * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)

    def get_action(self, state):
        #FIXME フレームスキップは有効？？
        action = self.repeated_action

        #ACTION_INTERVAL間隔で行動選択を行う（それ以外では行動をリピート）
        if self.time % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.time < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
            else:
                print(f"q_values.eval in get_action(): {self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]})}")
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        #εを線形に減少させる
        if self.epsilon > FINAL_EPSILON and self.time >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        training_data_batch = []

        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        terminal_batch = np.array(terminal_batch) + 0
        #target_networkで次の状態のQ値を計算
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        training_data_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        #誤差最小化
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={self.s: np.float32(np.array(state_batch) / 255.0), self.a: action_batch, self.training_data: training_data_batch})

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        action = self.repeated_action

        if self.time % ACTION_INTERVAL == 0:
            if random.random() <= 0.05:
                action = self.random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.time += 1

        return action

    def run(self, state, action, reward, terminal, observation):
        #次の状態を作成
        next_state = np.append(state[:, :, 1:], observation, axis=2)

        #報酬を1, -1でクリッピング
        reward = np.sign(reward)

        #Replay Memoryへの保存
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.time > INITIAL_REPLAY_SIZE:
            if self.time % TRAIN_INTERVAL:
                self.train_network()

        if self.time % TARGET_UPDATE_INTERVAL == 0:
            self.sess.run(self.update_target_network)

        #ネットワークを保存
        if self.time % SAVE_INTERVAL == 0:
            save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.time))
            print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            if self.time >= INITIAL_REPLAY_SIZE:
                states = [self.total_reward, self.total_q_max / float(self.duration), self.duration, self.total_loss / float(self.duration) / float(TRAIN_INTERVAL) ]
                for i in range(len(states)):
                    self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(states[i])})
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.time < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.time < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.time, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.time += 1

        return next_state


#FIXME: リサイズの次元の検討
def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT), mode="constant") * 255)
    return np.reshape(processed_observation, (FRAME_WIDTH, FRAME_HEIGHT, 1))


#大枠
def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    if TRAIN: # Train mode
        for _ in range(NUM_EPISODES):
            terminal = False #エピソード終了判定
            observation = env.reset() #初期画面を返す
            for _ in range(random.randint(1, NO_OP_STEPS)): #1~30フレーム分なにもしない
                last_observation = observation
                #stepが返す値は4つ、　observation(object):環境の情報をもつオブジェクト, reward(float):　前のアクションで得られた報酬, done(bool): エピソードが終了したかどうか, info(dict): デバック用情報
                observation, _, _, _ =env.step(0) #何もしない行動をして次画面を返す
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state) #行動選択
                observation, reward, terminal, _ = env.step(action)
                env.render() #画面出力
                processed_observation = preprocess(observation, last_observation) #画面の前処理
                state = agent.run(state, action, reward, terminal, processed_observation) #学習を行い次の状態を返す
    else:   #Test mode
        env.monitor.start(ENV_NAME + '-test')
        for _ in range(NUM_EPISODES_AT_TEST):
            terminal =False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_acction_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render() #画面出力
                processed_observation = preprocess(observation, last_observation) #画面の前処理
                state = np.append(state[1:, :, :], processed_observation, axis=0)


if __name__ == '__main__':
    main()




import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque

#変数
ENV_NAME = 'Breakout-v0'
NUM_EPISODES = 12000    #episode数
NO_OP_STEPS = 30    #何もしない最大ステップ数
INITIAL_EPSILON = 1.0 #ε-greedy法のε初期値
FINAL_EPSILON = 0.1 #εの終値
EXPLORATION_STEPS = 1000000 #εの減少フレーム数
STATE_LENGTH = 4    # 状態を構成するフレーム数
FRAME_WIDTH = 84    # リサイズ後のフレーム幅
FRAME_HEIGHT = 84   #　リサイズ後のフレーム高さ
LEARNING_RATE = 0.00025     #RMSPropで使われる学習率
MOMENTUM = 0.95     #RMSPropで使われるモメンタム
MIN_GRAD = 0.01     #RMSPropで使われる0で割るのを防ぐための値

#Agentクラス アルゴリズムが書かれてるクラス

class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS #εの減少率
        self.time_step = 0
        self.repeated_action = 0 #フレームスキップ間にリピートする行動を保持するための変数

        #Replay Memoryの初期化
        self.replay_memory = deque()

        #Q Networkの構築
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        #Target Networkの構築
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        #定期的なTargetNetworkの更新処理
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in xrange(len(target_network_weights))]

        #誤差関数、最適化のための処理の構築
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        #Sessionの構築
        self.sess = tf.InteractiveSession()

        #変数の初期化(Q Networkの初期化)
        self.sess.run(tf.initialize_all_variables())

        #Target Networkの初期化
        self.sess.run(self.update_target_network)

    #Q Network と Target Network の構築関数
    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        #TODO: ここ２行の理解
        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    #最適化の処理(エラークリップ)
    def build_training_op(self, q_network_wights):
        a = tf.placeholder(tf.int64, [None])    #行動
        y = tf.placeholder(tf.float32, [None])  #教師信号 r + γQ(s', a':θ)

        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)   #行動をone hot vectorに変換 num_actions次元のone hot vectorに変換
        q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)  #行動のQ値の計算 self.q_values * a_one_hot して 要素全て足し算(q_value以外全部0なのでq_valueが取り出される)

        #エラークリップ TODO:最適化アルゴリズムとかよくわかってない
        error = tf.abs(y - q_value)     #|y-q_value|?
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part) #誤差関数

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)   #最適化アルゴリズムの定義
        grad_update = optimizer.minimize(loss, var_list=q_network_wights)   #誤差最小化

        return a, y, loss, grad_update



#大枠

env = gym.make(ENV_NAME)
agent = Agent(num_actions=env.action_space.n)

for _ in xrange(NUM_EPISODES):
    terminal = False #エピソード終了判定
    observation = env.reset() #初期画面を返す
    for _ in xrange(random.randint(1, NO_OP_STEPS)): #1~30フレーム分なにもしない
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



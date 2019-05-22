import sys
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda, Conv2D, Concatenate
from keras.regularizers import l2
from keras.utils import plot_model
from rl_utils.networks import conv_block

class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, tau, k, dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.dueling = dueling
        # Initialize Deep Q-Network
        self.model = self.network(k, dueling)
        self.model.compile(Adam(lr), 'mse')
        # Build target Q-Network
        self.target_model = self.network(k, dueling)
        self.target_model.compile(Adam(lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.summary()
        plot_model(self.target_model, to_file='../models/rl_policy.png', show_shapes=True)

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self, k, dueling):
        """ Build Deep Q-Network
        """
        inp = Input((self.state_dim))
        # inp_z = Input((k, latent_dim))
        inp_loc = Input((2,))
        # If we have an image, apply convolutional layers
        if(len(self.state_dim) > 2):
            # Images
            # x = Reshape((self.state_dim[1], self.state_dim[2], -1))(inp)
            # x = conv_block(x, 32, (3, 3))
            # x = conv_block(x, 64, (3, 3))
            # x = conv_block(x, 128, (3, 3))

            x = Conv2D(8, (3,3), strides=(2,2))(inp)
            x = Conv2D(16, (3,3), strides=(2,2))(x)
            x = Conv2D(32, (3,3), strides=(2,2))(x)
            x = Conv2D(64, (3,3), strides=(2,2))(x)

            # mobilenet = MobileNetV2(include_top=False)
            # mobilenet.trainable = True
            # x = mobilenet(x)

            # generator = build_generator()
            # generator.load_weights('models/generator.064.h5', by_name=True)
            # generator.summary()
            # generator.trainable = False
            # x = generator([inp, inp_z])

            # x = TimeDistributed(AveragePooling2D())(x)
            # x = TimeDistributed(AveragePooling2D())(x)
            # x = TimeDistributed(AveragePooling2D())(x)

            # x = TimeDistributed(Flatten())(x)
            x = Flatten()(x)
            x = Dense(256)(x)

            # y = RepeatVector(128)(inp_loc)
            # y = Flatten()(y)
            # y = RepeatVector(512)(inp_loc)
            # y = Reshape((k,-1))(y)
            # x = Lambda(lambda z: K.expand_dims(z, axis=1), output_shape=(K.int_shape(x)[1], 1))(x)

            y = Dense(256)(inp_loc)
            
            # print(K.int_shape(y))
            # print(K.int_shape(x))
            # y = Lambda(lambda z: K.expand_dims(z, axis=1), output_shape=(1, K.int_shape(y)[1]))(y)

            x = Concatenate(axis=1)([x, y])

            # # print(K.int_shape(x))

            # x = Lambda(lambda z: K.expand_dims(z, axis=1), output_shape=(1, K.int_shape(x)[1]))(x)
            # x = LSTM(512)(x)
        else:
            x = Flatten()(inp)
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)

        if(dueling):
            # Have the network estimate the Advantage function as an intermediate layer
            x = Dense(self.action_dim + 1, activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_dim,))(x)
        else:
            x = Dense(self.action_dim, activation='linear')(x)
        return Model([inp, inp_loc], x)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
            # print(inp.shape)

        # len()
        # inp = np.reshape(inp, (inp.shape[1], inp.shape[2], -1))
        # print(inp[0].shape)
        # print(inp[0].shape)

        # print(inp[1].shape)
        return self.model.predict(self.reshape(inp))

    def target_predict(self, inp):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x[0].shape) < 4:
            new_x0 = np.expand_dims(x[0], axis=0)
        else:
            new_x0 = x[0]
        # try:
        if len(np.array(x[1]).shape) < 2:
            new_x1 =  np.expand_dims(x[1], axis=0)
        else:
            # print(x[1].shape)
            new_x1 = x[1]

        # if (type(x[1]) is float):
        #     new_x1 = np.expand_dims(x[1], axis=0)
        # else:
        #     new_x1 = np.expand_dims(x[1], axis=1)

        # except:
            # new_x1 = np.expand_dims(x[1], axis=0)
        return [new_x0, new_x1]

    def save(self, path):
        if(self.dueling):
            path += '_dueling'
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

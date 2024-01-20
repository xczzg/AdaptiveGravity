# _*_ coding:utf-8 _*_
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tensorflow.python.ops.numpy_ops import np_config
from keras.callbacks import Callback
np_config.enable_numpy_behavior()


class GenerateRelation(layers.Layer):
    def __init__(self, num_nodes, exponent):
        super(GenerateRelation, self).__init__()

        self.a1 = pd.read_csv(r'\Chicago\Dis_TAZ.csv')[['Dis_TAZ']].values.reshape((num_nodes,num_nodes))+1e-2
        self.a = tf.Variable(self.a1,trainable=True,dtype=tf.float32)
        self.a1 = self.a1 ** exponent
        self.exponent = exponent

    def call(self,inputs):
        x = inputs
        x1 = tf.matmul(x, x,transpose_b=True)
        attention = (x1 / (self.a ** self.exponent))
        X = tf.expand_dims(tf.reduce_sum(attention,axis=-1), axis=-1)
        gravity = x1 / self.a1
        attentionre = attention

        return X, [gravity, attentionre]

"""STHGCN"""
class AdaptiveG(layers.Layer):
    def __init__(self, hidden_units,mean, std, num_nodes, exponent):
        super(AdaptiveG, self).__init__()
        self.mean = mean
        self.std = std
        self.relationlayerlist = []

        for i in range(1):
            self.relationlayerlist.append(
                GenerateRelation(num_nodes,exponent)
            )

    def att(self,inputs):
        x = inputs  # (batch_size,num_node, num_time,2)
        nodes_embedding,attention = self.relationlayerlist[0](x)

        return nodes_embedding* self.std + self.mean, attention

    def call(self, inputs):
        out,att = self.att(inputs)
        return out

def mae_loss(label,pred):
    loss = tf.abs(tf.subtract(pred, label))
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss


aaaaa = []
class PrintPredictionsCallback(Callback):
    def __init__(self, input_data, save_path, scale = 5):
        super(PrintPredictionsCallback, self).__init__()
        self.x1 = input_data
        self.scale = scale
        self.pearlastone = 1
        self.pear_slope_lastone = 0
        self.pear_slope = []
        self.pear_slope_slope = []
        self.save_path = save_path
        self.acc = []

    def on_epoch_end(self, epoch, logs=None):
        pre, a = self.model.layers[1].att(self.x1)
        gravity, attentionre = a
        gravity = tf.reduce_sum(gravity,axis=0)
        attentionre = tf.reduce_sum(attentionre, axis=0)
        gravity = gravity[~np.eye(gravity.shape[0], dtype=bool).reshape(gravity.shape[0], -1)]
        attentionre = attentionre[~np.eye(attentionre.shape[0], dtype=bool).reshape(attentionre.shape[0], -1)]
        pc1 = pearsonr(attentionre.reshape((-1, 1))[:,0], gravity.reshape((-1, 1))[:,0])
        self.pear_slope.append(abs(pc1[0]-self.pearlastone)) #求pearson斜率绝对值
        self.pearlastone = pc1[0] #保存上一个epoch的Pearson 指数
        determin = 0

        val = pd.read_csv(r'\Chicago\odall.csv')[['volume']].values
        val = val.reshape((-1, 1))
        val = np.nan_to_num(val)
        pcga = pearsonr(gravity.reshape((-1, 1))[:,0], val[:,0])
        pc = pearsonr(attentionre.reshape((-1, 1))[:,0], val[:,0])
        aaaaa.append([pcga[0], pc[0], pc1[0]])
        pd.DataFrame(np.array(aaaaa)).to_csv(r'\Chicago\Figure2.csv')

        if (epoch+1) % self.scale ==0:
            #self.model.save_weights(self.save_path.format(epoch+1)) #save model
            val = pd.read_csv(r'\Chicago\odall.csv')[['volume']].values
            val = val.reshape((-1, 1))
            val = np.nan_to_num(val)
            pc = pearsonr(attentionre.reshape((-1, 1))[:, 0], val[:, 0])
            self.acc.append(pc)

            #stopping mechanism
            mean_slope = np.mean(self.pear_slope)
            self.pear_slope_slope.append(mean_slope - self.pear_slope_lastone)
            self.pear_slope_lastone = mean_slope
            self.pear_slope = []
            if len(self.pear_slope_slope) > 3:
                if determin == 0:
                    if self.pear_slope_slope[-4] < 0:
                        if self.pear_slope_slope[-3] < 0:
                            determin+=1
                            if self.pear_slope_slope[-2] > 0:
                                if self.pear_slope_slope[-1] > 0:
                                    self.model.stop_training = True
                                    print('best performance in',epoch+1-2 * self.scale, 'epoch')
                                    print('stopping pearson value with actual flow is', self.acc[-4])
                if determin > 0:
                    if self.pear_slope_slope[-3] < 0:
                        if self.pear_slope_slope[-2] > 0:
                            if self.pear_slope_slope[-1] > 0:
                                self.model.stop_training = True
                                print('best performance in', epoch+1 - 2 * self.scale, 'epoch')
                                print('stopping pearson value with actual flow is', self.acc[-4])



def main(num_nodes, mean, std,exponent,
         x_train, y_train, x_validation, y_validation, x_test, hidden_units):
    input_node_information = keras.layers.Input(shape=(num_nodes,1),
                                                dtype=tf.float32)

    ourmodel = AdaptiveG(hidden_units,mean, std, num_nodes,exponent)

    out = ourmodel(input_node_information)
    model = keras.Model(inputs=input_node_information,
                        outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(0.1), #LR change to 1 when it is conducted in LA
                  loss=mae_loss)

    x1 = np.concatenate([x_train,x_validation,x_test], axis=0)
    save_path = r'\epochs_60_{}.tf' #path for trained model
    printalist = PrintPredictionsCallback(x1,save_path)
    history = model.fit(x=x_train, y=y_train, callbacks=[printalist], batch_size=batch_size, epochs=epochs,
                        validation_data=(x_validation, y_validation), verbose=2)

    """load weight on the best epoch to get generated flows"""
    # model.load_weights(r'\Chicago\train_model\epochs_15_35.tf')
    # history = 0
    # pre2 = model.predict(x=x_test)
    # x1 = np.concatenate([x_train,x_validation,x_test], axis=0)
    # pre,a= ourmodel.att(x1)
    # a1sum = np.expand_dims(np.sum(np.expand_dims(np.sum(a[0],axis=-1) - np.max(a[0],axis=-1),axis=-1),axis=-2),axis=-1)
    # a1 = np.nan_to_num(a[0]/a1sum) * np.expand_dims(np.sum(x1,axis=1),axis=-1)
    # a2sum = np.expand_dims(np.sum(np.expand_dims(np.sum(a[1],axis=-1) - np.max(a[1],axis=-1),axis=-1),axis=-2),axis=-1)
    # a2 = np.nan_to_num(a[1]/a2sum) * np.expand_dims(np.sum(x1,axis=1),axis=-1)
    # np.save(r'\relation_g_15_flow.npy',a1)
    # np.save(r'\relation_our_15_flow.npy', a2)

    return history

def zscore(x, mean, std):
    y = (x - mean) / std
    return y

def rezscore(y, mean, std):
    x = y * std + mean
    return x

if __name__ == '__main__':

    mean = 0
    std = 1
    batch_size = 10
    epochs = 1000
    hidden_units = 64
    exponent = 1.5 # it will be 2.55 when it is conducted in LA


    x_train = np.expand_dims(np.load(r'\train_x_Chicago_60.npy')[:,:,0],axis=-1)
    y_train = np.expand_dims(np.load(r'\train_y_Chicago_60.npy')[:,:,0],axis=-1)+0.0

    x_validation = np.expand_dims(np.load(r'\val_x_Chicago_60.npy')[:,:,0],axis=-1)
    y_validation = np.expand_dims(np.load(r'\val_y_Chicago_60.npy')[:,:,0],axis=-1)+0.0

    x_test = np.expand_dims(np.load(r'\test_x_Chicago_60.npy')[:,:,0],axis=-1)
    y_test = np.expand_dims(np.load(r'\test_y_Chicago_60.npy')[:,:,0],axis=-1)+0.0

    x_train = zscore(x_train,mean,std)
    x_validation = zscore(x_validation,mean,std)
    x_test = zscore(x_test,mean,std)
    num_nodes = y_train.shape[1]
    num_samples = y_train.shape[0]

    history = main(num_nodes, mean, std, exponent,
         x_train, y_train, x_validation, y_validation, x_test, hidden_units)

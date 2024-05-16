from scipy.io import loadmat
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(0)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0:],'GPU')
import tensorflow_probability as tfp


class rel_norm(tf.keras.losses.Loss):
    '''
    Compute the average relative l2 loss between a batch of true and predictions
    '''
    def __init__(self):
        super().__init__()
    def call(self, true, pred):
        '''
        true: (batch_size, L, d). 
        pred: (batch_size, L, d). 
        number of variables d=1
        '''
        rel_error  = tf.math.divide(tf.norm(tf.keras.layers.Reshape((-1,))(true-pred), axis=1), tf.norm(tf.keras.layers.Reshape((-1,))(true), axis=1))
        return tf.math.reduce_mean(rel_error)
        
def load_data(path, ntrain, ntest):
    
    R = np.transpose(np.load(path + "Random_UnitCell_rr_10.npy"), (1,0))[:,np.newaxis,:] #(2000,1,42)
    X = np.transpose(np.load(path + "Random_UnitCell_XY_10.npy"), (2,0,1)) #(2000,972,2)
    R = np.repeat(5*R-1, X.shape[1], 1) #(2000,972,42)
    X = np.concatenate((X,R), axis=-1) #(2000,972,46)
    Y       = np.transpose(np.load(path + "Random_UnitCell_sigma_10.npy"), (1,0))[...,np.newaxis]

    m_train = loadmat(path + "m_dist1.mat")["m_dist"][:ntrain,...]
    m_test  = loadmat(path + "m_dist2.mat")["m_dist"][-ntest:,...]
    
    m_test  = m_test / m_train.max()
    m_train = m_train / m_train.max()

    return m_train.astype("float32"), m_test.astype("float32"), X[:ntrain,...].astype("float32"), Y[:ntrain,...].astype("float32"), X[-ntest:,...].astype("float32"), Y[-ntest:,...].astype("float32")

class mlp(tf.keras.layers.Layer):
    '''
    A two-layer MLP with GELU activation.
    '''
    def __init__(self, n_filters1, n_filters2):
        super(mlp, self).__init__()

        self.width1 = n_filters1
        self.width2 = n_filters2
        self.mlp1 = tf.keras.layers.Dense(self.width1, activation='gelu', kernel_initializer="lecun_normal")
        self.mlp2 = tf.keras.layers.Dense(self.width2, kernel_initializer="lecun_normal")

    def call(self, inputs):
        x = self.mlp1(inputs)
        x = self.mlp2(x)
        return x
        
    def get_config(self):
        config = {
        'n_filters1': self.width1,
        'n_filters2': self.width2
        }
        return config

class MultiHeadPosAtt(tf.keras.layers.Layer):
    def __init__(self, n_head, hid_dim, locality):
        super(MultiHeadPosAtt, self).__init__()

        self.locality     = locality
        self.hid_dim      = hid_dim
        self.n_head       = n_head
        self.v_dim        = round(self.hid_dim/self.n_head)

    def build(self, input_shape):

        self.r = self.add_weight(
            shape=(1, self.n_head, 1, 1),
            trainable=True,
            name="dist",
        )

        self.weight = self.add_weight(
            shape=(self.n_head, self.hid_dim, self.v_dim),
            initializer="lecun_normal",
            trainable=True,
            name="weight",
        )
        self.built = True

    def call(self, m_dist, inputs):
        """
        m_dist: (batch, N, N)
        """
        scaled_dist = tf.expand_dims(m_dist, 1) * self.r**2 #(batch_size, n_heads ,L, L)
        if self.locality <= 100:
            mask         = tfp.stats.percentile(scaled_dist, self.locality, interpolation="linear", axis=-1, keepdims=True)
            scaled_dist  = tf.where(scaled_dist<=mask, scaled_dist, tf.float32.max)
        else:
            scaled_dist        = scaled_dist
        scaled_dist = -scaled_dist #(batch_size, n_heads ,L, L)
        att         = tf.nn.softmax(scaled_dist, axis=-1) #(batch_size, n_heads ,L, L)

        value      = tf.einsum("bnj,hjk->bhnk", inputs, self.weight)#(batch_size, n_head, L, v_dim)
        concat     = tf.einsum("bhnj,bhjd->bhnd", att, value) # (batch_size, n_head, L, v_dim)
        concat     = tf.transpose(concat, (0,2,1,3)) # (batch_size, L, n_head, v_dim)
        concat     = tf.keras.layers.Reshape((-1,self.hid_dim))(concat) # (batch_size, L, hid_dim)
        return tf.keras.activations.gelu(concat)

        
    def get_config(self):
        config = {
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        'locality': self.locality
        }
        return config

class PiT(tf.keras.Model):
    '''
    Position-induced Transfomer, built upon the multi-head position-attention mechanism.
    PiT can be trained to decompose and learn the global and local dependcencies of operators in partial differential equations.
    '''
    def __init__(self, out_dim, hid_dim, n_head, locality_encoder, locality_decoder):
        super(PiT, self).__init__()
        '''
        out_dim: number of variables
        hid_dim: encoding dimension (network width)
        n_head: number of heads in multi-head attention modules
        locality_encoder: quantile parameter of local position-attention in the Encoder, allowing to customize the size of receptive filed
        locality_decoder: quantile parameter of local position-attention in the Decoder, allowing to customize the size of receptive filed
        '''
        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.en_local = locality_encoder
        self.de_local = locality_decoder
        self.n_blocks = 4 # number of position-attention modules in the Processor
        
        # Encoder
        self.en_layer = tf.keras.layers.Dense(self.hid_dim, activation="gelu", kernel_initializer="lecun_normal")
        self.down     = MultiHeadPosAtt(self.n_head, self.hid_dim, locality=self.en_local)
        self.mlp1     = mlp(self.hid_dim, self.hid_dim)
        self.w1       = tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal")

        # Processor
        self.PA       = [MultiHeadPosAtt(self.n_head, self.hid_dim, locality=200) for i in range(self.n_blocks)]
        self.MLP      = [mlp(self.hid_dim, self.hid_dim) for i in range(self.n_blocks)]
        self.W        = [tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal") for i in range(self.n_blocks)]

        # Decoder
        self.up       = MultiHeadPosAtt(self.n_head, self.hid_dim, locality=self.de_local)
        self.mlp2     = mlp(self.hid_dim, self.hid_dim)
        self.w2       = tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal")
        self.de_layer = mlp(self.hid_dim, self.out_dim)
        
    def call(self, m_dist, inputs):

        # Encoder
        en    = self.en_layer(inputs)  # (batch_size, L_qry, hid_dim)
        x     = self.mlp1(self.down(m_dist, en)) + self.w1(en) # (batch_size, L_qry, hid_dim)
        x     = tf.keras.activations.gelu(x) # (batch_size, L_qry, hid_dim)

        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.PA[i](m_dist, x)) + self.W[i](x) # (batch_size, L_qry, hid_dim)
            x = tf.keras.activations.gelu(x) # (batch_size, L_qry, hid_dim)

        # Decoder
        de    = self.mlp2(self.up(m_dist, x)) + self.w2(x) # (batch_size, L_qry, hid_dim)
        de    = tf.keras.activations.gelu(de) # (batch_size, L_qry, hid_dim)
        de    = self.de_layer(de) # (batch_size, L_qry, out_dim)
        return de
    
    def get_config(self):
        config = {
        'out_dim': self.out_dim,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        'locality_encoder': self.en_local,
        'locality_decoder': self.de_local
        }
        return config

class MultiHeadSelfAtt(tf.keras.layers.Layer):
    '''
    Scaled dot-product multi-head self-attention
    '''
    def __init__(self, n_head, hid_dim):
        super(MultiHeadSelfAtt, self).__init__()

        self.hid_dim      = hid_dim
        self.n_head       = n_head
        self.v_dim        = round(self.hid_dim/self.n_head)

    def build(self, input_shape):

        self.q = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="lecun_normal",
            trainable=True,
            name="query",
        )    
        
        self.k = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="lecun_normal",
            trainable=True,
            name="key",
        )

        self.v = self.add_weight(
            shape=(self.n_head, input_shape[-1], self.v_dim),
            initializer="lecun_normal",
            trainable=True,
            name="value",
        )
        self.built = True

    def call(self, inputs):
        
        query       = tf.einsum("bnj,hjk->bhnk", inputs, self.q)#(batch, n_head, L, v_dim)
        key         = tf.einsum("bnj,hjk->bhnk", inputs, self.k)#(batch, n_head, L, v_dim)
        att         = tf.nn.softmax(tf.einsum("...ij,...kj->...ik", query, key)/self.v_dim**0.5, axis=-1)#(batch, n_heads, L, L)

        value      = tf.einsum("bnj,hjk->bhnk", inputs, self.v)#(batch, n_head, L, v_dim)
        
        concat     = tf.einsum("...nj,...jd->...nd", att, value)#(batch, n_head, L, v_dim)
        concat     = tf.transpose(concat, (0,2,1,3)) # (batch_size, L, n_head, v_dim)
        concat     = tf.keras.layers.Reshape((-1,self.hid_dim))(concat) # (batch_size, L, hid_dim)
        return tf.keras.activations.gelu(concat)
        
    def get_config(self):
        config = {
        'n_head': self.n_head,
        'hid_dim': self.hid_dim
        }
        return config  

class LiteTransformer(tf.keras.Model):
    '''
    Replace position-attention of the Processor in a PiT with self-attention
    '''
    def __init__(self, out_dim, hid_dim, n_head, en_local, de_local):
        super(LiteTransformer, self).__init__()

        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.en_local = en_local
        self.de_local = de_local
        self.n_blocks = 4

        # Encoder
        self.en_layer = tf.keras.layers.Dense(self.hid_dim, activation="gelu", kernel_initializer="lecun_normal")
        self.down     = MultiHeadPosAtt(self.n_head, self.hid_dim, self.en_local)
        self.mlp1     = mlp(self.hid_dim, self.hid_dim)
        self.w1       = tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal")
        
        # Processor
        self.PA       = [MultiHeadSelfAtt(self.n_head, self.hid_dim) for i in range(self.n_blocks)]
        self.MLP      = [mlp(self.hid_dim, self.hid_dim) for i in range(self.n_blocks)]
        self.W        = [tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal") for i in range(self.n_blocks)]

        # Decoder
        self.up       = MultiHeadPosAtt(self.n_head, self.hid_dim, self.de_local)
        self.mlp2     = mlp(self.hid_dim, self.hid_dim)
        self.w2       = tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal")
        self.de_layer = mlp(self.hid_dim, self.out_dim)
        
    def call(self, m_dist, inputs):

        # Encoder
        en    = self.en_layer(inputs)
        x     = self.mlp1(self.down(m_dist, en)) + self.w1(en)
        x     = tf.keras.activations.gelu(x)
        
        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.PA[i](x)) + self.W[i](x)
            x = tf.keras.activations.gelu(x)

        # Decoder
        de    = self.mlp2(self.up(m_dist, x)) + self.w2(x)
        de    = tf.keras.activations.gelu(de)
        de    = self.de_layer(de) 

        return de
    
    def get_config(self):
        config = {
        'out_dim': self.out_dim,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        'en_local':self.en_local,
        'de_local':self.de_local
        }
        return config

class Transformer(tf.keras.Model):
    '''
    Replace position-attention of a PiT with self-attention.
    '''
    def __init__(self, out_dim, hid_dim, n_head):
        super(Transformer, self).__init__()

        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.n_blocks = 4

        # Encoder
        self.en_layer = tf.keras.layers.Dense(self.hid_dim, activation="gelu", kernel_initializer="lecun_normal")
        self.down     = MultiHeadSelfAtt(self.n_head, self.hid_dim)
        self.mlp1     = mlp(self.hid_dim, self.hid_dim)
        self.w1       = tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal")
        
        # Processor
        self.PA       = [MultiHeadSelfAtt(self.n_head, self.hid_dim) for i in range(self.n_blocks)]
        self.MLP      = [mlp(self.hid_dim, self.hid_dim) for i in range(self.n_blocks)]
        self.W        = [tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal") for i in range(self.n_blocks)]

        # Decoder
        self.up       = MultiHeadSelfAtt(self.n_head, self.hid_dim)
        self.mlp2     = mlp(self.hid_dim, self.hid_dim)
        self.w2       = tf.keras.layers.Dense(self.hid_dim, kernel_initializer="lecun_normal")
        self.de_layer = mlp(self.hid_dim, self.out_dim)
        
    def call(self, inputs):

        # Encoder
        en    = self.en_layer(inputs)
        x     = self.mlp1(self.down(en)) + self.w1(en)
        x     = tf.keras.activations.gelu(x)

        # Processor
        for i in range(self.n_blocks):
            x = self.MLP[i](self.PA[i](x)) + self.W[i](x)
            x = tf.keras.activations.gelu(x)

        # Decoder
        de    = self.mlp2(self.up(x)) + self.w2(x)
        de    = tf.keras.activations.gelu(de)
        de    = self.de_layer(de) 

        return de
    
    def get_config(self):
        config = {
        'out_dim': self.out_dim,
        'hid_dim': self.hid_dim,
        'n_head': self.n_head,
        }
        return config


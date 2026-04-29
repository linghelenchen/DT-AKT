from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

def maxpooling_downsample_block(x, n_filters, block_name='block1'):
    # vgg style downsampling block
    # the block name shoulbe like "block1", 'block2' to 'block5'
    x_conv1 = layers.Conv2D(n_filters, 3, padding="same", activation='relu', name=block_name+'_conv1')(x) 
    x_conv2 = layers.Conv2D(n_filters, 3, padding="same", activation='relu', name=block_name+'_conv2')(x_conv1) 
    x = layers.MaxPooling2D(3, strides=2, padding="same", name=block_name+'_pool')(x_conv2)
    x = layers.BatchNormalization(name=block_name+'_batchnorm')(x)

    return x, x_conv1

def single_conv_block(x, n_filters, strides=1, block_name='block1'):
    
    x = layers.Conv2D(n_filters, 3, strides = strides, padding="same", activation='relu', name=block_name+'_conv1')(x)
    x = layers.BatchNormalization(name=block_name+'_batchnorm')(x)

    return x

def upsampling2d_conv_block(x, n_filters, size=2, block_name='Block1'):

    x  = layers.UpSampling2D(size = size, interpolation='bilinear', name=block_name+'upsampling2d')(x)
    x  = layers.Conv2D(n_filters, 3, padding="same", activation='relu', name=block_name+'_conv1')(x)
    x = layers.BatchNormalization(name=block_name+'_batchnorm')(x)

    return x

def upsampling2d_block(x, size=2, block_name='Block1'):

    x  = layers.UpSampling2D(size = size, interpolation='bilinear', name=block_name+'upsampling2d')(x)
    x = layers.BatchNormalization(name=block_name+'_batchnorm')(x)

    return x

def tranconv2d_block(x, n_filters, strides=2, block_name='Block1'):
    # solution 2
    # this upsamplong block is more stable than last one in prediction stage
    x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=strides, padding='same',                  
                                 activation='relu', name=block_name+'_convtranspose')(x)
    x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same',                 
                                 activation='relu', name=block_name+'_conv1')(x)      
    return x

def salient_points_block(x, n_filters, block_name='points', Ksize=3):
    # This block has been tested to learning score map whose most points close to 0
    x = layers.Conv2D(n_filters, kernel_size = 3, padding="same",
                         activation='relu', 
                         name=block_name+'_conv1')(x)     
    x = layers.Conv2D(1, kernel_size = 3, padding="same",
                         activation='relu', 
                         name=block_name+'_conv2')(x)   
    x = layers.BatchNormalization(name=block_name+'_batchnorm')(x)
    x = layers.Activation("sigmoid", name=block_name+'_sigmoid')(x)   
    # x = layers.Lambda(lambda x: soft_nmss(x, Ksize = Ksize, sigma=0.5), name=block_name+'_nmss')(x)   

    return x

def FCNs(img_size, load_weights=True, points_option='avg', name='FCNs'):  
    inputs = keras.Input(shape=img_size + (3,), name='rgb_image')

    x = inputs
    
    # scale 1
    x1, x1_conv1 = maxpooling_downsample_block(x, 64, block_name='block1')
    f_L0 = single_conv_block(x1_conv1, 16, block_name='feature_level0') 
    p_L0 = salient_points_block(f_L0, 1, block_name='pointsL0', Ksize=3)  
    
    # scale 2
    x1, x1_conv1 = maxpooling_downsample_block(x1, 128, block_name='block2')
    f_L1_vgg = upsampling2d_block(x1_conv1, size=2,  block_name='vggup_level1')
    f_L1 = single_conv_block(f_L1_vgg, 32, block_name='feature_level1')
    p_L1 = salient_points_block(f_L1, 2, block_name='pointsL1', Ksize=5)
    
    # scale 3
    x1, x1_conv1 = maxpooling_downsample_block(x1, 256, block_name='block3')    
    f_L2_vgg = upsampling2d_block(x1_conv1, size = 4,  block_name='vggup_level2')
    f_L2 = single_conv_block(f_L2_vgg, 64, block_name='feature_level2')
    p_L2 = salient_points_block(f_L2, 4, block_name='pointsL2', Ksize=7)

    x1_conv1 = single_conv_block(x1, 512, block_name='block4')  
    f_L3_vgg = upsampling2d_block(x1_conv1, size = 8, block_name='vggup_level3')
    f_L3 = single_conv_block(f_L3_vgg, 128,  block_name='feature_level3')
    p_L3 = salient_points_block(f_L3, 8, block_name='pointsL3', Ksize=9)

    features = layers.concatenate([f_L0, f_L1, f_L2, f_L3], name='dense_feature_con')        
    features = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='dense_feature')(features)

    if points_option == 'avg':        
        scores = layers.Average(name='score_map')([p_L0, p_L1, p_L2, p_L3]) 
    if points_option == 'max':      
        scores = layers.Maximum( name='score_map')([p_L0, p_L1, p_L2, p_L3]) 
    if points_option == 'concat':      
        scores = layers.concatenate([p_L0, p_L1, p_L2, p_L3], name='score_map')
    
    # scores = layers.Activation("sigmoid", name='score_map')(scores)     

    outputs = [scores, features]
    model = keras.Model(inputs, outputs, name=name)

    freezed_layers = ['block1_conv1', 'block1_conv2','block2_conv1', 'block2_conv2',
                        'block3_conv1', 'block3_conv2','block4_conv1']

    for layer_name in freezed_layers:
        layer = model.get_layer(layer_name) 
        layer.trainable = False
    
    return model

def soft_nms(batch, Ksize = 5, option='soft', alpha=1):
    if option == 'soft':
        maxpool_tensor = tf.nn.max_pool2d(batch, ksize=Ksize, strides=1, padding='SAME')
        mask = 1. / tf.math.exp(alpha * (maxpool_tensor - batch))
        suppressed_tensor = batch * mask
    if option == 'softmax':
        batch_exp = tf.math.exp(batch) 
        kernel = tf.constant(np.ones((Ksize, Ksize)), dtype=tf.float32)    
        kernel = kernel[:,:,tf.newaxis, tf.newaxis]    
        kernel = tf.tile(kernel, [1, 1, batch.shape[-1], 1])  
        batch_exp_conv = tf.nn.depthwise_conv2d(batch_exp, kernel, strides=[1, 1, 1, 1], padding='SAME')    
        mask = batch_exp / batch_exp_conv
        suppressed_tensor = batch * mask
    if option == 'hard':
        maxpool_tensor = tf.nn.max_pool2d(batch, ksize=Ksize, strides=1, padding='SAME')
        mask = tf.cast(tf.equal(batch, maxpool_tensor), dtype=tf.float32)
        suppressed_tensor = batch * mask
    return suppressed_tensor, mask

def soft_threshold(score_maps, threshold=0.5, option='hard'):
    if option=='hard':
        mask = tf.cast(score_maps > threshold, tf.float32)
    if option=='soft':
        mask = normalize_to_range(tf.maximum(score_maps - threshold, 0.))
    
    outputs = tf.multiply(score_maps, mask)
    return outputs, mask

def soft_nmss(score_maps, Ksize=3, sigma=0.5):
    # score_maps 4d[batch, heigh, width, 1]
    # a large ksize generate more sparse neighbors
    # a samll sigma generate more sparse results

    batch_size, height, width, channel = score_maps.shape

    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
                             np.exp(-((x - (Ksize - 1) / 2) ** 2 + (y - (Ksize - 1) / 2) ** 2) / (2 * sigma**2)), 
                             (Ksize, Ksize))
    # print('kernel\n',kernel)
    kernel /= kernel.sum()
    # print('kernel\n',kernel)

    kernel = tf.constant(kernel, dtype=tf.float32)    
    kernel = kernel[:,:,tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, channel, 1])

    score_maps_sp = tf.nn.depthwise_conv2d(score_maps, kernel, strides=[1, 1, 1, 1], padding='SAME')
    # print(score_maps_sp.shape)

    output_scores = normalize_to_range(score_maps_sp)
    
    return output_scores

def normalize_to_range(tensor):
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    
    # Using clip to avoid division by zero
    normalized_tensor = (tensor - min_val) / tf.clip_by_value(max_val - min_val, 1e-6, float('inf'))
    return normalized_tensor



if __name__ == '__main__':    
    img_size = (640, 640)
    fcn = FCNs(img_size, points_option='avg')

    weight_file = 'weights/SMTL_thr0_lrf10_fast_combotrain_coco_10epochs.h5'
    fcn.load_weights(weight_file, by_name=True, skip_mismatch=False)
    fcn.summary()

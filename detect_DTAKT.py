import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from model import FCNs, soft_nmss, soft_nms
import matplotlib
matplotlib.use('TkAgg') # Change backend after loading model

def imshow_full(img, cmap=None):
    height, width = img.shape[0],img.shape[1]
    # Calculate the aspect ratio
    aspect_ratio = width / height
    fig_height = 5
    fig_width = fig_height * aspect_ratio
    plt.figure(figsize=(fig_width, fig_height)) 
    plt.imshow(img, cmap=cmap) 
    plt.axis('off') 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show() 

def pointindice2xy(points_ind, width):
    points_x = tf.cast(points_ind % width, tf.float32)
    points_y = tf.cast(points_ind // width, tf.float32)
    points_xy = tf.stack([points_x, points_y], axis=-1)
    points_ij = tf.stack([points_y, points_x], axis=-1)
    return points_xy, points_ij

def GetSalientFeature(inputs, num_points=5000, mask_border=3):  

    score_maps, feature_maps = inputs[0], inputs[1]  

    batch, height, width, chanel = score_maps.shape
    # print('====score_map size',batch, height, width, chanel)
    if mask_border > 0:
        mask = np.ones((height, width), dtype=np.float32)
        # Set the first and last three rows and columns to 0
        mask[:mask_border, :] = 0
        mask[-mask_border:, :] = 0
        mask[:, :mask_border] = 0
        mask[:, -mask_border:] = 0
        mask = mask[:, :, tf.newaxis]
        score_maps = score_maps * mask
    
    scores = tf.reshape(score_maps, (-1, height*width))
    # print('====scores',scores)
    sel_scores, sel_inds = tf.math.top_k(scores, k=num_points)
    # print('sel_inds====',sel_inds)

    points_xy, points_ij = pointindice2xy(sel_inds, width) 
    # print('points_xy====',points_ij)
    sel_features = tf.gather_nd(feature_maps, tf.cast(points_ij, tf.int32), batch_dims=1)  
    # print('sel_features====',sel_features)  

    return sel_features, points_xy, sel_scores

def getFcnFeature(FCN, inputs, num_points=5000, Ksize=5, mask_border=3, method='fcn'):
    print('FCN feature=============>\n')
 
    score_map, dense_feature = FCN.predict(inputs)

    # score_map_thr, mask = soft_threshold(score_map, threshold=0.5, option='hard')
    score_map_nmss = soft_nmss(score_map, Ksize=Ksize, sigma=0.5)  
    score_map_nms, mask = soft_nms(score_map_nmss, Ksize=Ksize, option='softmax', alpha=2)

    descriptors, keypoints, scores = GetSalientFeature([score_map_nms, dense_feature], 
                                                        num_points=num_points, 
                                                        mask_border=mask_border)

    score_map = score_map[0]
    dense_feature = dense_feature[0]

    keypoints = keypoints.numpy()[0]
    descriptors = descriptors.numpy()[0].tolist()    
    scores = scores.numpy()[0].tolist()

    width = score_map.shape[1]
    height = score_map.shape[0]
    keypoints  = np.hstack([keypoints[:,0][:, np.newaxis]/width, keypoints[:,1][:, np.newaxis]/height]).tolist()    

    return score_map, keypoints, descriptors, scores

def disp_keypoints(image_np, keypoints):    
    if len(keypoints) == 0:
        return image_np

    pts = keypoints
    cv_pts = [cv2.KeyPoint(pt[0], pt[1], size=2) for pt in pts]

    # drawKeypoints function is used to draw keypoints 
    output_image = cv2.drawKeypoints(image_np, cv_pts, 0, (0, 0, 255), 
                                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
    return output_image

def extract(extractor_rgb, img_path, display=True):    
    width = extractor_rgb.input.shape[2]
    height = extractor_rgb.input.shape[1]

    image = cv2.imread(img_path)    
    original_height, original_width = image.shape[0], image.shape[1]
    resized_image = cv2.resize(image, (width, height)) 
    

    inputs = resized_image[np.newaxis,:,:,:].astype(np.float32)/resized_image.max()

    start_time = time.time()
    
    print('------------->extract features based on the proposed method')
    score_map, keypoints, descriptors, scores = getFcnFeature(extractor_rgb, inputs, 
                                                              num_points=5000, 
                                                              Ksize=5, 
                                                              mask_border=3,  
                                                              method='fcn')    
    elapsedTime = time.time() - start_time   
    print(f'------------->elatpsed time is {elapsedTime} seconds for {len(keypoints)} keypoints')     
    keypoints_original = [[pt[0]*original_width, pt[1]*original_height] for pt in keypoints]     

    if display:
        kps_image = disp_keypoints(image, keypoints_original)  
        imshow_full(kps_image)  

if __name__ == '__main__':
    img_size = (640, 640)
    extractor_rgb = FCNs(img_size, points_option='avg')

    # DTL-Fast in paper: SMTL_thr0_lrf10_fast_combotrain_coco_10epochs
    # PLS-Fast in paper: detector_hardnms5_avg_0005_solotrain_coco_epochs
    weight_file = 'weights/SMTL_thr0_lrf10_fast_combotrain_coco_10epochs.h5'
    extractor_rgb.load_weights(weight_file, by_name=True, skip_mismatch=False)
    extractor_rgb.summary()

    img_path = './TEST_IMGS/brooklyn.png'
    extract(extractor_rgb, img_path, display=True)

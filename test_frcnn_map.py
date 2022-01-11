from __future__ import division
import os
import cv2
import numpy as np
import pandas as pd
import sys
import pickle
from optparse import OptionParser
import time
import re
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score



sys.setrecursionlimit(40000)

#config = tf.ConfigProto()
config=tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
config.log_device_placement = True
#sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
#set_session(sess)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config.pickle")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                default="pascal_voc"),
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("-i", "--output_model_number", dest="model_iter", help="Models of Epoch step to use. Type this with leading spaces for the hdf5 files!"),
parser.add_option("--ovt", "--overlap_threshold", type="float", dest="overlap_threshold",
                help="Value of overlap threshold for non-max-suppression.", default=0.7)
parser.add_option("--iou", "--iou_threshold", type="float", dest="iou_threshold",
                help="Value of IoU to overpass for the box to ve counted as True Positive", default=0.5)

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")
    
config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if options.model_iter is not None:
    x = re.match("^(.+)(\.hdf5)$", C.model_path)
    C.model_path = x.group(1) + "_" + options.model_iter + x.group(2)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path
iou_thresh = options.iou_threshold

def get_map(pred, gt, f):
    T = {}
    P = {}
    iou_result = 0
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    #print(pred)
    #print(pred_probs)
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1']/fx
            gt_x2 = gt_box['x2']/fx
            gt_y1 = gt_box['y1']/fy
            gt_y2 = gt_box['y2']/fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = 0
            iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            iou_result += iou
            #print('IoU = ' + str(iou))
            if iou >= iou_thresh:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))
    for gt_box in gt:
        if not gt_box['bbox_matched']: # and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    return T, P, iou_result

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
        
    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio    

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img_ratio(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def format_img(img, C):
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
    
    if width <= height:
        f = img_min_side/width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side/height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width/float(new_width)
    fy = height/float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, fx, fy


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

begin = time.time()
T = {}
P = {}
iou_result = 0

filepath_list=[]
iou_list=[]
all_map=[]

all_imgs = []

classes = {}

all_dets = []
filenames=[]        
coord=[] 
cl_prob=[]

overlap_threshold = options.overlap_threshold

test_imgs, _, _ = get_data(options.test_path)

for idx, img_data in enumerate(test_imgs):
    
    print('{}/{}'.format(idx + 1,len(test_imgs)))
    
    st = time.time()
    
    filepath = img_data['filepath'] 
    filepath_list.append(filepath) 

    img = cv2.imread(filepath)
    
    X, ratio = format_img_ratio(img, C)

    if K.common.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=overlap_threshold)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            #if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
  
    det_iou=[]
    for key in bboxes:
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=overlap_threshold)

        
        for jk in range(new_boxes.shape[0]):
            
            (x1, y1, x2, y2) = new_boxes[jk,:]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)
            det_iou.append(det)
            filenames.append(str(filepath).split('/')[1])
       

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

            textLabel = ("{}: {}".format(key,int(100*new_probs[jk])))
            cl_prob.append((key,100*new_probs[jk]))
            print(textLabel)

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_PLAIN,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            
            t=(real_x1, real_y1, real_x2, real_y2) 
            coord.append(t)
    # if the model detects the targeted object, the image with the predicted bbox is saved in the results_imgs folder
            if det_iou!=[]:
                cv2.imwrite('./results_imgs/{}.png'.format(str((filepath).split('/')[1])[:-4]), img)    
        
       
    
    X, fx, fy = format_img(img, C)
    t, p, iou = get_map(det_iou, img_data['bboxes'], (fx, fy))
    iou_result += iou
            
    iou_list.append(iou)
    for key in t.keys():
        print("t.keys:",t.keys())
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
        #print('T:',T)
        #print('P:',P)
        #save_log_data = ('\n T = {} and P = {} for image {}'.format(T, P, filepath))
        #with open("./saving_TP.txt","a") as f:
        #    f.write(save_log_data)
        #save_log_data = ('\n T[key]= {} and P[key] = {} for image {}'.format(T[key], P[key], filepath))
        #with open("./saving_TPkeys.txt","a") as f:
        #    f.write(save_log_data)
        
    all_aps = []

    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
        save_log_data = ('\n {} AP: {} for image {} '.format(key, ap, filepath))
        with open("./results/saving_AP.txt","a") as f:
            f.write(save_log_data)
        
    save_log_data = ('\n IoU@{} = {} for image {} '.format(iou_thresh, iou, filepath))
    with open("./results/saving_iou.txt","a") as f:
        f.write(save_log_data)
    
    maps=np.mean(np.array(all_aps))
    all_map.append(maps)
    print('mAP = {}'.format(maps))
    


AP_total=average_precision_score(T[key], P[key])
print('AP_total = {}'.format(AP_total))

print('IoU@{} total = {}' .format(iou_thresh, (iou_result/len(test_imgs))))


df=pd.DataFrame({'file':filepath_list, 'IoU':iou_list , 'mAP':all_map})
df.to_csv('./results/metrics.csv')

df1 = pd.DataFrame(all_dets, columns=['x1', 'x2', 'y1', 'y2', 'class', 'prob'])
df2 = pd.DataFrame(filenames, columns=['filename'])
df3 = pd.DataFrame(coord, columns=['pred_x1', 'pred_y1','pred_x2', 'pred_y2'])  

prediction = pd.concat([df2, df1, df3], axis=1)
prediction = prediction.drop(['x1', 'x2', 'y1', 'y2'], axis=1)
prediction.to_csv('./results/predictions.csv') 

print('Completely Elapsed time = {}'.format(time.time() - begin))







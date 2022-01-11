from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import re
import pandas as pd



from keras import backend as K
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn.simple_parser import get_data
from keras_frcnn.dataframe import filenames_per_batch
from keras_frcnn.simple_parser_datagen import get_classes_batch
from keras_frcnn.dataframe import get_dataframe
from keras_frcnn.simple_parser_datagen import get_data_batch


sys.setrecursionlimit(40000)

# Parsing annotation file 

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network",
                  help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips",
                  help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips",
                  help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs",
                  help="Number of epochs.", default=1000)
parser.add_option("--batch_size", type="int", dest="batch_size",
                  help="Size of batch for training for Image Data Generator")
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path",
                  help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--ovt", "--overlap_threshold", type="float", dest="overlap_threshold",
                help="Value of overlap threshold for non-max-suppression.", default=0.7)


(options, args) = parser.parse_args()

# Raise error if name of input path not given
if not options.train_path:   # if filename is not given
    parser.error(
        'Error: path to training data must be specified. Pass --path to command line')
    
# Chose the right parser according to type of data 
if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple_datagen_val':
    from keras_frcnn.simple_parser_datagen_val import get_data_batch
else:
    raise ValueError(
        "Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# To save the weights of the model in .hdf5
C.model_path = options.output_weight_path
model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
    print('Output weights must have .hdf5 filetype')
    exit(1)
C.num_rois = int(options.num_rois)

# To select based model 
if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
else:
    print('Not a valid model')
    raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()

overlap_threshold = options.overlap_threshold


# Training Data

# get dataframe from train_images folder 
df_train = get_dataframe("train_images")
print("Number of training images:",len(df_train))

# use ImageDataGenerator to load images per batch
train_datagen = ImageDataGenerator(rescale=1./255.)

# set batch size 
batch_size= int(options.batch_size)
BATCH_SIZE = batch_size

# Create the batches of training images
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory='train_images/',
                                                    x_col="image_path",
                                                    y_col="class",
                                                    subset="training",
                                                    batch_size=BATCH_SIZE,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=(256, 256),
                                                    validate_filenames=True)

# to get the filenames of the images put in the different batches (returns a list of lists, the first list of this list contains the filenames of the first batch)
imgs_per_batch_train = filenames_per_batch(train_generator)
# just for checking, must be equal to the number of batches per epoch below 
print('length of list called imgs per batch train (must be equal to number of batches):',len(imgs_per_batch_train))

# to calculate the number of batches per epoch using the number of samples and the batch size 
batches_per_epoch = train_generator.samples // train_generator.batch_size + (train_generator.samples % train_generator.batch_size > 0)
print('Number of batches per epoch = {}'.format(batches_per_epoch))

# validation data
df_val = get_dataframe("val_images")
num_val_imgs = len(df_val)
print("Number of validation images:",len(df_val))

# calculate the batch size to fit the number of batches in training 
if num_val_imgs % batches_per_epoch == 0:
    BATCH_SIZE_VAL = num_val_imgs // batches_per_epoch
else:
    BATCH_SIZE_VAL = (num_val_imgs // batches_per_epoch)+1

print("batch size for validation:",BATCH_SIZE_VAL)

val_datagen = ImageDataGenerator(rescale=1./255.)

val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory='val_images/',
                                                x_col="image_path",
                                                y_col="class",
                                                subset="training",
                                                batch_size=BATCH_SIZE_VAL,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="categorical",
                                                target_size=(256, 256),
                                                validate_filenames=True)

# to get the filenames of the images put in the different batches
imgs_per_batch_val = filenames_per_batch(val_generator)
#print('number of imgs per batch for validation:',len(imgs_per_batch_val))

save_log_data = ('\n batch size training = {} , batch size validation = {} ,Number of training images ={}, Number of validation images={}'.format(BATCH_SIZE, BATCH_SIZE_VAL, len(df_train), len (df_val)))
with open("./saving_params.txt","a") as f:
    f.write(save_log_data)

# Model parameters 
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)


epoch_length = BATCH_SIZE
num_epochs = int(options.num_epochs)

classes_count, class_mapping = get_classes_batch(
    options.train_path, imgs_per_batch_train[0], 'train')
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Num classes (including bg) = {}'.format(len(classes_count)))

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
#print('number of anchors:', num_anchors)


if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(
    classes_count), trainable=True)

model_base = Model(img_input, shared_layers)
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)


try:
    print('loading weights from {C.base_net_weights}')
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights')


model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls( num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print("Config has been written to {}".format(config_output_filename))

# create empty lists to store values 
loss_valid = []
acc_valid=[]
loss_train = []
acc_train=[]
epoch_number = []

# timer for total training 
start_total = time.time()

print('..............Starting training ............\n')

# loop on epochs numbers
for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print("Epoch {} / {}".format(epoch_num + 1, num_epochs))
    # timer for epoch training 
    start_ep = time.time()

    iter_num = 0
    
    # loop over batches 
    for i in range(batches_per_epoch):

        print("Batch {} / {}".format(i + 1, batches_per_epoch))
        
        # get data for images in batch 
        train_imgs = get_data_batch(options.train_path, imgs_per_batch_train[i])
        print("Num train samples {}".format(len(train_imgs)))

        
        # get ground truth data
        data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.common.image_dim_ordering(), mode='train')
        
        # set initial values 
        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        best_loss = np.Inf
        
        # timer for batch training 
        start_batch = time.time() 
    
        while True:
            try:
                
                # 
                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print("Average number of overlapping bounding boxes from RPN = {} for {} previous iterations".format(mean_overlapping_bboxes, epoch_length))

                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=overlap_threshold, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou( R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(
                            pos_samples, C.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(
                            neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch( [X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                            ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

                iter_num += 1

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {0}'.format( mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {0}'.format(class_acc))
                        print('Loss RPN classifier: {0}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {0}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {0}'.format(loss_class_cls))
                        print('Loss Detector regression: {0}'.format( loss_class_regr))
                        print('Elapsed time for training batch: {}'.format(time.time()-start_batch))

            
                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0

                    loss_train.append(curr_loss)
                    acc_train.append(class_acc)
                    epoch_number.append(epoch_num+1)

                    if curr_loss < best_loss:
                        if C.verbose:
                            print("Total loss for training decreased from {} to {} saving weights ".format( best_loss, curr_loss))
                            save_log_data = '\nTotal loss decreased from {} to {} in epoch {}/{} in training, saving weights'.format(best_loss, curr_loss, epoch_num + 1, num_epochs)
                            with open("./saving_log.txt", "a") as f:
                                f.write(save_log_data)

                        best_loss = curr_loss
                        model_all.save_weights(model_path_regex.group(1) + "_" + '{:04d}'.format(epoch_num+1) + model_path_regex.group(2))

                    break
            except Exception as e:
                print('Exception: {}'.format(e))
                continue
            
    print("................Starting validation ............. \n")

    for j in range(batches_per_epoch):

        print("Batch {} / {}".format(j + 1, batches_per_epoch))
        
        # get validation data for this batch 
        val_imgs = get_data_batch(options.train_path, imgs_per_batch_val[j])
        print("Num val samples = {}".format(len(val_imgs)))
        
        
        validation_epoch_length=len(val_imgs)
        
        # get ground truth data
        data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.common.image_dim_ordering(), mode='val')

        # set variable for validation 
        losses_val = np.zeros((validation_epoch_length, 5))
        start_batch_val = time.time()
        val_best_loss = np.Inf
        val_best_loss_epoch = 0                
                
        progbar = generic_utils.Progbar(validation_epoch_length)
        
        while True:
            try:
                X, Y, img_data = next(data_gen_val)
                        
                val_loss_rpn = model_rpn.test_on_batch(X, Y)
            
                P_rpn = model_rpn.predict_on_batch(X)
                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=overlap_threshold, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
                        
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)
            
                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []
            
                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []
                        
                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))
            
                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
            
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)
                               
                val_loss_class = model_classifier.test_on_batch( [X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                losses_val[iter_num, 0] = val_loss_rpn[1]
                losses_val[iter_num, 1] = val_loss_rpn[2]
                
                losses_val[iter_num, 2] = val_loss_class[1]
                losses_val[iter_num, 3] = val_loss_class[2]
                losses_val[iter_num, 4] = val_loss_class[3]
                            
                iter_num += 1
            
                progbar.update(iter_num, [('rpn_cls_val', np.mean(losses_val[:iter_num, 0])), ('rpn_regr_val', np.mean(losses_val[:iter_num, 1])),
                                                  ('detector_cls_val', np.mean(losses_val[:iter_num, 2])), ('detector_regr_val', np.mean(losses_val[:iter_num, 3]))])
            
                if iter_num == validation_epoch_length:
                               
                    val_loss_rpn_cls = np.mean(losses_val[:, 0])
                    val_loss_rpn_regr = np.mean(losses_val[:, 1])
                    val_loss_class_cls = np.mean(losses_val[:, 2])
                    val_loss_class_regr = np.mean(losses_val[:, 3])
                    val_class_acc = np.mean(losses_val[:, 4])
            
                                
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []
                            
                            
                    val_curr_loss = val_loss_rpn_cls + val_loss_rpn_regr + val_loss_class_cls + val_loss_class_regr
                    iter_num=0
                    loss_valid.append(val_curr_loss)
                    acc_valid.append(val_class_acc)  
                 
                    if C.verbose:
                        print('[INFO VALIDATION]')
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(val_class_acc))
                        print('Loss RPN classifier: {}'.format(val_loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(val_loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(val_loss_class_cls))
                        print('Loss Detector regression: {}'.format(val_loss_class_regr))
                        print("current loss: %.2f: "%(val_curr_loss))
                        print('Elapsed time for validation: {}'.format(time.time() - start_batch_val))   
                    
               
            
                    if val_curr_loss < val_best_loss:
                        if C.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(val_best_loss,val_curr_loss))
                            save_log_data = '\nTotal loss decreased from {} to {} in epoch {}/{} in validation, saving weights'.format(val_best_loss,val_curr_loss,epoch_num + 1 ,num_epochs)
                            with open("./saving_log.txt","a") as f:
                                f.write(save_log_data)
                                            
                        val_best_loss = val_curr_loss
                        val_best_loss_epoch=epoch_num
                                
                        model_all.save_weights(model_path_regex.group(1) + "_val" + '{:04d}'.format(epoch_num+1) + model_path_regex.group(2))
               
                                    
                    break
                
            except Exception as e:
                #print('Exception: {}'.format(e))
                continue
            
            elapsed_ep = time.time() - start_ep
            print('Elapsed time for the epoch: {0}'.format(elapsed_ep))
            
print('Elapsed time for training and validation: {0}'.format(time.time()-start_total))


df = pd.DataFrame(list(zip(loss_train, loss_valid, acc_train, acc_valid, epoch_number)),columns=['training loss', 'validation loss', 'training_acc', 'valid_acc', 'epoch_number'])
df.to_csv('./results/trainval_history.csv')

print('Training complete, exiting.')


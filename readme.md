**KERAS-FASTER-RCNN**

Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (https://github.com/yhenon/keras-frcnn/)
This repo was cloned from https://github.com/kbardool/keras-frcnn

*The original code has been optimized for custom dataset. 
A validation step was added, allowing you to plot the loss values per epoch.
The images are loaded using ImageDataGenerator.*

# TO WORK ON YOUR CUSTOM DATASET:

# Open a terminal in the keras_FasterRCNN_CustomDataset directory

## To train the model: 

    • Load training images in the folder train_images 

    • Load validation images in the folder val_images 

    • Provide a .txt file containing annotations of BOTH training and validation data, with each line containing:  `filepath,x1,y1,x2,y2,class_name`
        Exemple : train_images/image001.png/215,312,279,391,handwritten
   	     	     val_images/image256.png/215,312,279,391,handwritten

    • Run the following command in terminal:
        ◦ python train_frcnn_val.py -o simple_datagen_val -p annotation_file.txt --num_epochs 100 --batch_size 500 (adapt parameters)

    • If you want to continue a previous training, you can load the weights using option --input_weight_path './filename.hdf5' 

    • The default overlap threshold value for non-max-suppression is 0.7, if you want to change it, use option --ovt and set your own value
    
    • Train and validation loss values are saved in the results folder


## To evaluate the model on ANNOTATED data:

    • Provide a .hdf5 file corresponding to the training weights you want to load

    • Provide the config.pickle file corresponding to training

    • Load images in the folder named test_images

    • Provide a .txt file with each line containing:  `filepath,x1,y1,x2,y2,class_name`
        Exemple: test_images/image008.png/215,312,279,391,handwritten

    • Run the following command
        ◦ python test_frcnn_map.py -p annotationfile.txt -o simple 

        If option --input_weight_path is not set in the command line, the file model_frcnn.hdf5 in the keras-master-frcnn folder is taken to load weights.
        The default overlap threshold value for non-max-suppression is 0.7 and the default iou threshold value is 0.5. 
        If you want to change them, use options --ovt and --iou to set your own values

    ==> The images with the predicted bounding boxes will be saved in the results_imgs folder. If no box was predicted, the image is not saved. 

    ==> file metrics.csv containing IoU and mAP values for each image and file predictions.csv containing the coordinates of the predicted boxes, 
    the predicted class and the probability of the object to belong to the predicted class are saved in the results folder 


## To make predictions on NON annotated data :

    • Provide a .hdf5 file corresponding to the training weights you want to load

    • Load images in a folder named test_images

    • If you want to display all the boxes predicted by the model for each image 
        ◦ Run the following command:
            python test_frcnn_all_datagen.py -p test_images --batch_size 200
 
    In the parser options you can modify the overlap_threshold for non-max-suppression by using -- ovt option and change the bbox threshold using --bbt option
    Nb: changing the bbt threshold has an effect on the number of True / False Positives


    • If you want to display ONLY the box predicted with the best score for each image 
        ◦ Run the following command in terminal
            python test_frcnn_best_datagen.py -p test_images --batch_size 200
 
        In the parser options you can modify the overlap_threshold for non-max-suppression by using -- ovt option and you can also change the bbox threshold 
        using --bbt option


# NOTES: 

- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].
- This repo was developed using python 2.7 
- If you run out of memory, try reducing the number of ROIs that are processed simultaneously. Try passing a lower -n to train_frcnn.py. Alternatively, try reducing the image size from the default value of 600 (this setting is found in config.py).



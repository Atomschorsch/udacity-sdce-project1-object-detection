#!/usr/bin/python
# Module for visualization tooling

#from utils import get_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import tensorflow as tf
# Data generator
import json
import numpy as np

colors = ['y','r','g']
class_names = ['class 0','class 1', 'class 2']

def visualize_tf_record_dataset(dataset, n_show = -1, x_max = 5, y_max = 5, show_pred_classnames = True, show_gt_classnames = False):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """

    # Make subplots with loop over images
    filenames = [data_item['image/filename'].numpy() for data_item in dataset]
    num_images = len(filenames)
    if n_show == -1:
        n_show = num_images
    subplot_dim, _ = get_subplot_dims(num_images, x_max = x_max, y_max = y_max)
    max_num = x_max * y_max
    

    for idx_file, current_image_data in enumerate(dataset.take(n_show)):
        axs_idx = idx_file % max_num
        # Handle multiple subplot figures
        if axs_idx == 0:
            if sum(subplot_dim) > 2:
                # Multiple images
                fig, axs = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(18, 18)) # 
                axs = axs.reshape(min(max_num, subplot_dim[0]* subplot_dim[1]))
            else:
                # Only one image
                plt.figure()
                axs = [plt.gca()]
        # Handle image
        raw_image = current_image_data['image/encoded'].numpy()
        decoded_image = tf.image.decode_image(raw_image)    
        axs[axs_idx].imshow(decoded_image)
        axs[axs_idx].get_xaxis().set_visible(False)
        axs[axs_idx].get_yaxis().set_visible(False)
        axs[axs_idx].set_title(current_image_data['image/filename'].numpy())
        # Ground truth boxes
        gt_boxes = convert_tf_record_metadata(current_image_data)
        # image_data = get_image_data(ground_truth, current_image_data)
        if gt_boxes:
            plot_boxes(axs[axs_idx], gt_boxes, is_ground_truth = True)

        # Prediction boxes
        image_data = []
        if image_data:
            plot_boxes(axs[axs_idx], image_data, is_ground_truth = False)

        if sum(subplot_dim) > 2 and axs_idx == max_num-1:
            axs = axs.reshape(subplot_dim)
            fig.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()


def convert_tf_record_metadata(image_item):
    '''Current metadata of current imaga to dict'''
    # {
    #     "filename": "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\custom\cat2.jpg",
    #     "boxes": [[10,10,50,50],[100,100,110,120]],
    #     "classes": [2,1],
    # }
    width = image_item['image/width'].numpy()
    height = image_item['image/width'].numpy()
    box_x_mins = image_item['image/object/bbox/xmin'].values.numpy()*width
    box_x_maxs = image_item['image/object/bbox/xmax'].values.numpy()*width
    box_y_mins = image_item['image/object/bbox/ymin'].values.numpy()*height
    box_y_maxs = image_item['image/object/bbox/ymax'].values.numpy()*height
    classes_text = image_item['image/object/class/text'].values.numpy()
    classes = image_item['image/object/class/label'].values.numpy()
    ret_dict = {
        'classes_text': classes_text,
        'classes': classes
    }
    boxes = []
    for idx in range(len(box_x_mins)):
        boxes.append([box_y_mins[idx],box_x_mins[idx], box_y_maxs[idx],box_x_maxs[idx]])
    ret_dict['boxes']=boxes
    return ret_dict
    




def show_box(ax, box, classname, color, dashed = False):
    # x and y must be switched for matplotlib?
    ax.add_patch(plt.Rectangle((box[1],box[0]),box[3]-box[1],box[2]-box[0],linewidth=1,edgecolor=color,facecolor='none',linestyle='--' if dashed else '-'))
    if classname:
        ax.text(box[2],box[1], classname, color=color, horizontalalignment = 'right')
        #ax.annotate(classname, xy=(box[0],box[1]),xytext=(box[0]+50,box[1]+50), arrowprops=dict(facecolor='black', shrink=0.05))

def get_image_list(ground_truth, predictions = []):
    filenames = [image['filename'] for image in ground_truth + predictions]
    return set(filenames)

def get_coarse_resolution(num):
    x_dim = 1
    y_dim = 1
    while x_dim*y_dim < num:
        if x_dim<=y_dim:
            x_dim+=1
        else:
            y_dim+=1
    return (y_dim,x_dim)


def get_subplot_dims(num_images, x_max = 5, y_max = 5):
    """Get subplot dimensions based on number of images"""
    max_images = x_max * y_max
    num_subplots = math.ceil(num_images / max_images)
    subplot_dim = get_coarse_resolution(min(num_images, max_images))
    return subplot_dim, num_subplots

def get_current_subplot_position(current_idx, x_dim, y_dim):
    """For subplots to derive from index to grid position. Unnecessary if shape is flattened and can be used in for loop."""
    x_pos = current_idx % x_dim
    y_pos = current_idx // y_dim
    return x_pos, y_pos

def get_image_data(image_array, filename):
    """Get image data if image is available in array"""
    current_prediction = [image for image in image_array if image['filename'] == filename]
    assert len(current_prediction)<=1, "Multiple prediciton images match, should not happen"
    if len(current_prediction)>0:
        return current_prediction[0]
    else:
        return None

def plot_boxes(ax, image_data, is_ground_truth = False):
    for idx, box in enumerate(image_data['boxes']):
        current_class = image_data['classes'][idx]
        classname = "" if is_ground_truth else class_names[current_class]
        # Show gt without label (should be mostly obvious) and as dashed box
        dashed = True if is_ground_truth else False
        color = colors[current_class]        
        show_box(ax, box, classname, color, dashed=dashed) 

def viz(ground_truth, predictions = [], x_max = 5, y_max = 5, show_pred_classnames = True, show_gt_classnames = False):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """

    # Make subplots with loop over images
    filenames = get_image_list(ground_truth, predictions)
    num_images = len(filenames)
    subplot_dim, _ = get_subplot_dims(num_images, x_max = x_max, y_max = y_max)
    max_num = x_max * y_max
    

    for idx_file, file in enumerate(filenames):
        axs_idx = idx_file % max_num
        # Handle multiple subplot figures
        if axs_idx == 0:
            if sum(subplot_dim) > 2:
                # Multiple images
                fig, axs = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(18, 18)) # 
                axs = axs.reshape(min(max_num, subplot_dim[0]* subplot_dim[1]))
            else:
                # Only one image
                plt.figure()
                axs = [plt.gca()]
        # Handle image
        img = mpimg.imread(file)        
        axs[axs_idx].imshow(img)
        axs[axs_idx].get_xaxis().set_visible(False)
        axs[axs_idx].get_yaxis().set_visible(False)
        axs[axs_idx].set_title(os.path.basename(file))
        # Ground truth boxes
        image_data = get_image_data(ground_truth, file)
        if image_data:
            plot_boxes(axs[axs_idx], image_data, is_ground_truth = True)

        # Prediction boxes
        image_data = get_image_data(predictions, file)
        if image_data:
            plot_boxes(axs[axs_idx], image_data, is_ground_truth = False)

        if sum(subplot_dim) > 2 and axs_idx == max_num-1:
            axs = axs.reshape(subplot_dim)
            fig.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

def add_json(container, filename, boxes, classes):
    container.append({
        "filename": filename,
        "boxes": boxes,
        "classes": classes,
    })

def test_data_generator(num_images):
    """ Generator to generate random ground_truth and prediction json structures on CatsAndDogs dataset images"""
    ground_truth = []
    predictions = []

    data_root_dir = "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\PetImages"
    cat_dir = os.path.join(data_root_dir, "Cat")
    dog_dir = os.path.join(data_root_dir, "Dog")
    for idx in range(num_images):
        # Choose cat or dog
        is_cat = bool(np.random.randint(0,1+1))
        # TODOCreate random gt boxes
        image_index = np.random.randint(0,12000+1)
        image_dir = cat_dir if is_cat else dog_dir
        image_class = 1 if is_cat else 2
        image_path = os.path.join(image_dir, str(image_index) + '.jpg')
        if not os.path.exists(image_path):
            continue
        gt_boxes = []
        gt_classes = []
        for box_idx in range(np.random.randint(0,10+1)):
            x1 = np.random.randint(0,100+1)
            y1 = np.random.randint(0,100+1)
            x2 = x1 + np.random.randint(30,100+1)
            y2 = y1 + np.random.randint(30,100+1)
            gt_boxes.append([x1,y1,x2,y2])
            gt_classes.append(np.random.randint(1,2+1))
        add_json(ground_truth, image_path, gt_boxes, gt_classes)
        # Only add predictions in ~ 19/20 cases
        pred_boxes = []
        pred_classes = []
        if np.random.randint(0,20+1):
            for box_idx in range(np.random.randint(0,10+1)):
                x1 = np.random.randint(0,100+1)
                y1 = np.random.randint(0,100+1)
                x2 = x1 + np.random.randint(30,100+1)
                y2 = y1 + np.random.randint(30,100+1)
                pred_boxes.append([x1,y1,x2,y2])
                pred_classes.append(np.random.randint(1,2+1))
        add_json(predictions, image_path, pred_boxes, pred_classes)

    return ground_truth, predictions

if __name__ == "__main__": 
    # ground_truth, _ = get_data()

    # ground_truth = [{
    #     "filename": "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\custom\cat2.jpg",
    #     "boxes": [[10,10,50,50],[100,100,110,120]],
    #     "classes": [2,1],
    # },{
    #     "filename": "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\custom\cat1.jpg",
    #     "boxes": [[20,10,50,50],[100,10,110,500]],
    #     "classes": [1,1],
    # },{
    #     "filename": "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\custom\dog2.jpg",
    #     "boxes": [[10,10,50,50],[100,100,110,120]],
    #     "classes": [2,1],
    # },{
    #     "filename": "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\custom\dog1.jpg",
    #     "boxes": [[20,10,50,50],[100,10,110,500]],
    #     "classes": [1,1],
    # }]
    # predictions=[{
    #     "filename": "\\\\spacestation2\home\Drive\Projekte\Tensorflow\CatsAndDogs\custom\cat2.jpg",
    #     "boxes": [[14,10,50,50]],
    #     "classes": [2],
    # }]
    ground_truth, predictions = test_data_generator(100)
    viz(ground_truth, predictions)
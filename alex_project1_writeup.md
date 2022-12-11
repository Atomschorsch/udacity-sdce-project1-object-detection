# Project 1 - Object Detection
# Udacity Nanodegree - Self Driving Car Engineer 
# Writeup

## Project overview
Object detection is one of the main crucial tasks during video perception in self driving cars or driver assistance systems.
In this project, each necessary step from set up and data analysis due to cross validation, training, evaluation and augmentation shall be practiced and implemented, due to the lessons learned in the first chapter and with example images from the Waymo Open dataset.

## Set up
After heavy difficulties to set up a local system, I have finally managed to run the repo inside of a container on wsl2 on windows with 
* GPU usage
* routing ports for tensorboard
* plotting images from matplotlib outside of the container

All timings are measured on a system with Intel I13900K and RTX4090.

My setup including instructions and a description of difficulties along with necessary fixes are described in [alex_container_instructions.md](alex_container_instructions.md).

### **Scripting**
I have implemented some convenient scripts and helper classes (`./helpers`) to automize some of the steps defined in the project task and to reuse code for re-occuring tasks. These will be mentioned during each step, if relevant.

### **Code versioning**
During the implementation, I have used git / local gitlab instance for commmits and branches to save my increments. The progress and implementation steps should be understandable and visible via `gitk` or in any git gui.
### **External sources**
When using any code snippets from stack overflow or other sources during my research, I have tried to mark this or refer to the source in the code. In case I have forgotten or overseen some of them, I apologize for this.

I have put quite some effort into the docker setup and convenient scripts with the hope to be able to reuse those as templates for future machine learning projects.

## Questions to the tutor / corrector
Since this topic is new for me, I think I have implemented some functions way too complicated, and I guess for most of them are way better and integrated functions already present.
Could you please check my code, and give some hints, in case there already good library functions for code I have implemented? Especially regarding tfrecord handling, visualization and all the tooling stuff.
I have also marked some questions in the code with `# QUESTION TO TUTOR`. Could please refer to those?

## Dataset
### Dataset analysis
**Dataset overview:**  
After using the provided download_process.py script, the data consists of 100 tfrecord files with a total of 1997 images and labels, taken from the Waymo Open dataset.

To be honest I had quite some difficulties with the tfrecord format and had to do quite some research and test scripts to get along with it. I played around with it in the file
`alex_print_image_project1.py`, which was then later moved to `helpers/project1.py` for project1 specific functionality.

After that, I tried to identify generic / common parts (which are not project1 specific) and added some helper classes to get a first impression of the data and to visualize it to my demands:
- `./helpers/exploratory_analysis.py`
- `./helpers/visualization.py`

**Analysis and visualization functions:**

My implemented helper class `./helpers/exploratory_analysis.py` contains the functions
- `get_classes_info`:

    Show data attributes of data stored in tfrecord file. This was necessary for me to understand what fields are available in an unknown tfrecord and to understand how to decode it. (Later I learned, that this is part of the proto files, but I could not find them in my repo.)

    Output:
    ```python
    tf.train.Example structure:
    - "image/encoded" (bytes_list): "\377\330\377\340\000\020JFIF\000\001\001\001\001,\001,\000\000\377\333\000C\000\002\001\001\001\001\001\002\001\001\001\002\002\002\002\002\004\003\0
    - "image/filename" (bytes_list): "segment-11940460932056521663_1760_000_1780_000_with_camera_labels_0.tfrecord"
    - "image/format" (bytes_list): "jpg"
    - "image/height" (int64_list): 640
    - "image/object/bbox/xmax" (float_list): 0.4230337142944336
    - "image/object/bbox/xmin" (float_list): 0.40658605098724365
    - "image/object/bbox/ymax" (float_list): 0.5190880298614502
    - "image/object/bbox/ymin" (float_list): 0.5013245344161987
    - "image/object/class/label" (int64_list): 1
    - "image/object/class/text" (bytes_list): "vehicle"
    - "image/source_id" (bytes_list): "segment-11940460932056521663_1760_000_1780_000_with_camera_labels_0.tfrecord"
    - "image/width" (int64_list): 640
    ```
- `show_dataset_basics`

    This prints the basic dataset information (number of images, the existing classes and their distribution) to the console. It will also show some basic histograms on width, height and class distribution:

    Output:
    ```python
    Dataset info:
        1997 elements from 1997 files
        Classes:
        0  : 0
        1  b'vehicle': 35673
        2  b'pedestrian': 10522
        3  : 0
        4  b'cyclist': 270
    ```
    ![Dataset overview histogram](writeup_files/images/eda/project1_dataset_histogram.png)

Both functions are called within the project task `Exploratory Data Analysis.ipynb`.

#### **Implications:**
- Available information per image:
    - actual image
    - format
    - width / height
    - original filename
    - boxes with class / label

    This is the minimum information we need. Images for training and boxes/labels for evaluation.
- There are 1997 images. This is not a big number for data driven methods.
    - -> TODO implications on data split?
    - -> TODO implications on cross-validation?    
    - -> TODO further implications
- All images have the same resolution (width x height). This means we don't need any additional layers in our model to adapt to differen image sizes
- The histogram "Number of boxes per image" in above diagram shows that there are quite a lot of images with > 30 boxes / objects, even some with > 70 boxes. This means that we will probably have very crowded images with a lot of occlusions, and maybe also a lot of tiny objects.
    - We could try soft nms (non max suppression) in the improvment section.
- There are a lot of vehicle objects, only a few pedestrians and almost no cyclists (f). So it is doubtable that the model will perform well on unknown images for pedestrian or cyclist detection
    - --> We should shortly check that we have a good class distribution in each of the data splits.

Having analyzed the basics and gain some information about data distribution and quality, we need to have a closer look into the data itself:
- What is the quality of our data?
- Do we need a cleanup of the data?
- Are there outliers?

For this, I have implemented the helper class `helpers/visualization.py` based on one of the exercises we have made during the first chapter.  
This can visualize big arbitrary sets of images from the dataset, shows the images in matrix plots and adds the boxes with labels. It is useful to have a closer short look on a bigger set of images and identify abnormalities.

![Matrix visualization](writeup_files/images/eda/project1_eda_visualization.png)
![Matrix visualization2](writeup_files/images/eda/project1_eda_matrix_view2.png)

Implications from the visualization:
- The quality of the data seems quite okay, since multiple environments are mirrored in the data with a lot of variety in different aspects:
    - Variety in light conditions (day/night)
    - Variety in image quality (partly occlusions / waterd drops on camera lens)
    - Variety on different environments (low / high traffic, urban / city / landscape environments, different weather)
    - Variety on density (no objects / > 60 images), see histograms

- There are areas in most of the images where almost no objects can be found (upper left / right corner, bottom of image, see yellow area).

    ![Image zones](writeup_files/images/eda/project1_eda_image_zones.png)

    If this has any impact on our model we will see during training. During augmentation we could somehow move the images a little to check the performance of objects in the yellow areas.

- Assumption from above regarding occlusion and tiny objects has proven true, and also above mentioned implications. I don't expect the model to detect the boxes which are only a few pixels in size.

    ![Tiny boxes](writeup_files/images/eda/project1_eda_small_boxes.png)

### Data split and Cross validation
This section should detail the cross validation strategy and justify your approach.
The task to split the data into a train, val and test subset seemed easy on the first view, but opened some heavy questions.
Regarding percentages, I have chosen the ones given from the lessons: `75% train, 15% val, 10% test`, to have the most images for training, but keeping two sufficiently large sets for validation and training.

Going strictly for the task description, it would be to just split a number of files into three different sets.  
The problem I have with this approach, that it creates quite homogenous subsets, which can be very bad for training.  
If we assume that all images from one recording drive are in one file, and we also assume that during one recording drive we would have mostly similar conditions for all the images regarding weather, environment and object density, this would be mostly similar images in one file.
So if there would be only one drive (=one file) with a night drive, that would land in the test set, we would not have any images at night in the training set and vice versa.

So the approach I was going for to not split the files, but the file content.  
I implemented both approaches in `helpers/split.py`:  
- `split_files`: Take all processed files and divide them into three sets of files
- `split_images`: Take all processed files, take all the contained images, shuffle the image set and divide them into three sets.

After executing the `create_split.py`, I have the three expected folders at `/mnt/data/` including tfrecord files.

**Train set:**  
    ![Train set basics](writeup_files/images/split/train_basic.png)
    ![Train set histogram](writeup_files/images/split/train_histo.png)
    ![Train set visualization](writeup_files/images/split/train_set_mixed2.png)

**Val set:**  
    ![Val set basics](writeup_files/images/split/val_basic.png)
    ![Val set histogram](writeup_files/images/split/val_histo.png)
    ![Val set visualization](writeup_files/images/split/val_set_mixed.png)

**Test set:**  
    ![Test set basics](writeup_files/images/split/test_basic.png)
    ![Test set histogram](writeup_files/images/split/test_histo.png)
    ![Test set visualization](writeup_files/images/split/test_set_mixed.png)

#### **Split summary**
The number of the images sums up to the complete image count of the dataset and also matches the intended percentages.  
By shuffling the images before splitting them up, the distribution of classes is roughly the same in all splits, which is good for training and testing.  
From the visualization of the datasets, we also see that all 3 data splits have a similar distribution and variety of conditions due to the shuffling. All sets contain
- light: day / night
- weather: good / bad
- lens: clean / wet
- density: few / many objects
- area: urban / landscape

## Training
### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Own adaptions:
I had to override the model parameter `eval_config.metrics_set` with `coco_detection_metrics`, where the original parameter value `coco_detection_metrics` has thrown the error `'numpy.float64' object cannot be interpreted as an integer`. This fix has been provided via [https://knowledge.udacity.com/questions/657618](https://knowledge.udacity.com/questions/657618).

#### Evaluation of reference training
Training time: ~20min  
Eval time: ~3 min  
# TODO repeat reference training, is gone?
### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
# TODO Improvements nach Lektionsvorschlägen machen
Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model.

    One obvious change consists in improving the data augmentation strategy. The preprocessor.proto file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: Explore augmentations.ipynb. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

    Keep in mind that the following are also available:
        experiment with the optimizer: type of optimizer, learning rate, scheduler etc
        experiment with the architecture. The Tf Object Detection API model zoo offers many architectures. Keep in mind that the pipeline.config file is unique for each architecture and you will have to edit it.



### First experiment experiment0:
From lesson: Overfit a single batch without lr annealing, by scaling up epochs (from 25000 to 75000) and using constant learning rate (0.002)  
Training time:    
-   expected ~ 1 hour  
-    actual: ~ 1 hour

Eval time: ~ 6 min

Result:
This has shown significant improvment right from the beginning. The impacting factor seems to be the adaption of the learning rate.

### Second experiment1:
Implication from last experiment: adaptions on learning rate and optimizer seem to have a big impact on performance. We will now try `rms_prop_optimizer` with `exponential_decay_learning_rate`.  
Training time: ~1:15

Result:
Performance worse than experiment0

### Third experiment2:
Implication from last experiment: Don't use rms_prop_optimizer. Try `adam_optimizer`  with `exponential_decay_learning_rate`, only 50000 steps. 

Performance is also worse than experiment0
-> Choice for `momentum_optimizer ` with lr annealing.


### Fourth experiment3:
After test of all three available optimizers, the choice falls for `momentum_optimizer` with lr annealing, since loss seems to stagnate from step 40000 in experiment0
```
momentum_optimizer    {
      learning_rate {
        exponential_decay_learning_rate  {
          initial_learning_rate: 0.002
          decay_steps: 10000
          decay_factor: 0.95
        }}}
...
num_steps: 60000
```
Also non-max-suppression could help on big groups of objects that we have in the dataset, but it looks like this is already enabled in the pipeline config via `batch_non_max_suppression`. At [post_processing.prot](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/post_processing.proto) I have also found a parameter `soft_nms_sigma` which I thought would be related to soft non-max-suppression, but I have not found any instructions on how to configure this kind of sigma value and what it implies.

Result:  




# TODO
try soft nms (non max suppression) to improve on groups of overlapping objects
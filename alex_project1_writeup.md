# Project 1 - Object Detection
# Udacity Nanodegree - Self Driving Car Engineer 
# Writeup

## Project overview
Object detection is one of the main crucial tasks during video perception in self driving cars or driver assistance systems.
In this project, each necessary step from set up and data analysis due to cross validation, training, evaluation and augmentation shall be practiced and implemented, due to the lessons learned in the first chapter.

## Set up
After heavy difficulties to set up a local system, I have finally managed to run the repo inside of a container on wsl2 on windows. My setup and the difficulties along with necessary fixes are described in [alex_container_instructions.md](alex_container_instructions.md).

I have implemented some convenient scripts and helper classes (./helpers) to automize some of the steps defined in the project task and to reuse code for re-occuring tasks. These will be mentioned during each step, if relevant.
This section should contain a brief description of the steps to follow to run the code for this repository.

## Dataset
### Dataset analysis
Dataset overview:
After using the provided download_process.py script, the data consists of 100 tfrecord files with a total of 1997 images and labels, taken from the Waymo Open dataset.

To be honest I had quite some difficulties with the tfrecord format and had to do quite some research and test scripts to fully understand it. I played around with it in the file
`alex_print_image_project1.py`.
After that, I added some helper classes to get a first impression of the data and to visualize it to my demands:
- `./helpers/exploratory_analysis.py`
- `./helpers/visualization.py`

Coming to the actual analysis and its implications:

This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.

![Dataset overview histogram](writeup_files/images/project1_dataset_histogram.png)

### Cross validation
This section should detail the cross validation strategy and justify your approach.

## Training
### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
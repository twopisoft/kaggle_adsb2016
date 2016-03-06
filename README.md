# Kaggle Annual Data Science Bowl 2016

## Introduction
This application is based on the dynamic programming implementation of [Active (Dynamic) Contours](https://en.wikipedia.org/wiki/Active_contour_model) algorithm. Since the application is purely algorithmic, no training phase is needed and hence no trained model is provided. The training data was used to fine-tune the implementation. The application also uses some part of [Left Ventricle Segmentation Tutorial](https://gist.github.com/ajsander/fb2350535c737443c4e0#file-tutorial-md) published on Kaggle. Specifically, the way the application reads and maintains data in memory data and calculates ROIs is taken from the tutorial.

## Requirements

### Hardware/OS
This application was developed on MacBook Air (2011) running OSX El Capitan (10.11.3)

### Python
Python Version 2.7.11 was used to develop. Other python requirements are listed in the `requirements.txt` file.

### Other 3rd-party
OpenCV 3.1.0 is required for some image processing operations.

## How to train the model
The application is purely algorithmic, no training phase is needed.

## How make new predictions
1. Choose a base directory for storing your data.
2. Create a directory `validate`.
3. Put your studies under the `validate` directory. (i.e follow the same directory structure convention as used for the provided data)
4. Open the `settings.json` file and set the value of `basepath` to be the base directory.
5. Run the application as follows:

`python snake.py`

## Settings.json

This file defines data paths and other options for the application. The explanation follows:

* `basepath`: The base directcory for data. This is the directory where the `train` or `validate` directories exist.
* `ns`: Number of studies. `null` means all. If `ns` is not null, then application runs on `ns` numbers of samples randomly chosen from the studies.
* `validate`: `true` or `false`. Run the application on validation/test data. Note that the validate data must be stored under the `validate` directory under the `basepath`
* `train`: `true` or `false`. Run the application on training data. Note that the training data must be stored under the `train` directory under the `basepath`.

With the provided `settings.json` file, the application will run over all the validation examples.












# Dried Spice Classifier (Machine Learning Code)

* [Background](#background)
* [Project Overview](#project-overview)
* [Libraries and Dependencies](#libraries-and-dependencies)
* [Directories](#directories)
* [Techniques](#techniques)
* [Results](#results)

## Background
Dried and crushed Italian spices are difficult to differentiate. A spice mix-up would be disastrous for any recipe. An algorithm that can classify easily confused spices can prevent mishaps when following a recipe, especially for Italian recipes.

## Project Overview
Dried spice classifiers were built using three types of machine learning models (CNN, SVM, Random Forest), modeling techniques, and image augmentation. The optimal individual classifier was Random Forest; however, an ensemble model using the stacking technique was utilized as the final dried spice classifer (for experimentation purposes). 


## Libraries and Dependencies
Language
- Python 3.9.5

Environment
- Jupyter Lab/Notebook

Standard Modules
- glob
- os
- shutil
- pickle

Third Party Modules
- keras
- matplotlib
- numpy
- pandas
- PIL
- scipy
- seaborn
- scikit-learn
- tensorflow

## Directories
There are two dataset folders that contain datasets from raw to model-ready.

| Folder | Description |
| ----- | ----- |
| dataset | Contains images that are separated by labeled folders (e.g., dried_parsley, non_spice) |
| model_dataset | Contains images that are separated into a training, validation, and test folder. The images are further separated by labeled folders. |

The project workflow follows the programs listed below sequentially:
| Programs | Description |
| ----- | ----- |
| 1_Data_Collection.ipynb | dsfsd |
| 2_Data_Exploratopm.ipynb | Understand image data, analyze RGB relationships, and visualize with graphs |
| 3_Data_Augmentation.ipynb | Augment image data, specifically dried spice images |
| 4_1_Modeling_CNN.ipynb | Fit and evaulate data with multiple CNN models and tune parameters to achieve the best CNN model |
| 4_2_Modeling_SVM.ipynb | Fit and evaulate data with multiple SVM models and tune parameters to achieve the best SVM model |
| 4_3_Modeling_RF.ipynb | Fit and evaulate data with multiple Random Forest models and tune parameters to achieve the best Random Forest model |
| 4_4_Modeling_Stacking.ipynb | Take best SVM, CNN, and Random Forest models, create an ensemble model using stacking, and build a pipeline from image input to prediction; Outputs: cnn_model.h5, feat_extract.h5,  rf_model.pkl, svm_model.pkl |

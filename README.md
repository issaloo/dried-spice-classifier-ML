# Dried Spice Classifier (Machine Learning Side)

* [Background](#background)
* [Project Overview](#project-overview)
* [Libraries and Dependencies](#libraries-and-dependencies)
* [Directories](#directories)
* [Techniques](#techniques)
* [Results](#results)

## Background
Dried and crushed Italian spices are difficult to differentiate. A spice mix-up would be disastrous for any recipe. An algorithm that can classify easily confused spices can prevent mishaps when following a recipe, especially for Italian recipes.

## Project Overview
Dried spice classifiers were built using three types of machine learning models (CNN, SVM, Random Forest), modeling techniques, and image augmentation. The optimal individual classifier was Random Forest; however, an ensemble model (stacking) was utilized as the final dried spice classifer (for experimentation purposes). 


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

## Techniques
### Data Preparation
- Feature Extraction (using CNN)
- Image Augmentation (e.g., vertical flip, shear, brightness, scaling)
- 
### Modeling
- Test Train Split
- Random Forest
  - Hyperparameter Tuning
    - Grid Search
    - Class Weight
- Support Vector Machines (SVM)
  - Hyperparameter Tuning
    - Grid Search
    - Class Weight
- Convolutional Neural Networks (CNN)
  - Model Architecture Tuning
    - Max Pooling
    - Dropout
    - Activation
      - Softmax
      - ReLu
      - Leaky ReLu
  - Stacking
    - Equal Voting
    - Weighted Voting
- Model Evaluation
  - Confusion Matrix
  - Classification Report
    - Precision
    - Recall
    - F1-Score
  - Metrics for Comparing Multi-Class Models
    - Cohen's Kappa
    - Matthews Correlation Coefficient

## Results
The initial image dataset was imbalanced 470 non-spice images and 100 images for each spice. Spice images were split 70%-15%-15% for training, validation, and test and augmented, while non-spice images was set to 15 images for validation and test set - leaving a pool of 440 images for training. A generator was used to streamline data augmentation and train/evaluate a CNN model with a batch size of 32 and an epoch of 75. The top layer of the trained CNN model was dropped in order to be used as a feature extractor for SVM and Random Forest. After tuning hyperparameters for SVM, Random Forest, and CNN (not the feature extractor), an ensemble model was created that combined all three models (i.e., stacking). Using the ensemble model with CNN having twice the voting power, the test data achieved these accuracy results:

|       | Precision | Recall | F1-Score | Support |
| ----- | ----- | ----- | ----- | ----- |
| Dried Basil | 0.18 | 0.13 | 0.15 | 15 |
| Dried Oregano | 0.20 | 0.13 | 0.16 | 15 |
| Dried Parsley | 0.53 | 0.60 | 0.56 | 15 |
| Dried Thyme | 0.28 | 0.33 | 0.30 | 15 |
| Non-Spice | 0.68 | 0.87 | 0.76 | 15 |
| **Accuracy** | | | | |
| Macro Avg| 0.37 | 0.41 | 0.39 | 75 |

Comparing to a naive model of 0.25 recall (i.e., random guessing) as a baseline, the macro average recall exceed that value. Next steps to improving the model may include expanding the dried spice dataset and testing more models (e.g., ResNet, GoogLeNet).

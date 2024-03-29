# Dog-vs-Cat-Classification-with-Transfer-Learning-using-VGG16
Implementation of binary classification with **Dog and Cat images** using **VGG16** architecture and **Transfer Learning** techniques.

## About The Project

* This project focuses on binary image classification, distinguishing between images of dogs and cats. It leverages the power of the VGG16 architecture and Transfer Learning techniques to achieve highly accurate classification results.
* The pre-trained weights from the **ImageNet** dataset, which includes a wide range of object categories, are used to enhance the model's ability to recognize and classify dog and cat images.
* This repository provides a comprehensive implementation of the classification process and serves as a valuable resource for exploring the world of Transfer Learning with VGG16.

## Technology Used
* Python
* Numpy
* Pandas
* Matplotlib
* Scikit-learn
* Keras
* Tensorflow

## Dataset Used
  - This project uses collections of dog and cat images obtained from the Asirra (Animal Species Image Recognition for Restricting Access) dataset, which is available on [Dogs vs. Cats | Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats).
  - The amount of data used in this project:
    - Training data:
      - Dog: 500 images
      - Cat: 500 images
    - Testing data
      - Dog: 100 images
      - Cat: 100 images
  - Sample Images
    - Cat
      1 | 2 | 3 | 4 | 5
      :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
      ![Cat 1](data_dogvscat/data/train/cat.1.jpg) | ![Cat 2](data_dogvscat/data/train/cat.10.jpg) | ![Cat 3](data_dogvscat/data/train/cat.100.jpg) | ![Cat 4](data_dogvscat/data/train/cat.200.jpg) | ![Cat 5](data_dogvscat/data/train/cat.300.jpg)
    - Dog
      1 | 2 | 3 | 4 | 5
      :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
      ![Dog 1](data_dogvscat/data/train/dog.1.jpg) | ![Dog 2](data_dogvscat/data/train/dog.10.jpg) | ![Dog 3](data_dogvscat/data/train/dog.100.jpg) |  ![Dog 4](data_dogvscat/data/train/dog.200.jpg) |  ![Dog 5](data_dogvscat/data/train/dog.300.jpg)

## Workflow
- Data Preparation
- Label Encoding
- Data Preprocessing (Data Augmentation)
- Data Splitting
- Model Building
- Model Training
- Model Testing & Evaluation

## Algorithms/ Methods
* This project applies **Transfer Learning** methods by utilizing the **VGG16 architecture** with pre-trained weights sourced from **ImageNet**, encompassing around 1000 object categories.
* The training process uses a **fine-tuned method**, which allows all of the layers to update their weights during the training process.
* **Parameters:**
  * Epoch: 200
  * Batch = 10 (100 steps per epoch, because the number of training data is 1000)
  * Loss/ Cost Function = Binary Cross Entropy
  * Optimizer = Mini Batch Gradient Descent
  * Learning rate = 0.00001
  * Momentum = 0.0
  * Metrics = Accuracy

## Model Evaluation
### Graph of Epoch & Accuracy
![images/epoch%20&%20accuracy.png](images/epoch%20&%20accuracy.png)

### Confusion Matrix
|   | Actual Positive | Actual Negative |
|---|-----------------|-----------------|
| Predicted Positive |       93        |        7        |
| Predicted Negative |        5        |       95        |

### Classification Report
|    | Precision | Recall | F1-Score | Support |
|----|-----------|--------|----------|---------|
| Cat |   0.95    |  0.93  |   0.94   |   100   |
| Dog |   0.93    |  0.95  |   0.94   |   100   |
|----|-----------|--------|----------|---------|
|Accuracy|          |         |   0.94   |   200   |
|Macro Avg|  0.94   |  0.94  |   0.94   |   200   |
|Weighted Avg|  0.94  |  0.94  |   0.94   |   200   |

## Contributors
* [Linggar Maretva Cendani](https://github.com/LinggarM) - [linggarmc@gmail.com](mailto:linggarmc@gmail.com)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
- Asirra (Animal Species Image Recognition for Restricting Access) dataset: [Dogs vs. Cats | Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats).

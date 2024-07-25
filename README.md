

# Leaf Classification with Convolutional Neural Networks (CNN)

## Introduction

This project aims to classify images of leaves into different species using a Convolutional Neural Network (CNN). The project involves data preprocessing, building and training a CNN model, and evaluating its performance. This README provides an overview of the project structure, datasets, and the steps involved in developing the model.

## Project Structure

```
Leaf-Classification-CNN/
├── data/
├   ├── Folio/
│   ├── train/
│   ├── test/
│   └── val/
├── notebooks/
│   └── Leaf_Classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

## Datasets

The  Folio dataset used for this project contains images of leaves from various plant species. Each image is labeled with the corresponding species. The dataset is divided into training, validation, and test sets. You can download the dataset from [Kaggle](https://www.kaggle.com) or any other relevant source.

## Requirements

The project requires the following libraries and frameworks:

- PyTorch 
- NumPy
- pandas
- Matplotlib

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

The data preprocessing steps include:

1. **Resizing Images**: All images are resized to 128x128 pixels.
2. **Normalization**: Pixel values are normalized to the range [0, 1].
3. **Splitting Data**: The dataset is split into training, validation, and test sets.

The preprocessing script is located in `src/data_preprocessing.py`.

## Model Architecture

The CNN model architecture, wich is based on the resnet-50 archtecture, includes:

- Convolutional layers for feature extraction
- Pooling layers for down-sampling
- Fully connected layers for classification
- BatchNormalization for stability
- Dropout layers for regularization

The model is defined in `src/model.py`.

## Training the Model

The model is trained using the training dataset and validated using the validation dataset. The training script includes:

- Data augmentation techniques to improve model robustness
- Early stopping to prevent overfitting
- Hyperparameter tuning

You can find the training script in `src/train.py`.

## Evaluating the Model

The model is evaluated on the test dataset using various metrics such as accuracy, precision, recall, and F1-score. The evaluation script is located in `src/evaluate.py`.

## Results

The results of the model, including performance metrics and visualizations of training and validation loss/accuracy, are documented in `notebooks/Leaf_Classification.ipynb`.

## Conclusion

This project demonstrates the use of Convolutional Neural Networks for image classification. It covers data preprocessing, model building, training, and evaluation, providing a comprehensive overview of the workflow involved in developing a CNN-based image classification model.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Leaf-Classification-CNN.git
cd Leaf-Classification-CNN
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the data preprocessing script:

```bash
python src/data_preprocessing.py
```

4. Train the model:

```bash
python src/train.py
```

5. Evaluate the model:

```bash
python src/evaluate.py
```

## Acknowledgements

- The dataset providers
- TensorFlow, Keras, and PyTorch communities

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

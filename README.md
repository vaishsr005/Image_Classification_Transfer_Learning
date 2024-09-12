# DL - Image Classification - Transfer_Learning

# Binary Classification of Ants and Bees Using Transfer Learning with MobileNetV2

This project demonstrates a binary classification task to distinguish between ants and bees using a deep learning model. The approach leverages **transfer learning** with the pre-trained **MobileNetV2** model from TensorFlow Hub, which is used as a feature extractor.

## Dataset Preprocessing

1. **Data Loading**: The dataset consists of images of ants and bees, which are loaded from a compressed ZIP file.
2. **Resizing**: All images are resized to a standard resolution of 224x224 pixels to match the input dimensions required by the pre-trained model.
3. **RGB Conversion**: The images are converted to RGB format for consistency.
4. **Labeling**: Images are labeled as either `ant` (0) or `bee` (1) based on the file names.
5. **Normalization**: The image data is converted into **NumPy arrays** and normalized by dividing pixel values by 255 to scale them between 0 and 1, which improves model performance.

## Train-Test Split

- The dataset is split into training and test sets with a 90-10 ratio. The training set consists of 354 images, while the test set contains 40 images.
- Both sets are scaled to ensure consistent input for the neural network.

## Transfer Learning with MobileNetV2

- The pre-trained **MobileNetV2** model, available on TensorFlow Hub, is utilized for transfer learning. MobileNetV2 is efficient for image classification due to its smaller size and fewer parameters, making it suitable for edge devices.
- The model is loaded using `KerasLayer` from TensorFlow Hub. The `trainable=False` argument ensures that the pre-trained model's weights are frozen, meaning only the newly added layers are trained.
- The pre-trained model outputs feature vectors of size 1280, representing high-level features of the input images.

## Building the Model

The custom **Sequential** model consists of the following layers:
- **Input Layer**: Handles images of size 224x224 with 3 color channels (RGB).
- **Lambda Layer**: Applies the pre-trained **MobileNetV2** model to the input data, extracting feature vectors.
- **Dropout Layer**: A dropout rate of 0.5 is used to prevent overfitting by randomly setting 50% of the units to 0 during training.
- **Dense Layer**: A dense layer with 2 output units for binary classification (`ants` or `bees`), where the number of neurons matches the number of classes.

## Model Compilation and Training

- The model is compiled with the **Adam optimizer**, which adjusts learning rates for faster convergence.
- **Loss Function**: The **SparseCategoricalCrossentropy** loss function is used for multi-class classification.
- **Early Stopping**: An early stopping callback is used to monitor validation loss and halt training if there is no improvement after 3 epochs, preventing overfitting.
- The model is trained on the training set, with validation performed on the test set. The training is set to run for 20 epochs, but it stops early based on the early stopping condition.

**Result**
------------
- This model has a predictive accuracy of 92.5%, Precision is 92.69% and F1 Score is 0.9253.

**Libraries and Tools**
-----------------------
- **TensorFlow/Keras**: For building and training the neural network model.
- **OpenCV**: For image processing tasks.
- **NumPy**: For numerical computations and handling image data.
- **Matplotlib**: For visualizing the images and results.

**Conclusion**
--------------
By leveraging the power of neural networks and image processing libraries, this project aims to deliver an effective solution for classifying images of ants and bees. The techniques demonstrated in this workbook can be extended to other image classification tasks, making it a valuable resource for similar projects.

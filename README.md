# DL - Image Classification - Transfer_Learning

**Overview**
------------
This Python project is focused on a machine learning project that involves classifying images of ants and bees. The primary objective is to develop a neural network model capable of accurately distinguishing between images of ants and bees.

**Approach**
---------------
- **Printing the Name of Images**: Initial step to ensure the correct images are loaded.
- **Importing Dependencies**: Importing necessary libraries and packages for image processing and machine learning.
- **Displaying Images**: Visualizing images of ants and bees to understand the dataset.
- **Resizing Images**: Standardizing the size of all images to ensure consistency in the dataset.
- **Creating Labels**: Assigning labels to the images where ants are labeled as 0 and bees as 1.
- **Converting Images to Numpy Arrays**: Preparing the images for model training by converting them into numpy arrays.
- **Train-Test Split**: Dividing the dataset into training and test sets, with 354 images for training and 40 images for testing.
- **Building the Neural Network**: Developing a neural network model to classify the images.
- **Validation Data**: Further splitting the training data into training and validation sets for model evaluation.
- **Predictive System**: Implementing a predictive system to classify new images based on the trained model.

**Result**
------------
- Currently, this model has a predictive accuracy of 72.5%.

**Libraries and Tools**
-----------------------
- **TensorFlow/Keras**: For building and training the neural network model.
- **OpenCV**: For image processing tasks.
- **NumPy**: For numerical computations and handling image data.
- **Matplotlib**: For visualizing the images and results.

**Conclusion**
--------------
By leveraging the power of neural networks and image processing libraries, this project aims to deliver an effective solution for classifying images of ants and bees. The techniques demonstrated in this workbook can be extended to other image classification tasks, making it a valuable resource for similar projects.

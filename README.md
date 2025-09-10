# Galaxy Zoo Challenge: Galaxy Detection and Classification

This repository contains the code and solution for the Kaggle Galaxy Zoo challenge. The goal of this project is to classify galaxies based on their morphology using images from the Galaxy Zoo 2 project.

-----

## 📖 Dataset

The dataset for this project is from the **Kagle Galaxy Zoo challenge**. It consists of a large number of JPG images of galaxies and a solution file with probability distributions for 37 different morphological categories.

The dataset includes:

  * **images\_training**: 61,578 JPG images of galaxies for training.
  * **solutions\_training**: Probability distributions for the classifications for each of the training images.
  * **images\_test**: 79,975 JPG images of galaxies for testing.

-----

## 📂 Project Structure

This repository contains two main Jupyter Notebooks:

1.  **`galaxy-detection-all-questions (1).ipynb`**: This notebook contains the complete workflow for solving the Galaxy Zoo challenge. It includes:

      * Data loading and preprocessing.
      * Building a Convolutional Neural Network (CNN) using **Keras** with a **TensorFlow** backend.
      * Training the model on the galaxy images.
      * Evaluating the model's performance.

2.  **`Final_of_Galaxy_Detection_v1_0.ipynb`**: This notebook is a more detailed and descriptive version of the project. It provides:

      * A comprehensive description of the dataset and the morphological questions from the Galaxy Zoo 2 project.
      * Code for setting up the environment and visualizing the galaxy images.
      * A combined and polished solution, referencing the work from the other notebook.

-----

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python (3.6 or higher)**: The code is written in Python.
  * **Pip**: Python's package installer, which usually comes with Python.
  * **Git**: A version control system to clone the repository.
  * **(Optional but Recommended) NVIDIA GPU with CUDA**: For training the neural network, a GPU is highly recommended to speed up the process.

#### Key Libraries

This project relies on several core Python libraries for machine learning and data science. You can install them all using the `requirements.txt` file.

  * **TensorFlow & Keras**: The backbone of our project for building and training the deep learning model.
  * **NumPy**: Used for efficient numerical operations and handling the image data as arrays.
  * **Pandas**: Essential for reading and manipulating the tabular data from the solution files.
  * **Matplotlib**: Used for visualizing the galaxy images and plotting the model's training history.
  * **Scikit-learn**: Utilized for splitting the data into training and validation sets.
  * **Pillow (PIL)**: Needed for loading and processing the JPG image files.

### Installation

1.  Clone the repository:
    ```bash
    git clone 
    ```
2.  Navigate to the project directory:
    ```bash
    cd galaxy-detection
    ```
3.  Install the required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  Download the dataset from the [Kaggle Galaxy Zoo challenge](https://www.google.com/search?q=https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) and place the files in a designated `data` directory within the project folder.

-----

## 🔬 Methodology

The project follows these steps to classify the galaxies:

1.  **Data Loading and Preprocessing**: The galaxy images are loaded, and the corresponding probability distributions are read from the solution file. The images are preprocessed to be suitable for input to the neural network.
2.  **Model Building**: A Convolutional Neural Network (CNN) is built using Keras. The model architecture is designed to learn features from the galaxy images and predict the morphological classifications.
3.  **Training**: The CNN is trained on the training dataset.
4.  **Evaluation**: The performance of the trained model is evaluated using appropriate metrics to determine its accuracy in classifying the galaxies.

-----

## 📊 Results

The results and performance metrics of the model can be found in the `galaxy-detection-all-questions (1).ipynb` notebook. The notebook includes visualizations of the training process and the final accuracy of the model.

-----

## 🤝 Contributing

Contributions are welcome\! If you have any suggestions or improvements, please feel free to create a pull request or open an issue.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

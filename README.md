# Pulmonary Embolism Detection via Image Clustering

## Overview

This project aims to detect pulmonary embolism from medical images (specifically DICOM images) using an unsupervised learning approach. The goal is to cluster the medical images based on their features and then use the resulting clusters to identify if a patient has a pulmonary embolism or not. The process involves several steps, including data preprocessing, feature extraction, image clustering, and visualization of results.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Extraction](#feature-extraction)
4. [Image Clustering](#image-clustering)
5. [Evaluation and Visualization](#evaluation-and-visualization)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Project Setup

### Dataset

The dataset consists of DICOM images representing different medical scans, including those potentially showing pulmonary embolism. For the clustering algorithm to work, you will need a directory of these DICOM images.

The images are located in a folder structure, with each subfolder containing DICOM files for specific patients or scans.

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or Google Colab (for running the code)
- Required libraries (see below)

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/pulmonary-embolism-clustering.git
cd pulmonary-embolism-clustering
```

2. Create a Python virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the necessary Python libraries for the project. Below are the required libraries:

```txt
numpy
matplotlib
pydicom
scikit-image
scikit-learn
opencv-python
```

4. If you're using Google Colab, you can directly upload the images using the code provided below in the "Usage" section.

## Data Preprocessing

The data preprocessing step involves reading DICOM images, converting them to a uniform format, and applying any necessary transformations.

### Preprocessing Steps

1. **Reading DICOM Files**: The DICOM images are read using the `pydicom` library. This library allows extraction of pixel data and metadata from DICOM files.
2. **Rescaling and Normalization**: Images are rescaled to ensure uniformity in size and pixel intensity. Conversion of image data to unsigned 8-bit integers is performed using the `img_as_ubyte` function from the `skimage` library.
3. **Noise Reduction**: Various image enhancement techniques (e.g., Gaussian smoothing, median filtering) are applied to reduce noise.

Example code snippet for loading and preprocessing images:

```python
import pydicom
from skimage import img_as_ubyte
import numpy as np

def preprocess_image(dicom_image):
    # Convert the DICOM image's pixel data to unsigned 8-bit integer
    image_uint = img_as_ubyte(dicom_image.pixel_array)
    return image_uint
```

## Feature Extraction

Feature extraction involves identifying relevant patterns in the images that can be used for clustering. Common features for medical image analysis include pixel intensities, texture patterns, and other structural features.

### Feature Extraction Steps

1. **Pixel Intensity**: Flatten the image and extract raw pixel intensity values.
2. **Texture Features**: Use methods like the **Gray Level Co-occurrence Matrix (GLCM)** and **Local Binary Patterns (LBP)** to capture textural features from the image.

Example code for extracting features:

```python
from skimage.feature import greycomatrix, local_binary_pattern

def extract_features(image):
    # Extract texture features using GLCM
    glcm = greycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_features = glcm.flatten()

    # Extract texture features using Local Binary Pattern (LBP)
    lbp_features = local_binary_pattern(image, P=8, R=1, method='uniform').flatten()

    # Combine all features into a single feature vector
    features = np.concatenate([image.flatten(), glcm_features, lbp_features])
    return features
```

## Image Clustering

We apply unsupervised clustering methods to group the images into clusters based on the features extracted.

### Clustering Algorithms

We experiment with different clustering techniques such as:

- **K-means**: Partitions images into a fixed number of clusters based on their feature similarity.
- **Hierarchical Clustering**: Builds a hierarchy of clusters based on similarity.
- **DBSCAN**: Density-based clustering algorithm that can handle clusters of arbitrary shapes.

Example code for clustering using K-means:

```python
from sklearn.cluster import KMeans

# Assume `feature_matrix` is a NumPy array of feature vectors
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(feature_matrix)
```

## Evaluation and Visualization

Evaluation of clustering algorithms can be challenging in unsupervised settings. However, the clusters can be visually inspected by using dimensionality reduction techniques such as PCA (Principal Component Analysis) or t-SNE.

### Visualization of Clusters

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_matrix)

# Plot the clusters
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
plt.title("Clustered DICOM Images")
plt.show()
```

## Usage

### Running the Code

1. Place your DICOM files in the `my_images/` folder or upload them using Google Colab.
2. Run the Jupyter notebook or Python script (`main.py`) to preprocess the images, extract features, apply clustering, and visualize the results.

Example of using Google Colab for uploading files:

```python
from google.colab import files
import zipfile

# Upload the zip file containing DICOM images
uploaded = files.upload()

# Extract the contents
with zipfile.ZipFile('your_file.zip', 'r') as zip_ref:
    zip_ref.extractall("/content/my_images/")
```

### Running on Google Colab

1. Upload your DICOM image zip file.
2. Install the dependencies using `!pip install -r requirements.txt`.
3. Run the cells in the notebook to preprocess the data and perform clustering.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, create a new branch, and submit a pull request.

### How to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

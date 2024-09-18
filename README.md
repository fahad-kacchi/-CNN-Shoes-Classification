Here is an updated README.md with a note about checking the file paths:

---

# Shoe Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of shoes into three categories: Slippers, Sandals, and Boots.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview

The project uses a deep learning approach to classify shoe images into three categories: Slippers, Sandals, and Boots. The model is built using Keras with TensorFlow as the backend.

## Model Architecture

The CNN architecture consists of:
1. **Convolutional Layers**: Three convolutional layers with 32, 64, and 128 filters, respectively, each followed by MaxPooling layers.
2. **Flattening Layer**: Converts the 2D matrix data into a 1D vector.
3. **Fully Connected Layers**: One hidden layer with 512 units using ReLU activation, followed by an output layer with 3 neurons for the three shoe categories (softmax activation).

### Model Summary:

```
Total params: 44,397,635 (169.36 MB)
Trainable params: 44,397,635 (169.36 MB)
Non-trainable params: 0 (0.00 B)
```

## Data Preprocessing

The dataset is divided into training and validation sets. Image augmentation techniques such as rotation, zoom, and shift are applied to improve model generalization.

**Key Augmentation Techniques:**
- Rotation: 20 degrees
- Zoom: 20%
- Width and Height shift: 20%
- Horizontal Flip

## Training

The model is compiled using the Adam optimizer and categorical crossentropy loss function. The model is trained for 20 epochs with the training and validation data.

```python
classifier.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
```

## Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/your-username/shoe-classification-cnn.git
   cd shoe-classification-cnn
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is in the proper structure:
   - `train/` directory with subfolders for each shoe class (Slippers, Sandals, Boots)
   - `test/` directory with subfolders for each shoe class

4. **Note:** Before running the code, ensure that all file paths (such as training, validation, and test data) are correctly set based on your local environment.

## Usage

To train the model:

```python
result = classifier.fit(training_set, epochs=20, validation_data=validation_set)
```

To make predictions:

```python
def model_output(path):
    raw_img = image.load_img(path, target_size=(224, 224))
    raw_img = image.img_to_array(raw_img) / 255
    raw_img = np.expand_dims(raw_img, axis=0)
    predictions = reload_model.predict(raw_img)[0]
    class_indices = training_set.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_class = class_labels[np.argmax(predictions)]
    plt.imshow(cv2.imread(path))
    plt.axis('off')
    plt.show()
    print('Predicted Label:', predicted_class)
```

Make sure to **check and update all file paths** (such as the image dataset path) before running the code, especially for loading data and saving models.

## Results

After training for 20 epochs, the model achieved a training accuracy of **87.6%** and a validation accuracy of **81.9%**.

## License

This project is licensed under the MIT License.

---

Let me know if you need further changes!

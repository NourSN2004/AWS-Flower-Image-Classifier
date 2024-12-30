# AWS-Flower-Image-Classifier

This repository contains an **`.ipynb`** Jupyter Notebook designed for **Google Colab** to train and deploy an image classification model using **PyTorch**. You can follow the step-by-step instructions to load data, train a custom classifier on a pre-trained architecture (like VGG16), and make predictions on new images.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Getting Started in Colab](#getting-started-in-colab)  
4. [Notebook Structure](#notebook-structure)  
5. [Training the Model](#training-the-model)  
6. [Making Predictions](#making-predictions)  
7. [Results](#results)  
8. [Troubleshooting & Tips](#troubleshooting--tips)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## 1. Project Overview

This project showcases how to:
- **Load and preprocess** an image dataset in Colab.  
- **Train** a deep neural network with **PyTorch** using a **pre-trained architecture** (e.g., VGG16).  
- **Optimize hyperparameters** (learning rate, epochs, etc.) to maximize validation accuracy.  
- **Predict** the top-K most likely classes for new images, using a custom command within the notebook.  

---

## 2. Key Features

1. **Google Colab-Friendly**:  
   - No need to install anything locally.  
   - Access a GPU runtime for accelerated training.  

2. **Pre-trained Networks**:  
   - Uses well-known architectures (e.g., VGG16, ResNet) from `torchvision.models` as a starting point.  

3. **Data Augmentation**:  
   - Random rotations, flips, and normalization are applied to improve model generalization.  

4. **Hyperparameter Tuning**:  
   - Experiment with different learning rates, batch sizes, hidden units, and optimizers.  

5. **Inference & Visualization**:  
   - Predict classes on unseen images.  
   - Display top-K predictions with probability scores.

---

## 3. Getting Started in Colab

### 3.1. Open the Notebook in Google Colab

1. **Option A: From GitHub**  
   - In Google Colab, go to **File > Open notebook**.  
   - Click the **GitHub** tab and paste your repository’s URL.  
   - Select the `.ipynb` file and click **Open**.

2. **Option B: Upload the Notebook**  
   - Download the `.ipynb` file from GitHub to your local machine.  
   - In Colab, go to **File > Upload notebook**.  
   - Select the `.ipynb` file and upload.

### 3.2. Check Runtime Settings

1. Click **Runtime > Change runtime type**.  
2. Under **Hardware accelerator**, select **GPU**.  
3. Click **Save**.

> **Note**: Using a GPU can significantly reduce training time.

---

## 4. Notebook Structure

A typical structure in the `.ipynb` might include:

1. **Imports & Environment Setup**  
   - Install/upgrade any required libraries (e.g., `torch`, `torchvision`, `pillow`, etc.).  
2. **Mount Google Drive (Optional)**  
   - Useful if your dataset or checkpoint files are in Google Drive.  
3. **Data Loading & Preprocessing**  
   - Data augmentation and normalization transforms for training, validation, and testing sets.  
4. **Model Definition**  
   - Load a pre-trained model (e.g., `torchvision.models.vgg16`).  
   - Replace the classifier with a custom architecture.  
5. **Training**  
   - Set hyperparameters (learning rate, epochs, etc.).  
   - Use GPU if available.  
   - Track training/validation loss and accuracy over epochs.  
6. **Validation & Testing**  
   - Evaluate the model on the validation and/or test dataset.  
7. **Saving & Loading Checkpoints**  
   - Save model state dictionary and classifier hyperparameters.  
   - Load them later for inference.  
8. **Inference**  
   - Predict top-K classes for a new image.  
   - (Optionally) map class indices to category names.  
9. **Visualization**  
   - Display predicted classes with probabilities.

---

## 5. Training the Model

Inside the notebook, you’ll find cells dedicated to training. Here’s the general workflow:

1. **Load Data**  
   - Update any dataset paths if necessary (e.g., Colab paths or Google Drive paths).  
2. **Apply Data Transforms**  
   - Define `transforms.Compose([ ... ])` for training, validation, and testing sets.  
3. **Build the DataLoader**  
   - Use `DataLoader` with appropriate batch sizes and transformations.  
4. **Instantiate the Model**  
   - `model = models.vgg16(pretrained=True)` (or any other supported architecture).  
   - Replace the classifier with a custom feedforward network.  
5. **Define Loss & Optimizer**  
   - `criterion = nn.NLLLoss()` (for example).  
   - `optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)` (or your choice).  
6. **Train**  
   - Loop through epochs, train on the training set, and validate on the validation set.  
   - Print training/validation loss and accuracy per epoch.  

**Run each cell in order**—Colab will store the variables in memory as you go.

---

## 6. Making Predictions

After training:

1. **Load Checkpoint (if needed)**  
   - You can load the model from a saved checkpoint if you want to avoid retraining.  
   - For example:  
     ```python
     checkpoint = torch.load('model_checkpoint.pth')
     model.load_state_dict(checkpoint['state_dict'])
     ```
2. **Predict on a Single Image**  
   - You might have a function like `predict(image_path, model, topk=5)` in a cell.  
   - This function transforms the image, runs a forward pass, and returns the top-K classes with probabilities.  
3. **Optionally Map Classes to Category Names**  
   - If you have `cat_to_name.json`, load it and map numeric categories to actual names (e.g., `rose`, `daisy`, etc.).  

---

## 7. Results

During training, you should see:

- **Validation Accuracy**: How well your model is performing during training.  
- **Validation Loss Trend**: Whether the model is converging over epochs.  


## 8. Troubleshooting & Tips

- **Runtime Disconnects**:  
  - Colab sessions may disconnect if idle for too long. Consider shorter training or regularly interacting with the notebook to keep it alive.  
- **Out of Memory**:  
  - Large batch sizes may cause GPU memory errors. Reduce `batch_size` if needed.  
- **Check Data Paths**:  
  - If you’re mounting Google Drive, ensure the path to your dataset is correct (e.g., `/content/drive/MyDrive/flowers/`).  
- **Installing Additional Packages**:  
  - If you need extra libraries, install them in a cell via `!pip install <package>`.

---

## 9. Contributing

1. **Fork** the repository on GitHub.  
2. **Create** a new branch for your feature (`git checkout -b feature/your-feature`).  
3. **Commit** your changes (`git commit -m 'Add feature'`).  
4. **Push** to your branch (`git push origin feature/your-feature`).  
5. **Open** a Pull Request on GitHub.

---

## 10. License

This project is licensed under the **[MIT License](LICENSE)**. Feel free to use, modify, and distribute as permitted.

---

### Acknowledgments

- **AWS AI & ML Nanodegree** for the training framework and project inspiration.  
- **PyTorch Community** for extensive documentation and resources.  
- **Google Colab** for providing a free, easy-to-use environment for GPU-accelerated model training.

---

**Enjoy classifying images in Colab!**



Here is a clean, GitHub-ready **README.md** you can paste directly into your repository.

---

# 👁️ Eye Disease Detection using ResNet50 with Explainable AI (Grad-CAM)

This project implements an **eye disease classification system** using **transfer learning with ResNet50** and provides **model interpretability** via **Grad-CAM visualizations**. The goal is not only to achieve strong predictive performance but also to explain *why* the model makes its decisions — a critical requirement in medical AI applications.

---

## 📌 Project Overview

Medical image classification models often behave as black boxes. In healthcare, predictions without explanations are risky. This project:

✅ Trains a deep learning classifier for ocular diseases
✅ Uses a **pre-trained ResNet50** model (transfer learning)
✅ Evaluates performance using **K-Fold Cross-Validation**
✅ Applies **Grad-CAM** for visual explanations
✅ Demonstrates preprocessing & augmentation strategies

---

## 🧠 Model Architecture

* **Base Model:** ResNet50 (Pre-trained on ImageNet)
* **Technique:** Fine-tuning final layers
* **Framework:** PyTorch
* **Input Sizes Tested:**

  * 128 × 128
  * 224 × 224
  * 448 × 448

---

## 📂 Dataset

The model is designed for the **ODIR5K (Ocular Disease Intelligent Recognition)** dataset.

Typical dataset structure:

```
classified_images/
    train/
        Disease_1/
        Disease_2/
        ...
    test/
```

Each class folder contains retinal images corresponding to a disease category.

---

## ⚙️ Features Implemented

### ✔ Data Preprocessing

* Image resizing
* Tensor conversion
* Dataset loading using `ImageFolder`

### ✔ Training Strategy

* Transfer learning with ResNet50
* Cross-validation (K-Fold)
* Accuracy tracking

### ✔ Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report

### ✔ Explainable AI

* Grad-CAM heatmaps
* Visual explanation of predictions
* Highlighting disease-relevant regions

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install numpy pandas torch torchvision timm scikit-learn matplotlib pillow opencv-python kneed
```

---

## 🚀 Usage

### **1️⃣ Train the Model**

Update dataset paths inside the notebook/script and run:

```python
train_and_evaluate_resnet_kfold(dataset, n_epochs=10, device=device, k=5)
```

---

### **2️⃣ Run Grad-CAM Visualization**

```python
evaluate_with_gradcam(
    img_size=224,
    model_path="resnet50_model.pth",
    sample_image_path="sample.jpg"
)
```

This generates a heatmap showing **where the model is looking**.

---

## 🔬 Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping):

* Uses gradients flowing into the final convolution layer
* Produces a localization heatmap
* Highlights important regions influencing predictions

This helps validate whether the model focuses on **clinically meaningful features**.

---

## 📊 Example Outputs

* Disease classification predictions
* Confusion matrices
* Grad-CAM heatmaps over retinal images

---

## 🎯 Key Learnings

✔ Transfer learning works effectively for medical imaging
✔ Image resolution significantly affects performance
✔ Explainability improves trust in AI systems
✔ Grad-CAM helps detect model failure cases

---

## ⚠️ Important Notes

* Dataset paths are currently hardcoded (update before running)
* GPU recommended for training
* Grad-CAM requires trained model weights

---

## 🧩 Dependencies

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-Learn
* OpenCV
* Matplotlib
* Pillow

---

## 📜 License

This project is for **academic / research purposes**.
Verify dataset usage rights before distribution.

---

## 🙌 Acknowledgements

* ResNet Architecture – He et al.
* Grad-CAM – Selvaraju et al.
* ODIR5K Dataset



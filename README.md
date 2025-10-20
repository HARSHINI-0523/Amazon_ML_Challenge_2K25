# Smart Product Pricing Challenge 2025

This repository contains a complete machine learning solution for the **Smart Product Pricing Challenge**. The goal is to predict product prices using a multimodal approach that combines textual descriptions and product images.

## 🎯 The Challenge (What They Asked)

The core task was to build a regression model that accurately predicts the price of 75,000 e-commerce products.

### Key Requirements & Constraints:

* **Input Data**: Product information including `catalog_content` (text) and an `image_link`.
* **Target**: Predict the `price` for each product in the test set.
* **Evaluation Metric**: The model's performance is judged by the **Symmetric Mean Absolute Percentage Error (SMAPE)**. A lower SMAPE score is better.
* **Strict Prohibition**: Using any external methods to look up prices (e.g., web scraping, APIs) was strictly forbidden and would lead to disqualification. The model had to learn pricing patterns solely from the provided training data.

---

## 🚀 Our Solution (What We Did)

We developed a robust, multi-stage pipeline in a Google Colab notebook, leveraging both text and image data to create a powerful ensemble model. The workflow was designed to be resilient to common issues like memory crashes and runtime disconnects.

### Methodology Overview

Our solution followed a structured 7-stage workflow:

1.  **Data Setup & Prototyping**: Established a clean project structure and built a small, end-to-end pipeline on a 10% data sample for rapid debugging.
2.  **Text-Only Baseline Model**: Created a strong baseline using only `catalog_content` to set a benchmark score.
3.  **Image Feature Extraction**: Processed all 75,000 product images using a pre-trained deep learning model.
4.  **Fusion Modeling**: Combined text and image features to build a superior multimodal model.
5.  **Optimization & Ensembling**: Fine-tuned the model's hyperparameters and combined multiple models to maximize accuracy.
6.  **Final Prediction & Submission**: Trained the final models on the full dataset and generated the submission file.

### Stage-by-Stage Breakdown:

#### 1️⃣ **Text-Only Baseline Model (Stage 3)**

* **Technique**: We used a `TfidfVectorizer` to convert the cleaned product descriptions into a numerical matrix, considering both single words and two-word phrases (n-grams).
* **Model**: A `LightGBM` regressor was trained on these text features.
* **Result**: This provided a solid baseline **SMAPE of 62.02**. The goal from this point was to beat this score.

#### 2️⃣ **Image Feature Extraction (Stage 4)**

* **Technique**: A pre-trained **ResNet50** model was used to analyze each product image and convert it into a 2048-dimensional feature vector (embedding).
* **Challenge**: Processing 75,000 images on Google Colab presented major challenges with storage space and runtime limits.
* **Solution**: We implemented a **robust, memory-efficient batch-processing script**. This script automatically downloads images in small chunks (e.g., 500 at a time), extracts the features, saves the tiny `.npy` feature files to Google Drive, and then deletes the image chunk to free up space. This process is fully resumable, meaning if Colab crashed, we could restart the script, and it would automatically continue from where it left off without losing progress.

#### 3️⃣ **Fusion Model & Optimization (Stages 5 & 6)**

* **Fusion**: The text features (TF-IDF matrix) and image features (`.npy` vectors) were concatenated into a single, rich feature matrix.
* **Performance Boost**: Training a `LightGBM` model on this fused data immediately improved the score, dropping the **SMAPE to 61.47**. This confirmed that the visual information from images was valuable for predicting prices.
* **Tuning**: We used the `Optuna` library to automatically search for the best hyperparameters for our `LightGBM` model. To manage Colab's RAM limits, we ran a limited number of trials (10), which provided a significant performance boost without causing the session to crash.
* **Ensembling**: The final prediction was created by averaging the outputs of two different powerful models: our tuned **LightGBM** and an **XGBoost** model, both trained on the full text+image data. This ensemble technique helps to generalize better and reduce individual model errors.

#### 4️⃣ **Final Submission (Stage 7)**

The final, tuned ensemble models were trained on 100% of the training data. The same robust batch-processing pipeline was then used to generate features for the 75,000 test set images. Finally, predictions were made on the test set, and the `submission.csv` file was generated.

### 📁 Folder Structure

The project was organized with the following structure for clarity and reproducibility:

ML_Challenge_2025/

├── dataset/

│     ├── train.csv

│     ├── test.csv

│     └── temp_image_batch/  (temporary folder used during processing)

├── src/

│     └── utils.py

├── notebooks/

│     └── SMART PRODUCT PRICING CHALLENGE.ipynb

└── outputs/

├── image_features/       (contains 75k .npy files for train set)

├── test_image_features/  (contains 75k .npy files for test set)

└── submission.csv        (the final prediction file)

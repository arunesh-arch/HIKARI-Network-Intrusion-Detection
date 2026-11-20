
# 🛡️ HIKARI Network Intrusion Detection System

### Machine Learning-Based Network Traffic Classification

A machine learning-based Network Intrusion Detection System (NIDS) developed using Python and the HIKARI 2021 dataset. The project identifies network traffic as either **Benign** or **Attack** by training and comparing multiple machine learning classification algorithms.

The system automatically selects the best-performing model based on the F1-score and provides a simple interactive web interface using Streamlit for network traffic prediction.

---

## 📌 Project Overview

Network security is an important aspect of modern computing environments. Malicious network traffic can indicate unauthorized access, suspicious activities, or potential cyberattacks.

This project aims to build a machine learning-based intrusion detection system that analyzes network flow characteristics and classifies traffic into two categories:

- **Benign:** Normal network traffic.
- **Attack:** Potentially malicious network traffic.

The system uses multiple machine learning algorithms, evaluates their performance using classification metrics, and selects the most suitable model for prediction.

---

## ✨ Key Features

- Binary classification of network traffic (Benign vs Attack).
- Training and comparison of multiple machine learning algorithms.
- Automatic best-model selection using F1-score.
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- UID-based network traffic detection.
- CSV-based batch traffic prediction.
- Interactive Streamlit dashboard.
- Attack probability estimation.
- Downloadable prediction results.

---

## 🤖 Machine Learning Algorithms

The following classification algorithms are implemented and compared:

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification baseline |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble of decision trees |
| Extra Trees | Randomized tree-based ensemble |

### Model Selection

All algorithms are trained and evaluated using a validation dataset.

The model with the highest validation F1-score is automatically selected as the best model.

The selected model is then evaluated on an independent test dataset.

---

## 📊 Model Evaluation Metrics

The following performance metrics are used to evaluate the classification algorithms:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted attacks that are actual attacks |
| Recall | Proportion of actual attacks successfully detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to distinguish between classes |
| PR-AUC | Precision-recall performance, especially useful for imbalanced data |

**Note:** Since network traffic datasets can contain imbalanced classes, F1-score, Recall, and PR-AUC are particularly important when evaluating intrusion detection performance.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning algorithms |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Streamlit | Web application interface |

---

## 📂 Project Structure

```text
HIKARI-Network-Intrusion-Detection/
│
├── app.py
│   └── Streamlit application
│
├── train_model.py
│   └── Data preprocessing, model training,
│       evaluation and model selection
│
├── requirements.txt
│   └── Required Python libraries
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Excludes datasets and model files
│
└── Local Files (Not uploaded to GitHub)
    │
    ├── ALLFLOWMETER_HIKARI2021.csv
    ├── hikari_best_model.pkl
    └── model_comparison.csv
```

---

## 📁 Dataset Information

This project uses the **HIKARI 2021 network traffic dataset**.

The dataset contains network flow information and associated traffic labels used for supervised machine learning classification.

The dataset is intentionally not included in this GitHub repository.

Users must independently obtain the dataset and verify its applicable license and usage conditions.

### Dataset Setup

After obtaining the dataset, place it in the project directory using the following filename:

```text
ALLFLOWMETER_HIKARI2021.csv
```

The training script expects the dataset to be available in the same directory as `train_model.py`.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/HIKARI-Network-Intrusion-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd HIKARI-Network-Intrusion-Detection
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Before running the application, obtain the dataset and place it in the project directory.

Execute the training script:

```bash
python train_model.py
```

The script performs the following operations:

1. Loads the HIKARI dataset.
2. Removes unnecessary columns.
3. Handles missing and infinite values.
4. Removes duplicate records.
5. Separates input features and target labels.
6. Splits the dataset into training, validation, and testing sets.
7. Trains multiple machine learning algorithms.
8. Evaluates and compares model performance.
9. Selects the best-performing model using validation F1-score.
10. Evaluates the selected model on the test dataset.
11. Saves the selected model locally.

### Generated Files

After successful training, the following files are generated:

| File | Purpose |
|---|---|
| `hikari_best_model.pkl` | Selected trained model and feature information |
| `model_comparison.csv` | Performance comparison of algorithms |

These generated files remain local and are not included in the GitHub repository.

---

## 🌐 Running the Web Application

Once the model has been trained successfully, start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

Default local address:

```text
http://localhost:8501
```

---

## 🔍 Application Features

### 1. UID-Based Traffic Detection

Users can:

1. Enter a network flow UID.
2. Upload the HIKARI dataset containing that UID.
3. Search for the corresponding network flow.
4. Extract its network traffic features.
5. Generate a prediction using the selected machine learning model.

The result is displayed as:

- ✅ **BENIGN:** Traffic classified as normal.
- 🚨 **ATTACK:** Traffic classified as potentially malicious.

The application also displays the estimated attack probability when supported by the selected model.

**Note:** UID is used only to locate the corresponding network flow. Classification is performed using network traffic features.

### 2. CSV-Based Traffic Detection

Users can upload a HIKARI-compatible CSV file containing network flow features.

The application:

- Processes the uploaded records.
- Predicts the traffic category for each record.
- Displays the total number of records.
- Displays benign and attack traffic counts.
- Generates a traffic distribution chart.
- Provides downloadable prediction results.

---

## 🔄 System Workflow

```text
          HIKARI 2021 Dataset
                   |
                   v
          Data Preprocessing
                   |
                   v
          Train / Validation /
             Test Split
                   |
                   v
          Multiple ML Models
                   |
                   v
          Model Evaluation
                   |
                   v
          Best Model Selection
                   |
                   v
          Save Trained Model
                   |
                   v
          Streamlit Application
                   |
          +--------+--------+
          |                 |
          v                 v
       UID Search        CSV Upload
          |                 |
          v                 v
       Feature           Feature
       Extraction        Extraction
          |                 |
          v                 v
       Prediction        Prediction
          |                 |
          +--------+--------+
                   |
                   v
           Benign / Attack
```

---

## 🔒 Data and Privacy

- The original dataset is not distributed through this repository.
- Trained model files are not committed to GitHub.
- Dataset and generated model files are excluded through `.gitignore`.
- Users should independently verify dataset permissions before downloading or using it.

---

## 🚀 Future Improvements

Potential improvements include:

- Integration of additional machine learning algorithms.
- Hyperparameter optimization.
- Improved detection of minority attack classes.
- Real-time network traffic monitoring.
- More detailed attack classification.
- Model explainability using feature importance.
- Improved dashboard visualizations.
- Deployment to a cloud hosting platform.

---

## 🎯 Project Objective

The main objective of this project is to develop a machine learning-based network intrusion detection system that can distinguish benign and potentially malicious network traffic while providing an accessible interface for network traffic analysis.

---

Screenshots:

<img width="913" height="403" alt="image" src="https://github.com/user-attachments/assets/4c585758-3bb3-4045-bd8f-caade6ca60ff" />
Fig: Home Page


<img width="881" height="416" alt="image" src="https://github.com/user-attachments/assets/1e005aaa-64d9-47f1-a6a9-e2b1b7995ba2" /> 
Fig: Prediction of UID


<img width="943" height="385" alt="image" src="https://github.com/user-attachments/assets/4664803c-e946-4235-b3a3-fcc150de1e8c" />
Fig: Dataset Upload


<img width="919" height="274" alt="image" src="https://github.com/user-attachments/assets/f3e23a93-d34a-4624-904c-9d16634b056b" />
Fig: Dataset Preview


<img width="917" height="422" alt="image" src="https://github.com/user-attachments/assets/6c552027-b303-4858-a6f1-2807582fe494" />
Fig: Traffic Distribution & Prediction Results




## 👨‍💻 Author

**Developed as an AI / Machine Learning Project**

Project: HIKARI Network Intrusion Detection System

Technologies: Python | Machine Learning | Scikit-learn | Streamlit

---

## ⭐ Acknowledgements

This project uses the HIKARI 2021 dataset for network traffic classification research and educational purposes.


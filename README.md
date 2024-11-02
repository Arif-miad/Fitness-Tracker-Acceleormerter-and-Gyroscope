

<body>
<p align="center">
  <a href="mailto:arifmiahcse952@gmail.com"><img src="https://img.shields.io/badge/Email-arifmiah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/Arif-miad"><img src="https://img.shields.io/badge/GitHub-%40ArifMiah-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/arif-miah-8751bb217/"><img src="https://img.shields.io/badge/LinkedIn-Arif%20Miah-blue?style=flat-square&logo=linkedin"></a>

 
  
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801998246254-green?style=flat-square&logo=whatsapp">
  
</p>

<h1>Fitness Tracker Accelerometer and Gyroscope Dataset Analysis</h1>

This project utilizes accelerometer and gyroscope data collected from participants performing various exercises. With sensor readings along different axes and classifications based on exercise type and intensity, this dataset is ideal for analyzing motion patterns, building activity recognition models, and training machine learning algorithms for fitness and health tracking.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Data Dictionary](#data-dictionary)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Import Libraries](#1-import-libraries)
  - [2. Custom Models](#2-custom-models)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
```python
set_df=df[df["Set"]==1]
plt.plot(set_df["Accelerometer_y"])
```
![output](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f4.png)
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f4.png)
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f3.png)
  - [4. Remove Outliers](#4-remove-outliers)
```python
outlier_col=list(df.columns[:6])
df[outlier_col[:3]+["Label"]].boxplot(by="Label",figsize=(20,10),layout=(1,3))
df[outlier_col[3:6]+["Label"]].boxplot(by="Label",figsize=(20,10),layout=(1,3))
```
![output](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f5.png)
```python
df[outlier_col[:3]+["Label"]].plot.hist(by="Label",figsize=(20,10),layout=(3,3))
df[outlier_col[3:6]+["Label"]].plot.hist(by="Label",figsize=(20,10),layout=(3,3)
```
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f6.png)
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f7.png)
```python
label="bench"
for col in outlier_col:
    dataset=mark_outliers_iqr(df[df["Label"]==label],col)
    plot_binary_outliers(dataset,col,col+"_outlier",True)
```
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f8.png)
  - [5. Feature Engineering](#5-feature-engineering)
```python
subset=df_pca[df_pca["Set"]==35]
subset[["pca_1","pca_2","pca_3"]].plot()
```
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f9.png)
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f10.png)
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f11.png)
  - [6. Model Evaluation](#6-model-evaluation)
```python
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(projection="3d")
for c in df_cluster["Cluster"].unique():
    subset=df_cluster[df_cluster["Cluster"]==c]
    ax.scatter(subset["Accelerometer_x"],subset["Accelerometer_y"],subset["Accelerometer_z"],label=c)
plt.legend()
plt.show()
```
![](https://github.com/Arif-miad/Fitness-Tracker-Acceleormerter-and-Gyroscope/blob/main/f12.png)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)


## Dataset Overview

The dataset captures time-series sensor data from fitness trackers, covering accelerometer and gyroscope readings along the x, y, and z axes. Each record includes:
- **Exercise Type**: Label for the exercise or movement performed.
- **Exercise Intensity**: Category indicating the intensity level (e.g., heavy, medium).
- **Participant ID**: An identifier for each participant.

This data can be used to explore motion patterns, identify outliers, and develop machine learning models to classify exercises and intensity levels.

## Project Structure

```
.
├── README.md               # Project documentation
├── data                    # Folder for dataset files
│   └── 01_Data_Processed.csv
├── notebooks               # Jupyter Notebooks for analysis
│   └── analysis.ipynb
├── src                     # Source code for model training and evaluation
│   ├── custom_models.py    # Custom machine learning models
│   └── feature_engineering.py
└── requirements.txt        # Required libraries
```

## Data Dictionary

| Column            | Description |
|-------------------|-------------|
| `epoch (ms)`      | Timestamp in milliseconds, representing the exact time of data recording. |
| `Accelerometer_x` | X-axis acceleration value from the fitness tracker. |
| `Accelerometer_y` | Y-axis acceleration value from the fitness tracker. |
| `Accelerometer_z` | Z-axis acceleration value from the fitness tracker. |
| `Gyroscope_x`     | X-axis rotational velocity (gyroscope) reading. |
| `Gyroscope_y`     | Y-axis rotational velocity (gyroscope) reading. |
| `Gyroscope_z`     | Z-axis rotational velocity (gyroscope) reading. |
| `Participants`    | Identifier for the individual performing the exercise. |
| `Label`           | Type of exercise or movement (e.g., bench press, overhead press). |
| `Category`        | Intensity of the exercise (e.g., heavy, medium). |
| `Set`             | Set number or batch identifier for the recorded session. |

## Installation

To install the required packages, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

The project includes several steps to clean, preprocess, analyze, and model the data. Below is a summary of the process:

### 1. Import Libraries

The analysis begins by importing essential libraries like Pandas, Matplotlib, Seaborn, and Scikit-Learn, as well as custom scripts.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
```

### 2. Custom Models

We implement machine learning models, including:
- **KNeighborsClassifier**
- **LocalOutlierFactor** for anomaly detection
- **DecisionTreeClassifier**
- **GaussianNB**

### 3. Exploratory Data Analysis (EDA)

EDA includes visualizations and summary statistics to understand data distributions, relationships, and identify potential anomalies.

```python
# Example: Visualizing accelerometer x-axis data by exercise type
sns.lineplot(x='epoch (ms)', y='Accelerometer_x', hue='Label', data=df)
plt.title("Accelerometer X-Axis Over Time by Exercise Type")
plt.show()
```

### 4. Remove Outliers

Using **LocalOutlierFactor** to identify and remove outliers in the dataset. This helps improve the robustness of machine learning models by filtering out noise.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor()
outliers = lof.fit_predict(df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z']])
df = df[outliers == 1]  # Retain non-outliers
```

### 5. Feature Engineering

Generate new features based on existing data to improve model performance. For instance, creating composite features like the total acceleration across all three axes.

```python
df['total_acceleration'] = (df['Accelerometer_x']**2 + df['Accelerometer_y']**2 + df['Accelerometer_z']**2) ** 0.5
```

### 6. Model Evaluation

Evaluate models using metrics like accuracy, precision, recall, and F1-score. This helps determine the best model for activity recognition.

```python
from sklearn.metrics import classification_report

# Train and evaluate KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Results

The results from each model, along with evaluation metrics, are compared to determine the most accurate model for exercise recognition and intensity classification.

## Future Work

- **Improvement of Feature Engineering**: Explore additional composite features.
- **Incorporation of Deep Learning Models**: Test recurrent neural networks (RNNs) for better time-series modeling.
- **Real-time Activity Recognition**: Develop a pipeline for real-time prediction and feedback.

## License

This project is licensed under the MIT License.




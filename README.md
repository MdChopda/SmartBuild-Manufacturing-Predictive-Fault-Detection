# 🏭 SmartBuild Manufacturing: Predictive Fault Detection & Inventory Optimization

**Author:** Maulik Dilipbhai Chopda  
**Tech Stack:** Python, XGBoost, Pandas, Scikit-Learn

## 📖 Executive Summary
This project focuses on optimizing a manufacturing production line by implementing predictive analytics to solve two key business problems: automated material planning and early fault detection.

The analysis revealed a massive cost disparity in the production process:
* **Cost to produce a faulty unit:** 150 EUR (Machine + Material)
* **Cost to discard raw material:** 10 EUR
* **Impact:** By predicting errors before the machine runs, this model saves **140 EUR per detected error**.

## 📂 Project Structure
```text
├── data/                     # CSV datasets (Production_Log_01.csv, Machine_Settings_Log_01.csv)
├── docs/                     
│   └── SmartBuild_Report.pdf # Detailed project presentation and analysis
├── weight_prediction.py      # Q1: Regression Model (Material Prediction)
├── error_detection.py        # Q2: Classification Model (Fault Detection)
├── requirements.txt          # List of python dependencies
└── README.md                 # Project documentation

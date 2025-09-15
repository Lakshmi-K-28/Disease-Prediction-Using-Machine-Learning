# Disease-Prediction-Using-Machine-Learning
This project predicts diseases based on patient symptoms using multiple machine learning models such as Decision Tree, Random Forest, SVM, and Naive Bayes. It includes data preprocessing, class balancing with RandomOverSampler, model evaluation with Stratified K-Fold, and an ensemble approach for robust predictions.

Features
•	✔️ Data preprocessing with label encoding and RandomOverSampler for class balance
•	✔️ Model evaluation with Stratified K-Fold Cross-Validation
•	✔️ Trained classifiers: Decision Tree, Random Forest, Support Vector Classifier (SVC), Gaussian Naive Bayes
•	✔️ Confusion Matrix visualization for performance analysis
•	✔️ Ensemble prediction (majority voting) for robustness
•	✔️ User-friendly function to predict disease from given symptoms
Project Structure
📁 disease-prediction-ml
│── disease_prediction.ipynb   # Main notebook with implementation
│── improved_disease_dataset.csv # Dataset used
│── README.md                  # Documentation
│── requirements.txt           # Dependencies
Tech Stack
Programming Language: Python
Libraries: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, imbalanced-learn
Model Performance
Model	Accuracy (%)
Support Vector Classifier	60.53
Gaussian Naive Bayes	37.98
Random Forest	68.98
Ensemble Model	60.64
Example Prediction
Input Symptoms: skin_rash, fever, headache
{
  "Random Forest Prediction": "Peptic ulcer disease",
  "Naive Bayes Prediction": "Impetigo",
  "SVM Prediction": "Peptic ulcer disease",
  "Final Prediction": "Peptic ulcer disease"
}
Installation & Usage
1.	Clone the repository:
   git clone https://github.com/your-username/disease-prediction-ml.git
   cd disease-prediction-ml
2.	Install dependencies:
   pip install -r requirements.txt
3.	Run the notebook or script:
   jupyter notebook disease_prediction.ipynb
Future Improvements
•	➡️ Integrate with a web application (Flask/Django/Streamlit) for interactive use
•	➡️ Improve accuracy with advanced models like XGBoost, LightGBM, Neural Networks
•	➡️ Use real-world medical datasets for more reliable predictions

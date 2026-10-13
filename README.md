\# AI-Powered Manufacturing Quality Intelligence System



\## Overview



The AI-Powered Manufacturing Quality Intelligence System is a machine learning application designed to support manufacturing operations through data-driven predictions. It uses machine sensor readings and operational parameters to predict machine failures, identify anomalous operating conditions, and assess product quality.



The project combines data analysis, preprocessing, machine learning, experiment tracking, API development, an interactive dashboard, and containerization.



\## Objectives



\- Predict potential machine failures for predictive maintenance.

\- Identify anomalous machine operating conditions.

\- Predict whether a manufactured product passes quality requirements.

\- Compare Logistic Regression, Random Forest, and Artificial Neural Network (ANN) models.

\- Track experiments and manage selected models using MLflow.

\- Serve predictions through FastAPI.

\- Provide an interactive user interface using Streamlit.

\- Package the application using Docker.



\## Prediction Tasks



| Task | Target | Purpose |

|---|---|---|

| Machine Failure Prediction | `machine\_failure` | Predict potential machine failure |

| Anomaly Detection | `is\_anomaly` | Identify unusual operating conditions |

| Quality Prediction | `quality\_pass` | Predict product quality pass/fail status |



\## Dataset



The project uses a synthetic manufacturing dataset containing 6,000 records and 22 original columns.



The data includes machine identifiers, production-line and machine-type information, timestamps, maintenance history, operating measurements, machine health indicators, quality scores, and prediction targets.



\*\*Note:\*\* Because the dataset is synthetic, the results demonstrate the project workflow and should not be interpreted as validated performance on real manufacturing equipment.



\## Technologies Used



\- \*\*Programming:\*\* Python

\- \*\*Data analysis:\*\* Pandas, NumPy

\- \*\*Visualization:\*\* Matplotlib, Seaborn

\- \*\*Machine learning:\*\* Scikit-learn

\- \*\*Deep learning:\*\* TensorFlow / Keras

\- \*\*Experiment tracking and model registry:\*\* MLflow

\- \*\*API:\*\* FastAPI, Uvicorn, Pydantic

\- \*\*Dashboard:\*\* Streamlit

\- \*\*Containerization:\*\* Docker, Docker Compose

\- \*\*Version control:\*\* Git and GitHub



\## Machine Learning Workflow



1\. Explore the dataset and examine distributions, relationships, and target variables.

2\. Preprocess timestamps and engineer time-based features.

3\. Encode categorical features and scale numerical features.

4\. Split the data into training and testing sets.

5\. Train and evaluate Logistic Regression, Random Forest, and ANN models.

6\. Compare model performance using classification metrics.

7\. Track experiments and manage selected models through MLflow.

8\. Serve predictions through FastAPI and display results in Streamlit.



The current preprocessing pipeline uses an 80:20 train-test split with `random\_state=42`.



\## Model Comparison



The models were compared using accuracy, precision, recall, F1-score, and ROC-AUC.



| Prediction Task | Model | Accuracy | F1-score |

|---|---|---:|---:|

| Machine Failure | Random Forest | 98.08% | 90.21% |

| Anomaly Detection | ANN | 96.67% | 93.01% |

| Quality Prediction | Random Forest | 99.75% | 99.86% |



These are results from the synthetic dataset and the current evaluation split. They should be interpreted with the dataset's limitations in mind.



Detailed results and the comparison visualization are available in `reports/algorithm\_comparison.csv` and `reports/algorithm\_comparison.png`.



\## MLflow



MLflow is used to track experiments and manage registered models.



The configured model registry includes:



\- `ManufacturingMachineFailure`

\- `ManufacturingAnomalyDetection`

\- `ManufacturingQualityPrediction`



Selected model versions are assigned the `champion` alias in the registry.



\## Application Components



\### FastAPI Backend



The backend provides prediction endpoints:



\- `GET /` — basic API information

\- `POST /predict/failure` — machine failure prediction

\- `POST /predict/anomaly` — anomaly prediction

\- `POST /predict/quality` — quality prediction



Interactive API documentation is available at `/docs`.



\### Streamlit Dashboard



The dashboard collects machine operating information and presents the predictions returned by the backend.



\### Docker



Docker Compose runs the backend and frontend as separate services.



\- Backend: port `8000`

\- Frontend: port `8501`



\## Project Structure



```text

Manufacturing Quality Intelligence/

├── data/

│   ├── raw/

│   └── processed/

├── notebooks/

│   └── 01\_eda.ipynb

├── src/

│   ├── preprocess.py

│   └── train.py

├── models/

├── reports/

│   ├── algorithm\_comparison.csv

│   ├── algorithm\_comparison.png

│   └── mlflow\_model\_registry.png

├── backend/

│   ├── main.py

│   ├── schemas.py

│   └── requirements.txt

├── frontend/

│   ├── app.py

│   └── requirements.txt

├── Dockerfile.backend

├── Dockerfile.frontend

├── docker-compose.yml

├── .dockerignore

├── .gitignore

└── README.md

```



\## Running the Application with Docker



\### Prerequisites



\- Docker Desktop

\- Docker Compose

\- Project model files and preprocessing artifacts in the `models/` directory



\### Start the services



From the project root, run:



```bash

docker compose up --build -d

```



\### Access the application



\- \*\*Streamlit dashboard:\*\* http://localhost:8501

\- \*\*FastAPI documentation:\*\* http://localhost:8000/docs

\- \*\*FastAPI root:\*\* http://localhost:8000/



\### Stop the services



```bash

docker compose down

```



\## Running MLflow



To launch the MLflow UI in an environment where MLflow is installed and the project's tracking data is available, run:



```bash

mlflow ui

```



Then open http://localhost:5000.



\## Evaluation



The project evaluates model performance using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices where applicable. The comparison report provides a basis for selecting a model for each prediction task.



\## Limitations



\- The dataset is synthetic and may not represent the full complexity of real manufacturing environments.

\- Random train-test splitting may not fully measure generalization to unseen machines or future time periods.

\- Potential target leakage and correlations between engineered features and targets require further investigation.

\- High evaluation scores on synthetic data do not guarantee equivalent real-world performance.

\- Further validation with representative industrial data is required before operational deployment.



\## Future Improvements



\- Validate the models on real manufacturing sensor data.

\- Evaluate generalization across machines, production lines, and time periods.

\- Investigate potential data leakage and strengthen validation.

\- Add automated monitoring and model performance tracking.

\- Improve deployment, testing, and reproducibility documentation.



\## Author



Developed as an AI and machine learning project demonstrating an end-to-end manufacturing quality intelligence workflow.


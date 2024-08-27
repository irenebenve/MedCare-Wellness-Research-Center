# MedCare Wellness Research Center EDA - Predicting Physical Health

## Team Members
- Irene Benvenuti 288521
- Michele Giraudi 281971
- Justin Chisenga 289911

---

## [Section 1] Introduction

This project aims to predict the "Physical Health" score using the MedCare Wellness Research Center dataset, focusing on elderly populations. By accurately predicting these scores, the project seeks to support proactive healthcare interventions that improve the quality of life in this demographic.

Our approach begins with a thorough Exploratory Data Analysis (EDA) to uncover patterns, validate assumptions, and detect anomalies. We will preprocess the data through cleaning and feature engineering to prepare it for modeling. The project will then explore whether predicting "Physical Health" is best approached as a regression, classification, or clustering problem. Multiple models will be tested, with cross-validation used to identify the optimal approach.

The steps include:

- Data Cleaning: Address missing values, outliers, and inconsistencies.
- Descriptive Statistics: Understand the dataset through key statistical measures.
- Data Visualization: Explore relationships between variables using visual tools.
- Feature Engineering: Enhance the dataset’s predictive power with new features.
The outcome will provide a strong foundation for data-driven health interventions aimed at improving elderly individuals' physical health.
---

## [Section 2] Methods


### Overview
This section provides a detailed description of the methods used in the project, focusing on feature selection, algorithm choice, training strategy, and environment setup. Our approach was designed to build an effective model for predicting the "Physical Health" score, balancing accuracy and interpretability while ensuring that the environment is easily reproducible.

### Proposed Ideas

1. **Feature Selection and Engineering**
   - **Rationale:** We identified and selected features based on their relevance to physical health, informed by domain knowledge and statistical analysis during exploratory data analysis (EDA). We engineered additional features to capture potential interactions and enhance model performance.
   - **Process:** 
     - Conducted correlation analysis to identify significant relationships between features and the target variable.
     - Engineered new features, such as composite lifestyle scores, to encapsulate the combined effects of related attributes.
     - Selected the final set of features based on their statistical significance and impact on model accuracy during preliminary testing.

2. **Algorithm Selection**
   - **Rationale:** Given the continuous nature of the "Physical Health" score, the problem was framed as a regression task. We selected algorithms that balance simplicity and predictive power, allowing for effective modeling of the underlying data patterns.
   - **Algorithms:**
     - **Linear Regression:** Chosen as a baseline due to its simplicity and ease of interpretation. It provides a reference point for comparing more complex models.
     - **Random Forest Regressor:** Selected for its ability to handle non-linear relationships and interactions between features. It was particularly useful in capturing the diverse health metrics and their complex interdependencies.
     - **Gradient Boosting Regressor:** Opted for its strength in modeling complex patterns and improving accuracy through iterative learning. This model was particularly chosen for its robustness in handling both noise and overfitting, which was crucial given the variability in the dataset.

3. **Training Overview**
   - **Rationale:** To ensure robust model performance and avoid overfitting, we implemented cross-validation and hyperparameter tuning.
   - **Process:**
     - **Cross-Validation:** Utilized k-fold cross-validation (k=3) to evaluate model performance consistently across different data subsets, ensuring generalizability. This helped in mitigating the effects of data imbalance and providing a more accurate assessment of model performance.
     - **Hyperparameter Tuning:** Employed `RandomizedSearchCV` to fine-tune model parameters for Random Forest and Gradient Boosting, optimizing model performance based on validation metrics. Key parameters tuned included the number of estimators, maximum depth of trees, and learning rate for Gradient Boosting.

4. **Design Choices**
   - **Feature Scaling:** Applied standardization using `StandardScaler` to ensure that all features contributed equally to model training, particularly important for gradient boosting where the magnitude of features can influence model outcomes.
   - **Categorical Data Handling:** 
     - Used Label Encoding for ordinal categorical features where the order matters (e.g., health rating scales).
     - Applied One-Hot Encoding for nominal categorical features to prevent introducing unintended ordinal relationships (e.g., gender, ethnicity).
   - **Model Evaluation:** Selected RMSE and R² as evaluation metrics, providing clear insights into model accuracy and explanatory power. RMSE was chosen for its interpretability in the context of physical health scores, and R² was used to assess the proportion of variance explained by the model.

### Environment Setup

To replicate our environment, use the following Conda commands:

- **Python Version:** 3.8
- **Key Libraries:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`

Commands to set up the environment:

```bash
conda create -n physical_health_prediction python=3.8
conda activate physical_health_prediction
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```
---

### [Section 3] Experimental Design

Our experimental design aimed to identify the best model for predicting the "Physical Health" feature from the MedCare Wellness Research Center dataset. The experiment involved the following key steps:

#### Purpose
The primary purpose of our experiments was to evaluate and compare the performance of different regression models to predict the "Physical Health" score based on various health and lifestyle features.

#### Experimental Setup
1. **Baseline Model: Linear Regression**
   - We began with a simple Linear Regression model as our baseline. This model was chosen for its interpretability and to provide a reference point for evaluating the performance of more complex models.
   - The model was trained on the training dataset and evaluated using Root Mean Squared Error (RMSE) on the test set.

2. **Advanced Models: Random Forest and Gradient Boosting**
   - We then moved on to more sophisticated models: Random Forest Regressor and Gradient Boosting Regressor. These models were selected due to their ability to capture complex relationships in the data.
   - For each model, hyperparameter tuning was conducted using RandomizedSearchCV to identify the optimal parameters.

3. **Evaluation Metrics**
   - **Root Mean Squared Error (RMSE):** This metric was chosen to measure the average magnitude of the errors in our predictions, with a lower RMSE indicating better model performance.
   - **R² Score:** We also calculated the R² score to understand the proportion of variance in the dependent variable that our model could explain.

#### Cross-Validation
To ensure the robustness of our findings, we used cross-validation. Specifically, we employed a 3-fold cross-validation approach during hyperparameter tuning to mitigate the risk of overfitting and to ensure that our models generalize well to unseen data.

---

### [Section 4] Results

#### Main Findings:
The experiments revealed that the **Gradient Boosting Regressor** performed the best among the models tested, achieving the lowest Root Mean Squared Error (RMSE) of **3.40**. This model outperformed both the Linear Regression and Random Forest Regressor models, which had RMSEs of **3.61** and **3.40** respectively. Although the Random Forest Regressor tied with Gradient Boosting in terms of RMSE, the Gradient Boosting Regressor was ultimately selected due to its more consistent performance and its robustness to overfitting, as observed during cross-validation.

The R² score further supported these findings, indicating that the Gradient Boosting model explained a significant portion of the variance in the "Physical Health" feature. This suggests that the Gradient Boosting model is well-suited for predicting physical health outcomes based on the available features in the dataset.

#### Placeholder Figure
The figure below illustrates the comparison between the actual and predicted physical health scores. The blue line represents the actual values, while the orange line represents the predicted values. The close alignment between the two lines across most samples highlights the model's accuracy in predictions.

![Actual vs Predicted Physical Health Scores](/Users/carlogiraudi/Desktop/AI and Machine Learning/EDA Project/Graphs/output.png)



### Section 5: Conclusions

#### Summary of Findings
The main takeaway from our analysis is that the Gradient Boosting model emerged as the best performer in predicting the "Physical Health" score among elderly individuals, closely followed by the Random Forest model. The Gradient Boosting model provided a good balance between accuracy and generalization, as evidenced by its lower RMSE and higher R² scores compared to the baseline Linear Regression model. This suggests that more complex models that can capture non-linear relationships are better suited for this type of prediction task.

#### Unanswered Questions and Future Work
While our model performed well, there are still questions that remain unanswered. For instance, the model's predictions could potentially be improved by incorporating additional data sources or by engineering new features that capture aspects of physical health not currently included in the dataset. Furthermore, future work could explore the impact of using ensemble techniques to combine predictions from multiple models or the use of deep learning methods, which might capture even more intricate patterns in the data. Longitudinal studies tracking changes in physical health over time could also provide valuable insights and improve prediction accuracy.

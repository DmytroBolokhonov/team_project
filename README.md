# Team Project

# Project overview video:

Ronak Patel Video https://drive.google.com/drive/folders/15v_m8gkb1I9S4zbHdgMoqfI0D_cgw-Pf?usp=drive_link


# Pharmaceutical Spending Analysis

## Project Overview

This project analyzes pharmaceutical spending data obtained from the Organisation for Economic Co-operation and Development (OECD). The main objective is to explore global spending trends, investigate relationships among spending variables, and prepare the dataset for advanced analytics and modeling.

## Data Sources

The data is derived from the following sources:

- **Pharmaceutical Spending Data**: OECD  
- **Population Data**: DataHub  

### Data Fields

| **Field**       | **Description**                             |
|------------------|---------------------------------------------|
| `LOCATION`      | Country code                                |
| `TIME`          | Year of the data                           |
| `PC_HEALTHXP`   | Percent of health spending                 |
| `PC_GDP`        | Percent of GDP                             |
| `USD_CAP`       | US dollars per capita                      |
| `FLAG_CODES`    | Additional metadata flags                  |
| `TOTAL_SPEND`   | Total pharmaceutical spending              |

---

## Preprocessing Steps

### Data Loading
- The raw dataset is loaded and saved for reference.

### Data Cleaning
- Missing values in the `FLAG_CODES` column are replaced with "Unknown".
- Data types are standardized for consistency.

### Data Transformation
- Growth rates for `TOTAL_SPEND` and `USD_CAP` are calculated for time-series analysis.
- Outliers in `TOTAL_SPEND` are identified and removed based on Z-scores.

### Data Saving
- The processed dataset is saved in the `data/processed` directory for further analysis.

---

## Exploratory Analysis

### Pairwise Relationships
A pairplot was used to visualize relationships among the variables. Key observations:
- **Positive correlations** between:
  - `TIME` and `TOTAL_SPEND` (indicating increasing spending over time).
  - `PC_GDP` and `TOTAL_SPEND` (countries with higher GDP allocation to health tend to spend more).
- **Weak correlation** between `PC_HEALTHXP` and `USD_CAP`.

---

## Scatterplot Analyses

### 1. TIME vs. TOTAL_SPEND
- **Observation**: A clear upward trend in `TOTAL_SPEND` over time, indicating global increases in pharmaceutical spending.
- **Insights**:
  - The linear regression model fits well for lower spending values but struggles with outliers at the higher end.
  - Spending appears to grow non-linearly over time.
- **Recommendations**:
  - Use polynomial regression or log transformations to better capture non-linear growth patterns.
- **Visualization**:  
### TIME vs TOTAL_SPEND
### TIME vs TOTAL_SPEND
![Scatter Plot: TIME vs TOTAL SPEND](./TIME%20vs.%20TOTAL_SPEND.png)


---

### 2. PC_HEALTHXP vs. USD_CAP
- **Observation**: A weak positive correlation exists between the percentage of health spending and per capita spending.
- **Insights**:
  - Higher `PC_HEALTHXP` values slightly increase `USD_CAP`, but the variance is large.
  - Other factors likely influence per capita spending significantly.
- **Recommendations**:
  - Group countries by income or healthcare system type to explore stronger relationships.
  - Perform feature engineering to introduce new explanatory variables.
- **Visualization**:
- 
### PC_HEALTHXP vs USD_CAP
![Scatter Plot: PC_HEALTHXP vs USD_CAP](PC_HEALTHXP%20vs.%20USD_CAP.png)

---

### 3. PC_GDP vs. TOTAL_SPEND
- **Observation**: A positive correlation is evident, with higher GDP allocation to health linked to increased pharmaceutical spending.
- **Insights**:
  - Outliers with extremely high `TOTAL_SPEND` values skew the linear regression line.
- **Recommendations**:
  - Investigate outliers to identify specific countries driving this trend.
  - Consider normalizing `TOTAL_SPEND` by population to improve comparability.
- **Visualization**:  
### PC_GDP vs TOTAL_SPEND
![Scatter Plot: PC_GDP vs TOTAL_SPEND](PC_GDP%20vs.%20TOTAL_SPEND.png)


---

### 4. Pairplot Summary
- **Trends**:
  - Strong correlations between `TIME` and `TOTAL_SPEND`.
  - Moderate correlation between `PC_GDP` and `TOTAL_SPEND`.
- **Outliers**:
  - Significant outliers in `TOTAL_SPEND` and `USD_CAP` influence trends.
  - Removing outliers helps reveal clearer relationships.
### Pairplot Summary
![Pairplot Summary](Pairplot%20Summary.png)

# Additional Project Questions and Answers

## 1. What are the key variables and attributes in your dataset?
The dataset contains the following key variables:
- **LOCATION**: Country code identifying the region.
- **TIME**: Year of the data.
- **PC_HEALTHXP**: Percent of health expenditure.
- **PC_GDP**: Percent of GDP allocated to pharmaceutical spending.
- **USD_CAP**: Pharmaceutical spending per capita in US dollars.
- **TOTAL_SPEND**: Total pharmaceutical spending in the respective country.
- **FLAG_CODES**: Metadata flags for additional information.

---

## 2. How can we explore the relationships between different variables?
We can explore relationships using:
- **Scatterplots**: To identify correlations (e.g., `TIME` vs. `TOTAL_SPEND`, `PC_GDP` vs. `TOTAL_SPEND`).
- **Pairwise Plots**: To visualize all possible variable pairings for correlations.
- **Correlation Matrix**: To numerically quantify relationships between variables.
- **Regression Analysis**: To identify and model linear or non-linear trends.

---

## 3. Are there any patterns or trends in the data that we can identify?
Yes, the following trends and patterns were observed:
- Pharmaceutical spending (`TOTAL_SPEND`) has increased significantly over time (`TIME`).
- Countries with higher GDP allocation to health (`PC_GDP`) tend to have higher total pharmaceutical spending.
- Weak correlation between `PC_HEALTHXP` and per capita spending (`USD_CAP`), suggesting other factors are at play.

---

## 4. Who is the intended audience for our data analysis?
The analysis is intended for:
- **Policymakers**: To understand global trends in pharmaceutical spending and allocate resources effectively.
- **Healthcare Economists**: To analyze spending patterns and identify potential inefficiencies.
- **Researchers**: To explore socioeconomic factors influencing healthcare spending.

---

## 5. What is the question our analysis is trying to answer?
The analysis aims to answer:
- How has pharmaceutical spending evolved globally over time?
- What factors influence total pharmaceutical spending, and how significant are their effects?
- Are there patterns or anomalies in spending across countries that warrant further investigation?

---

## 6. Are there any specific libraries or frameworks that are well-suited to our project requirements?
Yes, we used:
- **Pandas**: For data preprocessing and manipulation.
- **NumPy**: For numerical computations.
- **Matplotlib and Seaborn**: For data visualization.
- **Scikit-learn**: For regression modeling and machine learning.
- **Statsmodels**: For statistical analysis and time-series decomposition.

---

## 7. How can we iterate on our design to address feedback and make iterative improvements?
- **Solicit feedback** from stakeholders and adjust visualizations or analysis based on their needs.
- Incorporate **interactivity** in visualizations using tools like Plotly or Tableau.
- **Refine models** based on validation metrics and feedback from domain experts.
- Use **user testing** to ensure insights are intuitive and actionable.

---

## 8. What best practices can we follow to promote inclusivity and diversity in our visualization design?
- Use **color-blind-friendly palettes**.
- Provide **text-based alternatives** for visualizations (e.g., descriptive captions).
- Avoid **biased terminology** or conclusions.
- Ensure visualizations are **accessible** (e.g., readable font sizes, tooltips for interactivity).

---

## 9. How can we ensure that our visualization accurately represents the underlying data without misleading or misinterpreting information?
- Use **appropriate scales** (e.g., logarithmic for large value ranges).
- Avoid **distorted axes** that exaggerate trends.
- Include **error bars** or uncertainty intervals in visualizations.
- Clearly label visualizations and describe assumptions in captions.

---

## 10. Are there any privacy concerns or sensitive information that need to be addressed in our visualization?
- The dataset does not contain personally identifiable information (PII); however:
  - Ensure that country-level data is anonymized if required by specific stakeholders.
  - Confirm compliance with GDPR or other data privacy laws for regional analysis.

---

## 11. What are the specific objectives and success criteria for our machine learning model?
- **Objective**: Predict total pharmaceutical spending (`TOTAL_SPEND`) based on socioeconomic indicators.
- **Success Criteria**:
  - Achieve high accuracy (low RMSE, high R²).
  - Identify key predictors influencing spending trends.

---

## 12. How can we select the most relevant features for training our machine learning model?
- Use **correlation analysis** to identify strong predictors.
- Apply **feature importance techniques** (e.g., using Random Forest or Gradient Boosting).
- Conduct **recursive feature elimination (RFE)** to systematically remove irrelevant features.

---

## 13. Are there any missing values or outliers that need to be addressed through preprocessing?
Yes:
- **Missing Values**:
  - The `FLAG_CODES` column had missing values, which were replaced with "Unknown".
- **Outliers**:
  - Significant outliers in `TOTAL_SPEND` were identified and removed using Z-scores to improve model accuracy.

---

## 14. Which machine learning algorithms are suitable for our problem domain?
- **Linear Regression**: For understanding linear trends.
- **Polynomial Regression**: To capture non-linear relationships.
- **Random Forest or Gradient Boosting**: For robust feature importance analysis and prediction.
- **Time-Series Models**: For forecasting trends over time.

---

## 15. What techniques can we use to validate and tune the hyperparameters for our models?
- Use **cross-validation** to evaluate model performance.
- Apply **grid search** or **random search** for hyperparameter tuning.
- Track metrics like RMSE, R², and MAE for different model configurations.

---

## 16. How should we split the dataset into training, validation, and test sets?
- **Training Set**: 70% of the dataset for model training.
- **Validation Set**: 15% for hyperparameter tuning.
- **Test Set**: 15% for evaluating final model performance.

---

## 17. Are there any ethical implications or biases associated with our machine learning model?
- Country-level data may reflect biases related to socioeconomic conditions or reporting accuracy.
- Ensure that the model does not perpetuate inequalities (e.g., overestimating spending for wealthier countries while underestimating for developing ones).

---

## 18. How can we document our machine learning pipeline and model architecture for future reference?
- Use **Jupyter Notebooks** or **Markdown documents** to document:
  - Preprocessing steps.
  - Feature engineering.
  - Model architecture.
  - Evaluation metrics.
- Save models and pipelines using **joblib** or **pickle** for reproducibility.


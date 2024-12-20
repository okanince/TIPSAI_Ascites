# TIPS AI Ascites Project
This is the computer code for our study "**Improving Clinical Decisions in IR: Interpretable Machine Learning Models for Predicting Ascites Improvement after Transjugular Intrahepatic Portosystemic Shunt Procedures**" published at JVIR. 

doi: 10.1016/j.jvir.2024.09.022

## Abstract:
 
**Purpose**: To evaluate the potential of interpretable machine learning (ML) models to predict ascites improvement in patients undergoing transjugular intrahepatic portosystemic shunt (TIPS) placement for refractory ascites.

**Materials and methods**: In this retrospective study, 218 patients with refractory ascites who underwent TIPS placement were analyzed. Data on 29 demographic, clinical, and procedural features were collected. Ascites improvement was defined as reduction in the need of paracentesis by 50% or more at the 1-month follow-up. Univariate statistical analysis was performed. Data were split into train and test sets. Feature selection was performed using a wrapper-based sequential feature selection algorithm. Two ML models were built using support vector machine (SVM) and CatBoost algorithms. Shapley additive explanations values were calculated to assess interpretability of ML models. Performance metrics were calculated using the test set.

**Results**: Refractory ascites improved in 168 (77%) patients. Higher sodium (Na; 136 mEq/L vs 134 mEq/L; P = .001) and albumin (2.91 g/dL vs 2.68 g/dL; P = .03) levels, lower creatinine levels (1.01 mg/dL vs 1.17 mg/dL; P = .04), and lower Model for End-stage Liver Disease (MELD) (13 vs 15; P = .01) and MELD-Na (15 vs 17.5, P = .002) scores were associated with significant improvement, whereas main portal vein puncture was associated with a lower improvement rate (P = .02). SVM and CatBoost models had accuracy ratios of 83% and 87%, with area under the curve values of 0.83 and 0.87, respectively. No statistically significant difference was found between performances of the models in DeLong test (P = .3).

**Conclusions**: ML models may have potential in patient selection for TIPS placement by predicting the improvement in refractory ascites.

Thank you for reviewing the code. Please do not hesitate to contact if you have any questions.

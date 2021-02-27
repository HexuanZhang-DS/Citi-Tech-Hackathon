![](PredictROI.gif)

# Introduction

Submission for the Jan 2021 Citi Tech Hackathon. Univest is a two-sided marketplace for Income Share Agreements (ISAs). Students sign up and indicate how much money they need for tuition, then they get offers from investors and choose the best one. Investors submit offers to fund the ISA. They can expect extremely low risk and medium to high returns. 

In our MVP students can create a request for an ISA. This is posted to our backend and stored in a SQLite DB. Our machine learning algorithms then go to work calculating the expected yearly ROI over a 10 year timespan after graduation. Investors can search for this ISA requests and see the expected ROI.


# Data Analysis

![WallStreetBets](https://upload.wikimedia.org/wikipedia/en/f/f0/WallStreetBets.png)

### Aim:
To predict investor's return on investment (ROI).

### Strategy:
Use data that can be collected for college students (e.g. major, gender, number of siblings, political views) to train a model that will predict the student's future income after graduation. Use that prediction to calculate the expected ROI for the investor.

### Data:
The data used for the analysis and modelling was collected from General Social Survey ([GSS](https://gss.norc.org/)). 10 datasets ranging from the year 2000 to 2018 were combined. Only individuals who fit the target market were kept (individuals within the age range of 21 to 45 who attended college or above). 
The idea was to see what salaries these individuals were earning and then use back-datable features in the dataset (e.g. gender, parents' highest level of education) to create a model that can predict salaries using these features. As the features are back-datable we could ask for that information from our student clients and predict their future salary. That prediction is then used to determine the expected ROI.  

### Breakdown:

### [EDA, Visualization and Data Wrangling:](https://github.com/ricotomo/Citi-Tech-Hackathon/blob/data_analytics/EDA.ipynb)
A notebook containing the initial analysis of the data:

- Relationship between numerical features and income.
- Distributions of numerical features.
- Correlations (heatmap). 
- Relationship between categorical features and income -> our analysis showed that some unexpected datapoints such as the number of grandparents born in the USA correlates strongly with income post-graduation, so when the students sign up for an ISA we ask them this info. 
- Feature engineering -> dealing with null values, using predictive models to predict null values. 


### [Modelling, ML Regressors, Keras Feed Forward Neural Network:](https://github.com/ricotomo/Citi-Tech-Hackathon/blob/data_analytics/modelling.ipynb)

- Comparing regression models on a "large" dataset with predicted features.
- Comparing regression models on a "small" dataset without predicted features (dropna). 
- Artificial Neural Network 



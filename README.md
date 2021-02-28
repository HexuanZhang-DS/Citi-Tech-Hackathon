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

__________________________________________________________________

### EDA, Visualization and Data Wrangling

#### Data Overview

<table class="tableizer-table">
<thead><tr class="tableizer-firstrow"><th></th><th>OCC10</th><th>SIBS</th><th>AGE</th><th>EDUC</th><th>PAEDUC</th><th>MAEDUC</th><th>DEGREE</th><th>PADEG</th><th>MADEG</th><th>MAJOR1</th><th>MAJOR2</th><th>DIPGED</th><th>SECTOR</th><th>BARATE</th><th>SEX</th><th>RACE</th><th>RES16</th><th>REG16</th><th>FAMILY16</th><th>MAWRKGRW</th><th>INCOM16</th><th>BORN</th><th>PARBORN</th><th>GRANBORN</th><th>POLVIEWS</th><th>INCOME</th></tr></thead><tbody>
 <tr><td>0</td><td>Broadcast and sound engineering technicians an...</td><td>1.0</td><td>26.0</td><td>16.0</td><td>16.0</td><td>16.0</td><td>BACHELOR</td><td>BACHELOR</td><td>GRADUATE</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>MALE</td><td>WHITE</td><td>CITY GT 250000</td><td>W. SOU. CENTRAL</td><td>MOTHER & FATHER</td><td>YES</td><td>NaN</td><td>YES</td><td>BOTH IN U.S</td><td>1.0</td><td>SLGHTLY CONSERVATIVE</td><td>$8 000 TO 9 999</td></tr>
 <tr><td>1</td><td>Advertising and promotions managers</td><td>6.0</td><td>44.0</td><td>14.0</td><td>12.0</td><td>12.0</td><td>JUNIOR COLLEGE</td><td>HIGH SCHOOL</td><td>HIGH SCHOOL</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>FEMALE</td><td>WHITE</td><td>BIG-CITY SUBURB</td><td>E. NOR. CENTRAL</td><td>MOTHER & FATHER</td><td>YES</td><td>NaN</td><td>YES</td><td>BOTH IN U.S</td><td>1.0</td><td>LIBERAL</td><td>$7 000 TO 7 999</td></tr>
 <tr><td>2</td><td>First-line supervisors of office and administr...</td><td>0.0</td><td>44.0</td><td>18.0</td><td>11.0</td><td>11.0</td><td>GRADUATE</td><td>HIGH SCHOOL</td><td>HIGH SCHOOL</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>MALE</td><td>WHITE</td><td>TOWN LT 50000</td><td>W. SOU. CENTRAL</td><td>MOTHER & FATHER</td><td>YES</td><td>NaN</td><td>YES</td><td>BOTH IN U.S</td><td>ALL IN U.S</td><td>SLIGHTLY LIBERAL</td><td>$50000 TO 59999</td></tr>
 <tr><td>3</td><td>Dispatchers</td><td>8.0</td><td>40.0</td><td>16.0</td><td>10.0</td><td>10.0</td><td>HIGH SCHOOL</td><td>LT HIGH SCHOOL</td><td>LT HIGH SCHOOL</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>MALE</td><td>BLACK</td><td>TOWN LT 50000</td><td>W. SOU. CENTRAL</td><td>MOTHER & FATHER</td><td>YES</td><td>NaN</td><td>YES</td><td>BOTH IN U.S</td><td>ALL IN U.S</td><td>MODERATE</td><td>$25000 TO 29999</td></tr>
 <tr><td>4</td><td>Software developers, applications and systems ...</td><td>7.0</td><td>37.0</td><td>16.0</td><td>NaN</td><td>13.0</td><td>BACHELOR</td><td>NaN</td><td>HIGH SCHOOL</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>NaN</td><td>MALE</td><td>WHITE</td><td>COUNTRY,NONFARM</td><td>W. SOU. CENTRAL</td><td>MOTHER</td><td>YES</td><td>NaN</td><td>YES</td><td>BOTH IN U.S</td><td>NaN</td><td>LIBERAL</td><td>$75000 TO $89999</td></tr>
</tbody></table>

#### Income Breakdown

![](image/income.png)

Income is in ranges, we recoded it into numeric variables by taking the lower bound of the range. We took the lower bound instead of the average of a range because some ranges have no upper bound in the data.

After recoding, the income distribution looks as follows: 

![](image/income2.png)

#### Income by other features

##### Majors
![](image/majorbyincome.png)

##### Degrees
![](image/degreebyincome.png)

##### Diplomas
![](image/diplomabyincome.png)

##### Father's Education
![](image/fatheredubyincome.png)

##### Mother's Education
![](image/motheredubyincome.png)

##### Siblings
![](image/sibbyincome.png)

##### Guardians
![](image/guardianbyincome.png)

##### Parents were born in the US
![](image/parentsbornusbyincome.png)

##### Gradparents were born in the US
![](image/grandparentsbornus.png)

##### Sex
![](image/sexbyincome.png)

##### Political Views
![](image/polviews.png)

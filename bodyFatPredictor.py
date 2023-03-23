#import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 
import seaborn as sns
from scipy.stats import boxcox

# --------------------------------- Data Preprocessing ----------------------------------------#
bodyFat = pd.read_csv("D:/Codes/Data sets/bodyfat.csv")

independentVariables = bodyFat.drop(['BodyFat'], axis=1)
dependentVariables = bodyFat['BodyFat']

# --------------------------------- first linear regression ----------------------------------------#

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(independentVariables, dependentVariables)

# Make predictions using the testing set
bodyFatPrediction = regr.predict(independentVariables)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(dependentVariables, bodyFatPrediction))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(dependentVariables, bodyFatPrediction))


# --------------------------------- Cooks distance ----------------------------------------#

# outlier detection using cooks distance
def removeOutliersCooksDistance(bodyFat, targetCol):
    independentVariables = bodyFat.drop([targetCol], axis=1)
    dependentVariables = bodyFat[targetCol]

    # Calculate cooks distance
    def cooksDistance(bodyFatPrediction, bodyFatPredictionI, m, d):
        cd = np.linalg.norm(bodyFatPrediction-bodyFatPredictionI)**2/(m*d)
        return cd
    # Calculate cooks distance for each data point
    def getCooksDistance(independentVariables, dependentVariables, bodyFatPrediction):
        cooksDist = []
        mses = []
        independentVariables = np.array(independentVariables)
        dependentVariables = np.array(dependentVariables)
        bodyFatPrediction = np.array(bodyFatPrediction)
        outliers = []
        for i in range(len(bodyFatPrediction)):
            xI = np.delete(independentVariables, i, axis=0)
            yI = np.delete(dependentVariables, i, axis=0)
            regr.fit(xI, yI)
            bodyFatPredictionI = regr.predict(xI)
            mses.append(mean_squared_error(yI, bodyFatPredictionI))
            testBodyFatPrediction = np.delete(bodyFatPrediction, i, axis=0)
            cooksDist.append(cooksDistance(testBodyFatPrediction, bodyFatPredictionI, mses[i], independentVariables.shape[1]))
        return cooksDist

    independentVariables = np.array(independentVariables)
    dependentVariables = np.array(dependentVariables)
    regr.fit(independentVariables, dependentVariables)
    dependentVariablesPred = regr.predict(independentVariables)

    cooksDistances = getCooksDistance(independentVariables, dependentVariables, dependentVariablesPred)
    outlierIndex = np.argmax(cooksDistances) 
    bodyFatOutlierRemoved = bodyFat.drop(index=[outlierIndex])
    return bodyFatOutlierRemoved


bodyfatClean = removeOutliersCooksDistance(bodyfat, 'BodyFat')

# --------------------------------- Linear regression after removing outliers ----------------------------------------#
independentVariablesClean = bodyfatClean.drop(['BodyFat'], axis=1)
dependentVariablesClean = bodyfatClean['BodyFat']
regr.fit(independentVariablesClean, dependentVariablesClean)
bodyfatPredictionClean = regr.predict(independentVariablesClean)
mseClean = mean_squared_error(dependentVariablesClean, bodyfatPredictionClean)
r2Clean = r2_score(dependentVariablesClean, bodyfatPredictionClean)

print("Coefficients(After removing outliers): \n", regr.coef_)
print("Mean squared error(After removing outliers): %.2f" % mseClean)
print("Coefficient of determination(After removing outliers): %.2f" % r2Clean)


plt.scatter(dependentVariables[:250], bodyfatPrediction[:250])
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs predicted values for original data')
plt.show()

plt.scatter(dependentVariablesClean[:250], bodyfatPredictionClean[:250])
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs predicted values for cleaned data')
plt.show()


# --------------------------------- Box Cox Transformation ----------------------------------------#


dependentVariablesClean = np.array(dependent_variables_clean) + 0.01
dependentVariablesBoxcox, lambdaBoxcox = boxcox(dependentVariablesClean)

# Fit a linear regression model to the transformed data
regr.fit(independent_variables_clean, dependentVariablesBoxcox)
bodyfatPredictionBoxcox = regr.predict(independent_variables_clean)

# Inverse transform the predicted output variable to get the original scale
dependentVariablesPred = np.power((bodyfatPredictionBoxcox * lambdaBoxcox + 1), 1/lambdaBoxcox)

# Check for missing values in dependentVariables and dependentVariablesPred
print('Missing values in dependentVariables:', np.isnan(dependentVariablesClean).sum())
print('Missing values in dependentVariablesPred:', np.isnan(dependentVariablesPred).sum())

# Remove rows with missing values from dependentVariables and dependentVariablesPred
dependentVariables = dependentVariablesClean[~np.isnan(dependentVariablesPred)]
dependentVariablesPred = dependentVariablesPred[~np.isnan(dependentVariablesPred)]

# Calculate MSE and R-squared for the transformed and original data
mseBoxcox = mean_squared_error(dependentVariablesBoxcox, bodyfatPredictionBoxcox)
r2Boxcox = r2_score(dependentVariablesBoxcox, bodyfatPredictionBoxcox)

# Print the MSE, R-squared, and lambda value
print('MSE (Box-Cox transformed):', mseBoxcox)
print('R-squared (Box-Cox transformed):', r2Boxcox)
print('Lambda value:', lambdaBoxcox)

residuals = dependent_variables_clean - bodyfat_prediction_clean
stdResiduals = residuals / np.sqrt(mean_squared_error(dependent_variables_clean, bodyfat_prediction_clean))

plt.scatter(bodyfat_prediction_clean, stdResiduals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Standardized Residuals')
plt.show()

plt.scatter(dependentVariablesBoxcox[:250], bodyfatPredictionBoxcox[:250])
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs predicted values after BOX COX transformation')
plt.show()

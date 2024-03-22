import pandas as pd  # for importing data from csv file
import numpy as np
import matplotlib.pyplot as plt  # for visualisation

# used for statistical modeling, including classification, regression, etc.
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error

# data preprocessing essential : no values should be empty


data = pd.read_csv('winequality-red.csv', sep=';')
print(data.corr()['quality'])

quality = data['quality']
data = data.drop(
    ['fixed_acidity', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH'],
    axis=1)

data_np = data.to_numpy()  # converting to 2D array from panda object

# finding strong correlation

# results for correlation : volatile_acidity  , sulphates , alcohol, citric_acid

# initialising subplot

'''
plt.scatter(data['volatile_acidity'], data['quality'])
plt.xlabel('volatile_acidity')
plt.ylabel('quality')
plt.show()

plt.scatter(data['sulphates'], data['quality'])
plt.xlabel('sulphates')
plt.ylabel('quality')
plt.show()
'''
# training the model : -1 for last column
X_train, Y_train = data_np[:, :4], data_np[:, -1]

sklearn_model = LinearRegression().fit(X_train, Y_train)
sklearn_y_predictions = sklearn_model.predict(X_train)
print(sklearn_y_predictions)


mean_absolute_error(sklearn_y_predictions, Y_train), mean_squared_error(sklearn_y_predictions, Y_train)

predictions_data = pd.DataFrame({

    'volatile_acidity': data['volatile_acidity'],
    'citric_acid': data['citric_acid'],
    'sulphates': data['sulphates'],
    'alcohol': data['alcohol'],
    'Sklearn quality predictions' : sklearn_y_predictions


})

print(predictions_data)
# Assuming the following :
# quality[i] = b0 + b1*fixed_acidity[i] + b2*volatile_acidity[i] + ... + b11*alcohol[i];
# Generally : y[i] = b0 + b1*x1[i] + b2*x2[i] + ... + b11*x11[i] + ERROR
# model : y_res = b0_hat + b1_hat*x1[i] + ... + b11_hat*x11[i]

def get_predictions(model, X):
    '''
    Obtain the predictions for the given model and inputs.

    model: np.array of Floats with shape (p,) of parameters
    X: np.array of Floats with shape (n, p-1) of inputs : input matrix


    Returns: np.array of Floats with shape (n,).
    '''

    (n, p_minus_one) = X.shape
    p = p_minus_one + 1

    new_X = np.ones(shape=(n, p))
    new_X[:, 1:] = X

    return np.dot(new_X, model)  # dot product


# testing te model
test_model = np.array([1, 2, 1, 4, 3])  # b0, b1, b2, b3, b4
get_predictions(test_model, X_train)  # this is an arbitrary model
predictions_data['Test Predictions'] = get_predictions(test_model, X_train)
print(predictions_data)

mean_absolute_error(predictions_data['Test Predictions'], Y_train)

from numpy.linalg import inv

def get_best_model(X, Y):

    """
    Returns the model with the parameters that minimize the MSE.

    X: np.array of Floats with shape (n, p-1) of inputs
    y: np.array of Floats with shape (n,) of observed outputs

    Returns: np.array of shape (p,) representing the model.
    """

    (n, p_minus_1) = X.shape
    p = p_minus_1 +1

    new_X = np.ones(shape=(n,p))
    new_X[:, 1:] = X

    return np.dot(np.dot(inv(np.dot(new_X.T, new_X)), new_X.T), Y)

best_model = get_best_model(X_train, Y_train)
predictions_data['Best Predictions'] = get_predictions(best_model, X_train)
print(predictions_data)





'''

predictions_data = pd.DataFrame({
    'fixed_acidity': data['fixed_acidity'],
    'volatile_acidity': data['volatile_acidity'],
    'citric_acid': data['citric_acid'],
    'residual_sugar': data['residual_sugar'],
    'chlorides': data['chlorides'],
    'free_sulfur_dioxide': data['free_sulfur_dioxide'],
    'total_sulfur_dioxide': data['total_sulfur_dioxide'],
    'density': data['density'],
    'pH': data['pH'],
    'sulphates': data['sulphates'],
    'alcohol': data['alcohol'],


})

print(predictions_data)

#plotting the data


def get_predictions(model, X):

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sklearn.tree

# Function to generate data
def generate_data(n_samples=1000):
    # Mean and standard deviation for the two Gaussian distributions in the first column
    mean_class0, std_class0 = -1, 0.8
    mean_class1, std_class1 = 1, 0.8

    # Class 0 data generation
    class_0 = np.random.normal(loc=mean_class0, scale=std_class0, size=(n_samples // 2, 1))
    # The second column is centered around origin
    class_0_second_column = np.random.normal(loc=0, scale=1, size=(n_samples // 2, 1))
    class_0_second_column_sign = np.sign(class_0_second_column)
    class_0_second_column += class_0_second_column_sign * 0.5

    ### !!!   ChatGPT FAILED to solve the case
    ### !!!!  "The second column helps discriminate in the cases where the first column is overlapping."

    # Class 1 data generation
    class_1 = np.random.normal(loc=mean_class1, scale=std_class1, size=(n_samples // 2, 1))
    # The second column is distributed above and below the values of the first class
    class_1_second_column = np.random.uniform(size=(n_samples // 2, 1)) - 0.5

    # Uniform noise for the third column
    noise = np.random.uniform(low=-2, high=2, size=(n_samples, 1))

    # Combine the columns
    data_0 = np.hstack((class_0, class_0_second_column, noise[:n_samples // 2], np.zeros((n_samples // 2, 1))))
    data_1 = np.hstack((class_1, class_1_second_column, noise[n_samples // 2:], np.ones((n_samples // 2, 1))))

    # Concatenate Class 0 and Class 1 data
    data = np.vstack((data_0, data_1))
    np.random.shuffle(data)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Gaussian_Mixture', 'Second_Column', 'Uniform_Noise', 'Target'])

    return df

def plot_class_histograms(df, feature_column, target_column, bins=30, alpha=0.5):
    """
    Plot histograms for the specified feature in the dataframe,
    one for each class in the target column.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - feature_column: String, name of the column in df to plot the histograms for.
    - target_column: String, name of the target column in df that contains the class labels.
    - bins: Integer, number of bins for the histogram.
    - alpha: Float, transparency level of the histograms.
    """

    # Filter the dataframe by class
    class_0_df = df[df[target_column] == 0]
    class_1_df = df[df[target_column] == 1]

    # Plot histograms
    plt.hist(class_0_df[feature_column], bins=bins, alpha=alpha, label='Class 0')
    plt.hist(class_1_df[feature_column], bins=bins, alpha=alpha, label='Class 1')

    # Add title and labels
    plt.title(f'Histogram of {feature_column} by Class')
    plt.xlabel(feature_column)
    plt.ylabel('Frequency')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def generate_random_labels(df):
    """
    Generates a numpy array containing a random distribution of 0s and 1s.
    The length of the array is the same as the number of rows in the dataframe.

    Parameters:
    - df: Pandas DataFrame.

    Returns:
    - A numpy array of randomly distributed 0s and 1s.
    """
    n_rows = df.shape[0]
    random_labels = np.random.choice([0, 1], size=n_rows)
    return random_labels


def evaluate_accuracy(predicted_labels, df, target_column):
    """
    Evaluates the accuracy of randomly generated labels against the actual labels in the dataframe.

    Parameters:
    - df: Pandas DataFrame containing the actual labels.
    - target_column: String, the name of the column in df that contains the actual labels.

    Returns:
    - Accuracy of the random labels as a float.
    """
    # Actual labels
    actual_labels = df[target_column].values

    # Calculate accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)
    return accuracy


def hand_made_empirical_classifier(df):
    predictions = np.ones(df.shape[0], dtype=int)
    predictions[df["Gaussian_Mixture"] < 0] = 0
    # Try 2nd column
    # Try combine. Weighted? Voting? Random?

    return predictions


def decision_tree_classifier(df):
    clf = sklearn.tree.DecisionTreeClassifier()
    X = df.loc[:, ['Gaussian_Mixture', 'Second_Column', 'Uniform_Noise']]
    Y = df.loc[:, ["Target"]]

    clf.fit(X, Y)

    predictions = clf.predict(X)  # Overfitting!
    return predictions


def train_DecisionTree(df):
    clf = sklearn.tree.DecisionTreeClassifier()
    X = df.loc[:, ['Gaussian_Mixture', 'Second_Column', 'Uniform_Noise']]
    Y = df.loc[:, ["Target"]]

    clf.fit(X, Y)

    return clf

def eval_DecisionTree(clf, df):
    X = df.loc[:, ['Gaussian_Mixture', 'Second_Column', 'Uniform_Noise']]
    predictions = clf.predict(X)
    return predictions


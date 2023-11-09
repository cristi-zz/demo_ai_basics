import first_classifier


def test_save_some_data():
    df = first_classifier.generate_data(n_samples=100)
    print(df.head())
    df.to_csv("data/train_sample.csv", index=False)


def test_plot_some_histograms():
    # Example usage:
    # Assuming 'df' is the dataframe you generated, 'Gaussian_Mixture' is the feature column, and 'Target' is the target column.
    df = first_classifier.generate_data(n_samples=1000)
    first_classifier.plot_class_histograms(df, 'Gaussian_Mixture', 'Target')
    first_classifier.plot_class_histograms(df, 'Second_Column', 'Target')
    first_classifier.plot_class_histograms(df, 'Uniform_Noise', 'Target')


def test_randomly_setted_labels():
    df = first_classifier.generate_data(n_samples=1000)
    # Example usage:
    # Assuming 'df' is your pandas DataFrame
    random_labels = first_classifier.generate_random_labels(df)
    # print(random_labels)
    assert random_labels.shape[0] == 1000


def test_randomly_setted_labels_acuracy():
    df = first_classifier.generate_data(n_samples=1000)
    # Example usage:
    # Assuming 'df' is your pandas DataFrame
    random_labels = first_classifier.generate_random_labels(df)
    # Assuming 'df' is your pandas DataFrame and 'Target' is the name of the target column
    accuracy = first_classifier.evaluate_accuracy(random_labels, df, 'Target')
    print(f"Accuracy of the random labels: {accuracy*100:.2f}%")


def test_handmade_classifier():
    df = first_classifier.generate_data(n_samples=1000)
    pred_labels = first_classifier.hand_made_empirical_classifier(df)
    accuracy = first_classifier.evaluate_accuracy(pred_labels, df, 'Target')
    print(f"Accuracy of our handmade classifier:{accuracy*100:.2f}%")


def test_sklearn_classifier():
    df = first_classifier.generate_data(n_samples=1000)
    pred_labels = first_classifier.decision_tree_classifier(df)
    accuracy = first_classifier.evaluate_accuracy(pred_labels, df, 'Target')
    print(f"Accuracy of our Decision Tree classifier:{accuracy*100:.2f}%")


def test_sklearn_classifier_properly():
    df = first_classifier.generate_data(n_samples=1000)
    clf = first_classifier.train_DecisionTree(df)

    df_test = first_classifier.generate_data(n_samples=10000)
    pred_labels = first_classifier.eval_DecisionTree(clf, df_test)

    accuracy = first_classifier.evaluate_accuracy(pred_labels, df_test, 'Target')
    print(f"Accuracy of our Decision Tree classifier:{accuracy*100:.2f}%")

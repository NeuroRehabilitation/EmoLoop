import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, LeavePGroupsOut

try:
    from physiological_data.lib.peak_detector import peak_detector
except (ImportError, ModuleNotFoundError):
    from physiological_data.lib.peak_detector import peak_detector


def closestIndex(data, value):
    return np.argmin(np.abs(np.array(data) - value))


def strIndex(input, corr):
    result = []
    for i, value in enumerate(input):
        if value == corr:
            result.append(i)
    return np.ravel(result)


def windowing(signal, sampling_rate=1000, time_window=0.25, overlap=0):

    app_window = int(sampling_rate * time_window)

    num_windows = int(signal.shape[0] / app_window)
    print(f"# of Windows: {num_windows, sampling_rate, time_window, signal.shape[0]}")
    signal_windows = np.zeros(shape=(num_windows, app_window, signal.shape[1]))

    for i, win in enumerate(range(0, signal.shape[0], app_window)):
        if win + app_window < len(signal) and i < num_windows:
            signal_windows[i] = signal[win : win + app_window]
    return signal_windows


def standardizeFeatures(features, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
    features = features - mean
    features = features / std
    return features, mean, std


def normalizeFeatures(features, mean=None, max_value=None):
    if mean is None and max_value is None:
        mean_value = np.mean(features)  # Calculate the mean of the signal
        abs_features = np.abs(features)  # Calculate the absolute of the signal
        max_value = np.max(features)  # Get the maximum value of the absolute signal

    normalised_features = (features - mean_value) / max_value
    return normalised_features, mean, max_value


def quantize(data, d=1000):
    scaled_data = data * d
    rounded_data = np.around(scaled_data)
    quantised_data = np.array(rounded_data, dtype=np.int)
    return quantised_data


# def plotDataset(features, labels):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     task, baseline = [], []
#     for i in range(len(labels)):
#         if labels[i] == 'task':
#             task.append(features[i])
#         else:
#             baseline.append(features[i])
#     task, baseline = np.array(task), np.array(baseline)
#     ax.scatter(task[:,0], task[:,1], task[:,2], zdir='z', s=20, color='blue', depthshade=True)
#     ax.scatter(baseline[:,0], baseline[:,1], baseline[:,2], zdir='z', s=20, color='red', depthshade=True)
#     plt.show()


def saveToTSV(features, labels):
    np.savetxt("features.tsv", features, delimiter="\t")
    np.savetxt("labels.tsv", np.reshape(labels, (-1, 1)), fmt="%s")


def classify(features, labels, groups=None, n_estimators=100):
    if groups is not None:
        participants = np.array(list(map(int, groups)))
        model = LeavePGroupsOut(n_groups=1)
        iterator = model.split(features, np.ravel(labels), participants)
    else:
        model = KFold(n_splits=10, shuffle=True, random_state=42)
        iterator = model.split(features)
    acc, f1 = [], []
    for i, (train_index, test_index) in enumerate(iterator):
        # For each iteration, we divide our dataset in train and test set.
        samples_train, samples_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        samples_train, mean, std = normalizeFeatures(samples_train)
        samples_test, _, _ = normalizeFeatures(samples_test, mean, std)
        saveToTSV(
            np.vstack([samples_train, samples_test]),
            np.vstack([labels_train, labels_test]),
        )

        # Build the random forest clasdsifier.
        random_forest = RandomForestClassifier(
            n_estimators=n_estimators, criterion="entropy", n_jobs=-1, random_state=42
        )

        # random_forest = DecisionTreeClassifier(random_state=42)

        # Train the classifier on the training set.
        random_forest = random_forest.fit(samples_train, np.ravel(labels_train))

        # Test the classifier on the testing set.
        results = random_forest.predict(samples_test)

        # This step is not necessary for the classification procedure, but is important to store the values
        # of accuracy to calculate the mean and standard deviation values and evaluate the performance of the classifier.
        acc.append(accuracy_score(np.ravel(labels_test), results) * 100)
        f1.append(
            f1_score(
                [0 if x == "task" else 1 for x in labels_test],
                [0 if x == "task" else 1 for x in results],
            )
            * 100
        )
        print(
            f"{i} - Accuracy: {accuracy_score(labels_test, results)*100:.2f}; F1-Score: {f1_score([0 if x =='task' else 1 for x in labels_test], [0 if x =='task' else 1 for x in results])*100:.2f}%"
        )
    return acc, f1, random_forest


# if __name__ == '__main__':
#     print(closestIndex([-1, 1, 2, 0, 35], .5))

try:
    from acquisition import *
except (ImportError, ModuleNotFoundError):
    from lib.acquisition import *

folders = ['05_11_2020_1',
           '05_11_2020_2',
           '10_11_2020_2',
           '27_11_2020',
           '11_12_2020_1',
           '11_12_2020_2'
           ]

features, labels, participants = [], [], []

for folder in folders:
    path = os.path.join('..\..\acquisitions\Acquisitions', folder)
    acquisition = Acquisition(r'..\acquisitions\Acquisitions\11_12_2020_1')
    timestamps, initialLabels = acquisition.getPsychoTimestamps(), acquisition.getPsychoActivity()
    acquisition.getBiosignalsSensors()
    acquisition.segmentWindowingBiosignals(acquisition.signal, timestamps, initialLabels)
    acquisition.extractFeatures(acquisition.segmentedBiosignals)
    if len(features) == 0:
        features, labels, participants = acquisition.getDataset()
    else:
        featuresP, labelsP, participantP = acquisition.getDataset()
        features = np.vstack(features, featuresP)
        labels = np.vstack(labels, labelsP)
        participants = np.vstack(participants, participantP)

saveToTSV(features, labels)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
################# Classification #################
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, LeaveOneGroupOut

# model = KFold(n_splits=10, shuffle=True, random_state=42)
model = LeaveOneGroupOut()

acc, f1 = [], []
for train_index, test_index in model.split(features, labels, participants):
    # For each iteration, we divide our dataset in train and test set.
    samples_train, samples_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # Build the random forest clasdsifier.
    random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1, random_state=42)

    # Train the classifier on the training set.
    random_forest = random_forest.fit(samples_train, labels_train)

    # Test the classifier on the testing set.
    results = random_forest.predict(samples_test)

    # This step is not necessary for the classification procedure, but is important to store the values
    # of accuracy to calculate the mean and standard deviation values and evaluate the performance of the classifier.
    acc.append(accuracy_score(labels_test, results)*100)
    f1.append(f1_score([0 if x =='task' else 1 for x in labels_test], [0 if x =='task' else 1 for x in results])*100)

print(f"Accuracy: {np.mean(acc):.2f} +- {np.std(acc):.2f}%")
print(f"F1-Score: {np.mean(f1):.2f} +- {np.std(f1):.2f}%")

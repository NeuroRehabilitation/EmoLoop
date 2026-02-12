from acquisition import *


def concatenateSegments(segments):
    out = np.empty((segments.shape[0], segments.shape[1] * segments.shape[2]))
    for i in range(segments.shape[0]):
        out[i] = np.concatenate(segments[i])
    return out


folders = [
    "05_11_2020_1",
    "05_11_2020_2",
    "10_11_2020_2",
    "27_11_2020",
    "11_12_2020_1",
    "11_12_2020_2",
]

# folders = ['05_11_2020_1',]


segments, labels, group = [], [], []
for i, folder in enumerate(folders):
    path = os.path.join(r"..\acquisitions\Acquisitions", folder)

    acquisition = Acquisition(path)

    timestamps, intialLabels = (
        acquisition.getPsychoTimestamps(),
        acquisition.getPsychoActivity(),
    )

    acquisition.deleteArtifacts()
    acquisition.getBiosignalsSensors(ALL, rightPos=True)
    acquisition.convertSensors()
    acquisition.segmentWindowingBiosignals(
        acquisition.signal, timestamps, intialLabels, timeWindow=0.128
    )

    if len(segments) == 0:
        segments = concatenateSegments(acquisition.segmentedBiosignals)
        print(segments.shape)
        labels = np.array(acquisition.labels)
        group = np.array([folder] * segments.shape[0])
    else:
        segments = np.vstack(
            [segments, concatenateSegments(acquisition.segmentedBiosignals)]
        )
        labels = np.concatenate([labels, acquisition.labels])
        group = np.concatenate(
            [group, [folder] * acquisition.segmentedBiosignals.shape[0]]
        )

dataset = np.hstack([segments, labels.reshape(-1, 1), group.reshape(-1, 1)])
np.save("dataset_0_128s", dataset)

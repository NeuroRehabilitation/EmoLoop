import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch import optim

try:
    from dataLoader import SensorsDataset
except (ImportError, ModuleNotFoundError):
    from lib.dataLoader import SensorsDataset


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._features = None

        self.encoder_hidden_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=kwargs["input_shape"])

        self.encoder_output_layer.register_forward_hook(self.save_outputs_hook(self.encoder_output_layer))

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, self._features

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features = output.detach()
        return fn


def train_classifier(epochs, train_loader):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=1024).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    features, labels = None, None

    for epoch in range(epochs):
        loss = 0
        for batch_features, batch_labels, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = torch.tensor(batch_features).view(1, -1)
            print(batch_features.size)
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs, feat = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

            if epoch == epochs - 1:
                if type(features) != type(None):
                    features = np.vstack([features, feat])
                    labels = np.concatenate([labels, batch_labels])
                else:
                    features = feat
                    labels = batch_labels

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    return model, features, labels


if __name__ == '__main__':
    groups = ['05_11_2020_1',
           '05_11_2020_2',
           '10_11_2020_2',
           '27_11_2020',
           '11_12_2020_1',
           '11_12_2020_2'
           ]

    data = SensorsDataset(sensors=[1], groups=groups[:-1])

    model, features, labels = train_classifier(20, data)
    random_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    random_forest = random_forest.fit(features, labels)

    print(features.shape)

    del data
    test_loader = SensorsDataset(sensors=[1], groups=groups[-1])
    result = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    features, labels = [], []
    for test_batch, test_labels in test_loader:
        test_batch = test_batch.view(-1, 1024).to(device)
        output, feat = model(test_batch)
        train_loss = criterion(output, test_batch)
        loss = train_loss.item()

        features.append(feat)
        labels.append(test_labels)

        result += loss
    print("loss = {:.6f}".format(result / len(test_loader)))

    features, labels = np.vstack(features), np.concatenate(labels)
    results = random_forest.predict(features)

    print(accuracy_score(results, labels))

    # print(features, features.shape)

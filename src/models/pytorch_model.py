# code heavily inspired and at times copied from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
import pandas as pd
import torch
from torch.utils.data import Dataset
import string
import unicodedata
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
from dotenv import load_dotenv
import os

allowed_characters = string.ascii_letters + ".,;''"
n_letters = len(allowed_characters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


def letterToIndex(letter):
    return allowed_characters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class NamesDataset(Dataset):
    def __init__(self, data_dir, brewery_dir):
        data_df = pd.read_csv(data_dir)
        data_df['beer_name'] = data_df['beer_name'].apply(unicodeToAscii)
        brewery_df = pd.read_csv(brewery_dir)
        brewery_ids = data_df['brewery_id'].tolist()
        id_to_location = dict(zip(brewery_df['id'], brewery_df['location']))
        print(data_df[data_df['beer_name'].isna()])

        # get the data to list
        self.data = data_df['beer_name'].tolist()
        self.data_tensor = data_df['beer_name'].apply(lineToTensor).tolist()
        self.labels = [id_to_location[id] for id in brewery_ids]

        # get rid of the empty strings
        empty_string_indices = [i for i, x in enumerate(self.data) if x == '']
        for empty_string_index in sorted(empty_string_indices, reverse=True):
            del self.data[empty_string_index]
            del self.data_tensor[empty_string_index]
            del self.labels[empty_string_index]

        # prepare the label tensor
        self.labels_tensor = []
        labels_set = set(self.labels)
        # gives you the unique labels
        self.labels_unique = list(labels_set)
        for idx in range(len(self.labels)):
            self.labels_tensor.append(torch.tensor([self.labels_unique.index(self.labels[idx])], dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.labels_tensor[idx], self.data_tensor[idx], self.labels[idx], self.data[idx]


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output


def train(rnn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad()  # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:  # for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses


def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval()  # set to eval mode
    with torch.no_grad():  # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())  # numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


if __name__ == "__main__":
    load_dotenv()
    dataset = NamesDataset(data_dir=os.getenv("PATH_TO_BEERADVOCATE_BEERS"),
                           brewery_dir=os.getenv("PATH_TO_BEERADVOCATE_BREWERIES"))
    train_set, test_set = torch.utils.data.random_split(dataset, [.85, .15],
                                                        generator=torch.Generator().manual_seed(2024))
    n_hidden = 128
    rnn = CharRNN(n_letters, n_hidden, len(dataset.labels_unique))
    start = time.time()
    all_losses = train(rnn, train_set, n_epoch=27, report_every=5, learning_rate=0.15)
    end = time.time()
    print(f"training took {end - start:.2f} seconds")
    evaluate(rnn, test_set, classes=dataset.labels_unique)
    torch.save(rnn, "full_model.pth")
    print("model saved")


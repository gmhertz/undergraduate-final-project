import glob
import os
import matplotlib.pyplot as plt

import librosa
import numpy as np
from keras import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size // 2)

def extract_features( sub_dirs, file_ext="*.wav"):
    window_size = hop_length * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(sub_dir, file_ext)):
            sound_clip, _ = librosa.load(fn, sr=sample_rate)
            print('Extracting features from: ' + fn)
            label = fn.split('/')[0].split('_')[0]
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
                    logspec = librosa.power_to_db(melspec, ref=np.max)
                    logspec = logspec / 80 + 1
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)
    features = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    np_labels = np.array(labels, dtype=np.int)
    unique, counts = np.unique(np_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    return np.array(features), np_labels

def load_date():
    file_url = '/home/marcelo/final_project/test'
    tr_sub_dirs = ["0", "1"]
    #tr_features, tr_labels = extract_features(tr_sub_dirs)
    #np.savez(file_url, tr_features, tr_labels)
    #return tr_features, tr_labels
    npread = np.load(file_url + '.npz')
    return npread['arr_0'], npread['arr_1']

def create_model():
    model = Sequential()

    # Clearly an overfitting model
    '''
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(bands, frames, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    '''

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(bands, frames, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
def plot_epoch(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    epochs = range(1, len(loss_values) + 1)
    print("Epochs: " + str(epochs))
    print("Traing Loss: " + str(loss_values))
    print("Validation loss: " + str(val_loss_values))
    print("Training Accuracy: " + str(acc_values))
    print("Validation Accuracy: " + str(val_acc_values))


    '''
    plt.subplot(211)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()
    '''

def train_and_evaluate_model(model, xtrain, ytrain, xtest, ytest):
    history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=32, epochs=50, verbose=1)
    plot_epoch(history)
    y_predicted_train = model.predict_classes(xtrain)
    y_predicted_test = model.predict_classes(xtest)

    f1_score_train = skm.roc_auc_score(ytrain, y_predicted_train)
    f1_score_test = skm.roc_auc_score(ytest, y_predicted_test)
    print("F1-score train: %.4f" % f1_score_train)
    print("F1-score test: %.4f" % f1_score_test)

    return f1_score_train, f1_score_test

#seed = 123
#np.random.seed(seed)  # for reproducibility

bands = 60
frames = 40
hop_length=256
n_fft=1024

sample_rate = 8000

n_folds = 2
X, Y = load_date()

# Get metrics
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

f1_train_scores = []
f1_test_scores = []

for i, (train, test) in enumerate(skf.split(X, Y)):
        print("Running Fold", i+1, "/", n_folds)
        model = None # Clearing the NN.
        model = create_model()

        # Generate batches from indices
        xtrain, xtest = X[train], X[test]
        ytrain, ytest = Y[train], Y[test]
        f1_train, f1_test = train_and_evaluate_model(model, xtrain, ytrain, xtest, ytest)
        f1_train_scores.append(f1_train)
        f1_test_scores.append(f1_test)

print("F1 train: " + str(f1_train_scores))
print("F1 test: " + str(f1_test_scores))
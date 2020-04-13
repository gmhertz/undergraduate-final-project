# Identification of Aedes Aegypti mosquito through audio using Machine Learning Techniques

## The research

This project was my thesis to graduate in Electrical Engineering at [Federal University of Rio Grande do Sul](http://www.ufrgs.br/).

You can read it in Portuguese only: [PDF Format](https://github.com/marceloschreiber/undergraduate-final-project/blob/master/Monografia.pdf)

### Abstract

(copy from the document for convenience)

Mosquitos are a vector of infectious diseases, such as malaria, dengue and yellow fever, and affect
thousands of people annually. One way to combat these vectors is by mapping these species
geographically. This work explores the possibility of identifying the Aedes aegypti mosquito in
specific, using its audio and Machine Learning techniques. So that future works can use the
classifier model proposed here to assist in the mapping of mosquito species. Each species has a
different flight frequency, different studies show different frequency ranges, and usually, it’s
contained between 200 and 700 Hz. It was used a dataset containing cellphone flight recordings of
20 species, two of these species being subclassified into strains, totaling 23 different classes. All
recordings were resampled to 8 kHz and segments that didn’t have a sound of mosquitoes flying
were removed. Convolutional neural networks were trained using supervised learning and K-fold
cross-validation with 10-folds. The recordings’ spectrograms were used as a feature so that the
flight frequency of the mosquito is represented visually along the time. Three classifiers were
trained to compare their results: binary classifier between Aedes aegypti vs remainder classes,
multiclass classifier and an ensemble composed by binary classifiers (where each binary classifier
performs the classification between Aedes aegypti vs a specific species). The performance metrics
obtained were higher than those found in other works, which do not use Machine Learning to
perform the classification. The multiclass classifier had an accuracy of 80.00% ±1.3, whereas the
binary classifier Aedes aegypti vs remainder classes had an accuracy of 97.78% ±0.73 and the
ensemble shown the best result with an accuracy of 98.34 ±0.27. Therefore, it is possible to identify
mosquito species, especially Aedes aegypti, using their flight sound and Machine Learning
techniques.

## Dataset

The original dataset used was from Stanford: [Using mobile phones as acoustic snesors for high-throughput mosquito serveillance](http://web.stanford.edu/group/prakash-lab/cgi-bin/labsite/publications/). The original recordings had audio portions that were silence and did not have the flying sound of a mosquito. These parts were manually removed. All audios were resampled to 8 kHz as part of normalization.

New Dataset: [Google drive link](https://drive.google.com/file/d/1omRl9IPmYSn4SzBwXIRogzSKTPqgq2VU/view?usp=sharing)

## Files breakdown:

### General

- `plot_mel_spec.py` - Comparison between Hertz and Mel scale
- `data_cleaning/audio_duration.py` - Print total audio duration of the files
- `data_cleaning/decimator.sh` - Uses the `sox` utility to resampled to 8 kHz

### Binary Classifier

- `binary_classifier/train_cnn.py` - It loads the audio file from a specific directory, inside this directory, it must have directories named: **0** and **1**. **0** must have all the audios for _Aedes Aegypti_ and **1** all the audios for the other specie
- `binary_classifier/ensemble.py` - After running the file above for all species it was used their output models to create an ensemble. In order to run it's necessary to perform the same steps: Change the directory and copy the audio of _Aedes Aegypti_ to directory **0** and all other audios to folder **1**
- `binary_classifier/overfitting_example.py` - Used when I was trying to verify if I was overfitting my model

### Jupyter Notebooks

- `jupyter_notebooks/features_study.ipynb` - Used in the very beginning when I was trying to understand the dataset and generate plots for the final document.
- `jupyter_notebooks/results.ipynb` - Used after I had trained the models and was performing a comparison between them

### Multiclass Classifier

- `multiclass_classifier/train_cnn.py` - It's necessary to separate the audios between folders **0** to **22**. If in doubt of the ordering check the file below based on the labels order. **0** for _Aedes Aegypti_, **1** for _Aedes Albopictus_, etc.
- `multiclass_classifier/confusion_matrix.py` - To generate the confusion matrix based on the results of the file above

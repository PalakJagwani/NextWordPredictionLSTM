## Next-Word Prediction using LSTM - Hamlet Dataset
## Project Overview
This project utilizes an LSTM-based (Long Short-Term Memory) deep learning model to predict the next word in a sequence of words. The model was trained on the Shakespeare's Hamlet dataset, aiming to generate the next word given a sequence of previous words. This project was completed as part of my learning journey in Krish Naik's Data Science and Machine Learning course on Udemy.

## Dataset
The dataset used in this project is the text of Shakespeare's Hamlet, which is a well-known play written in Early Modern English. 

## Model Overview
- Model Type: LSTM (Long Short-Term Memory)

## Libraries Used:
- TensorFlow
- Keras
- NumPy
- Pandas

## Architecture
- Embedding Layer: Transforms input tokens (words) into dense vectors of fixed size.
- LSTM Layers: Two LSTM layers for learning the sequential dependencies of the words.
- Dense Layer: A fully connected layer to predict the next word.
- Activation: Softmax to output probabilities for each word in the vocabulary.

## Steps Involved
- Data Preprocessing:
- Tokenization and vectorization of the Hamlet dataset.
- Sequence padding to ensure uniform input length.
- Splitting the data into training and testing sets.

## Model Building:

Implementing an LSTM model with an embedding layer, two LSTM layers, and a dense layer to predict the next word.

Compiling the model using categorical cross-entropy loss and Adam optimizer.

## Training:

The model was trained for 50 epochs, and the loss and accuracy were monitored.

Evaluation:
The model was evaluated on the test data, achieving an accuracy of 47% for the next-word prediction task.

## Conclusion
This project aimed to predict the next word in a sequence using a dataset from Shakespeare's Hamlet. With an accuracy of 47%, the model successfully generated contextually relevant words, but there is room for improvement. Future improvements could involve:

Using more advanced models (e.g., Bidirectional LSTM, GRU, or Transformer-based models).
Further refining the dataset and preprocessing techniques.
This project provided hands-on experience in working with text data, LSTM networks, and deep learning techniques in general.

## Installation
To run the project, ensure you have the required dependencies installed:
```
pip install tensorflow numpy pandas matplotlib
```

## Download or clone the repository.
```
https://github.com/PalakJagwani/NextWordPredictionLSTM.git
cd NextWordPredictionLSTM
```

## Learnings from Krish Naik’s Course
- This project was completed as part of the Data Science and Machine Learning Bootcamp by Krish Naik on Udemy. The course provided in-depth knowledge on deep learning techniques, including:
- Understanding and building LSTM-based models.
- Working with text data and applying tokenization and vectorization.
- The concepts learned in the course played a crucial role in building and fine-tuning this next-word prediction model.
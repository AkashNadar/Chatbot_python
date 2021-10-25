import random
import json
import pickle
import numpy as np

#reduce the word to its stem (work working worked works) all words are same
import nltk
from numpy.core.defchararray import mod
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize is splitting of sentences into the words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        #appending a tuple
        documents.append((word_list, intent['tag']))
        # checking the class is in the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#  lemmatizer means that it takes all the training data from the jason and puts together in to the list
# ['hi', 'how are you', 'bye'] etc
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
#avoid duplicate set() avoid duplicates 
words = sorted(set(words))

classes = sorted(set(classes))

# for saving into a file 
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# ML(Wanted to convert the words into the numeric values so that we can do nural-networks)
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    words_patterns = document[0]
    words_patterns = [lemmatizer.lemmatize(word.lower()) for word in words_patterns]
    for word in words:
        bag.append(1) if word in words_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# features and label to train our nural network
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Nural network part
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#scales the results in outputlayer %of how likely to have the output 
model.add(Dense(len(train_y[0]), activation='softmax'))

#Stochastic gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Done")
from data import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, MaxPooling2D, Conv2D, LSTM, GRU, Bidirectional, BatchNormalization, Embedding
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from keras.optimizers import SGD

base_model = load_model("C:/Users/yagao/Documents/ASO/data/ASO.keras")

x = base_model.output
x = Flatten(name='ft_flatten')(x)
x = Dense(128, activation='relu', name='ft_dense1')(x)
x = Dropout(0.5, name='ft_dropout1')(x)
x = Dense(64, activation='relu', name='ft_dense2')(x)
x = Dropout(0.5, name='ft_dropout2')(x)
x = Dense(32, activation='relu', name='ft_dense3')(x)
x = Dropout(0.5, name='ft_dropout3')(x)
prediction = Dense(2, activation='softmax', name='ft_res')(x)

model = Model(inputs=base_model.input, outputs=prediction)


for layer in base_model.layers:
    layer.trainable = True

model.compile(loss=BinaryCrossentropy(),optimizer=SGD(lr=0.05, momentum=0.2),metrics=['accuracy'])

print(model.summary())

(df, seqs, structures, labels) = load_sequence_data("C:/Users/yagao/Documents/ASO/data/fine_tune.csv")

trainAttrX, testAttrX, trainX, testX, trainStructX, testStructX, trainY, testY = train_test_split(df, seqs, structures, labels, test_size=0.1, shuffle=True, random_state=123)

trainAttrX, testAttrX = process_sequence_attributes(trainAttrX, testAttrX,
                                                    sc_path="C:/Users/yagao/PycharmProjects/CNNproject/ASOscalers.joblib")

preds = base_model.predict([testAttrX, testX, testStructX])
probs = np.array(preds[:,1])
y_test = np.array(testY[:,1])
fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=100, mode='max')

model.fit([trainAttrX, trainX, trainStructX], trainY,
            validation_data=([testAttrX, testX, testStructX], testY),
            batch_size=64, epochs=300, verbose=1, callbacks=early_stop)
model.save("C:/Users/yagao/Documents/ASO/data/ASO2.keras")

preds = model.predict([testAttrX, testX, testStructX])
probs = np.array(preds[:,1])
y_test = np.array(testY[:,1])
fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


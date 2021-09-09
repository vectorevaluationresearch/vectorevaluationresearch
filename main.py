from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from sklearn.metrics import classification_report


def svm_model(input_length, input_dim, hidden_dim):
    input_data = Input(shape=(input_length, input_dim))
    input_data_f = Flatten()(input_data)
    hidden = Dense(hidden_dim, activation='relu')(input_data_f)
    hidden = Dense(2, kernel_regularizer=l2(0.01))(hidden)
    output = Activation('softmax')(hidden)
    model = Model(input_data, output)
    model.compile(loss='hinge', optimizer='adadelta', metrics=['Precision', 'Recall', 'AUC'])
    return model


def model_1(data, split):
    processed_data = []
    for i in range(len(data)):
        processed_data.append(data[i][2])
    return np.array(processed_data[:split]), np.array(processed_data[split:])


def model_2(data, split):
    processed_data = []
    for i in range(len(data)):
        subprocess = []
        for j in range(len(data[i][1])):
            label = data[i][1][j]
            if label == 1:
                subprocess.append(np.concatenate((data[i][2][j], np.zeros((768,))), axis=0))
            else:
                subprocess.append(np.concatenate((np.zeros((768,)), data[i][3][j]), axis=0))
        processed_data.append(subprocess)
    return np.array(processed_data[:split]), np.array(processed_data[split:])


def model_3(data, split):
    processed_data = []
    for i in range(len(data)):
        subprocess = []
        for j in range(len(data[i][1])):
            label = data[i][1][j]
            if label == 1:
                subprocess.append(data[i][2][j])
            else:
                subprocess.append(data[i][3][j])
        processed_data.append(subprocess)
    return np.array(processed_data[:split]), np.array(processed_data[split:])


def model_4(data, split):
    processed_data = []
    for i in range(len(data)):
        subprocess = []
        for j in range(len(data[i][1])):
            label = data[i][1][j]
            if label == 1:
                subprocess.append(data[i][2][j])
        processed_data.append(subprocess)
    for i in range(len(data)):
        max_length = len(data[i][2][0])
        if len(processed_data[i]) < max_length:
            if len(processed_data[i]) > 0:
                processed_data[i] = np.concatenate((processed_data[i], np.zeros((max_length - len(processed_data[i]), 768))), axis=0)
            else:
                processed_data[i] = np.zeros((max_length - len(processed_data[i]), 768))
    return np.array(processed_data[:split]), np.array(processed_data[split:])


def get_label_data(data, split):
    processed_data = []
    for i in range(len(data)):
        processed_data.append(data[i][0])
    return np.array(processed_data[:split][:]).reshape(-1, 2), np.array(processed_data[split:][:]).reshape(-1, 2)


if __name__ == '__main__':
    split = 180
    input_data = np.load(os.path.join(os.getcwd(), "/processed/data.npy"), allow_pickle=True)
    label_data, test_label = get_label_data(input_data, split)
    input_1, test_1 = model_1(input_data, split)
    input_2, test_2 = model_2(input_data, split)
    input_3, test_3 = model_3(input_data, split)
    input_4, test_4 = model_4(input_data, split)
    m_1 = svm_model(len(input_1[0]), len(input_1[0][0]), 128)
    m_2 = svm_model(len(input_2[0]), len(input_2[0][0]), 128)
    m_3 = svm_model(len(input_3[0]), len(input_3[0][0]), 128)
    m_4 = svm_model(len(input_4[0]), len(input_4[0][0]), 128)
    m_1.fit(input_1, label_data, batch_size=128, epochs=50)
    m_2.fit(input_2, label_data, batch_size=128, epochs=50)
    m_3.fit(input_3, label_data, batch_size=128, epochs=50)
    m_4.fit(input_4, label_data, batch_size=128, epochs=50)
    out_1 = m_1.predict(test_1)
    out_1 = np.argmax(out_1, axis=1)
    test_label = np.argmax(test_label, axis=1)
    print(classification_report(test_label, out_1))
    out_2 = m_2.predict(test_2)
    out_2 = np.argmax(out_2, axis=1)
    print(classification_report(test_label, out_2))
    out_3 = m_3.predict(test_3)
    out_3 = np.argmax(out_3, axis=1)
    print(classification_report(test_label, out_3))
    out_4 = m_4.predict(test_4)
    out_4 = np.argmax(out_4, axis=1)
    print(classification_report(test_label, out_4))


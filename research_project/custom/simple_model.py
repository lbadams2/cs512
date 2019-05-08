import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


TRAIN_PATH = 'research_project/nltk/train.txt'
DEV_PATH = 'research_project/nltk/dev.txt'
TEST_PATH = 'research_project/nltk/test.txt'

def read_train_file(path):
    sentences = []
    with open(path) as f:
        sentence = []
        for line in f:
            if line.isspace():
                if not sentence:
                    continue
                sentences.append(sentence)
                sentence = []
            else:
                tokens = line.rstrip('\n').split()
                sentence.append(tokens)
    return sentences

def create_word_idx():
    word_map = {}
    tag_map = {}
    cur_ind = 0
    for path in [TRAIN_PATH, DEV_PATH, TEST_PATH]:
        with open(path) as f:
            for line in f:
                if not line.isspace():
                    tokens = line.rstrip('\n').split()
                    word = tokens[0]
                    if word not in word_map:
                        word_map[word] = cur_ind
                        cur_ind += 1
    word_map['ENDPAD'] = cur_ind

    tag_map['B-PER'] = 0
    tag_map['I-PER'] = 1
    tag_map['B-ORG'] = 2
    tag_map['I-ORG'] = 3
    tag_map['B-LOC'] = 4
    tag_map['I-LOC'] = 5
    tag_map['B-MISC'] = 6
    tag_map['I-MISC'] = 7
    tag_map['O'] = 8
    return word_map, tag_map

def run():
    sentences = read_train_file(TRAIN_PATH)
    word_map, tag_map = create_word_idx()
    max_len = max([len(s) for s in sentences])
    X = [[word_map[w[0]] for w in s] for s in sentences]
    n_words = len(word_map)
    n_tags = 9
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
    y = [[tag_map[w[2]] for w in s] for s in sentences]
    y_testing = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag_map["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y_testing]

    dev_sentences = read_train_file(DEV_PATH)
    dev_max_len = max([len(s) for s in dev_sentences])
    X_dev = [[word_map[w[0]] for w in s] for s in dev_sentences]
    X_dev = pad_sequences(maxlen=max_len, sequences=X_dev, padding="post", value=n_words - 1)
    y_dev = [[tag_map[w[2]] for w in s] for s in dev_sentences]
    y_dev = pad_sequences(maxlen=max_len, sequences=y_dev, padding="post", value=tag_map["O"])
    y_dev = [to_categorical(i, num_classes=9) for i in y_dev]

    test_sentences = read_train_file(TEST_PATH)
    test_max_len = 33
    X_test = [[word_map[w[0]] for w in s] for s in test_sentences]
    X_test = pad_sequences(maxlen=test_max_len, sequences=X_test, padding="post", value=n_words - 1)
    y_test = [[tag_map[w[2]] for w in s] for s in test_sentences]
    y_test = pad_sequences(maxlen=test_max_len, sequences=y_test, padding="post", value=tag_map["O"])

    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=20, input_length=max_len)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(n_tags, activation="relu"))(model)  # softmax output layer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    history = model.fit(X, np.array(y), batch_size=32, epochs=5, verbose=1)
    model.save('simple_model.h5')
    return model, tag_map, X_test, y_test

def predict(model, X_test, y_test):
    '''
    test_sentences = read_train_file(TEST_PATH)
    word_map, tag_map = create_word_idx()
    n_words = len(word_map)
    n_tags = 9
    test_max_len = max([len(s) for s in test_sentences])
    test_max_len = 33
    X_test = [[word_map[w[0]] for w in s] for s in test_sentences]
    X_test = pad_sequences(maxlen=test_max_len, sequences=X_test, padding="post", value=n_words - 1)
    y_test = [[tag_map[w[2]] for w in s] for s in test_sentences]
    y_test = pad_sequences(maxlen=test_max_len, sequences=y_test, padding="post", value=tag_map["O"])
    #y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]
    '''

    total_predicted = 0
    total_correct = 0
    actual_entity_count = 0
    for i, x in enumerate(X_test):
        x_arr = np.array([x])
        p = model.predict(x_arr)
        p = np.argmax(p, axis=-1)
        tags = y_test[i]
        matching = False
        for j, pred_tag in enumerate(p[0]):
            if pred_tag == 8:
                if matching and tags[j] == 8:
                    total_correct += 1
                matching = False
                continue
            if pred_tag % 2 == 0:
                total_predicted += 1
            if tags[j] % 2 == 0:
                actual_entity_count += 1
            if pred_tag == tags[j]:
                matching = True
            else:
                matching = False
    precision = total_correct / total_predicted
    recall = total_correct / actual_entity_count
    print('precision:', precision)
    print('recall:', recall)
    f1 = 2 * precision * recall / (precision + recall)
    print('f1:', f1)

def pred2label(pred, idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

if __name__ == '__main__':
    model, tag_map, X_test, y_test = run()
    '''
    idx2tag = {i: w for w, i in tag_map.items()}
    test_pred = model.predict(X_test, verbose=1)
    pred_labels = pred2label(test_pred, idx2tag)
    test_labels = pred2label(y_test, idx2tag)
    print(classification_report(test_labels, pred_labels))
    '''
    predict(model, X_test, y_test)


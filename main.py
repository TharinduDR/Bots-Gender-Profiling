from random import seed

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed


def main():
    print("Start executing")
    seed(726)
    set_random_seed(726)

    pre_train = pd.read_csv("data/pan_data_en.tsv", sep='\t')

    rows = []
    for index, row in pre_train.iterrows():
        for tweet in row['text'].split(", '"):
            if len(tweet.split()) > 3:
                new_row = [row['index'], row['humanBot'], row['sex'], tweet]
                rows.append(new_row)

    header = ["id", "human", "gender", "tweet"]

    train_full = pd.DataFrame(rows, columns=header)

    train, test = train_test_split(train_full, test_size=0.2)

    # train["tweet"] = train["tweet"].str.lower()
    # test["tweet"] = test["tweet"].str.lower()
    #
    # train["tweet"] = train["tweet"].apply(lambda x: clean_text(x))
    # test["tweet"] = test["tweet"].apply(lambda x: clean_text(x))
    #
    # train['doc_len'] = train["tweet"].apply(lambda words: len(words.split(" ")))
    # max_seq_len = np.round(train['doc_len'].mean() + train['doc_len'].std()).astype(int)
    #
    # ## some config values
    # embed_size = 300  # how big is each word vector
    # max_features = None  # how many unique words to use (i.e num rows in embedding vector)
    # maxlen = max_seq_len  # max number of words in a question to use #99.99%
    # path = "/data/fasttext/crawl-300d-2M-subword.vec"
    # model_path = "models/lstm_attention_weights_best.h5"
    #
    # ## fill up the missing values
    # X = train["tweet"].fillna("_na_").values
    # X_test = test["tweet"].fillna("_na_").values
    #
    # ## Tokenize the sentences
    # tokenizer = Tokenizer(num_words=max_features, filters='')
    # tokenizer.fit_on_texts(list(X))
    #
    # X = tokenizer.texts_to_sequences(X)
    # X_test = tokenizer.texts_to_sequences(X_test)
    #
    # ## Pad the sentences
    # X = pad_sequences(X, maxlen=maxlen)
    # X_test = pad_sequences(X_test, maxlen=maxlen)
    #
    # ## Get the target values
    # Y = train['human'].values
    #
    # le = LabelEncoder()
    #
    # le.fit(Y)
    # encoded_Y = le.transform(Y) - 1
    #
    # word_index = tokenizer.word_index
    # max_features = len(word_index) + 1
    #
    # embedding_matrix = load_fasttext(word_index, path, max_features)
    # model = lstm_attention(maxlen, max_features, embed_size, embedding_matrix)
    #
    # y_test, bestscore = train_reducing_lr(model, X, encoded_Y, X_test, model_path)
    #
    # y_test = y_test.reshape((-1, 1))
    # pred_test_y = (y_test > np.mean(bestscore)).astype(int)
    # test['predictions'] = le.inverse_transform(pred_test_y + 1)
    #
    # print(accuracy_score(test["human"], test['predictions']))

    train.to_csv("Data/train_en.tsv", sep='\t', encoding='utf-8')
    test.to_csv("Data/test_en.tsv", sep='\t', encoding='utf-8')

    print("finished executing")


if __name__ == "__main__":
    main()

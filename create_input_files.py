from utils import create_input_files, train_word2vec_model

if __name__ == '__main__':
    create_input_files(csv_folder='./data',
                       output_folder='./outdata',
                       # sentence_limit=15,
                       # word_limit=20,
                       # min_word_count=5)
                       sentence_limit=30,
                       word_limit=200,
                       min_word_count=3)

    train_word2vec_model(data_folder='./outdata', algorithm='skipgram')

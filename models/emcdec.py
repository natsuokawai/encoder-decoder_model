from keras.models import Model
from keras.layers import Input, Embebdding, Dense, LSTM


def EncDec(train_X, train_Y, tokenizer_en, tokenizer_ja):
    emd_dim = 256
    hd_dim = 256
    en_vocab_size = len(tokenizer_en.word_index) + 1
    ja_vocab_size = len(tokenizer_ja.word_index) + 1
    train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.02, random_state=42)
    train_X = pad_sequences(train_X, padding='post') # padding
    train_Y = pad_sequences(train_Y, padding='post') # padding
    seqX_len = len(train_X[0])
    seqY_len = len(train_Y[0])

    """
    encoder
    """
    encoder_inputs = Input(shape=(seqX_len,))
    # embedding english vocabulary, en_vocab_size(input_shape) -> emb_dim(embedding_dimension)
    encoder_embedded = Embedding(en_vocab_size, emb_dim, mask_zero=True)(encoder_inputs)
    # get internal state of LSTM by return_state=True (referrence: https://keras.io/layers/recurrent/)
    _, *encoder_states = LSTM(hid_dim, return_state=True)(encoder_embedded)

    """
    decoder
    """
    decoder_inputs = Input(shape=(seqY_len,))
    decoder_embedding = Embebdding(ja_vocab, emb_dim)
    # embedding japanese vocabulary
    decoder_embedded = decoder_embedding(decoder_inputs)
    decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
    # decode embedded sequences
    decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    deocder_dense = Dense(ja_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    return model

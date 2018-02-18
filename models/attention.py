from keras.models import Model
from keras.layers import Input, Permute, Activation, Embedding, Dense, LSTM, concatenate, dot
from keras import backend as K

def Attention(train_X, train_Y, tokenizer_en, tokenizer_ja):
    emb_dim = 256
    hid_dim = 256
    att_dim = 256
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
    encoder_embebdded = Embedding(en_vocab_size, emb_dim, mask_zero=True)
    encoded_seq, *endoer_states = LSTM(hid_dim, return_sequences=True, return_state=True)

    """
    decoder
    """
    decoder_inputs = Input(shape=(seqY_len,))
    decoder_embedding = Embedding(ja_vocab, embb_dim)
    decoder_embedded = decoder_embedding(decoder_inputs)
    decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
    decoded_seq, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

    """
    attention
    """
    score_dense = Dense(hid_dim)
    score = score_dense(decoded_seq)
    score = dot([score, encoded_seq], axes=(2,2))
    attention = Activation('softmax')(score)
    context = dot([attention, encoded_seq], axes=(2,1))
    concat = concatenate([context, decoded_seq], axes=2)
    attentional = attention_dense(concat)
    output_dense = Dense(ja_vocab_size, activation='softmax')
    outputs = output_dense(attentional)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    return model

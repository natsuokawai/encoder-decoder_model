from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from encdec import EncDec

def load_data(file_path):
    tokenizer = Tokenizer(filters="")
    whole_texts = []
    for line in open(file_path, encoding='utf-8'):
        whole_texts.append("<s> " + line.strip() + " </s>")

    tokenizer.fit_on_texts(whole_texts)

    return tokenizer.texts_to_sequences(whole_texts), tokenizer

def decode_sequence(input_seq, bos_eos, max_output_length = 1000):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array(bos_eos[0])
    output_seq= bos_eos[0]

    while True:
        output_tokens, *states_value = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
        output_seq += sampled_token_index

        if (sampled_token_index == bos_eos[1] or len(output_seq) > max_output_length):
            break

        target_seq = np.array(sampled_token_index)

    return output_seq


if __name__ = '__main__':
    train_X, tokenizer_en = load_data('/root/userspace/public/lesson4/train.en')
    train_Y, tokenizer_ja = load_data('/root/userspace/public/lesson4/train.en')

    model = EncDec(train_X, train_Y, tokenizer_en, tokenizer_ja)

    train_target = np.hstack((train_Y[:, 1:], np.zeros((len(train_Y),1), dtype=np.int32)))
    model.fit([train_X, train_Y], np.expand_dims(train_target, -1), batch_size=128, epochs=15, verbose=1, validation_split=0.2)

    # encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # decoder
    decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))] # for pointing state of decorder_lstm

    decoder_inputs = Input(shape=(1,))
    decoder_embedded = decoder_embedding(decoder_inputs) # use trained embedding layer
    decoder_outputs, *decoder_states = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs) # use trained LSTM
    decoder_outputs = decoder_dense(decoder_outputs) # use trained dense layer

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    detokenizer_en = dict(map(reversed, tokenizer_en.word_index.items()))
    detokenizer_ja = dict(map(reversed, tokenizer_ja.word_index.items()))

    text_no = 0
    input_seq = pad_sequences([test_X[text_no]], seqX_len, padding='post')
    bos_eos = tokenizer_ja.texts_to_sequences(["<s>", "</s>"])

    print('input sequences:', ' '.join([detokenizer_en[i] for i in test_X[text_no]]))
    print('decoded sequences:', ' '.join([detokenizer_ja[i] for i in decode_sequence(input_seq, bos_eos)]))
    print('correct sequences:', ' '.join([detokenizer_ja[i] for i in test_Y[text_no]]))

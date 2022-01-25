from flask import Flask, render_template, request
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from AttentionLayer import  AttentionLayer

from Tokenizer_text import Tokenizer_text

global max_len_text, max_len_summary

max_len_text = 100
max_len_summary = 15


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

#import model

a_file = open("C:\\Users\\ASUS\\PycharmProjects\\projetDeepL\\projetDL\\target_word_index\\reverse_target_word_index.pkl", "rb")
reverse_target_word_index = pickle.load(a_file)
a_file.close()
a_file = open("C:\\Users\\ASUS\\PycharmProjects\\projetDeepL\\projetDL\\target_word_index\\reverse_source_word_index.pkl", "rb")
reverse_source_word_index = pickle.load(a_file)
a_file.close()
a_file = open("C:\\Users\\ASUS\\PycharmProjects\\projetDeepL\\projetDL\\target_word_index\\target_word_index.pkl", "rb")
target_word_index = pickle.load(a_file)
a_file.close()

#
encoder_model = tf.keras.models.load_model('C:\\Users\\ASUS\\PycharmProjects\\projetDeepL\\projetDL\\encoder_model2.h5')
decoder_model = tf.keras.models.load_model('C:\\Users\\ASUS\\PycharmProjects\\projetDeepL\\projetDL\\decoder_model2.h5',custom_objects={'AttentionLayer': AttentionLayer})

##

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    #print('input_seq: {}, e_out: {} '.format(input_seq,e_out))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
       # print("sampled_token:",sampled_token)
        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
                stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # stop_condition = True
        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
        newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+reverse_source_word_index[i]+' '
    return newString

#########





def text2seq(input_text):
    newSeq=np.zeros((100,), dtype=int)
    for i in range(len(input_text)):
      newSeq[i]= list(reverse_source_word_index.keys())[list(reverse_source_word_index.values()).index(input_text[i])]
    return newSeq

# text_tok = text2seq("got infamous pink line death replaced warranty replacement issue second defective phone replaced seller".split())
# print(text_tok)

# text_tok = tokenizer_text(["live phone far great upgrade samsung galaxy works perfectly usa straight talk"])
# print(text_tok)
#
# print("Review:",seq2text(text_tok))
# print("Predicted summary:",decode_sequence(text_tok.reshape(1,max_len_text)))
#

Tokenizer_text = Tokenizer_text()
@app.route('/',methods=['POST'])
def predict():

    input_text = request.form['input_text']

    input_text2 = request.form['input_text2']
    if (len(input_text)!=0 and len(input_text2)!=0):

        text_tok = text2seq(Tokenizer_text.text_cleaner(input_text).split())
        ps = decode_sequence(text_tok.reshape(1, max_len_text))

        text_tok2 = text2seq(Tokenizer_text.text_cleaner(input_text2).split())
        ps2 = decode_sequence(text_tok2.reshape(1, max_len_text))
        return render_template('index.html',final=ps, text=input_text, final2=ps2, text2=input_text2)

    elif len(input_text)!=0 :
        text_tok = text2seq(Tokenizer_text.text_cleaner(input_text).split())
        ps = decode_sequence(text_tok.reshape(1, max_len_text))
        return render_template('index.html', final=ps, text=input_text)

    elif len(input_text2)!=0 :
        text_tok2 = text2seq(Tokenizer_text.text_cleaner(input_text2).split())
        ps2 = decode_sequence(text_tok2.reshape(1, max_len_text))
        return render_template('index.html', final2=ps2, text2=input_text2)
    else :
        return render_template('index.html', final=input_text, text=input_text)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
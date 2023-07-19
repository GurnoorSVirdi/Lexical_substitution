#!/usr/bin/env python
import string
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import defaultdict, OrderedDict

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List

# used for part 3
stop_words = stopwords.words('english')


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = []
    for word in wn.lemmas(lemma, pos=pos):
        for lem in word.synset().lemmas():
            # make sure to replace underscore with a space
            candidate = lem.name().replace('_', ' ')
            candidates.append(candidate)

    # make sure you remove the lemma
    set_candidates = set(candidates)
    set_candidates.remove(lemma)

    return list(set_candidates)


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context: Context) -> str:
    # get the lemma
    lemma = context.lemma
    # get the part of speech from context
    part_of_speech = context.pos
    # create a dictionary of the frequencies
    frequencies = defaultdict(int)

    # iterate through all lemmas given the lemma and pos
    for lem in wn.lemmas(lemma, pos=part_of_speech):
        # get the synset for each lemma
        curr_synset = lem.synset()
        for lex_lem in curr_synset.lemmas():
            # check if the name of the lexeme lemma is equal to the lemma if is not then go and add count
            if lex_lem.name() != lemma:
                fixed_lemma = lex_lem.name()
                # check for two words/ word phrase
                if '_' in lex_lem.name():
                    fixed_lemma = lex_lem.name().replace('_', ' ')
                # add a count to the current dictionary spot for the lemma
                frequencies[fixed_lemma] += lex_lem.count()

    # get the highest total occurence freq, which is what we wanna return
    #used stack overflow to find the max function
    highest_total_occurence_frequency = max(frequencies, key=lambda x: frequencies[x])
    # return this
    return highest_total_occurence_frequency


def wn_simple_lesk_predictor(context: Context) -> str:
    lemma = context.lemma
    part_of_speech = context.pos

    def calculate_overlap(token, word_context, synset):
        def count_synset_freq(synset):
            for lex_lem in synset.lemmas():
                if lex_lem.name() == lemma:
                    return lex_lem.count()
            return 0

        def normalize_word(x):
            normalized_words = []
            for word in x:
                if word not in stop_words:
                    normalized_words.append(word.lower())
            return normalized_words

        #get the normalized tokena and word contex ... make everything lowercase
        normalized_token = normalize_word(token)
        normalized_word_context = normalize_word(word_context)
        #find the intersection between the two
        word_intersection = set(normalized_token).intersection(set(normalized_word_context))
        #convert to list
        unique_word_intersection = list(set(word_intersection))
        #get the synset frequency using the method above
        synset_freq = count_synset_freq(synset)

        result_tuple = (len(unique_word_intersection), synset_freq)

        return result_tuple

    def synset_tokens(lexeme):
        tokens = tokenize(lexeme.definition())
        for ex in lexeme.examples():
            ex_tok = tokenize(ex)
            tokens += ex_tok
        return tokens

    results = defaultdict(int)
    for lexeme in wn.lemmas(lemma, pos=part_of_speech):
        tokens = synset_tokens(lexeme.synset())
        for hypernym in lexeme.synset().hypernyms():
            tokens += synset_tokens(hypernym)
        word_context = context.left_context + context.right_context
        results[lexeme.synset()] = calculate_overlap(tokens, word_context, lexeme.synset())

    #used stack overflow to find the way to get the first item in an ordered dict
    ordered_results = OrderedDict(sorted(results.items(), key=lambda x: (-x[1][0], -x[1][1], x[0])))
    option = next(iter(ordered_results))

    candidates = defaultdict(int)
    for lex_lem in option.lemmas():
        name = lex_lem.name().replace('_', ' ')
        if name != context.lemma:
            candidates[name] += lex_lem.count()

    if len(candidates) > 0:
        result = max(candidates, key=lambda x: candidates[x])
    else:
        result = None

    return result


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context : Context) -> str:
        #get all possible candidates given the lemma and the part of speech
        candidates = get_candidates(context.lemma, context.pos)
        #create a dictionary of all scores
        synonym_scores = defaultdict(int)
        #for each synonym in candidates
        for synonym in candidates:
            try:
                synonym_scores[synonym] = self.model.similarity(context.lemma, synonym)
            except:
                synonym_scores[synonym] = 0
        most_similar = max(synonym_scores, key=synonym_scores.get)
        return most_similar



class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        #step one was to get the candidates
        candidates = get_candidates(context.lemma, context.pos)
        #convert information into  suitable input representation of the Masked Model
        updated_context = context.left_context + ["[MASK]"] + context.right_context
        #store info
        input_tokens = self.tokenizer.encode(updated_context)
        #word_index
        word_index = self.tokenizer.convert_ids_to_tokens(input_tokens).index("[MASK]")
        #reshape tokens into an array
        input_arr = np.array(input_tokens).reshape((1, -1))
        #run the model/ predict
        #used Edstem #904 for verbose
        outputs = self.model.predict(input_arr, verbose=False)[0][0][word_index]
        #sort the best words in the list
        best_words = np.argsort(outputs)[::-1]

        # Select, from the set of wordnet derived candidate synonyms, the highest-scoring word in the target position (i.e. the position of the masked word)
        for word in self.tokenizer.convert_ids_to_tokens(best_words):
            if word in candidates:
                return word

        return None  # replace for part 5

    def bert_with_lemma_predict(self, context: Context) -> str:
        # Firstly, I realized from part 5, that Bert was the most accurate and precise so I decided to play
        # with this method to make a stronger output
        # my approach to this problem was seeing if i can use the lemma instead of the mask:
        #in order to accomplish this, I changed the Mask to lemma
        #get the lemma
        lemma = context.lemma

        # step one was to get the candidates
        candidates = get_candidates(lemma, context.pos)
        # convert information into  suitable input representation of the Masked Model
        updated_context = context.left_context + [lemma] + context.right_context
        # store info
        input_tokens = self.tokenizer.encode(updated_context)
        # word_index
        word_index = self.tokenizer.convert_ids_to_tokens(input_tokens).index(lemma)
        # reshape tokens into an array
        input_arr = np.array(input_tokens).reshape((1, -1))
        # run the model/ predict
        # used Edstem #904 for verbose
        outputs = self.model.predict(input_arr, verbose=False)[0][0][word_index]
        # sort the best words in the list
        best_words = np.argsort(outputs)[::-1]

        # Select, from the set of wordnet derived candidate synonyms, the highest-scoring word in the target position (i.e. the position of the masked word)
        for word in self.tokenizer.convert_ids_to_tokens(best_words):
            if word in candidates:
                return word

        return None




if __name__ == "__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # print(get_candidates('slow', 'a')) #part 1
        #print(smurf_predictor(context)) # smurfTest
        #prediction = smurf_predictor(context)
        # print(wn_frequency_predictor(context)) #part 2
        #prediction = wn_frequency_predictor(context)  #part 2
        #prediction = wn_simple_lesk_predictor(context)  # part 3
        #prediction = predictor.predict_nearest(context) #part4
        #my best predictor is part 5 for now.... the bert
        #prediction = BertPredictor().predict(context) #part 5

        #part6 TODO
        #changed the Bert predictor slightly
        prediction = BertPredictor().bert_with_lemma_predict(context)


        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))



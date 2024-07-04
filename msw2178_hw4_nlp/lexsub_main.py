#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

nltk.download('punkt')
from collections import Counter
import operator

# For part 3
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Part 4
'''
import numpy as np
import gensim

# Part 5
import transformers
import tensorflow as tf 
'''

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    # Retrieve all lexemes for the input lemma
    all_lexemes = wn.lemmas(lemma, pos = pos)

    # list to hold the synsets
    possible_synsets = []

    # Get the synset of every lexeme in all_lexemes
    for position in range(0, len(all_lexemes)):
        lexeme = wn.lemmas(lemma, pos = pos)[position]
        synset = lexeme.synset()
        possible_synsets.append(synset)

    # Get the lexemes for every synset and take the name
    # of every lexeme. Put this word in a list
    possible_words = []
    for synset in possible_synsets:
        for i in range(len(synset.lemmas())):
            possible_words.append(synset.lemmas()[i].name())

    # Converting the list to a set to remove duplicates
    set_possible_words = set(possible_words)

    # Making sure the output set does NOT include the input lemma
    set_possible_words.remove(lemma)

    # Go through the output set to replace '_' with ' ' in a multi-word
    for word in set_possible_words:
        if "_" in word:
            new_word = word.replace("_", " ")
            set_possible_words.remove(word)
            set_possible_words.add(new_word)

    # Return a list because that is what is asked for
    return list(set_possible_words)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2

    # Get the lemma of the Context object
    target_lemma = context.lemma
    # Get the pos of the Context object
    target_pos = context.pos

    # Retrieve all lexemes for the target_lemma
    all_lexemes = wn.lemmas(target_lemma, pos = target_pos)

    # Rertieve all synsets of those lexemes (bc we need synonyms!)
    possible_synsets = []
    for position in range(0, len(all_lexemes)):
        lexeme = wn.lemmas(target_lemma, pos = target_pos)[position]
        synset = lexeme.synset()
        possible_synsets.append(synset)

    # Get the lexemes of those synsets
    candidate_lemmas = []
    for synset in possible_synsets:
        lemma = synset.lemmas()
        candidate_lemmas.append(lemma)

    # Remove the input lemma 
    updated_candidate_lemmas = [lemma for lemma in candidate_lemmas if lemma != target_lemma]

    # Replace "_" with " " in the lemmas
    for list in updated_candidate_lemmas:
        for i in range(len(list)):
            name = list[i].name()
            if "_" in name:
               new_name = list[i].name().replace("_", " ")
               name = new_name


   # Make a dictionary/counter that maps the lemma to its frequency
   # Iterate through all synsets and all lemmas in each synset
   # Use count() on each lemma and add the count to the dictionary
    dictionary = {}
    counting_list = [lemma for sublist in updated_candidate_lemmas for lemma in sublist]
    for lemma in counting_list:
        if lemma in dictionary.keys():
            dictionary[lemma] += lemma.count()
        else:
            key = lemma
            count = lemma.count()
            dictionary[key] = count
    
   # Finally, sort your dictionary so that lemmas are ordered by frequency value
    sorted_dictionary = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
    
    # Return the lemma with the highest value in the sorted dictionary
    highest_lemma = next(iter(sorted_dictionary))

    return highest_lemma.name().lower()

def wn_simple_lesk_predictor(context : Context) -> str:

    # Take the context's word_form, left_context, and right_context
    # and make a sentence.
    target_word = context.word_form
    sentence = " ".join(context.left_context) + " " + target_word + " " + " ".join(context.right_context)

    # Tokenize, lemmatize, lower, and remove stop words from sentence.
    stop_words = stopwords.words('english')
    tokenized_sentence = [word.lower() for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words]
    lemmatized_sentence = [WordNetLemmatizer().lemmatize(word, context.pos) for word in tokenized_sentence]
   
    # Look at all the synsets that the target_lemma appears in
    target_lemma = context.lemma
    synsets = wn.synsets(target_lemma)

    # Compute overlap between each synset's definition and context of target_word (updated_sentence)
    result = ""
    for synset in synsets:
        tokenized_definition = [word.lower() for word in word_tokenize(synset.definition()) if word.isalpha() and word not in stop_words]
        lemmatized_definition = [WordNetLemmatizer().lemmatize(word, context.pos) for word in tokenized_definition]
        
        examples = synset.examples()
        examples = " ".join(examples)
        tokenized_examples = [word.lower() for word in word_tokenize(examples) if word.isalpha() and word not in stop_words]
        lemmatized_examples = [WordNetLemmatizer().lemmatize(word, context.pos) for word in tokenized_examples]

        #hypernyms
        hypers = synset.hypernyms()
        hyper_defs = []
        hyper_ex = []
        for synset in hypers:
            hyper_defs.append(synset.definition())
            hyper_ex.append(synset.examples())
        
        all_hyper_ex = ""
        for list in hyper_ex:
            all_hyper_ex += " ".join(list)

        tokenized_hypernym_definition = [word.lower() for word in word_tokenize( " ".join(hyper_defs)) if word.isalpha() and word not in stop_words]
        lemmatized_hypernym_definition = [WordNetLemmatizer().lemmatize(word, context.pos) for word in tokenized_hypernym_definition]
  
        tokenized_hypernym_examples = [word.lower() for word in word_tokenize(all_hyper_ex) if word.isalpha() and word not in stop_words]
        lemmatized_hypernym_examples = [WordNetLemmatizer().lemmatize(word, context.pos) for word in tokenized_hypernym_examples]

        updated_def = []
        for word in lemmatized_definition:
            updated_def.append(word)
        for word in lemmatized_examples:
            updated_def.append(word)
        for word in lemmatized_hypernym_definition:
            updated_def.append(word)
        for word in lemmatized_hypernym_examples:
            updated_def.append(word)

        set_def = set(updated_def)
        set_context = set(lemmatized_sentence)
        result = set_def.intersection(set_context)

        # Make sure to NOT include the original word
        if target_word in result:
           result.remove(target_word)

    result = str(result).replace("{", "").replace("}", "")

    # For ties and 0 overlaps
    lemmas = []
    for synset in synsets:
        lemma = synset.lemmas()
        lemmas.append(lemma)

    if len(result) == 0 or len(result) > 1:
        dictionary = {}
        counting_lemmas = [lemma for sublist in lemmas for lemma in sublist]

        for lemma in counting_lemmas:
            if lemma in dictionary.keys():
                dictionary[lemma] += lemma.count()
            else:
                key = lemma
                count = lemma.count()
                dictionary[key] = count
    
        sorted_dictionary = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
        result = next(iter(sorted_dictionary)).name()
    
    return result

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:

        '''
        Goal: obtain a set of possible synonyms from Wordnet (can use P1)
        and then return the synonym that is the MOST similar to the target word
        '''

        set_synonyms = get_candidates(context.lemma, context.pos)

        dictionary = {}
        for synonym in set_synonyms:
            if synonym in dictionary.keys():
                dictionary[synonym] += self.model.similarity(context.word_form, synonym)
            else:
                try:
                    key = synonym
                    count = self.model.similarity(context.word_form, synonym)
                    dictionary[key] = count
                except KeyError:
                    continue


        sorted_dictionary = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))

        most_similar_synonym = next(iter(sorted_dictionary))
        
       
        return most_similar_synonym


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:

        # Obtain a set of candidate synonyms (for example, by calling get_candidates)
      get_candidates(context.lemma, context.pos)

      # Convert the information in context into a suitable masked input representation
      # for the DistilBERT model (see example above).
      target_word = context.word_form
      context_sen = " ".join(context.left_context) + " " + target_word + " " + " ".join(context.right_context)
      tokens_list = self.tokenizer.tokenizer(context_sen)
      # Mask the target word and store the index of the masked target word
      mask_index = 0
      for token in tokens_list:
        for i in range(len(tokens_list)):
          if token == target_word:
            mask_index = i
            token = '[MASK]'

      input_toks = self.tokenizer.encode(context_sen)

      # Run the DistilBERT model on the input representation.
      input_mat = np.array(input_toks).reshape((1,-1))
      outputs = self.model.predict(input_mat)
      predictions = outputs[0]

      # Select, from the set of wordnet derived candidate synonyms,
      # the highest-scoring word in the target position (i.e. the
      # position of the masked word). Return this word.
      best_words = np.argsort(predictions[0][mask_index])[::-1]
      tokenized_best_words = self.tokenizer.convert_ids_to_tokens(best_words[:10])
      winner = tokenized_best_words[0]

      return winner

def part_six_predictor(context : Context) -> str:
  
    '''
     Attempt at redoing Part 3
    '''
    target_word = context.word_form
    sentence = " ".join(context.left_context) + " " + target_word + " " + " ".join(context.right_context)

    syn = lesk(sentence, context.lemma,context.pos)
    result = syn.lemmas()[0].name()

    return result

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).
    '''
    Note: Part 3 ended up being my best predictor.
    '''

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)
        prediction = wn_simple_lesk_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))


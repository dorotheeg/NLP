#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string


# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    for syn in wn.synsets(lemma,pos):
        for l in syn.lemmas():
            if l.name() not in possible_synonyms and l.name() != lemma:
                possible_synonyms.append(l.name())
    
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    possible_synonyms = []
    dict = {}
    counter = 0
    temp = ""
    m = 0
    for syn in wn.synsets(context.lemma, context.pos):
        for l in syn.lemmas():
            if l.name() != context.lemma:
                m = l.count()
                dict[l.name()] = dict.get(l.name(), 0) + l.count()
                '''if m > 0:
                    counter = m
                    temp = context.lemma
    print("TEMP", temp)'''

    
    '''maxy = possible_synonyms[0] 
    print("LIST", possible_synonyms)
    for i in possible_synonyms:
        curr_frequency = possible_synonyms.count(i) 
        if(curr_frequency > counter): 
            counter = curr_frequency 
            maxy = i '''
    maxy = max(dict, key=dict.get)
  
    
    #print(maxy) 
    
    return maxy # replace for part 2
    '''sset = wn.synsets(context.lemma, context.pos)
    final_count = 0
    d = {}
    for synset in sset:
        for w in synset.lemmas():
            if w.name() != context.lemma:
                d[w.name()] = d.get(w.name(), 0) + w.count()
    result = max(d, key=d.get)
    if "_" in result:
        return result.replace("_"," ")
    print(d)
    return result # replace for part 2
'''

def wn_simple_lesk_predictor(context):
    '''compare left and right context with:
    (1) the definitions in the sysnet .definition()
    (2) the examples of the sysnet .examples()
    (3) the definitions of all the all the hypernyms of the synset and .hypernyms().definition()
    (4) the examples of all the hypernyms of the synset .hypernyms().examples() '''

    #get contexts
    #print(context.lemma)
    left = (context.left_context)
    right = context.right_context
    tots = set(left+right)
    #print("tots first", (tots))
    '''seperator = ' '
    tots = str(seperator.join(left + right))
    print("tots", (tots))
    #print("tots TOK", tokenize(tots))
    print("tots SET", set(tots))
    tots = set(tots) - set(stopwords.words('english'))'''

    ls = wn.synsets(context.lemma, context.pos)
    #ls = wn.synsets(context.lemma, context.pos)
    deff = {}
    ex = {}
    h_deff = {}
    h_ex = {}
    final = []

    counter = 0
    

    for each in ls:
        #get definitions
        #print("each", each)
        #deff = (each.definition())
        #print(deff)
        deff = tokenize(each.definition())
        deff = set(deff) - set(stopwords.words('english'))
        #print("deff", deff)

        #get examples
        temp = (each.examples())
        #print("temp", temp)
        for i in temp:
            ex = tokenize(i)
            ex = set(ex) - set(stopwords.words('english'))
            #print("ex", ex)
            inter = tots.intersection(ex)
            if counter < len(inter):
                #print("inter 1",inter)
                #print("tots 1", (tots))
                #print(ex)
                final.append(inter)
                counter = len(inter)

        inter = tots.intersection(deff)
        if counter < len(inter):
            #print("inter 2",inter)
            #print("tots 2", (tots))
            #print(deff)
            final.append(inter)
            counter = len(inter)
        

        for every in each.hypernyms():
            h_deff = tokenize(every.definition())
            h_deff = set(h_deff) - set(stopwords.words('english'))

        
            #get examples
            h_temp = (every.examples())
            for j in h_temp:
                h_ex = tokenize(j)
                h_ex = set(h_ex) - set(stopwords.words('english'))
                inter = tots.intersection(h_ex)
                if counter < len(inter):
                    #print("inter 3", inter)
                    #print("tots 3", (tots))
                    #print(h_ex)
                    final.append(inter)
                    counter = len(inter)
            
            inter = tots.intersection(h_deff)
            if counter < len(inter):
                #print("inter 4",inter)
                #print("tots 4", (tots))
                #print(h_deff)
                #print(every.name())
                final.append(inter)
                counter = len(inter)
            
            


    if counter == 0:
        #print("OLD", wn_frequency_predictor(context))
        return wn_frequency_predictor(context)
    #print(final)
    else:
        #print("FINAL", final)
        return final.pop().pop()

        #get hypernyms definitions
        #h_deff = set(each.hypernyms().definition())

        #get hypernyms examples
        #h_ex = set(each.hypernyms().examples())
    
    

    
    
    return None #replace for part 3        
   
class Word2VecSubst(object):
    
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        #print("HERE")
        

    def predict_nearest(self,context):
        #print("HELP")

        final = ""
        
        possible_synonyms = []
        for syn in wn.synsets(context.lemma,context.pos):
            for l in syn.lemmas():
                if l.name() not in possible_synonyms and l.name() != context.lemma:
                    possible_synonyms.append(l.name())

        cur = -10
        


        
        for each in possible_synonyms:
            #print(each)
  
            if "_" in each:
                #print("Here")
                each.replace("_"," ")
                #print(each)
            
            #print(self.model.similarity(each, context.lemma))
            try:
                if cur < self.model.similarity(each, context.lemma):
                    final = each
                    #print(final)

            except:
                pass
                
        #how the eff do I test

        return final # replace for part 4

    

    def predict_nearest_with_context(self, context):

        # sum left and righ side vectors, removing stop and only allowing the last 5 in each dir
        # run the get syn
        # compare the vector of contexts with the syn
        # return the most similar one

        # sum left and righ side vectors, removing stop and only allowing the last 5 in each dir
        seperator = " "
        left = (seperator.join((context.left_context)))
        right = (seperator.join((context.right_context)))
        #print(left)
        left = tokenize(left)
        right = tokenize(right)

        #print("VOCABBBBB", self.model.vocab)
        
        #self.model.vocab =  {k.lower(): v for k, v in self.model.vocab.items()}

        lefty = []
        righty = []
        
        for syn in left:
            if syn in self.model.vocab:
                lefty.append(syn)


        for syn in right:
            if syn in self.model.vocab:
                righty.append(syn)
            
                    
    
        lefty = set(lefty) - set(stopwords.words('english'))
        righty = set(righty) - set(stopwords.words('english'))

        lefty = list(lefty)
        if len(lefty) > 5:
            lefty = lefty[-5:]
            #print(left)
            
        righty = list(righty)
        if len(righty) > 5:
            righty = righty[:5]
            #print(right)
            
        tots = set(lefty+righty)



        # run the get syn'
        possible_synonyms = []
        for syn in wn.synsets(context.lemma,context.pos):
            for l in syn.lemmas():
                if l.name() not in possible_synonyms and l.name() != context.lemma:
                    possible_synonyms.append(l.name())
        
        def cos(v1,v2):
            return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        
        # compare the vector of contexts with the syn
        cur = 0
        final = ""
        for syn in possible_synonyms:
            if syn in self.model.vocab:
                pass
            else:
                possible_synonyms.remove(syn)

                
        vector = np.zeros(len(self.model.wv[context.lemma])) + self.model.wv[context.lemma]
        
        for x in tots:
            vector = vector + self.model.wv[x]

        
            
        
        for each in possible_synonyms:
            #print(each)
  
            if "_" in each:
                #print("Here")
                each.replace("_"," ")
                #print(each)

            #possible_synonyms = set(possible_synonyms) - set(stopwords.words('english'))
            #print("PS", possible_synonyms)
            #print("TOTOS",tots)
            #print("OTHER", self.model.wv[each])
            
            #print(cos(vector, self.model.wv[each]))
            #print(self.model.similarity(each, context.lemma))
            try:
                #print(self.model.similarity(self.model.wv[each], tots))
                if cur < cos(vector, self.model.wv[each]):
                    cur = cos(vector, self.model.wv[each])
                    #print(cos(vector, self.model.wv[each]))
                    final = each
                    

            except:
                pass


        print(cur)
        return final # replace for part 5


    def competition(self, context):

        # sum left and righ side vectors, removing stop and only allowing the last 5 in each dir
        # run the get syn
        # compare the vector of contexts with the syn
        # return the most similar one

        # sum left and righ side vectors, removing stop and only allowing the last 5 in each dir
        seperator = " "
        left = (seperator.join((context.left_context)))
        right = (seperator.join((context.right_context)))
        #print(left)
        left = tokenize(left)
        right = tokenize(right)

        #print("VOCABBBBB", self.model.vocab)
        
        self.model.vocab =  {k.lower(): v for k, v in self.model.vocab.items()}

        lefty = []
        righty = []
        
        for syn in left:
            if syn in self.model.vocab:
                lefty.append(syn)


        for syn in right:
            if syn in self.model.vocab:
                righty.append(syn)
            
                    
    
        lefty = set(lefty) - set(stopwords.words('english'))
        righty = set(righty) - set(stopwords.words('english'))

        lefty = list(lefty)
        #if len(lefty) > 5:
         #   lefty = lefty[-5:]
            #print(left)
            
        righty = list(righty)
        #if len(righty) > 5:
         #   righty = righty[:5]
            #print(right)
            
        tots = set(lefty+righty)


        deff = []
        ex = []
        # run the get syn'
        possible_synonyms = []
        for syn in wn.synsets(context.lemma,context.pos):
            for l in syn.lemmas():
                if l.name() not in possible_synonyms and l.name() != context.lemma:
                    possible_synonyms.append(l.name())




        ls = wn.synsets(context.lemma, context.pos)

        for each in ls:

            deff = tokenize(each.definition())
            deff = set(deff) - set(stopwords.words('english'))
            #print("deff", deff)

            #get examples
            temp = (each.examples())
            #print("temp", temp)
            for i in temp:
                ex = tokenize(i)
                ex = set(ex) - set(stopwords.words('english'))

        foo = list(possible_synonyms) + list(deff) + list(ex)
        possible_synonyms = set(foo)
                           
        
        def cos(v1,v2):
            return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        
        # compare the vector of contexts with the syn
        cur = 0
        final = ""
        cand = []
        for syn in possible_synonyms:
            cand.append(syn)

                
        vector = np.zeros(len(self.model.wv[context.lemma]))+ self.model.wv[context.lemma]

        
        for x in tots:
            vector = vector + self.model.wv[x]

        cand = set(cand) - set(stopwords.words('english'))

        cap = ""
        

        if cand:
            for each in cand:
                if " " in each:
                    each = each.replace(" ","_")
                if each in self.model.vocab:
                    if cur < cos(vector, self.model.wv[each]):
                        cur = cos(vector, self.model.wv[each])
                        final = each
                try:
                    if each.capitalize() in self.model.vocab:
                        cap = each.capitalize()
                        if cur < cos(vector, self.model.wv[cap].capitalize()):
                            cur = cos(vector, self.model.wv[cap].capitalize())
                            final = each
                except:
                    pass
                            
        print(cur)
        return final # replace for part 5

if __name__=="__main__":

    #print(get_candidates('slow', 'a'))
    #print(wn_frequency_predictor(context))
    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #print(predictor.predict_nearest())
    #print(predictor.predict_nearest(context))
    
    for context in read_lexsub_xml(sys.argv[1]):
        
        #print(Word2VecSubst.predict_nearest(model.self, context))
        #print(context)  # useful for debugging
        prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
        print(wn_simple_lesk_predictor(context))
        print(wn_frequency_predictor(context))
    print(predictor.predict_nearest_with_context(context))
    print(predictor.competition(context))
        

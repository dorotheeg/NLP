"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        table, probs = parser.parse_with_backpointers(tokens)
        search = grammar.startsymbol
        parse = table[(0, len(tokens))]

        if search not in parse:
            return False
        else:
            return True

        
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(defaultdict)
        probs = defaultdict(defaultdict)
        inside_table = defaultdict()
        inside_probs = defaultdict()

        for i in range(len(tokens)):
            search = self.grammar.rhs_to_rules[(tokens[i],)]
            for each in search:
                toks = tokens[i]
                num = math.log(each[2])
                table[(i,i + 1)][each[0]] = toks
                probs[(i,i + 1)][each[0]] = num

            #print(table)
        bleh = 0
        j = 0
        for k in range(2,len(tokens)+1):
            for i in range(len(tokens)+1):
                    #print("k :", k)
                    #print("i :", i)
                    #print("j :", j)
                    times = k + i
                    for j in range (i + 1, times):
                        if table.get((i,j)) != None and table.get((j,times)) != None:
                            for first in table[(i,j)]:
                                #key1= item1.key
                                for second in table[(j,times)]:
                                    #key2 = item2.key
                                    one = probs[(i, j)][first]
                                    two = probs[(j, times)][second]
                                    three = math.log(each[-1])
                                    #if probs.get(each[0]):
                                    if probs[(i, times)]:
                                        #print("HEREEEEE")
                                        newbie = one + two + three
                                        orig = probs[(i, times)][each[0]]
                                        if orig < newbie: #replace the prob with the most likely prob
                                            probs[(i, times)][each[0]] = newbie
                                            table[(i,times)][each[0]] = ((first,i,j),(second,j,times))
                                    else:
                                        if self.grammar.rhs_to_rules.get((first, second)) != None :
                                            for each in self.grammar.rhs_to_rules.get((first, second)):
                                                one = probs[(i, j)][first]
                                                two = probs[(j, times)][second]
                                                three = math.log(each[-1])
                                                probs[(i, times)][each[0]] = one + two + three
                                                table[(i, times)][each[0]] = ((first, i, j), (second, j, times))
                 

                    
                                        
                    
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    each = chart[(i,j)]
    each1 = chart[(i,j)][nt]
    table = chart
    if isinstance(each1, tuple):
        zero = each1[0][0]
        zero_one = each1[0][1]
        two_two = each1[0][2]
        one_zero = each1[1][0]
        one_one = each1[1][1]
        one_two = each1[1][2]

        #return (nt,get_tree(chart,zero,zero_one,two_two),get_tree(table,one_zero,one_one,one_two))
        return (nt,get_tree(table,zero_one,two_two,zero),get_tree(table,one_one,one_two,one_zero))
        
    else:
        #return None
        return (nt, each1)

    
   
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        print( parser.is_in_language(toks))
        #toks = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
        #print( parser.is_in_language(toks))
        #toks =['flights', 'from','miami', 'to', 'cleveland','.']
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        print(probs)
        tree = get_tree(table, 0, len(toks), grammar.startsymbol)
        #print(tree)

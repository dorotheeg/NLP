"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # (note all nonterminal
        # symbols are upper-case) and that all probabilities for the same 
        # lhs symbol sum to 1.0. 
        
        # TODO, Part 1
        tots = []

        for x in self.lhs_to_rules.items():
          if (x[0]) != (x[0].upper()):
              print("Not all upper case")
              return False
          for y in x[1]:
            if (y[0]) != (y[0].upper()):
              print("Not all upper case")
              return False

        for z in self.lhs_to_rules.keys():
          prob = 0.0
          tots.clear()
          for a in self.lhs_to_rules[z]:
            tots.append(float(a[-1]))
          prob = fsum(tots)
          if prob > (1 + .00001 ) or prob < (1 - .00001 ):
            print("Sum does not equal 1")
            return False   

        print("The grammar is a valid PCFG in CNF")
        return True 


if __name__ == "__main__":
    # with open(sys.argv[1],'r') as grammar_file:
    #     grammar = Pcfg(grammar_file)

    
     with open('atis3.pcfg','r') as grammar_file:
         grammar = Pcfg(grammar_file)
         check = grammar.verify_grammar()
         print(check)



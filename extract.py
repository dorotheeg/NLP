def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2
        #print(state)
        #print(words)
        #print(pos)

        temp = np.zeros(6)
        #print(state.stack)
        if len(state.stack) < 3:
            x = len(state.stack) +1
        else:
            x = 4

        for i in range(1,x):
            ind = state.stack[-i]
            if not words[ind]:
                temp[i-1] = 3
            elif pos[ind] == "NNP":
                temp[i-1] = 1
            elif pos[ind] == "CD":
                temp[i-1] = 0
            elif words[ind].lower() in self.word_vocab:
                temp[i-1] = self.word_vocab[words[ind].lower()]
            elif pos[ind] == "UNK":
                temp[i-1] = 2
            if len(state.stack) < 3:
                temp[i-1] = 4

        if len(state.buffer) < 3:
            x = len(state.buffer)+1
        else:
            x = 4
        #print(state.buffer)    
        for i in range(1,x):  
            ex = state.buffer[-i]
            if not words[ex]:
                #print("here")
                temp[i+2] = 3
            elif pos[ex] == "NNP":
                temp[i+2] = 1
            elif pos[ex] == "CD":
                temp[i+2] = 0
            elif words[ex].lower() in self.word_vocab:
                temp[i+2] = self.word_vocab[words[ex].lower()]
            elif pos[ex] == "UNK":
                temp[i+2] = 2  
            if len(state.buffer) < 3:
                temp[i+2] = 4
        #print(temp)
        return temp

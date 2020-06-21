def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            # TODO: Write the body of this loop for part 4
            features = np.array([self.extractor.get_input_representation(words, pos, state),])
            actions = self.model.predict(features)
            #piazza
            
            temp = []
            for i in range(len(actions[0])):
                temp.append(actions[0][i])
            #print("TEMP", temp)
            sort = sorted(enumerate(temp), reverse=True, key=lambda x:x[1])
            #https://docs.python.org/3/howto/sorting.html This helped
            #print("SORT", sort)

            ind = 0

            for ind in range(91) : #and boo == True:
                out = self.output_labels[sort[ind][0]]
                
                if out[0] == "left_arc":
                    if len(state.stack) > 0 :
                            state.left_arc(out[1]) #direction
                            break
                
                    elif len(state.stack) == 0:
                        state.shift()
                        #print("HEREEEE")
                        break
                        
                elif out[0] == "right_arc":
                    if len(state.stack) > 0:
                        state.right_arc(out[1]) 
                        break


                else: #shift
                    if len(state.buffer) > 1:
                        state.shift()
                        break
                        
                #print("EACH")
                ind += 1
            
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 

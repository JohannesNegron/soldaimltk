# -*- coding: utf-8 -*-
from soldaimltk.general.PhraseTokenizer import PhraseTokenizer
import numpy as np

class GramTokenizerNerRecognizer:
    
    def __init__(self, model, encoder):
        self.model = model
        self.phrase_tokenizer = PhraseTokenizer(space_char=' ')
        self.encoder = encoder
        
    def build_example(self, text):
        X = np.ndarray((1, self.encoder.entries))
        X[0][:] = self.encoder.encodePhrase(text)
        return X
    
    # Check for each word if its part of an entity and return it
    def extractEntites(self, text):
        entities = []
        current_entity = []
        for word in self.phrase_tokenizer.getTokens(text):
            vector = self.build_example(word)
            prediction = self.model.predict(vector)[0]
            print(word + " : " + str(prediction))
            if prediction == 1 and len(current_entity) > 0:
                entities.append(current_entity)
                current_entity = []
            if prediction == 2:
                print(word + " is a man's name")
                current_entity.append((word, 'name', 'male'))
            if prediction == 3:
                current_entity.append((word, 'name', 'female'))
            if prediction == 4:
                current_entity.append((word, 'lastname', 'neutral'))
        if len(current_entity) > 0:
            entities.append(current_entity)
        return self.processEntities(entities)
    
    # This method process the entities
    def processEntities(self, entities):
        print(entities)
        processed_entities = []
        for e in entities:
            i = 0
            male = 0
            entity = {}
            for name in e:
                if name[1] == 'name':
                    entity['name_' + str(i)] = name[0]
                if name[1] == 'lastname':
                    entity['lastname_' + str(i)] = name[0]
                    
                if name[2] == 'male':
                    male += 1
                if name[2] == 'female':
                    male -= 1
                i += 1
            entity['gender'] = 'unknown'
            
            if male > 0:
                entity['gender'] = 'male'
            if male < 0:
                entity['gender'] = 'female'
                
            processed_entities.append(entity)
            
        return processed_entities

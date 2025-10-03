import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import pymorphy3
import pandas as pd
import string

nltk.download('punkt')

morph = pymorphy3.MorphAnalyzer()
pairs = []

with open("text.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences = sent_tokenize(text)

for sentence in sentences:
    words = word_tokenize(sentence, language='russian')
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        
        if word1 in string.punctuation or word2 in string.punctuation:
            continue
            
        parsed1 = morph.parse(word1)[0] if morph.parse(word1) else None
        parsed2 = morph.parse(word2)[0] if morph.parse(word2) else None
        
        if not parsed1 or not parsed2:
            continue
        
        pos1, pos2 = parsed1.tag.POS, parsed2.tag.POS
        adj_pos, noun_pos = {'ADJF', 'ADJS'}, {'NOUN'}
        
        if (parsed1.tag.number == parsed2.tag.number and 
            parsed1.tag.case == parsed2.tag.case):
            
            # Прил + сущ или сущ + прил
            if (pos1 in adj_pos and pos2 in noun_pos) or (pos1 in noun_pos and pos2 in adj_pos):
                adj = parsed1 if pos1 in adj_pos else parsed2
                noun = parsed2 if pos1 in adj_pos else parsed1
                
                noun_lemma_parse = morph.parse(noun.normal_form)[0]
                if noun_lemma_parse.tag.gender:
                    inflected_adj = adj.inflect({
                        noun_lemma_parse.tag.gender, 
                        noun_lemma_parse.tag.number, 
                        noun_lemma_parse.tag.case
                    })
                    if inflected_adj:
                        adj_normalized = inflected_adj.word
                        noun_normalized = noun.normal_form
                        if pos1 in adj_pos:
                            pairs.append((adj_normalized, noun_normalized))
                        else:
                            pairs.append((noun_normalized, adj_normalized))
            
            # Сущ + сущ
            elif pos1 in noun_pos and pos2 in noun_pos:
                pairs.append((parsed1.normal_form, parsed2.normal_form))
            
            # Прил + прил
            elif pos1 in adj_pos and pos2 in adj_pos:
                if parsed1.tag.gender and parsed2.tag.gender:
                    inflected_adj1 = parsed1.inflect({parsed2.tag.gender, parsed1.tag.number, parsed1.tag.case})
                    if inflected_adj1:
                        pairs.append((parsed1.normal_form, parsed2.normal_form))


print("Пары слов (леммы):")
for word1, word2 in pairs:
    print(f"{word1} - {word2}")

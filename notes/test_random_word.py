import nltk
import random
from nltk.corpus import words

def generate_random_word(pos):
    word_list = [word for word in words.words() if nltk.pos_tag([word])[0][1] == pos]
    if word_list:
        random_word = random.choice(word_list)
        return nltk.pos_tag([random_word])[0][0]
    else:
        return None

# Example usage
random_noun = generate_random_word('NN')
random_verb = generate_random_word('VB')
random_adjective = generate_random_word('JJ')

print("Random Noun:", random_noun)
print("Random Verb:", random_verb)
print("Random Adjective:", random_adjective)

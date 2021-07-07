# %%
import re
import sqlite3

import numpy as np
import pandas as pd

from tqdm import tqdm
from contextlib import suppress

np.random.seed(0)

# %% load data
db_file_path = "data/database.sqlite"
ctx = sqlite3.connect(db_file_path)
df = pd.read_sql_query("SELECT * FROM scripts WHERE type = 'Dialogue'", ctx)
df.head()

# %% Markov Chain model
class MarkovChain():
	vocabulary:        list
	transition_matrix: np.ndarray

	def __init__(self, text: str) -> None:
		text_split = re.findall(r"[\w']+|[.,!?]", text)
		self.vocabulary = sorted(set(text_split))
		n = len(self.vocabulary)
		print(f"Length of vocabulary: {n}")

		A_n = np.zeros((n, n), dtype=int)
		self.transition_matrix = np.zeros((n, n), dtype=float)

		print(" -> training...")
		for k, word in tqdm(enumerate(text_split), total=len(text_split)):
			with suppress(IndexError):
				next_word = text_split[k+1]
				i, j = self.vocabulary.index(word), self.vocabulary.index(next_word)
				A_n[i][j] += 1
		self.transition_matrix = (A_n.T / np.sum(A_n, axis=1)).T
		print(" -> done\n")

	def predict(self, word: str, generative = False) -> str:
		try: i = self.vocabulary.index(word)
		except ValueError: raise ValueError(f"word '{word}' not in vocabulary")
		
		if generative:
			r = np.random.uniform()
			p_values = sorted(
				zip(
					self.transition_matrix[i],
					range(len(self.vocabulary))), 
				reverse=True)
			
			P = 0
			for p_value, index in p_values:
				P += p_value
				if r < P:
					return self.vocabulary[index]
		else: 
			return self.vocabulary[np.argmax(self.transition_matrix[i])]

	def predict_sentence(self, text: str, generative = False) -> str:
		word = text.split()[-1]
		current_word, sentence = word, [text]

		while current_word not in ".!?":
			if len(sentence) > 42:
				sentence.append("...")
				break
			sentence.append(self.predict(current_word, generative))
			current_word = sentence[-1]

		sentence = ' '.join(sentence)
		for punctuation in ".,!?":
			sentence = sentence.replace(f" {punctuation}", punctuation)

		return sentence.capitalize()

test_text = df.at[3, 'detail']
print(test_text, '\n')

test_model = MarkovChain(test_text)
print(test_model.predict('Hello'))
print(test_model.predict_sentence('Hello'))
print(test_model.predict_sentence('Hello', generative=True))

# %%
full_text = ' '.join(df.detail.values)
full_model = MarkovChain(full_text)

# %% static predict
phrases_to_predict = [
	"Hello", 
	"Hello", 
	"The weather", 
	"Ah, yes... The weather", 
	"Indeed",
	"Do",
	"No",
	"Goodbye",
	"Goodbye" 
]
for phrase in phrases_to_predict:
	print(full_model.predict_sentence(phrase) + '\n')

# %% generative predict
for phrase in phrases_to_predict:
	print(full_model.predict_sentence(phrase, generative=True) + '\n')

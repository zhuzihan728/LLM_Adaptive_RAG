from itertools import combinations
import spacy
def generate_paragraphs_(sentences, num_beams):
    n = len(sentences)
    x = n - num_beams
    indices = list(range(n))
    masked_combinations = combinations(indices, x)
    result_paragraphs = []

    for combo in masked_combinations:
        # Convert the combination to a set for quick lookup
        masked_set = set(combo)
        # Create the new paragraph by joining sentences that are not masked
        new_paragraph = ' '.join([sentences[i] for i in indices if i not in masked_set])
        result_paragraphs.append(new_paragraph)

    return result_paragraphs

def generate_paragraphs(sentences, num_beams):
    n = len(sentences)
    result_paragraphs = []
    for i in range(num_beams, n+1):
        temp = generate_paragraphs_(sentences, i)
        result_paragraphs.extend(temp)
    return result_paragraphs

# Example usage:
num_beams = 2  # Number of sentences for each sub-paragraph
sentences = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4", "Sentence 5"]
new_paragraphs = generate_paragraphs(sentences, num_beams)
for paragraph in new_paragraphs:
    print(paragraph)


# ret for each sub-paragraph

# aggregate ret-doc scores

# choose top ret-docs
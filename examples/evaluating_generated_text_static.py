from rouge_score import rouge_scorer

# Simulated LLM responses
generated_answers = [
    "The Eiffel Tower is located in London.",   # Incorrect
    "Berlin is the capital city of Germany.",   # Correct
    "Mount Everest is located in China."        # Incorrect
]

# Ground truth answers
reference_answers = [
    "The Eiffel Tower stands in Paris.",
    "The capital of Germany is Berlin.",
    "Mount Everest is located in Nepal."
]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Evaluate the first example
scores1 = scorer.score(reference_answers[0], generated_answers[0])
print("Example 1 ROUGE:", scores1)



# Evaluate all examples in a loop and calculate total ROUGE scores
total_scores = {'rouge1': 0, 'rougeL': 0}

for i, (reference, generated) in enumerate(zip(reference_answers, generated_answers), start=1):
    scores = scorer.score(reference, generated)
    print(f"Example {i} ROUGE:", scores)
    total_scores['rouge1'] += scores['rouge1'].fmeasure
    total_scores['rougeL'] += scores['rougeL'].fmeasure

# Calculate average ROUGE scores
num_examples = len(generated_answers)
average_scores = {key: value / num_examples for key, value in total_scores.items()}
print("Average ROUGE scores:", average_scores)


"""
Prints:

Example 1 ROUGE: {'rouge1': Score(precision=0.5714285714285714, recall=0.6666666666666666, fmeasure=0.6153846153846153), 'rougeL': Score(precision=0.5714285714285714, recall=0.6666666666666666, fmeasure=0.6153846153846153)}
Example 2 ROUGE: {'rouge1': Score(precision=0.8571428571428571, recall=1.0, fmeasure=0.923076923076923), 'rougeL': Score(precision=0.5714285714285714, recall=0.6666666666666666, fmeasure=0.6153846153846153)}
Example 3 ROUGE: {'rouge1': Score(precision=0.8333333333333334, recall=0.8333333333333334, fmeasure=0.8333333333333334), 'rougeL': Score(precision=0.8333333333333334, recall=0.8333333333333334, fmeasure=0.8333333333333334)}
Average ROUGE scores: {'rouge1': 0.7905982905982906, 'rougeL': 0.6880341880341879}

Comment:

You can see that the ROUGE scores are quite high for example 3 - the sentence structure is very much similar - just the actual fact - the country name is different.
This showcases the big limitation of n-gram based metrics. They cannot really judge the semantic meaning of the sentence - but just compare the available character sequences.

"""

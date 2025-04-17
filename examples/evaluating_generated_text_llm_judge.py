

from examples.llm_util import llm


generated_answers = [
    "The Eiffel Tower is located in London.",   # Incorrect
    "Berlin is the capital city of Germany.",   # Correct
    "Mount Everest is in China."                # Incorrect
]

# Define the judging prompt
judging_prompt = """
You are an impartial judge. Compare the generated answer on correctness.

Generated answer: {{?generated_answer}}

Only evaluate factual correctness, not style or phrasing.

Return your judgment as JSON with the following schema:
{
  "correctness": "Correct" | "Incorrect"
}
"""

correct_count = 0
total_count = len(generated_answers)

# Loop through answers and evaluate using LLM with judging prompt
for generated_answer in generated_answers:
    result = llm(
        prompt=judging_prompt,
        variables={"generated_answer": generated_answer},
        json_schema={
            "type": "object",
            "properties": {
                "correctness": {
                    "type": "string",
                    "enum": ["Correct", "Incorrect"]
                }
            },
            "required": ["correctness"]
        })
    
    if result['correctness'] == "Correct":
        correct_count += 1

    print(f"Generated: {generated_answer}\nJudgment: {result['correctness']}\n")

# Calculate evaluation score
eval_score = (correct_count / total_count) * 100
print(f"Evaluation Score: {eval_score}%")

"""
Prints:

Generated: The Eiffel Tower is located in London.
Judgment: Incorrect

Generated: Berlin is the capital city of Germany.
Judgment: Correct

Generated: Mount Everest is in China.
Judgment: Incorrect

Evaluation Score: 33.33%

Comment:

We are able to tell that the LLM is able to judge the correctness of the generated text. 
But you might ask: If the LLM generated the response in the first place and maybe got it false/in bad form/not relevant - why would the judge know better?
There is some scientific evidence that the LLM's do infact perform better when judged then when actually having to come up with the answer in the first place.
In practice - LLM as a judge can help us scale the evaluation of generated text - without depencende on manual human annotation.

"""

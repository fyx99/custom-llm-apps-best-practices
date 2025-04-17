Integrating LLM-based functionality into enterprise software has become more accessible than ever. However, evaluating the performance, accuracy, and utility of such systems is increasingly critical. Traditional approaches like unit testing and end-to-end integration testing fall short when dealing with non-deterministic components such as LLMs. This blog draws from our customer engagements to present practical strategies for benchmarking AI systems and ensuring the delivery of high-quality end product.



### Background

As part of SAP's Business AI Strategy, we recognize the vast potential for building customer-specific use cases. With the extensive suite of services available on the BTP, developers are empowered to create their own pro-code automations, business logic, and even sophisticated multi-agent systems. This enables them to leverage the power of Generative AI to address domain-specific business challenges and processes. One of the key offerings in this space is the Generative AI Hub, which provides access to a wide range of foundation models, including multimodal ones, from providers such as OpenAI, Anthropic, and others.

When developing business apps involving LLM's everybody comes to the point, where the first MVP is there and you start to collect feedback from the stakeholders. They demand certain answers to be different and off you go - enaging in improving the system by doing prompt engineering - adding data or adjusting parameters. But at a certain point in time - making changes to fit case 1 - will lead to a regression in case 2. At this point - it gets harder to make proper decisions on how to tweak the system moving forward. In this case it is advised to introduce benchmarks. Benchmarks are techniques to test your version of the AI-system with a set of examples and measure the output based on various approaches. 

The challenge:

The primary challenge with LLM-based systems lies in their inherent non-deterministic nature. Due to the way LLMs function, even identical or nearly identical inputs can produce varying outputs. 
Same when executing the same propmt but in different contexts. This behavior stems from the architecture of LLMs, which relies on sampling tokens from a probability distribution, introducing an element of randomness into their responses.

Let's look at 3 relevant patterns of LLM's Evals:

## 1. Evaluate Generated Text

Typical Use-Casese:
Writing-Assistants: Use Cases focused on enhanced writing for example - Goal Generation in SAP Successfactors
Copilots: Use-Cases involving text generated to be presented to the user:
Retrival-Augmented-Generation Systems: Knowledge base integrations to LLM's


Overview:
This paradigm focuses on the quality of free-form, unstructured outputs (e.g., paragraphs, dialogues, creative writing). Here, the goal is to determine whether the generated text is coherent, contextually relevant, factually accurate, and stylistically appropriate.

Key Considerations and Metrics:

Reference-Based vs. Reference-Free Metrics:

Reference-Based: Traditional metrics such as BLEU, ROUGE, and BERTScore measure n-gram overlap or semantic similarity with human-written references. These approaches are effective for well-defined tasks (e.g., translation or summarization) but may not capture creativity or nuance. E.g. given a generated Text "The Eifeltower is located in London" it is compared to a expected value "The Eifeltower is located in Paris". Metrics like BLEU now calculcate the textual overlap between to two word sequences. But be aware - these metrics sometimes focus on superficial patterns rather than deep semantics. Plus it requires us to actually have pre-written best case examples for any given input prompt. Usually something that is costly to construct.

Reference-Free / LLM-as-a-Judge: Emerging approaches leverage other LLMs to act as evaluators (or “judges”) using chain-of-thought (CoT) prompting, as seen in methods like G-Eval. These frameworks can assess quality dimensions like coherence, factuality, and relevancy by comparing outputs to human-like evaluation criteria. Essentially we instruct a seperate LLM to judge a generated text towards our own defined metrics. The beauty of this approach is that we are able to miimic a humans decision and taste by baking that into the prompting. The goal is to have a good enough overlap between an experts judgement and the LLM'Judges. Aligning LLM evaluators with human preferences requires careful prompt design and possibly iterative tuning based on a small eval dataset.

Lets have a look at the ROUGE metric example:

Given a set of questions - we now evaluate the generated answer from the LLM'system with the ground truth. 

```python
from rouge_score import rouge_scorer

# Simulated LLM responses
generated_answers = [
    "The Eiffel Tower is located in London.",   # Incorrect
    "Berlin is the capital city of Germany.",   # Correct
    "Mount Everest is in China."                # Incorrect
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
```

For this purpose we use the rouge_score library and calculate different Rouge values - ROUGE-1 and ROUGE-L. Explaination of 1 and l 
Then we print out the result. Thats already it for the first example. We can use this and wrap it in an eval loop and we are good.



Next, let's see how it is done with LLM-as-a-Judge:

Again, we start with the same answers - but this time we do not use reference answers - but base our evaluation on the LLM.

```python
generated_answers = [
    "The Eiffel Tower is located in London.",   # Incorrect
    "Berlin is the capital city of Germany.",   # Correct
    "Mount Everest is in China."                # Incorrect
]

# Define the judging prompt
    judging_prompt = """
You are an impartial judge. Compare the generated answer on correctness.

Generated answer: {generated_answer}

Only evaluate factual correctness, not style or phrasing.

Return your judgment as JSON with the following schema:
{
  "correctness": "Correct" | "Incorrect"
}
"""

correct_count = 0
total_count = len(generated_answers)

# Loop through answers and evaluate using LLM with judging prompt
for gen, ref in generated_answers:
    result = llm(
        prompt=judging_prompt,
        variables={"generated_answer": gen},
        json_schema={"correctness": {"type": "string", "enum": ["Correct", "Incorrect"]}}
    )
    
    if result['correctness'] == "Correct":
        correct_count += 1

    print(f"Generated: {gen}\nJudgment: {result['correctness']}\n")

# Calculate evaluation score
eval_score = (correct_count / total_count) * 100
print(f"Evaluation Score: {eval_score}%")

```

Using a LLM call fiven a structured schema - we can obtain a score of the correctness. 
We could also use the LLM-as-a-judge pattern to compare multiple versions of our LLM-system by having it pick the better answer.

So we can tell there is a broad variety of fine-granular metric evaluation possible with this approach. Always of course having the limitations given by the technology in mind.


## 2. Evaluate Structured Outputs

Embedded functionalities: Use-cases involving mainly structured outputs

These use-cases often times reduce the entropy of the data - for example using an LLM prompt to classify an email - generate a Graph based on a Prompt etc.

This is the most simple pattern to evaluate here. You can collect a set of examples - for example 30 emails - and hand-lable them with the expected classes. Similar to classical classification algorithms, you can now calculate metrics like accuracy and recall. Whenever possible this is a great way to put a numeric metric as a indicator of performance.

Given categorical values - like the mail class - you can go for an exact match eval to get the total accuracy over the benchmark set.
For numeric outputs you can use the classical deviation metrics like mean square error as a metric.
In the case of a structured output you can also have rule-based evaluators - that for example check the adherence to the format expected. e.g. is it parseable JSON? Yes or No.

Lets have a look at the mail classification exapmle:


```python
def classify_mail(mail_content):
    """ Classifies an email into predefined categories using a language model. """
    # Pseudo code for LLM-based classification
    categories = ["Business", "Personal", "Spam", "Support", "Sales"]
    prompt = f"""
    You are an email classification assistant. 
    Classify the following email into one or more of the following categories: {categories}.
    Email content: 
    {mail_content}
    """ 

    # Define the JSON schema for the structured output
    schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["Business", "Personal", "Spam", "Support", "Sales"]
            }
        },
        "required": ["category"]
    }

    # Validate the output against the schema
    result = llm(prompt, variables={"categories": categories, "mail_content": mail_content}, json_schema=schema)  # returning structured output

    return result

# Example usage
email = "Hi, I would like to inquire about the pricing for your enterprise solutions."
classification_result = classify_mail(email)
print(classification_result)
```
Our classify_mail content produces a structured output with the category field.
We can then use a benchmarking set consisting of some sample mails and their expected class to evaluate the accuracy and recall.

```python
from sklearn.metrics import accuracy_score, recall_score

# Define a test set with mail contents and their expected classifications
test_set = [
    {"mail_content": "Hi, I would like to inquire about the pricing for your enterprise solutions.", "expected_class": "Business"},
    {"mail_content": "Hey, how are you doing? Let's catch up soon!", "expected_class": "Personal"},
    {"mail_content": "Congratulations! You've won a free gift card. Click here to claim it.", "expected_class": "Spam"},
    {"mail_content": "I need help with resetting my account password.", "expected_class": "Support"},
    {"mail_content": "We have a special discount on our products this week. Don't miss out!", "expected_class": "Sales"}
]

# Collect predictions
predictions = []
true_labels = []

for test_case in test_set:
    mail_content = test_case["mail_content"]
    expected_class = test_case["expected_class"]
    true_labels.append(expected_class)
    
    # Call the classify_mail function
    result = classify_mail(mail_content)
    predictions.append(result["category"])

# Calculate accuracy and recall
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average="macro")

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
```
We can calculate the accuracy and recall of the classification model using the sklearn library.
In this example one of them is missclassified. But to be fair - it could apply to both categories - Business and Sales.
That would be a good time to investigate if the categories are too similar to each other or if we have to refine the prompting.



## 3. Evaluate actions taken

Agentic behavior: Use-cases involving interaction with tools

These cases essentially require a agent to do planning, use multiple tools and either just take actions, present structured output in a app or even interact with the user in text form.
In this case the most critical aspect is the trajectory the agent takes. Given a query to book a trip, you want your agents to come to the conclusion to take xyz steps. 
We can validate the trajectory or the resulting state of the environment to do so.

Let's have a look at some sample on how this could look like:

The agentic behaviour results in a trace of ordered tool_calls - having a tool_name and some tool arguments. They can be of any type - often times JSON.
In our case we have a Agent capable of using multiple tools to book a flight. 

We provide the agent the following tools:

```python
agent_tools = [
    {
        "name": "search_flight",
        "description": "Searches for available flights based on destination and date.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The destination city for the flight."
                },
                "date": {
                    "type": "string",
                    "description": "The date of the flight in YYYY-MM-DD format."
                }
            },
            "required": ["destination", "date"]
        }
    },
    {
        "name": "select_flight",
        "description": "Selects a specific flight from the search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {
                    "type": "string",
                    "description": "The unique identifier of the flight to be selected."
                }
            },
            "required": ["flight_id"]
        }
    },
    {
        "name": "book_flight",
        "description": "Books the selected flight.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {
                    "type": "string",
                    "description": "The unique identifier of the flight to be booked."
                }
            },
            "required": ["flight_id"]
        }
    },
    {
        "name": "lookup_weather",
        "description": "Retrieves the weather forecast for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location to get the weather forecast for."
                }
            },
            "required": ["location"]
        }
    }
]
```

Now looking at an example - we have a request from a user to book a flight at the 1st of May 2025 to Paris. Resulting in a trace of tool calls.
For our use-case and the response quality it is very important that the agent uses these tools in the correct order and with the input specified by the user.
We now compare the actual actions taken - and its sequence to the expected trace.

```python
# Example: Define an expected trace and actual trace using dictionaries
expected_trace = [
    {"tool_name": "search_flight", "inputs": {"destination": "Paris", "date": "2025-05-01"}},
    {"tool_name": "select_flight", "inputs": {"flight_id": "flight_1"}},
    {"tool_name": "book_flight", "inputs": {"flight_id": "flight_1"}}
]

actual_trace = [
    {"tool_name": "search_flight", "inputs": {"destination": "Paris", "date": "2025-05-01"}}, # Good tool, bad input
    {"tool_name": "select_flight", "inputs": {"flight_id": "flight_1"}},                      # Good
    {"tool_name": "lookup_wheather", "inputs": {"location": "london"}}                        # Bad tool
]


# Function to compare two tool calls represented as dictionaries
def compare_tool_calls(actual: dict, expected: dict) -> bool:
    return actual == expected

# Function to evaluate a trace (list of tool call dictionaries)
def evaluate_trace(actual_trace: list, expected_trace: list) -> bool:
    if len(actual_trace) != len(expected_trace):
        print(f"Trace lengths do not match: {len(actual_trace)} vs {len(expected_trace)}")
        return False
    
    for i, (actual, expected) in enumerate(zip(actual_trace, expected_trace)):
        if not compare_tool_calls(actual, expected):
            print(f"Mismatch at step {i}:")
            print(f"  Actual: {actual}")
            print(f"  Expected: {expected}")
            return False

    return True


# Evaluate if the actual trace matches the expected trace
is_valid = evaluate_trace(actual_trace, expected_trace)
print("Trace is valid:", is_valid)
```

We can check if the actions taken by the LLM are correct or not. In this case the prepared example is not correct - the LLM used a wrong tool and wrong input for the tool.
This is a simple example - but in practice we can use this to evaluate the LLM's performance and correctness of the actions taken.
Scaling the number of example prompts and expected traces allows us to evaluate the LLM's performance in a more robust way.

### Using evals to improve performance
When speaking with developers, we often encounter concerns about the performance of custom LLM applications. While some individual prompts may work well, they often fail to address the breadth of use cases users present. To tackle this, you can adopt a continuous improvement cycle using the following general schema:

1. **Define Metrics for Your Use Case**: Identify the key aspects to evaluate. Is factual accuracy the most critical factor? Or is it the style of the output or adherence to formatting requirements? Tailor your KPIs to reflect these priorities.

2. **Collect Data for Your MVP as a Benchmark Set**: Gather data from your initial users, include examples where the system fails, and ensure the dataset is diverse to cover a wide range of scenarios.

3. **Iterate and Improve**: Refine your prompts to make them clearer, add relevant business context, or rethink your approach. Consider using structured outputs, breaking down complex tasks into multiple prompts, or employing advanced techniques like Chain-of-Thought prompting. Experiment with different models and parameters, such as temperature. Additionally, review the raw prompts sent to the LLM—issues like unfilled template variables or formatting errors are common pitfalls.

4. **Run Evaluations for Each Change**: After making small adjustments, run evaluations and focus on failing examples to identify areas for further improvement.

5. **Repeat**: Continue this cycle to iteratively enhance the performance and reliability of your LLM application.

By following this structured approach, you can systematically address performance issues and ensure your application meets the needs of its users.

### Conclusion:

After exploring various approaches, it is evident that the choice of metrics and benchmarking techniques depends heavily on the specific use case. Some scenarios may even benefit from a combination of these methods. The sample code provided is intended to inspire you to develop your own evaluation code tailored to your needs.

I hope you found this guide helpful. Feel free to explore the complete code examples, and if you have any questions, don’t hesitate to leave a comment!

Integrating LLM-based functionality into enterprise software has become more accessible than ever. However, evaluating the performance, accuracy, and utility of such systems is increasingly critical. Traditional approaches like unit testing and end-to-end integration testing fall short when dealing with non-deterministic components such as LLMs. This blog draws from our customer engagements to present practical strategies for benchmarking AI systems and ensuring the delivery of high-quality end product.



### Background

As part of SAP's Business AI Strategy, we recognize the vast potential for building customer-specific use cases. With the extensive suite of services available on the BTP, developers are empowered to create their own pro-code automations, business logic, and even sophisticated multi-agent systems. This enables them to leverage the power of Generative AI to address domain-specific business challenges and processes. One of the key offerings in this space is the Generative AI Hub, which provides access to a wide range of foundation models, including multimodal ones, from providers such as OpenAI, Anthropic, and others.

When developing business applications involving LLMs, every team eventually reaches the point where the first MVP is ready, and feedback from stakeholders starts rolling in. Stakeholders often request that certain answers be adjusted, and naturally, you begin refining the system — whether through prompt engineering, adding data, or tweaking parameters.

However, as you address specific cases — optimizing for case 1, for example — you’ll often encounter unintended regressions in case 2. At this stage, making effective decisions about further adjustments becomes increasingly difficult.

This is where introducing benchmarks becomes essential. Benchmarks provide a structured way to evaluate your AI system using a predefined set of examples, allowing you to measure performance consistently across different scenarios and guide future improvements more systematically.

The challenge:

The primary challenge with LLM-based systems lies in their inherent non-deterministic nature. Because of the way large language models operate, even identical — or nearly identical — inputs can result in different outputs. The same applies when executing the same prompt in varying contexts.

This variability is rooted in the architecture of LLMs, which sample tokens from a probability distribution, introducing an element of randomness into their responses.

Let’s explore three relevant patterns in evaluating LLMs:


## 1. Evaluate Structured Data Generation

This area covers use cases where the output follows a predictable, structured format rather than open-ended text. Common examples include classifying emails into categories, extracting key information from documents, generating machine-readable formats like JSON or XML, or creating structured representations such as graphs or tables from text prompts. These scenarios are especially useful when integrating LLMs into automated workflows or downstream systems that rely on consistent, structured data.

This is the simplest pattern to evaluate. You can start by collecting a representative set of examples — for instance, 30 emails — and manually label them with the expected categories. Similar to classical classification tasks, you can then calculate performance metrics such as accuracy and recall. Whenever possible, applying these kinds of numeric metrics provides a clear, objective indicator of your system’s performance.

For categorical outputs — like email classes — an exact match evaluation is usually the most straightforward approach, allowing you to compute overall accuracy across the benchmark set.

For numeric outputs, you can rely on traditional deviation-based metrics, such as mean squared error (MSE), to quantify performance.

In cases involving structured outputs (e.g., JSON, XML), rule-based evaluators can also be useful — for example, to check whether the generated output adheres to the expected format. A basic but critical check might simply verify: Is the output valid JSON? Yes or No.

Let’s take a closer look at the email classification example:


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
Our classify_mail function generates a structured output, including a category field. To evaluate its performance, we can use a benchmark set consisting of sample emails along with their expected categories. This allows us to calculate key metrics like accuracy and recall, providing insights into how well the function is performing in classifying emails.

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
We can calculate the accuracy and recall of the classification model using the sklearn library. In this example, one of the emails is misclassified. However, it’s worth noting that the email could reasonably belong to either the Business or Sales category. This highlights a key moment to assess whether the categories are too similar or if the prompting needs refinement to reduce ambiguity.

## 1. Evaluate Text Generation

Typical use cases in this category include writing assistants, where the goal is to enhance text generation for applications such as goal generation in SAP SuccessFactors. Another example are copilot systems, where text is generated to be presented to the user, assisting with tasks or providing suggestions. Additionally, retrieval-augmented generation (RAG) systems are increasingly popular, as they integrate knowledge bases with LLMs to produce more context-aware and informative outputs.

This paradigm evaluates the quality of free-form, unstructured text generation, such as paragraphs, dialogues, or creative writing. The primary goals are to assess whether the generated text is coherent, contextually relevant, factually accurate, and stylistically appropriate.

Key Considerations and Metrics:
Reference-Based vs. Reference-Free Metrics:

Reference-Based Metrics:
Traditional metrics like BLEU, ROUGE, and BERTScore measure n-gram overlap or semantic similarity with human-written references. These approaches work well for well-defined tasks such as translation or summarization, but may not fully capture the creativity or nuance of the generated text.
For instance, if the LLM generates "The Eiffel Tower is located in London," and the expected reference is "The Eiffel Tower is located in Paris," metrics like BLEU will calculate the textual overlap between the two sequences and it will be quite a high result - because the sentence is very similar. However, it's important to note that these metrics sometimes focus more on superficial patterns than on deeper semantic meaning. Additionally, this approach requires having pre-written "ideal" examples, which can be costly and time-consuming to construct.

Reference-Free / LLM-as-a-Judge:
Emerging techniques leverage other LLMs to act as evaluators (or “judges”) of the generated text using chain-of-thought (CoT) prompting, as seen in methods like G-Eval. This allows LLMs to assess key quality dimensions such as coherence, factual accuracy, and relevance by comparing outputs to human-like evaluation criteria. In this approach, a separate LLM is instructed to judge the generated text based on defined metrics. The advantage here is that we can replicate human decision-making and preferences by embedding them into the prompting process. Aligning LLM evaluators with human judgment, however, requires careful prompt design and may involve iterative tuning based on a small evaluation dataset.

Example: ROUGE Metric Evaluation

To illustrate this, let’s consider a scenario where a set of questions is posed to an LLM, and the generated answers are compared against the ground truth. The ROUGE metric is often used here to measure the quality of the generated response by comparing it to the reference answers, providing insight into how well the model is performing in terms of recall, precision, and overlap with human-provided answers.

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

Using the rouge_score library, we calculate several ROUGE metrics, focusing on ROUGE-1 and ROUGE-L.

ROUGE-1 measures the overlap of unigrams (single words) between the generated text and the reference. It’s a basic evaluation metric for capturing lexical similarity.

ROUGE-L looks at the longest common subsequence between the generated text and the reference, which helps assess the fluency and coherence of the generated content.

You can see that the ROUGE scores are quite high for example 3 - the sentence structure is very much similar - just the actual fact - the country name is different.
This showcases the big limitation of n-gram based metrics. They cannot really judge the semantic meaning of the sentence - but just compare the available character sequences.

Next, let's see how it is done with LLM-as-a-Judge:

Now, let's explore how the evaluation works with LLM-as-a-Judge. This time, we won’t use reference answers. Instead, we base our evaluation on an LLM itself. The idea is to ask a separate LLM to evaluate the generated text based on predefined quality criteria, such as coherence, factual accuracy, and relevance.

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

By using an LLM with a structured schema, we can assess the correctness of the generated text. Additionally, the LLM-as-a-Judge pattern allows us to compare multiple versions of our LLM system, enabling the model to select the best response.

This approach provides a wide range of fine-grained metric evaluations. However, you might wonder: If the LLM generated the response in the first place and perhaps did so incorrectly, in poor form, or with irrelevant content, why would the judge LLM perform better?

Interestingly, there is scientific evidence suggesting that LLMs actually perform better when tasked with evaluating responses than when generating them initially. There’s an intuitive explanation for this: Just as humans often find it easier to judge whether something is well-written than to write it themselves, LLMs can more effectively assess quality than generate perfect responses.

In practice, using an LLM as a judge enables us to scale the evaluation of generated text without relying on manual human annotation.


## 3. Evaluate Tool Use

Use cases in this category involve agentic behavior, where agents interact with multiple tools to complete tasks. These can range from automated planning scenarios, where the agent must plan and execute a series of actions, to tool interaction, where the agent presents structured outputs, takes actions, or even interacts with users in a conversational manner. The most critical aspect of these use cases is the trajectory the agent takes. For example, in a task like booking a trip, the agent needs to logically follow a series of steps, such as searching for flights, selecting dates, and confirming details. Evaluating such tasks involves assessing both the trajectory and the resulting state of the environment.

In agentic behavior, an agent produces a trace of ordered tool calls, where each call includes a specific tool name and its corresponding arguments — typically structured as JSON. This trace outlines the sequence of steps the agent takes to complete a given task.

Let’s take a simple example: imagine an agent is asked to book a flight to Paris on May 1st, 2025. The agent will process this request and generate a series of tool calls as it works toward fulfilling the task.

For use cases like this, the quality of the response heavily depends on the agent using the correct tools in the correct order, while accurately passing along the input provided by the user. To evaluate this, we can compare the actual tool call sequence produced by the agent against a predefined expected trace — ensuring both the order and the arguments match the intended plan.

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
We can now check whether the actions taken by the LLM are correct or not. In this particular example, the agent’s response is incorrect — it used the wrong tool and provided the wrong input.

While this is a simple case, the same approach can be applied in practice to systematically evaluate the LLM’s decision-making and the correctness of the actions it takes. By scaling the number of example prompts and expected traces, we can build a more robust and reliable benchmark for assessing the agent’s performance across different scenarios.

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

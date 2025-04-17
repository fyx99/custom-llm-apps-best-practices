from examples.llm_util import llm


def classify_mail(mail_content):
    """ Classifies an email into predefined categories using a language model. """
    # Pseudo code for LLM-based classification
    categories = ", ".join(["Business", "Personal", "Spam", "Support", "Sales"])
    prompt = """
You are an email classification assistant. 
Classify the following email into one or more of the following categories: {{?categories}}.
Email content: 
{{?mail_content}}
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
if __name__ == "__main__":
    email = "Hi, I would like to inquire about the pricing for your enterprise solutions."
    classification_result = classify_mail(email)
    print(classification_result)
    
    
    
    
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_recall_fscore_support

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
    print(f"Mail: {mail_content}, Expected: {expected_class}, Actual: {result['category']}") 
    predictions.append(result["category"])

# Calculate accuracy and recall
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average="macro")

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")


# Calculate precision, recall, and F1-score per class
class_metrics = precision_recall_fscore_support(true_labels, predictions, labels=["Business", "Personal", "Spam", "Support", "Sales"])

# Extract recall and accuracy per class
classes = ["Business", "Personal", "Spam", "Support", "Sales"]
for idx, class_name in enumerate(classes):
    precision = class_metrics[0][idx]
    recall = class_metrics[1][idx]
    f1_score = class_metrics[2][idx]
    support = class_metrics[3][idx]
    print(f"Class: {class_name}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}, Support: {support}")


"""
Prints:

Mail: Hi, I would like to inquire about the pricing for your enterprise solutions., Expected: Business, Actual: Sales
Mail: Hey, how are you doing? Let's catch up soon!, Expected: Personal, Actual: Personal
Mail: Congratulations! You've won a free gift card. Click here to claim it., Expected: Spam, Actual: Spam
Mail: I need help with resetting my account password., Expected: Support, Actual: Support
Mail: We have a special discount on our products this week. Don't miss out!, Expected: Sales, Actual: Sales
Accuracy: 0.8
Recall: 0.8

Comment:

You can calculate the accuracy and recall of the classification model using the sklearn library.
In this example one of them is missclassified. But to be fair - it could apply to both categories - Business and Sales.
That would be a good time to investigate if the categories are too similar to each other or if we have to refine the prompting.

"""

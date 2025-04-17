
agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flight",
            "description": "Searches for available flights based on destination and date.",
            "strict": True,
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
                "required": ["destination", "date"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_flight",
            "description": "Selects a specific flight from the search results.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {
                        "type": "string",
                        "description": "The unique identifier of the flight to be selected."
                    }
                },
                "required": ["flight_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Books the selected flight.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {
                        "type": "string",
                        "description": "The unique identifier of the flight to be booked."
                    }
                },
                "required": ["flight_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Retrieves the weather forecast for a specific location.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city or location to get the weather forecast for."
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    }
]




from gen_ai_hub.proxy.native.openai.clients import OpenAI

client = OpenAI(api_version="2024-10-21")


def simulate_tool_response(tool_name):
    # Simulate tool execution (mock data for simplicity)
    if tool_name == "search_flight":
        tool_result = '{"flights": [{"flight_id": "FL123", "destination": "Paris", "date": "2025-05-10"}]}'

    elif tool_name == "select_flight":
        tool_result = '{"status": "Flight FL123 selected"}'

    elif tool_name == "book_flight":
        tool_result = '{"status": "Flight FL123 booked successfully", "booking_id": "BK9876"}'

    elif tool_name == "lookup_weather":
        tool_result = '{"forecast": "Sunny, 22°C"}' 
    return tool_result



# Define your system message
system_message = {
    "role": "system",
    "content": "You are a helpful travel assistant. Help the user book a flight using the available tools. Search, select, and book a flight step by step. Return the booking confirmation at the end."
}

# Start the conversation
messages = [
    system_message,
    {"role": "user", "content": "I want to book a flight to Paris on 2025-05-10."}
]

# trace of actions taken
trace = []

# Simplified loop to handle the agent's planning and execution
while True:
    response =  client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=agent_tools,
        tool_choice="auto",
        parallel_tool_calls=False,
        temperature=0.2
    )

    # If the model wants to call a tool
    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool_result = simulate_tool_response(tool_name)
        
        trace.append({      # logging the tool use sequence for later evaluation
            "tool_name": tool_name,
            "inputs": tool_args
        })

        # Add the assistant's tool call and the tool's response to the messages
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": tool_result
        })

    else:
        # Goal reached: return the assistant's final message
        print("Assistant:", response.choices[0].message.content)
        break


# Print the trace in a readable format
print("Trace of actions taken:")
for step, action in enumerate(trace, start=1):
    print(f"Step {step}:")
    print(f"  Tool Name: {action['tool_name']}")
    print(f"  Inputs: {action['inputs']}")
    
    

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



"""
Prints:

Assistant: Your flight to Paris on 2025-05-10 has been successfully booked. Your booking ID is BK9876. Safe travels!
Trace of actions taken:
Step 1:
  Tool Name: search_flight
  Inputs: {"destination":"Paris","date":"2025-05-10"}
Step 2:
  Tool Name: select_flight
  Inputs: {"flight_id":"FL123"}
Step 3:
  Tool Name: book_flight
  Inputs: {"flight_id":"FL123"}
Mismatch at step 2:
  Actual: {'tool_name': 'lookup_wheather', 'inputs': {'location': 'london'}}
  Expected: {'tool_name': 'book_flight', 'inputs': {'flight_id': 'flight_1'}}
Trace is valid: False

Comment:

We can check if the actions taken by the LLM are correct or not. In this case the prepared example is not correct - the LLM used a wrong tool and wrong input for the tool.
This is a simple example - but in practice we can use this to evaluate the LLM's performance and correctness of the actions taken.
Scaling the number of example prompts and expected traces allows us to evaluate the LLM's performance in a more robust way.

"""
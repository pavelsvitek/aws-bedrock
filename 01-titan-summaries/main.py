# Use the Converse API to send a text message to Titan Text G1 - Express.

import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="ca-central-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "amazon.titan-text-lite-v1"

# Start a conversation with the user message.
user_message = """Meeting transcript:
Miguel: Hi Brant, I want to discuss the workstream  for our new product launch
Brant: Sure Miguel, is there anything in particular you want to discuss?
Miguel: Yes, I want to talk about how users enter into the product.
Brant: Ok, in that case let me add in Namita.
Namita: Hey everyone
Brant: Hi Namita, Miguel wants to discuss how users enter into the product.
Miguel: its too complicated and we should remove friction.  for example, why do I need to fill out additional forms?  I also find it difficult to find where to access the product when I first land on the landing page.
Brant: I would also add that I think there are too many steps.
Namita: Ok, I can work on the landing page to make the product more discoverable but brant can you work on the additonal forms?
Brant: Yes but I would need to work with James from another team as he needs to unblock the sign up workflow.  Miguel can you document any other concerns so that I can discuss with James only once?
Miguel: Sure.
From the meeting transcript above, Create a list of action items for each person.
"""
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens":4096,"stopSequences":["User:"],"temperature":0,"topP":1},
        additionalModelRequestFields={}
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)


## API Calls for OpenAI, Claude, and Gemini

## CLaude API
The claude API call is similar to the OpenAI API call.

### Differences with OpenAI
1. As we can system message is provided separately from the user prompt.
2. We cannot pass a system message in the messages list like we do with OpenAI. Otherwise it will throw an error.
3. Also Max Tokens is an parameter that is optional for OpenAI but is required for Claude.
4. Anthropic supports streaming but with a different(different from create - claude.messages.stream ) method called stream and not 
   like openAI where we pass an additional parameter.

```
# Claude 3.7 Sonnet
# API needs system message provided separately from user prompt
# Also adding max_tokens

message = claude.messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

print(message.content[0].text)
```



### Gemini API
The Gemini API is similar to the OpenAI API but has some differences in the way it handles messages and parameters.
1. As we can see here an Object of type GenerativeModel is created with the model name and system instruction.
2. We pass the user prompt directly to the `generate_content` method of this GenerativeModel object.

```
# The API for Gemini has a slightly different structure.
# I've heard that on some PCs, this Gemini code causes the Kernel to crash.
# If that happens to you, please skip this cell and use the next cell instead - an alternative approach.

gemini = google.generativeai.GenerativeModel(
    model_name='gemini-2.0-flash',
    system_instruction=system_message
)
response = gemini.generate_content(user_prompt)
print(response.text)
```


## Adverserial Conversations between LLMs

### Prompt Structure
We will use the below prompt structure to have an adverserial conversation between two LLMs.
We can have longer list of messages in the conversation.
1. First we have a system message that sets the context for the conversation.
2. Then we have a user prompt that starts the conversation.
3. Then we have an assistant response that is the response from the first LLM.
4. Then we have a new user prompt that is the response from the second LLM.

**Note - This structure doees not have to necessarily be used for adverserial conversations. 
We can use this structure for any conversation between a user and an assistant.
Infact this is how we converse using popular UIs like chatgpt .
Everytime we have a conversation with something like chatgpt, we are actually sending a list of messages in this format.
the system prompt is same , but everytime there is a userprompt and an assistant response.
Then again there will be a user prompt and an assistant response and so on.**

**Note - Therefore for every conversation the entire conversation history is sent as input to the model.
Therefore we get the illusion that the model is able to remember the entire conversation.**

This entire message passes whould be within the context window of the model.

```
[
    {"role": "system", "content": "system message here"},
    {"role": "user", "content": "first user prompt here"},
    {"role": "assistant", "content": "the assistant's response"},
    {"role": "user", "content": "the new user prompt"},
]
```

### Now we will have a conversation between two LLMs gpt-4o-mini and llama3.2.
Code in Week 2 day 2.ipynb


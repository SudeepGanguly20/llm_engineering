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
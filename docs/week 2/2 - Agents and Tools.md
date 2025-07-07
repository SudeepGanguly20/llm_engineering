# Tools (Week 2 - Day 4)

![image.png](attachment:9c1b13e1-3480-40f5-bc66-1f62615f190b.png)


## This is How it Works -
1. We define the functions that we would want the LLM to use like if we want our LLM to use calculator as a tool , we define the calculator tool.
2. We describe what are the inputs , what are the outputs and when should the LLM use it.
3. When talking to the LLM we say can you do this task and you have access to this tool to do this task.
4. The LLM inturn when generating the output asks his to run the tool and pass the response to it.

Note - At this point the LLM would not be directly calling the tool.

![image.png](attachment:df25ce97-b462-469b-a886-61e96a66e341.png)
![image.png](attachment:ab8b0a7f-eddc-498b-9190-59bb4836bad5.png)

1. **Add data as knowledge** - one example is to call a database and get some data about a specific customer when the customer is asking the llm to check it's flight status.
2. **Taking actions** - book a meeting or book a flight ticket. Ofcourse it needs access to the api of the ticket booking site.
3. **Perform Calculation** - LLMs are not naturally good at calculations (No longer the case) , therefore we can add a tool that does the calculations.
4. **Modify UI** - We can ask LLM to modify our UI based on some data

## Check out the Tools section in the jupyter notebook for Day 4 under week 2

### How to add any function as a tool ?
#### There's a particular dictionary structure that's required to describe our function:
1. First we need to define the function that we want to use as a tool.

```
# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")
```

3. Then we need to create a dictionary that describes the function.
3. This dictionary should have the following
   1. `name` - The name of the function
   2. `description` - A brief description of what the function does
                        Note - Giving an example in the description is a very good idea.
   3. `parameters` - A dictionary that describes the parameters of the function
      - `type` - The type of the parameters, in this case it is an object
      - `properties` - A dictionary that describes the properties of the parameters
      - `required` - A list of required parameters
      - `additionalProperties` - A boolean value that indicates whether additional properties are allowed or not

```
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}
```


### Hooking this up with the LLM
1. First we need to create a list of tools that we want to use.
```
# And this is included in a list of tools:
tools = [{"type": "function", "function": price_function}]
```

2. Then we need to pass this list of tools to the LLM when we are making the API call.

```
response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
```
**The API request is exactly the same as the one we used before, but now we are passing the tools parameter.**

**Now when openai will build the prompt for the LLM, it will include the tools that we have defined. This tool or function is also going to be converted into a series of tokens 
and passed to the LLM . This works because the LLM is trained on a lot of code and it can understand the structure of the function.**


## But But , the LLM will not call the function directly . It will only inform us that it wants to call the function.
## This happen when the finish_reason is set to "tool_calls" in the response. 

This means , 
that Basically GPT will say , i do not have a output for you yet , instead , you run this tool and provide me the output of the tool and 
then i will give you the final output.

So if GPT wants us to run the tool -
```if response.choices[0].finish_reason=="tool_calls":```

So as part of asking us to run the tool, GPT will return a message that we will have to read. We read this message-
```message = response.choices[0].message```

We need to unpack the message to understand what GPT wants us to do. 
handle_tool_call(message) is a function that we will write to handle this message.
```response, city = handle_tool_call(message)```

Then we add the last response from the llm into the messages list so that we can continue the conversation by linking the last message.
```messages.append(message)```

We also add the response we got the tool call
```messages.append(response)```


Finally we call the LLM again.
Note - In the second LLm call we do not pass the tools parameter again because we do not expect the tools to be needed again.




# Agents (Week 2 - Day 5)

Now that we know about tools, we can use them to create agents.
![img_4.png](img_4.png)

1. Agents are programs that can use tools to perform tasks. 
2. They can be used to automate tasks, answer questions, and perform other actions.
3. They can also be integrated with traditional software applications to enhance their capabilities.
4. The LLM can also be responsible for planning the sequence of actions that the agent should take to achieve a goal.


In our case, we will create an artist agent that will generate images for us.
![img_5.png](img_5.png)
![img_6.png](img_6.png)



# Agentic AI
Agentic AI could be an umbrella term for AI systems that can act autonomously, make decisions, and adapt to changing environments.
It could be any of the below -

1. Breaking a complex problem into smaller steps, with multiple LLMs carrying out specialized tasks
2. The ability for LLMs to use Tools to give them additional capabilities
3. The 'Agent Environment' which allows Agents to collaborate
4. An LLM can act as the Planner, dividing bigger tasks into smaller ones for the specialists
5. The concept of an Agent having autonomy / agency, beyond just responding to a prompt - such as Memory


#Exercises to Do 
![img_7.png](img_7.png)
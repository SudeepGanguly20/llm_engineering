# More on Building RAG Applications with LangChain

#### Goal
![img_15.png](img_15.png)

#### Langchain Expressions Langauge
**Alternate to Coding using Langchain**

LCEL or Langchain Expressions Language is a way to write expressions in Langchain. 
It is similar to SQL, but it is designed specifically for Langchain. It allows you to write expressions that can be used to query and manipulate data in Langchain.

It is a yaml file and we can implement the steps that we want to implement in a descriptive way using this language.

![img_17.png](img_17.png)

As we see above , we mention the model , the tempreature, the prompt and the input variables.
There are different components like -
OPENAI_LLM and type is chatOpenAI
etc.


#### SO How does Langchain Works under the hood?
![img_18.png](img_18.png)


#### We will see how to use callbacks to see the prompts being sent by Langchain to the LLMs.
This is very useful to debug the prompts being sent by Langchain to the LLMs.
Because sometime we might see that the right chunks maynot be sent to the LLMs and the response that we might be getting from langchain may not be correct.

**Lets see the Problem First -**
If we go to the empoyees section of our InsuranLLM company knowledge base, we have the employee Maxine Thompson with below data.

```text
   She was recognized as Insurellm Innovator of the year in 2023, receiving the prestigious IIOTY 2023 award.    
```
When we try to ask the question on this employee to our chatAgent , we see below -
![img_19.png](img_19.png)

So lets see How to diagonise this problem using callbacks.

To hook up the callbacks, we need to do the following -
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

**it is as simple as adding the callbacks as a parameter as a list to the chain. We will be able to see the prompt as well with this**
**refer week 5 day 5.ipynb notebook for the code.**


# RAG Project

#Goal
![img_10.png](img_10.png)

# First since we will be using Langchain instead of native llm api calls therefore lets first see the abstractions that langchain provides.
1. LLMs - Langchain provides a simple interface to work with LLMs. We can use any LLM provider like OpenAI, HuggingFace, etc.
2. Retrievers - Langchain provides a simple interface to work with retrievers or vector stores. We can use any retriever like Chroma, Pinecone, etc.
3. Memory - Langchain provides a simple interface to work with memory. We can use any memory like Redis, etc to build a chat history or something.

In addition Langchain makes it super easy to build applications with LLMs by providing a set of tools and libraries to make it easier to work with LLMs.

Below is the 4 lines of code that we can use to build a RAG pipeline.
![img_11.png](img_11.png)

1. First we define that we will use OpenAI as our LLM provider and create a client for chat.
2. We use ConversationBufferMemory to keep track of the conversation history. return_messages=True means that we want to return the messages in the conversation history format and not a single long string.
3. Then we create a retriever using the as_retriever() method of the vector store. This retriever will be used to retrieve relevant documents from the vector store based on the user query.
4. Then we create a ConversationalRetrievalChain that will use the LLM and the memory to answer questions. We pass the LLM , the memory and the retriever and our conversational retrieval chain is ready to use.

week 5 , day 4.ipynb
#### We are still continuing with the same RAG project from last time where we build a conversational agent that can answer questions about our insurance company since we have stored the vector embeddings in the vector store chroma.




## Some Observations
1. With this conversational agent we only get data that is present in the vector store.

![img_12.png](img_12.png)

Wouldn't the llm still have access to its own knowledge, even though it is getting some extra knowledge from the vector store?

### Ans -Because LangChain's default system prompt is strict that the LLM should only respond based on information provided.

Excerpts from an anthropic blog -
Anthropic which warns of some of the challenges of using frameworks like LangChain:

```text
These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, 
and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts and 
responses, making them harder to debug. 
They can also make it tempting to add complexity when a simpler setup would suffice.
```
This is also the reason that even if we say hi to the chat , we get a non generic response -
![img_13.png](img_13.png)


2. For Lanchain we dont need to specific the system prompt for the LLM ? 

### Ans  -
It sets it behind the scenes using best practices.

LangChain: gets you up and running quickly without worrying about boilerplate prompting; 
but it can also be harder to understand what's going on behind the scenes.
LangChain provides a prompt templating system if you want to change the prompt.


3. Hallucinations
![img_14.png](img_14.png)


4. Building a RAG pipeline with PDF containing images and text.
You are building a RAG-based technical support agent using Langchain and OpenAI, with your product manuals stored as PDFs. You want the agent to answer customer queries with both text and images (if available) from the manuals. You are seeking guidance on extending the RAG example to support multimodal (text + image) responses.



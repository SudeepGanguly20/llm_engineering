# RAG

Until now we have seen the below ways to send input the LLMs -
1. Multishot Prompting
2. Passing tools to the LLM
3. Additional Context

RAG is a step further in sending more information to the LLMs. 

### The idea behind RAG is -
**1. We build a database of information that we want to use to answer questions.**
**2. Everytime user asks a question , we first search the database for relevant information.**
**3. if we find relevant information, we pass that information to the LLM along with the user question.**

![img.png](img.png)


### Lets First build a version of RAG without a database to get a clear idea.
#### Task -
![img_1.png](img_1.png)

Here we will read some product information from a file and store it in a dictionary.
Anytime a qustion comes in we will search the dictionary for relevant information and pass that to the LLM along with the user question.

**Implementation in day 1.ipynb**


## Encoding LLMs and Vector Embeddings
This is the most cruical idea behind RAG.

Almost all of the LLMs we have been talking about untill now are called as **Auto-regressive models**.
Then there are second category of models called **Auto-encoding models**.

### Difference between Auto-regressive and Auto-encoding models
**1. Auto-regressive models are given a set of past tokens and they predict the next token in the sequence. 
They keep repeating this process until they generate the entire sequence.**

**2. Auto-encoding models are given a full set of tokens past , present and future tokens, they create one output from the input tokens
that reflects the entire set of input tokens. Example - classification , sentiment analysis**

![img_2.png](img_2.png)

However there is another way that these models are used - vector embeddings.
Vector embeddings are a way to represent text in a numerical format that can be used for various tasks like similarity search, clustering, etc.
So text is converted into a series of numbers that represent the text in a way that the model can understand.

**Note - Vector embedding reflects the meaning of the text in a point in space is n dimensions.
Usually the text is converted into a vector of 768 or 1024 dimensions. 
for simplicity ,lets say if it represents in 3 dimension then the character a's position in the x,y and z dimenions is the vector embedding of 'a'**

![img_4.png](img_4.png)
![img_3.png](img_3.png)

Now if there are a bunch of blocks that have the same position in space, then they are similar to each other.
They do not have to be the same words but if the are in the same position in space or nearby, then there meanings are same.
So , things closer to each other in vectior embeddings should mean the same thing.
There is also something called vector maths that can be done.

Example -
The point in space that represents "King" is close to the vector that represents the word "Man"
We take the word King and subtract man from it. This means the vector will move back towards the direction of man
And then we add women which means we move forward in the direction of woman's vector.
This conceptually means we want to remove man from the meaning of the word king and added a woman to this word. This means our king
will be now a queen.
Remarkably this works and the vector that we get is very close to the vector that represents the word "Queen".


Therefore it is said that this vector embeddings are able to capture the meaning of the words they represent both in terms of their
meaning with respect to each others and in terms of similar words.


## Big Idea behind RAG
![img_5.png](img_5.png)
1. At the top we have something called a encoding model that is used to create vector embeddings from texts.
2. Then we store our vector embeddings in a database called vector database. SO we will store the text and the vector representing the text in the database.
3. Sequence is -
    1. User asks a question.
    2. We encode the question into a vector using the encoding model.**(Vectorizing the question)**
    3. We search the vector database for similar vectors to the question vector. **(we either get same vectors or vectors that are close in meaning to the question vector)**
    4. We get a list of texts that are similar to the question.
    5. We pass these texts along with the question to the LLM.
    6. The LLM generates an answer based on the question and the similar texts. 
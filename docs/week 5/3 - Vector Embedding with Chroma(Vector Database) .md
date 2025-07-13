## Vector Embedding Models
How exactly does a vector embedding work that is how is text coverted into numbers ?
![img_8.png](img_8.png)

1. word2vec - it uses a neural network to learn the vector representation of words and its meaning.
2. BERT - it uses a transformer architecture to learn the vector representation of words and its meaning.
3. OpenAI Embeddings - it uses a neural network to learn the vector representation of words and its meaning.


## We will use OpenAIEmbeddings for our use case. This is a auto-encodding model that takes in the entire set of token and gets a vector embedding for the entire set of tokens.
## we will use chroma as our vector database.

Week 5 - day 3.ipynb

![img_9.png](img_9.png)

### Task -
1. Add own chunks of text to the vector database and then see the representation to see if they are near or how far.
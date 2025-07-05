# Lets See Some Models

### Popular Open Source Models
![img_2.png](images/img_2.png)

### Three Ways to Use Models
![img_3.png](images/img_3.png) ![img_4.png](images/img_4.png)
1. **Chat interfaces** would be using the apis under the hood

2. **Native APIs** 
   1. Most frontier model providers have their own native apis to use them. 
   2. Langchain is a wrapper around the APIs of these models, allowing you to use them in a more structured way.
   3. Then there are apis provided by hyperscalers like amazon , azure and google that allow you to use these models that re hosted in their cloud.
   4. Amazon has amazon bedrock , google has vertex ai and azure has azure ml.

3. **Direct Inference**
    1. This is where you get the codes and the weights for the model and run it on your own hardware.
    2. This is the most flexible way to use models, but also the most resource-intensive.
    3. You need to have the hardware to run these models, which can be expensive.
    4. You also need to have the expertise to set up and run these models, which can be a barrier to entry for some people.
    5. There are two ways to do this:
       1. **Hugging Face**: Hugging Face is a platform that provides a wide range of models and tools for working with them. 
                             You can use their models directly or download them and run them on your own hardware. Normally to overcome
                             the issue of hardware, we may run it on something like google colab or a similar service.
      
       2. **Local Deployment**: This is where you download the model and run it on your own hardware. 
                                 This is the most flexible way to use models, but also the most resource-intensive.
                                 But here we do not have too much control over the model since it is fully compiled code.
   

### Running Models Locally
Ollama when we run on our local machine , also serves responses on the port 11434.
We can serve requests to the ollama model as -
```
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# Create a messages list using the same format that we used for OpenAI
messages = [
    {"role": "user", "content": "Describe some of the business applications of Generative AI"}
]

payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

response = requests.post(OLLAMA_API, headers=HEADERS, json=payload)
print(response.json()['message']['content'])
```


Here we are using the requests library to make a POST request to the ollama API.

Response -
![img_5.png](images/img_5.png)


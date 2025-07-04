# Setup

## **Install llama3.2 model from Ollama.**

*Note:* Ollama is a platform where there are multiple models. Once we install Ollama , we can run multiple models.

To see the models go here - [Link](your-link)

## Even Deepseek and Qwen from Alibaba is present here.
To run the llama3.2 model -<br>
ollama run llama3.2<br>



## Running Python with multiple versions using pyenv<br>
version 3.11.9 <br>
pyenv install 3.11.9<br>
pyenv global 3.12.3<br>



## Creating .env file
[//]: # (Create a .env file and add the OPEN AI Key with the keyword OPENAI_API_KEY)

Note - If we do ls in the folder then the .env file will not be listed since it is considered as a hidden file.


## Jupyter Lab
All the codes are present in Jupyter Notebooks.
For a revision of Jupyter Lab, there is a documentation Guide to Jupyter.ipynb in 
the week 1 directory.

[//]: # (Note - To run a command in jupyter notebook like we do in CLI we can use !)

[//]: # (Example - !pip install google)

[//]: # (Or !ls)



## Summary From Code -

### Setting Up openai
1. We create the .env file , put the OPENAI API Key in it.
2. Then we import dotenv python package 
    ```from dotenv import load_dotenv```
3. Then we invoke the load_dotenv() to get the data from the .env file
    ```load_dotenv(override=True)```


4. Then we have the below method openAI() that actually makes a connection
    ```openai = OpenAI()```


### Our WebSummary Application
1. We will write a program that will take as input any webpage 
2. Create a summary of the webpage and return

#### Implementation :
1. We write a class the Website class , that represents the website we are scrapping
2. We are using a package called Beautiful Soup that is used to parse webpages. (web scrapping)
3. We can use this package to get any data from the webpage like here we are getting the title.
4. We can also pluck out the styling , ,images , formatting etc. Esentially only retaining text data.

So when we run the code , we get data from this website-
![img.png](img.png)

In this format-
![img_1.png](img_1.png)



### Prompting for OpenAI
The API from OpenAI expects to receive messages in a particular structure.
It got so popular that Many of the other APIs share this structure:

It is a list of Dictionary-
With each dictionary having two fields role and content.
```
[
    {"role": "system", "content": "system message goes here"},
    {"role": "user", "content": "user message goes here"}
]
```

### Side Topic - How to use Ollama in Code
When we have ollama running in our local machine , then it also serves on the port 11343. ```http://localhost:11434/```

### API to Call GPT models
The model chat.completions.create makes the API calls.<br>
OpenAI calls it the completions API because we are asking it to complete the Conversation.
It takes two arguments , first is the model name and the second is the message 
``` 
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)
```


### Output -
We will discuss later on the response structure.
```response.choices[0].message.content```

Also in the method ```display_summary``` we are using the Markdown function of
Jupyter notebook to see the response in a marked down format.

Note- Gpt responds the output in marked down format since we asked that in the prompt-
This was part of the system prompt - <br>
```please provide a short summary of this website in markdown.```


# LLM Leaderboard and Benchmarks for Frontier LLMs

## Goals
![img_10.png](img_10.png)

## Six Essential Leaderboards

#### 1. HuggingFace Open Leaderboard
   - Only for open-source models<br>
   - Periodically updated and old one archived to include additional Benchmarks<br>

#### 2. HuggingFace BigCode Leaderboard
   **https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard**
   ![img_13.png](img_13.png)<br>
    - For open-source models<br>
    - This leaderboard is focused on code generation models.<br>
    - There are base models as well as other models that are fine-tuned for specific tasks.<br>
    - We can see the models and their performance on various programming languages like python , java etc.<br>
    - We select all in the filter then we see models that have been fine tuned for coding
   

#### 3. HuggingFace LLM Perf LeaderBoard
   **https://huggingface.co/spaces/optimum/llm-perf-leaderboard**
   ![img_14.png](img_14.png)

   - HuggingFace Leaderboard for open sourced LLMs 
   - Focus on Performance , cost and compute for LLMs.
   - The models are listed based on their performance on various factors like speed,memory and energy consumption etc.
   - There is also a second tab **Find Your Best Model** which gives us a better picture.
   - In x-axis it shows speed i.e. the time to generate 64 tokens , lower values of this is better models.
   - In y-axis , there is open llm score which measures accuracy.
   - The size of the dots represent the cost . A bigger dot means more expensive.
   - So our ideal model would be one to the left , high and smaller dot in size.
   - Lastly the color of the dot respresent the family of the model. There is a legend of the type in the chart
![img_15.png](img_15.png)
   - There is also a tab for the hardware like A100 or A10 etc , so we can find the model best suited for a particular hardware.

#### 4. HuggingFace Others Leaderboard
   - HuggingFace Leaderboard for open sourced LLMs 
   - Focus on other LLMs designed for specifc use cases like the medical leaderboard for medical use case , leaderboard for other languages.
   - We can go to HuggingFace Spaces and search for Leaderboards and we will see many of these leaderboards.

**Note - As with all other apps, HuggingFace Leaderboards are also built using Gradio and run on HuggingFace spaces.**

#### 5. vellum Leaderboard (https://www.vellum.ai/llm-leaderboard)
   - Vellum is a AI company that does benchmarking for LLMs.
   - It includes both open and closed source models.
   - Has leaderboard based on all the benchmarks we discussed previously
   - There are very exhaustive leaderboards for both open and closed source models.
   - It also has leaderboards for API cost and context window.
   - We can also compare two models
  ![img_18.png](img_18.png)

   - There is also comparision of all the models-
   ![img_19.png](img_19.png)

#### 6. SEAL LeaderBoard
   - SEAL leaderboard asses various expert skills.
   - It is by a company called Scale AI.
   - This company works on producing data sets. If there is a problem we want to solve using LLMs, scale AI can help us craft and curate the dataset.
   - It includes both open and closed source models.


Lets see these Leaderboards in detail.


## Chatbot Arena (https://lmarena.ai/)
![img_11.png](img_11.png)
1. We can compare various models based on their performance on various tasks.
2. No benchmarks are used here , rather human judgement is used after chatting with the models to decide on the model based on the
   task at hand.
3. Qualitative decision by the human by blindtesting the models. 
4. During the testing human won't know which model is which.
5. Really useful when we want to compare models based on their performance on specific tasks like coding, reasoning, etc.
6. Here as well the models are rated based on their performance on various tasks.
https://lmarena.ai/leaderboard
![img_20.png](img_20.png)

&. Below is how we vote. We chat with the models without knowing which model is which and then based on the output choose which we found better.
![img_21.png](img_21.png)
Once we vote , we get to see the model for which we voted.

The leaderboards are then decided based on these votes.


## LLM Commercial Use Cases -
![img_12.png](img_12.png)

1. Harvey.ai for law
    - Harvey.ai is a legal AI platform that helps lawyers automate their work.
    - It uses LLMs to analyze legal documents, generate contracts, and provide legal advice.
    - It can also be used to automate legal research and document review.

2. Nebula.io
   - Nubula uses LLMs for talent management and recruitment.

3. bloop.ai
    - Bloop is an AI-powered code conversion tool which can convert legacy code like cobol into java code etc.
    - It can add comments and test cases

4. Khanmigo
   - Khanmigo is an AI-powered tutor that helps students learn.
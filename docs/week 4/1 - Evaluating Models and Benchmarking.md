# Evaluating LLMs

### Goal
![img.png](img.png)

### How to Compare Models
**LLMs need to be evaluated for suitability for a given tasks.**
![img_1.png](img_1.png)

Lets start with the basics.

### 1. Basic Evaluation
1. Compare the folllowing basic features of the models:
   1. Open Sourced or Closed Sourced
   2. Release Date and knowledge Cutoff
   3. Paramaters 
        - This gives us an idea of the strength of the model and how much training data is needed to finetune it , cost of th model.
   4. Training Tokens 
        - This is the number of tokens that the model was trained on. More tokens generally mean better performance.
   5. Context Length 
        - This is the maximum number of tokens that the model can process in a single input.
        - Like we discussed all the previous conversation history is passed as input to the model. 
        - Also if we are using multi-shot prompting , then the entire prompt is also passed as input to the model , therefore the context length has to be good enough.



2. Costs related to the Model.
    1. Inference Cost
        - API Charges , Subscription or Runtime compute.
        - This is the cost of running the model for inference. 
        - This is generally measured in terms of cost per 1 million tokens.
        - API Cost includes the cost of sending the input tokens to the model and receiving the output tokens.
        - For open source models, this is the cost of running the model on your own hardware. This is the Runtime Compute Cost.
       
    For Open source models, the inference cost is lower but the runtime compute cost is higher.
    Therefore the choice of model depends on the use case and the budget.

    2. Training Cost -
        - This is the cost of training the model. 
        - For frontier models , unless we finetune the model, we do not have to worry about this cost.
        - For open source models that we want to make specialized models, this is the cost of training the model on your own hardware.

   3. Build Cost -
        - This is the cost of building solutions using the model.
      
3. Time to Market -
   - This is the time it takes to build a solution using the model.
   - This is generally lower for closed source models as they have ready to use APIs and SDKs.
   - For open source models, we have to build the solution from scratch, which takes more time.

4. Rate Limits -
   - There are rate limits on the number of requests that can be made to the model.
   - This is true for all major frontier model vendors.
   - Vendors also publish the rate limits for their models. So we need to consider this when building solutions.

5. Speed -
   - This is the speed at which the model can process the input and generate the output.
   - This is generally measured in terms of tokens per second.
   - This is important for real-time applications where we need to process the input and generate the output in real-time.

7. Latency -
   - There is a subtle difference between speed and latency i.e. the request response time.
   - This is the time it takes for the response to start coming for any request that we make.

8. License -
   - This is the license under which the model is released.
   - This is important for commercial applications where we need to comply with the license terms.
   - For open source models, we need to check the license terms and conditions before using the model.
   - For closed source models, we need to check the terms of service and privacy policy before using the model.
   - Even for open source models, there are some restrictions on the use of the model like until some revenue etc.



### 2. Advanced Evaluation
#### 1. Chinchilla Scaling Law
**Introduced by DeepMind, the Chinchilla Scaling Law states that the performance of a model can be improved by increasing 
the number of training tokens and the number of parameters in the model.**
![img_2.png](img_2.png)

So suppose we have a model with 8 billion parameters and we get to a point where we see diminishing returns in performance.
In that case adding more training data will not significanltly improve the performance of the model.

For making the model better, we can either increase the number of parameters . But How much more is enough ?
So it is propotional to the amount of traiining data that we have.
If we want to better the model by training it on double the amount of training data , then we need to double the number of parameters in the model.

**So to pass the same amount of training data that a 8 billion parametered model is already trained on , 
we need to double the number of parameters to 18 billion**

This also gives us a way to evaluate the opposite of this model as well.
**If we are already working on a model with 8 billion parameters and we think of using a 16billion parametered model, 
then we need to train it on roughly double the size of training data that the 8 billion parametered model was trained on.**

**Note - For transformers based models, the Chinchilla Scaling Law holds good.**


#### 2. Benchmarks
#### Genric Benchmarks
Benchmarks are used to evaluate the performance of the model on a specific task.
These are series of tests that are run on the model to measure its performance.
![img_3.png](img_3.png)

#### Specific Benchmarks
Then there are Benchmarks for specific tasks like question answering, summarization, etc.
![img_5.png](img_5.png)
ELO is a score that can be given in games that are zero-sum like chess where there will definitely a winner and definitely a looser.


#### Limitations of Benchmarks
1. Not consitently applied -
    - AI Companies releasing the benchmarks may apply it differently giving them better results for their models.
    - They could also use enhanced hardware to run the benchmarks, which may not be available to everyone.
    - This can lead to biased results and make it difficult to compare models fairly.
   
2. Too Narrow in Scope
    - Benchmarks are not always representative of real-world scenarios.
    - Like multiple choice questions are not always the best way to evaluate a model's performance.

3. Training Data Leakage
    - There is now way we can ensure that the training data used to train the model is not leaked into the benchmarking questions.
    - Models now are also being trainined on data of these benchmarks, which can lead to biased results.

4. Overfitting to Benchmarks
    - Models can be trained to perform well on benchmarks, but may not perform well on real-world tasks.
    - We can keep on re-running the models with some tweaks to get better results on the benchmarks.
    - This can lead to models that are not generalizable and do not perform well in real-world scenarios.

5. Frontier LLMs may be aware they are being evaluated. (Raised but not proven yet)
    - Here is how this is an issue - If we are asking the model question on safety and security, 
        then the model may be aware that it is being evaluated on these topics , then the model may change its approach to answer these questions.


**Note - Here we can see over time the benchmarks score improved and then exceeded human performance , therefore these benchmarks were not enough.**

### Hard, Next level benchmarks
Because there were limitations on the existing benchmarks, new benchmarks are being created to evaluate the models on more real-world scenarios.
![img_7.png](img_7.png)

1. GPQA (Google Proof Q&A).
  - This means resistance to google. 
  - These are questions that are designed to be difficult for models to answer, even with access to the internet (therefore google proof).
  - Normally these questions are PHD level questions that require deep understanding of the topic.
  - **Note - As of 10 July 2025 , Google Gemini 2.5 Pro is the highest scorer in GPQA**

2. BBHard (Big-Bench-Hard) 
   - is a benchmark that is designed to evaluate the models on a wide range of tasks.
   - It is a collection of hard tasks that are designed to be difficult for models to solve.
   - It includes tasks like reasoning, common sense, and knowledge.
   - It is designed to be more challenging than existing benchmarks and to evaluate the models on a wider range of tasks.

3. Math lv 5
    - This is a benchmark that is designed to evaluate the models on their mathematical reasoning abilities.
    - This is lvel 5 of High school competitive mathematics. 
    - It includes tasks like solving complex mathematical problems, reasoning about numbers, and understanding mathematical concepts.
    - It is designed to be more challenging than existing benchmarks and to evaluate the models on their mathematical reasoning abilities.

4. IFEval-
   - This is a benchmark that is designed to evaluate the models on their ability to understand and generate text.
   
5. MuSR (MultiStep Soft Reasoning)
   - This is a benchmark that is designed to evaluate the models on their ability to reason about multiple steps.
   - Evaluates Logical Reasoning, Mathematical Reasoning, and Common Sense Reasoning.

6. MMLU_PRO
   - MCQ but with 10 choices instead of 4 to take out luck out of equation.
   - Harder Questions than MMLU.


### HuggingFace LLM Leaderboard (https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- This is a leaderboard that ranks the open source models only based on their performance on various benchmarks.
- HuggingFace Leaderboard is part of HuggingFace space , so these are apps running on HuggingFace spaces.
- These are mostly gradio apps behind the scenes.
- Like we already discussed , initially HuggingFace was using the Basic 7 benchmarks to rank the models.
- But since with time the LLMs started to exceed human performance on these benchmarks, it was closed and new benchmarks were introduced.

- We can set some filters to see the models that we want to see.
![img_8.png](img_8.png)
**Note - There are also models that are finetuned on these benchmarks, so we can skip those.**

- We can also filter based on the parameters of the model, the training tokens, and the context length. 
  Therefore it is helpful to compare the models based on these parameters before running it on our boxes.

- We can also filter based on the precision of the models like float16 etc based on the quantization that we want to use.

- Then we see all the Benchmarks for the models.
![img_9.png](img_9.png)

- As we can see there are multiple benchmarks that are used to evaluate the models. Like GPQA, BBHard, Math lv 5, IFEval, MuSR, MMLU_PRO, etc.
- We can choose the model based on these benchmarks for the task that we want to perform. Like if want to do Math we can choose model\
   based on the Math lv 5 benchmark.


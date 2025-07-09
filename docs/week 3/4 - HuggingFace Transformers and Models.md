# Models in HuggingFace

## Goal for this lesson
![img_23.png](img_23.png)

## Models we will use -
![img_24.png](img_24.png)

## Aspects of the Models to focus on-
We will focus on the following aspects of the models -
1. **Quantization** - This is the technique of reducing the precision of the model weights to reduce the size of the model.
                    This way it is easier to fit the models into memory and load therefore making it run faster.

2. **Model Internals** - We will look at the model internals to understand how the model works and how it is structured.
   - We will look at the pytorch layers that sit inside the huggingface transformers library.

3. **Streaming** - Streaming Results.



## Steps -
https://colab.research.google.com/drive/1hhR9Z-yiqjUe7pJjVQw4c74z_V3VchLy?usp=sharing

First lets complete all the boilerplate code to get the models running in Google Colab.

1. Install the required libraries -
```bash
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0
```

2. Import the required libraries -
```python
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import gc
```

3. Login to HuggingFace -
```python
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```

4. Then we define our models that we want to use.
```python
# instruct models

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # If this doesn't fit it your GPU memory, try others from the hub
```

5. Lastly we define our message-
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]
```




## 1. Quantization-
When we do quantization, we are reducing the precision of the model weights to reduce the size of the model.

**The weights are generally in float32 format, which takes up a lot of memory. 
Therefore reducing reducing the precision to float16(16 bits) or int8(8 bits) can help reduce the size of the model.**

This process is called as **quantization**.

Now point to remmeber is even though we are reducing the precision of the model weights, the performance of the model does not degrade significantly.
It does becomes a little less accurate, but it is still able to perform well on most tasks. It is totally worth it to reduce the size of the model.


**Note - In practice we can even reduce the precision to 4 bits**
 
This process is particularly useful when we are trying to run the model on a device with limited memory, such as a for fine tunning.
There is a fine tuning approach called **QLoRA** which is a combination of quantization and LoRA.

Lets see the code -
```
# Quantization Config - this allows us to load the model into memory and use less memory

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```
We are using the `BitsAndBytesConfig` class from the `transformers` library to configure the quantization.
The paremters are as follows -
1. We are saying load the model in 4 bit precision (`load_in_4bit=True`). We could also do `load_in_8bit=True` to load in 8 bit precision.
2. `bnb_4bit_use_double_quant=True` - With this the quantization happens twice therefore saving more memory.

3. `bnb_4bit_compute_dtype=torch.bfloat16` - This is the data type that will be used for computation. We are using bfloat16 which is a 16 bit floating point format.
    This will improve the performance of the model while still saving memory.

4. `bnb_4bit_quant_type="nf4"` - 
     This is the quantization type that will be used. We are using nf4 which is a new quantization type that is more efficient than the previous types.
     SO when we have reduced the precision of the model weights to 4 bits , how to compress those numbers in 4 bit.
     nf4 is a 4 bit representation of a number.


### 2. Tokenization
Our next step is to tokenize the input messages.
```python
# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
```
The above code is mostly familiar from previous lesson.
The middle statement is new.

pad_token is a token that is used to pad the input token if more tokens are needed to fill in the prompt when fed into the model.
pad_token is used to make sure that the input token length is same as the output token length.

Here we do pad_token=tokenizer.eos_token because it is a good practice to set the pad_token to be same as the special_token.

Then we use the `apply_chat_template` method to convert the messages into input tokens.


### 3. Load the Model
```python
# The model

model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
```
This is creating the model object using the `AutoModelForCausalLM` class from the `transformers` library.
Just like we used AutoTokenizer to create the tokenizer, we are using AutoModelForCausalLM to create the model.
The `from_pretrained` method is used to load the model from the HuggingFace hub.

**A causal LLM is an autoregressive model that generates text by predicting the next token in a sequence based on the previous tokens.**
Most LLMs we have talked about so far are causal LLMs.

The parameters passed to the `from_pretrained` method are as follows:
1. We provide the model name or path to the model we want to load, in this case `LLAMA`.
2. The `device_map="auto"` argument is used to automatically map the model to the available GPU devices.
3. quantization_config=quant_config is used to pass the quantization configuration we defined earlier.

**This is how we build the model. This model is the real code that will run as python code on our google colab and will generate the output.**

**Under the hood, this is a pytorch model that is built using the huggingface transformers library.
There are layers of pytorch , neural networks**

**What actually happens when we run the above code -**
1. We connect to the HuggingFace hub and download the model files and weights and puts it in local disk of the google colab box in a temp file.
2. So this model is temporairily stored in the local disk of the google colab box.
3. The model is then loaded into memory and the quantization is applied to the model weights.

Output from Above two stpes- 
![img_25.png](img_25.png)

We can check for the memory utilized in storing and loading the model using below code-
```python
memory = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory:,.1f} MB")
```
![img_26.png](img_26.png)

### Lets Print the model at this point
```python
model
```

Output -
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```
This model object is a Neural Network, implemented with the Python framework PyTorch. 
The Neural Network uses the architecture invented by Google scientists in 2017: the Transformer architecture.

This is the layers of code representing the deep neural network that is used to generate the text.
1. First there is an Embedding layer that converts the input tokens into embeddings.
    - this takes tokens and turns them into 4,096 dimensional vectors.

2. There are then 32 sets of groups of layers called "Decoder layers". Each Decoder layer contains three types of layer: 
    - (a) self-attention layers 
    - (b) multi-layer perceptron (MLP) layers 
    - (c) batch norm layers.

    - Then we have multitron perceptron layers that process the output of the attention layers.
        1. There is an activation function that is applied to the output of the attention layers. 
        The activation function used here is SiLU (Sigmoid Linear Unit) which is a smooth and non-monotonic activation function.

3. There is an LM Head layer at the end; this produces the output.


**To go deeper into Transformers Architecture -**
- https://chatgpt.com/canvas/shared/680cbea6de688191a20f350a2293c76b
- https://www.youtube.com/playlist?list=PLWHe-9GP9SMMdl6SLaovUQF2abiLGbMjs
- https://github.com/huggingface/transformers


**Note - The number ```Embedding(128256, 4096)``` 128256 here represents the vocabulary size of the model.
Then we have the output ```(lm_head): Linear(in_features=4096, out_features=128256, bias=False)``` in out_features**



### 4. Running the Model
generate function is used to generate the output from the model that is running on our box.
```python
# OK, with that, now let's run the model!

outputs = model.generate(inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0]))
```
This is the code that actually runs the model and generates the output.
It takes the input . We also pass the max_new_tokens parameter which is the maximum number of tokens to generate.

Then the output is generated and printed using the `tokenizer.decode` method.
The `tokenizer.decode` method converts the output tokens back into text.

Output -
![img_27.png](img_27.png)


### At this point we have successfully run the model and generated the output. Lets do a cleanup since we are using free version of Google Colab. Otherwise we will run out of GPU
```python
# Clean up memory
# Thank you Kuan L. for helping me get this to properly free up memory!
# If you select "Show Resources" on the top right to see GPU memory, it might not drop down right away
# But it does seem that the memory is available for use by new models in the later code.

del model, inputs, tokenizer, outputs
gc.collect()
torch.cuda.empty_cache()
```


### 5. Packaging the Code into a Function
Now we will package the code into a function that can be used to generate text from any model.
The method takes two arguments , the model name and the messages and generates the output.

```python
# Wrapping everything in a function - and adding Streaming and generation prompts

def generate(model, messages):
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
  streamer = TextStreamer(tokenizer)
  model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
  outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
  del model, inputs, tokenizer, outputs, streamer
  gc.collect()
  torch.cuda.empty_cache()
```

This function does the same thing as the previous code but in a more structured way.
1. First line defines a tokenizer for the model by taking the model as input.
2. This line sets the pad token to be the same as the end of sentence token. (boilerplate code)
3. In line 3 we apply the chat template to the messages and convert them into input tokens suitable to the model we want to run.
   - .to("cuda") is used to move the input tokens to the GPU.

4. Line 4 is new . It is used to stream outputs. It is part of the HuggingFace 
   - To stream results, we simply replace:
        ```outputs = model.generate(inputs, max_new_tokens=80)
      With:
        ```streamer = TextStreamer(tokenizer)
           outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)```
     

5. Line 5 defines the model using the `AutoModelForCausalLM` class from the `transformers` library.
   - The model is loaded with the quantization config we defined earlier.
   - The model is also moved to the GPU using `device_map="auto"`.

6. Line 6 generates the output using the `generate` method of the model.
    - The `max_new_tokens` parameter is used to limit the number of tokens generated.
    - The `streamer` parameter is used to stream the output tokens as they are generated.

7. The last line is used to clean up the memory by deleting the model, inputs, tokenizer, outputs, and streamer objects.


### 6. Running the Function
![img_28.png](img_28.png)


### We can change the model and messages to generate different outputs.
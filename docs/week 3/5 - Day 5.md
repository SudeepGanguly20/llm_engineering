# Combining Frontier Model & Open Source Model for Audio-to-Text Summarization

## Goal
![img_29.png](img_29.png)

## Problem Statement
![img_30.png](img_30.png)

**Requirements:**
![img_31.png](img_31.png)

## Input -
For input we will use publicly available audio files on huggingFace.
https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing

**The file is already downloaded and kept in the above google drive and we will use it in our code.**

### Lets see the code now. First we have some boiler plate code to install the required packages.
https://colab.research.google.com/drive/1KSMxOCprsl1QRpt_Rq0UqCAyMtPqDQYx?usp=sharing#scrollTo=f2vvgnFpHpID

### Step 1 : Install the required packages
```bash
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai
```
We are installing the required huggingface packages and the openai package to use the GPT-4o model.


### Step 2 : Import the required packages
```python
import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
```


### Step 3 : Define the models to use
```python
# Constants

AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```
### As audio model we will whisper from Open AI and for summarization we will use llama from huggingFace.


### Step 4. Connect with Google Drive to get access to the audio file
```python
# New capability - connect this Colab to my Google Drive
# See immediately below this for instructions to obtain denver_extract.mp3

drive.mount("/content/drive")
audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"
```
Output -Mounted at /content/drive


### Step 5. Sign in into HuggingFace and OpenAI
```python 
# Sign in to HuggingFace Hub
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Sign in to OpenAI using Secrets in Colab
# Note here we have explictily pass the openai_api_key since it is not set in the environment variables.
openai_api_key = userdata.get('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)
```

### Step 6. Use OpenAI Whisper to transcribe the audio
```python
# Use the Whisper OpenAI model to convert the Audio to Text
# If you'd prefer to use an Open Source model, class student Youssef has contributed an open source version
# which I've added to the bottom of this colab

audio_file = open(audio_filename, "rb")
transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
print(transcription)
```
We pass the name of the model that is is Whisper-1 and the audio file that we want to transcribe.
Also we define the response format as text.


### Step 7. Use Llama to summarize the text
First we define the prompt using the template we already know.
```python
system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

```


The we need to do the steps we did in the previous sections to get the model and tokenizer.
1. First Quantize the model using BitsAndBytesConfig.
2. Load the model and tokenizer using the AutoModelForCausalLM and AutoTokenizer classes.
3. TOkenize the input messages using the tokenizer.
4. Pass the Input to the model and get the output.
5. Decode the output using the tokenizer.

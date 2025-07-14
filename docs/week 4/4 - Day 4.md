# Code Generation using Open Source Models using HuggingFace

### Goals
![img_26.png](img_26.png)

#### Reminder of the Problem Statement - Build a code generation tool that converts python code to CPP for Performance enhancement
![img_27.png](img_27.png)

### Step 1 : Evaluate the model that we want to use
For open source code generation models, we can use the HuggingFace BigCode Leaderboard.(https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

**Reminder - There are base models as well as other models that are fine-tuned for coding in this leaderboard and even more custmization like
base models fine tuned for cpp code.**

Some models in this leaderboard will show up with the flag EXT. This means the benchmark of such models is from external sources and not from the HuggingFace team.

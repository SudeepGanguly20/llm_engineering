# A Code conversion Tool using LLM

## Goal
![img_22.png](img_22.png)

### Challenge - Build a tool that converts python code to CPP for Performance using frontier Models

### Steps

#### Step 1 : Have a Prompt ready to evaluate the models.

Lets start with out prompt that we can use to evaluate the models.

```python
Prompt = '''Please reimplement the following python code in C++ with the fastest possible implementation for a M4 mac
Only respond with the C++ code and nothing else. Do not include any comments or explanations.
The only requirement is the c++ code prints the same result as the python code and runs fast.'''

pi = '''
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
'''
```
**Note - This is a known code that executes slowly.**


#### Step 2 : Use the prompt to evaluate the models.
Since we need coding models and we want to use frontier models , lets see the leaderboard for frontier models.
We already know we can use vellum or seal to see the leaderboard.
Also we have the lmarena to try out the prompt among different models.
![img_23.png](img_23.png)


Now that we have our LLM that we want to use , lets write the code to implement this use case.
Note- We will use gpt-4o-mini for this use case to save cost.


#### Step 3 : Implementation
The implementation is very simple. We just need to pass the prompt and the code to the model and get the response.
Lets see our actual Prompts -
```python
system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time."
```
Then our user Prompt-
```python
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python
```

### The prompt we generated needs tweaking around. This is because there is no syntax to write a prompt. It depends on the task and trying out different combination of words to get a optimal result.
Therefore the last line in user_prompt. That part was derived by trial and error with multiple models.


**Note - Complete Code in jupyter notebook week 4 day3.ipynb**


Then below we have an utility that will strip out anything that is not needed from the response of the model and save the code into a file.
**Note- The output included ```cpp something like this in the top. We do not neet it.

```python
# write to a file called optimized.cpp

def write_output(cpp):
    code = cpp.replace("```cpp","").replace("```","")
    with open("optimized.cpp", "w") as f:
        f.write(code)
```

Then our code to call the openai model and get the response.
```python
def optimize_gpt(python):    
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)
    write_output(reply)
```
Here we have set stram=True , therefore the results will stream.
This means that the results will come back in chinks .
Therefore with the for loop , we are printing each little chunk and write the chunks to a file.

**Same functionality with claude will be -**
```python
def optimize_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            print(text, end="", flush=True)
    write_output(reply)
```

Then we have the below code-
```python
exec(pi)
```
**The exec function runs a python code that has been stored in a string. This code is just for testing the python code.


**Lastly the response -**
```optimize_gpt(pi)

Skip to main panel
/week4/
def optimize_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            print(text, end="", flush=True)
    write_output(reply)
pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""
exec(pi)
optimize_gpt(pi)

```cpp
#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int iterations, int param1, int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    double result = calculate(100000000, 4, 1) * 4;
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Result: " << std::setprecision(12) << std::fixed << result << std::endl;
    std::cout << "Execution Time: " << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    return 0;
}
```


Also a file is created with the cpp code and saved in the same directly.

#### Now we can run both the python code and the generated cpp code and check the time it took to execute and we will see that the c++ code runs much faster than the python code.


### Note - there is hard python code as well in the bottom of the notebook that when converted s not converted properly by gpt but it works very well for claude.
The python code was for finding maximum subarray sum using brute force algorithm.
Claude understood the intent of the code and instead of applying brute force, it applied kadane's algorithm to solve the problem.
Therefore the performance of the piece of code improved by thousands of times.
But GPT 4o was trying to still convert it to brute force and therefore the performance was not as good as claude.


### Lastly, we integrated gradio with this solution.
```python
with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:", lines=10, value=python_hard)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
        convert = gr.Button("Convert code")

    convert.click(optimize, inputs=[python, model], outputs=[cpp])

ui.launch(inbrowser=True)
```

Output-
![img_24.png](img_24.png)


#### Next Even better ui is built where the code can be run in the UI itself. 
![img_25.png](img_25.png)



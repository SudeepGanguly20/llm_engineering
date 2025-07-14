#### Goal 
![img.png](img.png)

#### Fine Tuning-
Up until now we have been working with inference. THis is where the model is already trained and we are using it to get predictions. 
We tried various techniques to improve the inference capabiity like prompt engineering, prompt chaining , tools, multi shot prompting, RAG etc.
All of these techniques are useful when we want to use a pre-trained model for our use case.

Fine tuning is where we take a pre-trained model and train it further on our own data. 
This is useful when we have a specific use case that the pre-trained model does not cover.

Frontier models are large language models that are trained on a large amount of data and have large amounts of weights.
**With Fine tuning we will be able to figure out how we can tweak the parameters or weights , optimize then slightly based on our oqn data
so that the models get better and better at predicting the next token in our use case.**

![img_1.png](img_1.png)

Now training frontier models is not easy and not cheap .Therefore we will do that fine tuning using something called Transfer Learning.

**Transfer Learning is a technique where we take a pre-trained model and fine-tune it on our own data or continue the training on a very
particular dataset. The model will transfer all the knowledge it already has and we will be able to add more knowledge**

Now lets see the problem statement for the project we will be working on this week.
We work for an E commerce company , we want to build a model that can take the description of any product and estimate 
how much it cost based upon the description.
![img_2.png](img_2.png)


Lets see the Steps
### Step 1 : Finding and Crafting our Dataset
Data sources for fine tuning models can be from -
1. Own Propietary Data
2. Kaggle Datasets - Wonderful source of datasets for various use cases.
3. HuggingFace Datasets - HuggingFace has a lot of datasets for various use cases.
4. Synthetic Data - 
        We can generate our own data using other frontier LLMs but to generate data and then using this data to fine tune frontier model
        does not make sense. It is useful when we want to build our own small model or a cheaper model

6. Specialist Companies - There are specialist companies that can help us craft and curate the dataset.
   - Scale AI is one such company that helps us craft and curate the dataset.
   - They have a lot of datasets for various use cases.
   - They also have a lot of tools to help us with data annotation, data cleaning, etc.


**For our use case we will be using HuggingFace Datasets.**
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
This data is of amazon reviews of products from 2023.
It also has metadata about the products like the product title, product description, prices etc.


### Step 2 : Preprocessing the Dataset
There are several sub steps before using a dataset for fine tuning a model.

**1. Investigate the dataset**
   - We need to understand the dataset and what it contains.
   - We need to understand the data types, the number of rows, the number of columns, etc.
   - We need to understand the distribution of the data, the missing values, etc.

**2. Parsing the dataset**
   - We need to parse the dataset and convert it into a format that would be easier to handle.
   - We need to convert the dataset into a format that can be used by the model.
   - We need to convert the dataset into a format that can be used by the tokenizer.
   - We do not want to work with raw datasets rather with objects of data.

**3. Visualization of the dataset**
   - We need to visualize the dataset and understand the distribution of the data.
   - We need to understand the distribution of the data, the missing values, etc.
   - We can use various visualization techniques like histograms, box plots, etc. to visualize the data.

**4. Assess Data Quality**
   - We need to assess the quality of the data and understand the missing values, the outliers, etc.
   - Understand the limitation with our data so that we can take the right action.

**5. Curate the dataset (Crafting the dataset that is most suitable for our use case)**
   - We need to curate the dataset and remove the unnecessary data.
   - We need to remove the unnecessary columns, the unnecessary rows, etc.
   - We need to remove the duplicates, the missing values, etc.
   - We need to remove the outliers, etc.
   - If we find our dataset is imbalanced in some way then our model could potentially learn a particular balance of the data. 
     Therefore this is the time to rectify the imbalance .

**6. Save**
   - Uploading the dataset to HuggingFace hub so that it can be ready for the training.



### Our Goal for this lesson is to Curate the Data.
First of all lets investigate the dataset and understand what it contains.
![img_3.png](img_3.png)
This data set contains user reviews of products from Amazon along with some metadata like the price , product information etc.
across product categories like electronics, clothing, etc.

if we go into the HuggingFace folder that contains the data for the meta data that is data of the product description , pricing etc
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories

![img_4.png](img_4.png)
here we can see the data is divided into various categories like electronics, clothing, etc and we have the size of the dataset.

**The code to load the data and investigate it is in the week 6 , day 1.pynb**

Note - this data is also available in the page of the dataset in HuggingFace.
![img_5.png](img_5.png)


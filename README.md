# Text-Summarization
Text Summarization using a LSTM Encoder-Decoder model with Attention

![image](https://user-images.githubusercontent.com/64171895/150996495-5f271586-58cd-4296-a28a-69a6ca8664fb.png )

## Understanding the Problem Statement
Customer reviews can often be long and descriptive. Analysing these reviews manually, as you can imagine, is really time-consuming. This is where the brilliance of Natural Language Processing can be applied to generate a summary for long reviews.
Our objective here is to generate a summary for the phone reviews using the abstraction-based approach we learned about above.
## Architecture of the encoder-decoder model with attention 

<img src="https://user-images.githubusercontent.com/64171895/150996741-d231205d-80d1-45cf-979d-cb545eb20662.png" alt="image" width="65%"/>
### Implementation
### Step 1: Importing the Dataset
<!-- ![image](https://user-images.githubusercontent.com/64171895/150996890-351206fa-7cff-450d-ba55-a2042f425e8e.png) -->
<img src="https://user-images.githubusercontent.com/64171895/150996890-351206fa-7cff-450d-ba55-a2042f425e8e.png" alt="image" width="65%"/>

### Step 2: Cleaning the Data
Performing basic pre-processing steps is very important before we get to the model building part. Using messy and uncleaned text data is a potentially disastrous move. So in this step, we will drop all the unwanted symbols, characters, etc. from the text that do not affect the objective of our problem.
We will perform the below pre-processing tasks for our data:<br />
•	Convert everything to lowercase.<br />
•	Remove HTML tags.<br />
•	Contraction mapping.<br />
•	Remove (‘s).<br />
•	Remove any text inside the parenthesis ( ).<br />
•	Eliminate punctuations and special characters<br />
•	Remove stop words.<br />
•	Remove short words<br />

<img src="https://user-images.githubusercontent.com/64171895/150997096-5e6020f5-45ab-4980-b207-a6304675471e.png" alt="image" width="65%"/>

<!-- ![image](https://user-images.githubusercontent.com/64171895/150997096-5e6020f5-45ab-4980-b207-a6304675471e.png)
 -->
### Step 3: Determining the Maximum Permissible Sequence Lengths
Here, we will analyze the length of the reviews and the summary to get an overall idea about the distribution of length of the text. This will help us fix the maximum length of the sequence:
<!-- ![image](https://user-images.githubusercontent.com/64171895/150997182-ffe1110c-0108-456b-9e47-caecf37c4340.png) -->
<img src="https://user-images.githubusercontent.com/64171895/150997182-ffe1110c-0108-456b-9e47-caecf37c4340.png" alt="image" width="65%"/>

### Step 4: Tokenizing the Text
<img src="https://user-images.githubusercontent.com/64171895/150997273-f446ed12-2d9e-4229-a9da-a8952e38e39d.png" alt="image" width="65%"/>

<!-- ![image](https://user-images.githubusercontent.com/64171895/150997273-f446ed12-2d9e-4229-a9da-a8952e38e39d.png) -->

### Step 5: Model building:
We need to familiarize ourselves with a few terms which are required prior to building the model.
Return Sequences = True: When the return sequences parameter is set to True, LSTM produces the hidden state and cell state for every timestep
Return State = True: When return state = True, LSTM produces the hidden state and cell state of the last timestep only
Initial State: This is used to initialize the internal states of the LSTM for the first timestep
Stacked LSTM: Stacked LSTM has multiple layers of LSTM stacked on top of each other. This leads to a better representation of the sequence. 
We are building a 3 stacked LSTM for the encoder
<img src="https://user-images.githubusercontent.com/64171895/150997381-d482f406-01d7-4227-a4a1-141e94f72f99.png" alt="image" width="65%"/>
<img src="https://user-images.githubusercontent.com/64171895/150997402-61b12a10-73a1-48d9-a466-a2560a51d392.png" alt="image" width="65%"/>

<!-- ![image](https://user-images.githubusercontent.com/64171895/150997381-d482f406-01d7-4227-a4a1-141e94f72f99.png)
![image](https://user-images.githubusercontent.com/64171895/150997402-61b12a10-73a1-48d9-a466-a2560a51d392.png)
 -->
 

### Step 6: Generating Predictions
Now, we will set up the inference for the encoder and decoder. Here the Encoder and the Decoder will work together, to produce a summary. The Decoder will be stacked above the Encoder, and the output of the decoder will be again fed into the decoder to produce the next word.
<img src="https://user-images.githubusercontent.com/64171895/150997426-7eec2fd6-8bd6-40f6-b3dc-11934827be19.png" alt="image" width="65%"/>

<!-- ![image](https://user-images.githubusercontent.com/64171895/150997426-7eec2fd6-8bd6-40f6-b3dc-11934827be19.png) -->

 



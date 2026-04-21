Speech Emotion Change Prediction
===
This code implements the model discussed in the paper Speech emotion change prediction using Complex Recurrent Neurons. It converts speech conversations to complex fourier signals for training. Next, during testing the phase of the predicted signal is used to detect emotional transitions such as from 'Anger' to 'Frustration'.  

Requirements
---
This code is based on : 

Complex Valued Nonlinear Adaptive Filtering toolbox for MATLAB, Supplementary to the book:

"Complex Valued Nonlinear Adaptive Filters: Noncircularity, Widely Linear and Neural Models" by Danilo P. Mandic and Vanessa Su Lee Goh

English Conversations
---

<img width="1986" height="1212" alt="Image" src="https://github.com/user-attachments/assets/1a4270df-a604-4219-b27b-e2db9814ebdd" />

- We consider the audio signal in English conversations 
- Emotion changes from Happy to Excited

Complex recurrent neural network
---

- We train the complex recurrent network using audio samples from both 'Angry' and 'Frustration' emotions
- During testing we consider two consecutive uttrances in a conversation from both emotion classes

The code for training and testing is as follows :

rtrl_speech('audiodata','audiodata_labels.txt','ang','fru')
- audiodata is the directory with audio samples
- audiodata_label is the file with emotion labels
- we specify two emotions for which we want to predict the change in emotions

Classification
---





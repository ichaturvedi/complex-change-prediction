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
- Downsampling followed by Fourier transform is used to convert each audio to a complex vector

Complex recurrent neural network
---

- We train the complex recurrent network using pairs of consecutive audios from the same speaker
- We focus on a transition window of 8 seconds ( 4 from each audio )
- Most frequent emotion pairs and a balanced same emotion pair was used 

The code for training and testing is as follows :

carv_speech('metadata.csv','AllSentences')
- AllSentences is the directory with audio samples for each utterance
- metadata.csv is the file with emotion labels and change information

Ablation
---

- The code predicts F-measure for Change and No Change classes
- We provide results from only real and only imaginary features for comparison






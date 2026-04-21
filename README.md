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

- We train the complex recurrent network using audio samples from both 'Angry' and 'Frustration' emotions
- During testing we consider two consecutive uttrances in a conversation from both emotion classes
- Testing uses the trained model to predict the complex signal two consecutive utterances 

The code for training and testing is as follows :

rtrl_speech('audiodata','audiodata_labels.txt','ang','fru')
- audiodata is the directory with audio samples
- audiodata_label is the file with emotion labels
- we specify two emotions for which we want to predict the change in emotions

Classification
---

- Class 1 is when there is a change in emotions from 'ang' to 'fru'
- Class 2 is when there is no change in emotions using the label file
- We consider the maximum phase angle for each predicted sequence for classification

The code for classification is as follows : 

fmeaavg = computeAcc(changeFile, angleFile)
- changeFile has the target label as change or no change for two consecutive utterances
- angleFile has the maximum phase angle for each predicted sequence during testing
- output is the average F-measure over both classes



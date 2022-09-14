# Neural Error Correction for MLA In-Text Citations

Although using in-text citations is an integral part of the academic writing process, it is oftentimes a head-scratcher for many students. 
Currently, popular AI writing assistants do not make recommendations on specific formats like MLA. 
This AI-powered prototype aims to help fill the gap by detecting and correcting common mistakes with MLA in-text citations - such as the ones illustrated below.

![image](https://user-images.githubusercontent.com/84154105/190180864-690476d1-4efc-4833-98dc-6e435d6cd593.png)

## Under the Hood
Training data was created using pattern-based error generation (3 million observations). 
Machine translation from erroneous to correct observations was built using Sequence-to-Sequence (Seq2Seq) modeling with Keras. 
A pre-trained BERT NER model is leveraged during text pre-processing steps to identify author names.

![image](https://user-images.githubusercontent.com/84154105/190181079-ffd01095-9831-42c2-b96e-42405d10f90c.png)

## Demo
Click <a href="https://misha345a-seq2seq-error-correction-app-5ew83v.streamlitapp.com/" target="_blank">here</a> to try out the web app demo.

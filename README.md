# Bachelorthesis

- [x] Datensatz finden
- [x] add read papers to xzotero and try to create a label for Thesis
- [x] Check if i should read GLARA: Graph-based Labeling Rule Augmentation for Weakly
Supervised Named Entity Recognition
- [x] check if i can find the dataset with labels of the alibaba dataset 
- [x] search the links of the amazon and bio dataset and check if they are labelled (make a link in gchrome) 
- [x] read and response matthias and tatiana 
- [x] find 3 datasets and write down which one I choose and why
- [x] load them into 3 Notebooks with the name of the dataset
- [x] get advice from chatgpt on what the best way for dataset visualisation is
- [x] combine the whole dataset
- [x] create a wordcloud plot
- [x] find a model on which I want to fine tune on --> distilBert
- [x] find model on which I want to fine tune on and create in notebook 
- [ ] 
- [ ] clean data from JNLPBA (removing I and B)
- [ ] split data (ready for training)
- [ ] make training (finetuning)
- [ ] creating a strategy for making the JNLPBA weak
- [ ] take JNLPBA and make it weak


- [ ] fine tune again this time with weak JNLPBA
- [ ] strategy for seq2seq 
- [ ] seq2seq 
- [ ] compare all results
- [ ] find a model which I could fine tune with ACL19
- [ ] make ACL19 ready for fine tuning
- [ ] fine tuning of found model with ACL19
- [ ] NER with the result of the model fine tuned on ACL19
- [ ] strategy for seq2seq 
- [ ] seq2seq 
- [ ] compare all results 

## Notes

I need to search the scores for which i wanna meassure and than the NER scores. 


Meeting with Matthias:

- Dicussion about the sense of the alibaba dataset 
Because I want to do normal ner, weakly, 

- The advantage of demonstrating based on a benchmark dataset is that I can make the dataset weak and I can compare the performance really good -> i can see that it works really good. Than I can use this approaches on my weakly labelled. the benchmark dataset is like a test.

I have 3 approaches:

- NER
- Weakly
- T5

The thing is that in my opinion the alibaba dataset is weakly labelled already because 

1. The alibaba paper handles the dealing with weakly labelled data and they chooses this dataset for that

My alternative approach would be to take a benchmark dataset like jnlpba and perform normal ner on it. than make it weak and do again ner on it and than use t5 for seq2seq modelling and compare differences.

After this step I go back to the alibaba paper and do the same steps there but without comparison of the ner with the weakly ner. 

Reasons for the alternative approach are 

1. i can compare the performance between seq2seq with weak to ner with strong data - what i cannot do with the alibaba paper

The one thing which confused me all the time was that we wrote down this three steps but they are not really possible because the alibaba paper is 



According the style of the thesis. I write the thesis in notebook and convert this notebook to md and md to pdf. 

- Having one dataset ready for training/started fine tuning 
- having 5 pages written in my thesis and having the initial setup for my writing system (notebook, einleitung)
- Reading

Just realized that it could also be interesting to read about NER for itself. This gives me a little bit the view for what ner could be useful for. The goal for why doing this is to get an better overview of whats possible.

# Literature

## NLP 
- [ ] https://link.springer.com/content/pdf/10.1007/s11042-022-13428-4.pdf?pdf=button%20sticky

## NER 
- [ ] https://dl.acm.org/doi/abs/10.1145/3445965?casa_token=kqaz9U7KxyUAAAAA:cqsDkCCs99QtGCDQkzldoABzNVE3KII7Kq2euLJUsLsODP1EzRjY-AkriP-4Xd-ecYnQa-JttVyx

## NER with weakly labelled data
- [ ] https://aclanthology.org/2021.ecnlp-1.2.pdf
- [ ] https://arxiv.org/pdf/2209.15108.pdf

## Transformer
- [ ] https://arxiv.org/abs/1706.03762

## Auto-Labeling
- [ ] https://arxiv.org/pdf/2109.03784.pdf

## Seq2Seq

## LSTM/RNN

## Fine tuning

## JNLPBA
- [ ] https://arxiv.org/abs/2104.10344

Basically what I need to do is reading about nlp, ner, ner with weakly labelled data, transformer. 

# 1. Abstract
# 2. Einführung
    2.1 Motivation (bspw. Anwendungen von NER),
    2.2 Problem
    2.3 Ziele (bspw. Überblick über bestehende Ansätze, Analyse und Vergleich der drei gängigsten Ansätze),
    2.4 Struktur der Arbeit
# 3. Grundlagen
    3.1 Neuronale Netzwerke
    3.2 Deep Learning 
    3.3 NLP (Erklärung des gesamten Prozess von Wort zu Vektor und die verschiedenen Bereiche, wo diese Vektoren eingesetzt werden können (Textgenerierung, Textzusammenfassungen...))
    3.4 NER 
    3.5 Labelling 
    3.6 Seq2Seq
    3.7 LSTM/RNN 
    3.8 Transformer
    3.9 Fine tuning 
# 4. Tools
    4.2 PyTorch
    4.1 Huggingface 
    ...
    ...
# 5. Methode 
    5.1 Training mit schwach gelabelten Datensatz 
    5.2 Training mit selbst gelabelten Datensatz 
    5.3 Generatives Modell
# 6. Design der Experimente (inkl. Modell/e, Daten)
    6.1 Training mit schwach gelabelten Datensatz 
    6.2 Training mit selbst gelabelten Datensatz 
    6.3 Generatives Modell
# 7. Ergebnisse der Experimente
    7.1 Training mit schwach gelabelten Datensatz 
    7.2 Training mit selbst gelabelten Datensatz 
    7.3 Generatives Modell
# 8. Diskussion (inkl. wichtiger Ergebnisse, Einschränkungen und zukünftiger Arbeiten, Implikationen für Forschung und Praxis)
# 9. Zusammenfassung

- [ ] finding a model which I want to fine tune
- [ ] find out which data format my dataset needs to have for this model
- [ ]  

BERT, RoBERTa, ALBERT, and XLNet are transformer which could be used for fine tuning.

What can I do now? Should i just start with fine tuning a model?

Le


parallele computing would be really interesting. something in which i would be really interested in. chunking the file into thousands of chunks and than iterate each with an gpu instance. a parallel computing company would. would be really interesting to try this out in my thesis. 


I  want to check the distribution of tags in the dataset. i have 20k rows 

one row is like 0 1 01 0 0... - want to i

Now I should visualize the dataset a little bit to get to know how it really looks like. i sh d

Decisions for datasets (made because datasets appear on leaderboard):

CoNLL 2003
NCBI-disease
JNLPBA



Questions Matthias:

- can he share about the internal project in the company is exactly about?


Dadurch mein erster Ansatz ist ein NER modell mit einem schwach gelabelten datensatz zu trainieren brauche ich auf jedenfall ein Datenset welches gelabelt ist. Weil ein schwach gelabeltes datenset kann ich immernoch machen. Ich denke die zwei Datensets die in dem recent paper benutzt wurden eignen sich besser, weil da vorallem auch ein bio datenset mit dabei ist .

When doing the introduction. I get a really good overview of the topic. I really like the atmosphere here. When writing the introduction I can start writing it already in my Jupyter notebook. It will for sure a business idea of me to create something like the "thesis book". It will be the standard for. I basically can just convert a jupyter notebook to markdown and the markdown to pdf.

I need to make sense of the dataset so that I can use i

einleitung, grundlagen 1-2 wochen

Names: Max, Max, Yannick, Matthias, Sören 


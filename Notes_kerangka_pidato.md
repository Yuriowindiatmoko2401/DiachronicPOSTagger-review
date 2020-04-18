Assalamu Alaikum wr wb
- okay , hari ini kita akan coba me-review paper text analytic yang bertemakan POS Tagging
- POS tagging itu sendiri merupakan kedudukan kata pada suatu kalimat yaa , seperti kata kerja (verb) , Kata Benda (Noun) kata penghubung dst ..

saya memilih paper ini untuk di review , krn cukup update dan menarik sih paper ini (2019) 

Detecting Syntactic Change Using a Neural Part-of-Speech Tagger
- Mendeteksi perubahan sintaksis menggunakan Neural Nets POS Tagger 

Abstract
abstraksi

1.
latar belakang 
apa

analyze the tagger’s ability to implicitly learn temporal structure between years, and the extent to which this knowledge can be transferred to date new sentences
- menganalisa kemampuan POS tagger untuk scr implisit mempelajari struktur temporal linguistik yang berubah dari tahun ke tahun, yg dgn hal ini diharapkan model dpt mengetahui tanggal terutama tahun kapan kalimat tersebut dibuat

2.
menjelaskan hasil

learned year embeddings show a strong linear correlation between their first principal component and time. We show that temporal information encoded in the model can be used to predict novel sentences’ years of composition relatively well
- Model words embedding yang diterapkan disini menunjukkan korelasi yg kuat dari first PCA model tersebut terhadap Waktu (tahun) 
- Informasi tahun pembuatan dr tiap kalimat yang sudah di encode pada model dpt digunakan utk memprediksi tahun dibuatnya kalimat pada novel tersebut dengan cukup baik

3. 
Data yg digunakan 
American English over the course of the 19th, 20th, and early 21st centuries.
Corpus of Historical American English (COHA)
The COHA corpus is composed of documents dating from 1810 to 2009 and contains over 400 million words. 
- the genre mix of the texts is balanced in each decade, and
- includes fiction works,
- academic papers,
- newspapers, and
- popular magazines.

- oh iyaa disini cukup byk membahas diachronic apa sih itu ?
Introduction We define a diachronic language task as a standard computational linguistic task where the input includes not just text, but also information about when the text was written.


#### Batasan Metode

Because of computational constraints, 
we randomly selected 50,000 sentences from each decade for a total of 1,000,000 sentences. 
We selected an equal number of sentences from each decade to ensure a temporally balanced corpus.

We put 90% of these into a training set and 10% into a test set. (9/10 * 50.000) = 45.000 train, 5000 test each decade 
45.000 x 200 = 900.000 train set
5000 x 200 = 100.000 test set

- cut off all sentences at a maximum length of 50 words
- karena Only 6.95% of sentences in the full COHA corpus exceeded 50 words.

Texts in COHA are annotated (beranotasikan) with 
word, lemma, and POS information. 
The POS labels come in three levels of specificity, with the most specific level containing several thousand POS tags.

- We used the least specific label for our model, which still had 423 unique POS tags. (level one)

- Our model utilized pre-trained 300-dimensional Google News (Mikolov et al., 2013) word embeddings that were learned using a standard word2vec architecture. 1.5 GB

When there was no embedding available for a word in the corpus, we assigned the word an embedding vector drawn from a normal distribution, so that different unknown words would have different embeddings (follow normal distribution).

COHA corpus has words that not in word2vec(google_News) --> UNK

Due to computational constraints, we only included embeddings for the `600,000 most common words in the vocabulary`. 
(only top 600,000 words most common from COHA Corpus)

Other words were replaced by a special symbol UNK.


#### Methods
##### Network Architecture

We used a single-layer LSTM model (fig.1). 
For a given sentence from a document composed in the year with

- embedding t,

- the model’s input for the i th word in the sentence is the concatenation of the word’s embedding xi and t.

https://github.com/viking-sudo-rm/DiachronicPOSTagger For example,

- consider a sentence hello world! written in 2000. The input corresponding to hello would be the concatenation of

- the embedding for hello and

- the embedding of the year 2000.

The word embeddings were loaded statically (direct by word2vec GoogleNews). 
In contrast, year embeddings were `Xavier-initialized` and learned dynamically by our network. 

- We gave both the `word embeddings` and `year embeddings` a dimensionality of 300. 
We picked the `size of our LSTM layer to be 512. `

- Due to the size of our training set (45000) and our limited computational resources, we ran our network for just `one training epoch`. 
`Manual tweaking` `of the learning rate` and `batch size` `revealed that the network’s performance` `was not particularly sensitive to their values`.

- we set the 
`learning rate to 0.001` and the 
`batch size to 100.` 
`We did not incorporate dropout or regularization into our model since we did not expect overfitting`, `as we only trained for a single epoch`.

In order to `calibrate the performance of our LSTM`, we trained the following ablation models: (pengurangan) 
• An LSTM tagger without year input (only word embedding) 
• A feedforward tagger with year input (word n year) 
• A feedforward tagger without year input (only word)
`4 times training`

`All taggers were trained with identical hyperparameters to the original LSTM.` by default `tf.rnn.LSTM` 
`For the feedforward models`, the 
`LSTM layer` `was replaced` `by a feedforward layer` of size 512. 
(50 x 300) --> 15000 flatten --> 512 FF

The `lack of recurrent connections` in the feedforward models makes it impossible for these models to consider interactions between words. 
`FF tdk ada recurren networks`
Thus, `these models serve` `as a baseline` `that only considers` `relationships` `between single words` and `their POS tags–not syntax`.


- reduced the year embeddings to one-dimensional space using principal component analysis (PCA)



Temporal Prediction

- We used our model to compute the perplexity of each sentence in a given bucket at every possible year (1810-2009). 
- We then fit a curve to perplexity as a function of year using locally weighted scatterplot smoothing (LOWESS).

3 Results

3.1 Tagger Performance

test set
Feedforward	LSTM
Year	82.6	95.5
No Year	77.8	95.3

train set
Feedforward	LSTM
Year	82.6	95.6
No Year	77.7	95.4

Figure 2: The first principal component of the LSTM year embeddings correlates strongly with time (R2 = 0.89).

Figure 3: The first principal component of the feedforward year embeddings shows a weaker temporal trend than that of the LSTM (R2 = 0.68).

	Baseline	Feedforward	LSTM
Decade	50.0	26.6	12.5
Year	50.0	37.5	21.9

perplexity


3.3 Temporal Prediction

sentence -> LSTM embed -> PCA -> fit cure -> predicted year


sentence example

flow chart --->














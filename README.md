## 0. Imports


```python
import nltk
import collections
import numpy as np
from keras.datasets import imdb
from nltk.util import ngrams
```

## 1. IMDB Dataset


```python
(x_train, y_train), (x_test, y_test) = imdb.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
    17464789/17464789 [==============================] - 74s 4us/step
    

## 2. Pre-Processing

### 2.0. Index to word and vice versa


```python
word_index = imdb.get_word_index()
index_word = {i: word for word, i in word_index.items()}
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
    1641221/1641221 [==============================] - 6s 4us/step
    


```python
def mapper(sources, map_dict):
    destinations = []
    for source in sources:
        destination = [map_dict.get(element) for element in source]
        destinations.append([d for d in destination if d is not None])
    return np.array(destinations, dtype=object)
```


```python
x_train_words = mapper(x_train, index_word)
x_train_indexes = mapper(x_train_words, word_index)

x_test_words = mapper(x_test, index_word)
x_test_indexes = mapper(x_test_words, word_index)
```


```python
print(x_train[0])
print(x_train_words[0])
print(x_train_indexes[0])
```

    [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 22665, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 21631, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 19193, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 10311, 8, 4, 107, 117, 5952, 15, 256, 4, 31050, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 12118, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
    ['the', 'as', 'you', 'with', 'out', 'themselves', 'powerful', 'lets', 'loves', 'their', 'becomes', 'reaching', 'had', 'journalist', 'of', 'lot', 'from', 'anyone', 'to', 'have', 'after', 'out', 'atmosphere', 'never', 'more', 'room', 'titillate', 'it', 'so', 'heart', 'shows', 'to', 'years', 'of', 'every', 'never', 'going', 'villaronga', 'help', 'moments', 'or', 'of', 'every', 'chest', 'visual', 'movie', 'except', 'her', 'was', 'several', 'of', 'enough', 'more', 'with', 'is', 'now', 'current', 'film', 'as', 'you', 'of', 'mine', 'potentially', 'unfortunately', 'of', 'you', 'than', 'him', 'that', 'with', 'out', 'themselves', 'her', 'get', 'for', 'was', 'camp', 'of', 'you', 'movie', 'sometimes', 'movie', 'that', 'with', 'scary', 'but', 'pratfalls', 'to', 'story', 'wonderful', 'that', 'in', 'seeing', 'in', 'character', 'to', 'of', '70s', 'musicians', 'with', 'heart', 'had', 'shadows', 'they', 'of', 'here', 'that', 'with', 'her', 'serious', 'to', 'have', 'does', 'when', 'from', 'why', 'what', 'have', 'critics', 'they', 'is', 'you', 'that', "isn't", 'one', 'will', 'very', 'to', 'as', 'itself', 'with', 'other', 'tricky', 'in', 'of', 'seen', 'over', 'landed', 'for', 'anyone', 'of', "gilmore's", 'br', "show's", 'to', 'whether', 'from', 'than', 'out', 'themselves', 'history', 'he', 'name', 'half', 'some', 'br', 'of', "'n", 'odd', 'was', 'two', 'most', 'of', 'mean', 'for', '1', 'any', 'an', 'boat', 'she', 'he', 'should', 'is', 'thought', 'frog', 'but', 'of', 'script', 'you', 'not', 'while', 'history', 'he', 'heart', 'to', 'real', 'at', 'barrel', 'but', 'when', 'from', 'one', 'bit', 'then', 'have', 'two', 'of', 'script', 'their', 'with', 'her', 'nobody', 'most', 'that', 'with', "wasn't", 'to', 'with', 'armed', 'acting', 'watch', 'an', 'for', 'with', 'heartfelt', 'film', 'want', 'an']
    [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 22665, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 21631, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 19193, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 10311, 8, 4, 107, 117, 5952, 15, 256, 4, 31050, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 12118, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
    

### 2.1. Any data cleaning


```python
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\alise\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\stopwords.zip.
    


```python
def cleaner(sentences, stop_words):
    new_sentences = []
    for sentence in sentences:
        new_sentence = [word for word in sentence if word not in stop_words and len(word) > 2]
        new_sentences.append(new_sentence)
    return new_sentences
```


```python
x_train_words_cleaned = cleaner(x_train_words, stop_words)
x_train_indexes_cleaned = mapper(x_train_words_cleaned, word_index)

x_test_words_cleaned = cleaner(x_test_words, stop_words)
x_test_indexes_cleaned = mapper(x_test_words_cleaned, word_index)
```


```python
print(x_train_words[0])
print(x_train[0])
print(x_train_indexes_cleaned[0])
print(x_train_words_cleaned[0])
```

    ['the', 'as', 'you', 'with', 'out', 'themselves', 'powerful', 'lets', 'loves', 'their', 'becomes', 'reaching', 'had', 'journalist', 'of', 'lot', 'from', 'anyone', 'to', 'have', 'after', 'out', 'atmosphere', 'never', 'more', 'room', 'titillate', 'it', 'so', 'heart', 'shows', 'to', 'years', 'of', 'every', 'never', 'going', 'villaronga', 'help', 'moments', 'or', 'of', 'every', 'chest', 'visual', 'movie', 'except', 'her', 'was', 'several', 'of', 'enough', 'more', 'with', 'is', 'now', 'current', 'film', 'as', 'you', 'of', 'mine', 'potentially', 'unfortunately', 'of', 'you', 'than', 'him', 'that', 'with', 'out', 'themselves', 'her', 'get', 'for', 'was', 'camp', 'of', 'you', 'movie', 'sometimes', 'movie', 'that', 'with', 'scary', 'but', 'pratfalls', 'to', 'story', 'wonderful', 'that', 'in', 'seeing', 'in', 'character', 'to', 'of', '70s', 'musicians', 'with', 'heart', 'had', 'shadows', 'they', 'of', 'here', 'that', 'with', 'her', 'serious', 'to', 'have', 'does', 'when', 'from', 'why', 'what', 'have', 'critics', 'they', 'is', 'you', 'that', "isn't", 'one', 'will', 'very', 'to', 'as', 'itself', 'with', 'other', 'tricky', 'in', 'of', 'seen', 'over', 'landed', 'for', 'anyone', 'of', "gilmore's", 'br', "show's", 'to', 'whether', 'from', 'than', 'out', 'themselves', 'history', 'he', 'name', 'half', 'some', 'br', 'of', "'n", 'odd', 'was', 'two', 'most', 'of', 'mean', 'for', '1', 'any', 'an', 'boat', 'she', 'he', 'should', 'is', 'thought', 'frog', 'but', 'of', 'script', 'you', 'not', 'while', 'history', 'he', 'heart', 'to', 'real', 'at', 'barrel', 'but', 'when', 'from', 'one', 'bit', 'then', 'have', 'two', 'of', 'script', 'their', 'with', 'her', 'nobody', 'most', 'that', 'with', "wasn't", 'to', 'with', 'armed', 'acting', 'watch', 'an', 'for', 'with', 'heartfelt', 'film', 'want', 'an']
    [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 22665, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 21631, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 19193, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 10311, 8, 4, 107, 117, 5952, 15, 256, 4, 31050, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 12118, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
    [973, 1622, 1385, 458, 4468, 3941, 173, 256, 838, 112, 670, 22665, 480, 284, 150, 172, 112, 167, 21631, 336, 385, 172, 4536, 1111, 17, 546, 447, 192, 2025, 19, 1920, 4613, 469, 76, 1247, 17, 515, 17, 626, 19193, 62, 386, 316, 106, 2223, 5244, 480, 3785, 619, 1415, 28, 10311, 107, 5952, 256, 31050, 3766, 723, 476, 400, 317, 1029, 104, 381, 2071, 194, 7486, 226, 476, 480, 144, 5535, 28, 224, 104, 226, 1334, 4472, 113, 103, 5345, 19, 178]
    ['powerful', 'lets', 'loves', 'becomes', 'reaching', 'journalist', 'lot', 'anyone', 'atmosphere', 'never', 'room', 'titillate', 'heart', 'shows', 'years', 'every', 'never', 'going', 'villaronga', 'help', 'moments', 'every', 'chest', 'visual', 'movie', 'except', 'several', 'enough', 'current', 'film', 'mine', 'potentially', 'unfortunately', 'get', 'camp', 'movie', 'sometimes', 'movie', 'scary', 'pratfalls', 'story', 'wonderful', 'seeing', 'character', '70s', 'musicians', 'heart', 'shadows', 'serious', 'critics', 'one', 'tricky', 'seen', 'landed', 'anyone', "gilmore's", "show's", 'whether', 'history', 'name', 'half', 'odd', 'two', 'mean', 'boat', 'thought', 'frog', 'script', 'history', 'heart', 'real', 'barrel', 'one', 'bit', 'two', 'script', 'nobody', 'armed', 'acting', 'watch', 'heartfelt', 'film', 'want']
    

## 3. Build Models


```python
def get_vocab_size(x, y):
    x_dict = {0: [], 1: []}
    for i, sample in enumerate(x):
        class_idx = y[i]
        x_dict[class_idx] += sample

    p_vocab_size = len(collections.Counter(x_dict[1]))
    n_vocab_size = len(collections.Counter(x_dict[0]))

    return n_vocab_size, p_vocab_size
```


```python
def build_ngrams(x, y, n):
    ngrams_dict = {0: [], 1: []}

    for i, sample in enumerate(x):
        class_idx = y[i]
        sample_ngrams = ngrams(sample, n)
        ngrams_dict[class_idx].extend(sample_ngrams)

    p_ngrams_freq = collections.Counter(ngrams_dict[1])
    n_ngrams_freq = collections.Counter(ngrams_dict[0])

    return n_ngrams_freq, p_ngrams_freq
```


```python
def get_model(freqs, vocab_size, before_freqs=None):
    denominator = sum(freqs.values())
    model = dict()
    for gram, freq in freqs.items():
        if before_freqs is not None:
            denominator = before_freqs[gram[:-1]]
        value = (freq + 1) / (denominator + vocab_size)
        model[gram] = value
    return model
```


```python
vocab_size = get_vocab_size(
    x_train_indexes_cleaned,
    y_train
)
```


```python
print(vocab_size[0])
print(vocab_size[1])
```

    61337
    63648
    

### 3.1. Uni-Gram


```python
n = 1
unigram_freqs = build_ngrams(
    x_train_indexes_cleaned,
    y_train,
    n
)
```


```python
unigram_model = {
    0: get_model(unigram_freqs[0], vocab_size[0]),
    1: get_model(unigram_freqs[1], vocab_size[1])
}
```

### 3.2. Bi-Gram



```python
n = 2
bigram_freqs = build_ngrams(
    x_train_indexes_cleaned,
    y_train,
    n
)
```


```python
bigram_model = {
    0: get_model(bigram_freqs[0], vocab_size[0], unigram_freqs[0]),
    1: get_model(bigram_freqs[1], vocab_size[1], unigram_freqs[1])
}
```

### 3.3. Tri-Gram


```python
n = 3
trigram_freqs = build_ngrams(
    x_train_indexes_cleaned,
    y_train,
    n
)
```


```python
trigram_model = {
    0: get_model(trigram_freqs[0], vocab_size[0], bigram_freqs[0]),
    1: get_model(trigram_freqs[1], vocab_size[1], bigram_freqs[1])
}
```

## 4. Evaluate Model


```python
pos_count = np.count_nonzero(y_test)
neg_count = y_test.shape[0] - pos_count
neg_class_prob = neg_count / y_test.shape[0]

print(neg_class_prob)
```

    0.5
    


```python
def nb_predict(x_test, n, model, neg_class_prob, unk_factor, vocab_size):
    preds = []
    
    for sample in x_test:
        neg_prob = np.log(neg_class_prob)
        pos_prob = np.log(1 - neg_class_prob)

        ngrams_list = ngrams(sample, n)
        for gram in ngrams_list:
            gram_neg_prob = model[0].get(gram)
            gram_pos_prob = model[1].get(gram)
            
            # <UNK>
            if gram_neg_prob is None:
                gram_neg_prob = 1 / (unk_factor[0] + vocab_size[0])
            if gram_pos_prob is None:
                gram_pos_prob = 1 / (unk_factor[1] + vocab_size[1])

            gram_neg_prob = np.log(gram_neg_prob)
            gram_pos_prob = np.log(gram_pos_prob)

            neg_prob += gram_neg_prob
            pos_prob += gram_pos_prob

        if neg_prob > pos_prob:
            preds.append(0)
        else:
            preds.append(1)
    return np.array(preds)
```


```python
def calculate_metrics(y_r, y_p):
    tp = 0
    tn = 0
    fn = 0
    fp = 0

    for i in range(y_test.shape[0]):
        if y_r[i] == 1:
            if y_p[i] == 1:
                tp += 1
            elif y_p[i] == 0:
                fn += 1
        elif y_r[i] == 0:
            if y_p[i] == 0:
                tn += 1
            elif y_p[i] == 1:
                fp += 1

    a = (tp + tn) / (tp + tn + fn + fp)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1_score = (2 * p * r) / (p + r)

    return {
        'accuracy': a,
        'precision': p,
        'recall': r,
        'f1_score': f1_score,
    }
```

### 4.1 Unigram


```python
# Unigram
# With cleaning stop words

n = 1
model = unigram_model

preds = nb_predict(
    x_test_indexes_cleaned,
    n,
    model,
    neg_class_prob,
    vocab_size,
    vocab_size
)

calculate_metrics(y_test, preds)
```




    {'accuracy': 0.813,
     'precision': 0.861164958921813,
     'recall': 0.74632,
     'f1_score': 0.7996399948570694}




```python
for i in range(5):
    print(' '.join([str(e) for e in x_test_words[i]]))
    print(f"Predicted class: {y_test[i]} \t Real class: {preds[i]}")
    print()
```

    the wonder own as by is sequence i i jars roses to of hollywood br of down shouting getting boring of ever it sadly sadly sadly i i was then does don't close faint after one carry as by are be favourites all family turn in does as three part in another some to be probably with world uncaring her an have faint beginning own as is sequence
    Predicted class: 0 	 Real class: 0
    
    the as you world's is quite br mankind most that quest are chase to being quickly of little it time hell to plot br of something long put are of every place this consequence council of interplay storytelling being nasty not of you warren in is failed club i i of films pay so sequences mightily film okay uses to received wackiness if time done for room sugar viewer as cartoon of gives to forgettable br be because many these of reflection sugar contained gives it wreck scene to more was two when had find as you another it of themselves probably who interplay storytelling if itself by br about 1950's films not would effects that her box to miike for if hero close seek end is very together movie of wheel got say kong sugar fred close bore there is playing lot of scriptures pan place trilogy of lacks br of their time much this men as on it is telling program br silliness okay orientation to frustration at corner rawlins she of sequences to political clearly in of drugs keep guy i i was throwing room sugar as it by br be plot many for occasionally film verge boyfriend difficult kid as you it failed not if gerard to if woman in launching is police fi spooky or of self what have pretty in can so suit you good 2 which why super as it main of my i i  if time screenplay in same this remember assured have action one in realistic that better of lessons
    Predicted class: 1 	 Real class: 1
    
    the plot near ears recent halliburton cosmopolitan of him flicks frank br by excellent sans br of past loyalty near really all grief family four victim's to movie that obvious family brave movie is got say cosmopolitan with up comment this orks been of entertaining not be lamarr james in you seen vittorio castle's portrayed dirty in so washington ursula this you minutes no all station all after torments promising who aragorn horn noir' to contracted any by speed they is my as screams dirty in of full br pacino dignity need men of pitchfork popular really all way this behaviour this sturdy they is my no standard certainly near br an beach with this make imbecilic i i of fails ritt br of finished wear psycho cosmopolitan in learn in twice know by br be how rings epps with is seemed fails visually posthumous extremely movie scooping it's of ishtar like children is easily is thug br simply must well at although this family an br many not scene that it time seemed de ignored up they boat morning like well force of suggestion sent been history like story its disappointing same of club finch watching husband reviewer to although that around finch except to de impersonation br of you available but hours animals showing br of optimism than dead white splatter waiting film tenants to attentions this documentary in 3 eduardo of accents committee br of ann i i comes 9 it place this is overseas of scooping spradlin know of mode he bonus film were central to one oh is excellent cindy in can when from well people in characters' chief from leaving in mattia landers but is easily of lamas he historian speak this as today paul that against one will actual in could her plot bias error few grade marc go landers but be lot it oliver movie is dis picture tuning feel this of ensue like different just clichéd girl at finds is sweets no landers glory any is children's just moment like mixing any of ishtar leaving for as it even cliche to purchased is money easily egypt landers glory any is indiscernible i i liam film as digress set actually easily like outdated sequel any of ishtar ryan made film is jaayen br nook constant unit of 90s letting deep in act made of road in of spradlin movie convictions rural vhs of share in reaching fact of indiscernible polly spinal of 90s to them book are is unfamiliar mercy karen's mode they funniest is white courage fiver vegas wooden br of gender traditionally unfortunately of 1968 no of years hokey ishtar true up mattia landers but 3 all ordinary be oblivious to auer were deserve film clone prairie of creative br comes their kung who is assuming bias out new all it incomprehensible it episode much that's including i i cartoon of my certain no as rooting over you with way to cartoon of enough for that with way who is finished mornings they of rukh br for cupboard expressing stunts black that story at actual in can as movie is strands has though songs cosmopolitan action it's action his one me joshua's grass this second no all way scooping not lee warhol be moves br figure of you boss movie is snatched 9 br propaganda resumed scooping after at of smoke splendid snow saturday it's results this of load it's think class br think cop for games make southern things to it jolly who gladys if is boyfriend you which is tony by this make residents too not make above it even background
    Predicted class: 1 	 Real class: 1
    
    the was stick did as roles br on take as my was although except torture in perspective of goes he's was big people for was into out improved has that as with boy weapon of seems for ago film of performances production he time relationship not of grade great he jean misses was rather is boat say around thought to was well constructed except much take was story his people star of blood of over fun end this as on other of killer this as on it deborah film about history in of come br tested was saying was three her length has about to about unusual most was story one let's town of genre when is seriously would with long only king's to future deep i'm dvd have can about people friends of here other it especially fan often somewhere br doesn't characters for he means her seemed states by well potential can when it never means movie so night bad he seducing daughter film of unusual are of goes her them such of number big bad one left bloody
    Predicted class: 0 	 Real class: 0
    
    the just good because great cold watching is minute each shirley completely to was several as b i i as b gave compared rest not includes we if main that movie sometimes movie have sex man endearing of feet he played to faris from into pot have dissection man second hand in integrate watching his offering as b it other rudimentary to it taste bit i i in perfect as slowly truth was one in perfect only deliver sleazy has thrown not wonder classic as b satisfied at main that i i their among among without didn't later if for very pian didn't clearly aa didn't forget didn't
    Predicted class: 1 	 Real class: 0
    
    

### 4.2 Bigram


```python
# Bigram
# With cleaning stop words

n = 2
model = bigram_model

preds = nb_predict(
    x_test_indexes_cleaned,
    n,
    model,
    neg_class_prob,
    vocab_size,
    vocab_size
)

calculate_metrics(y_test, preds)
```




    {'accuracy': 0.80428,
     'precision': 0.8911858479893037,
     'recall': 0.6932,
     'f1_score': 0.7798227062052828}




```python
for i in range(5):
    print(' '.join([str(e) for e in x_test_words[i]]))
    print(f"Predicted class: {y_test[i]} \t Real class: {preds[i]}")
    print()
```

    the wonder own as by is sequence i i jars roses to of hollywood br of down shouting getting boring of ever it sadly sadly sadly i i was then does don't close faint after one carry as by are be favourites all family turn in does as three part in another some to be probably with world uncaring her an have faint beginning own as is sequence
    Predicted class: 0 	 Real class: 0
    
    the as you world's is quite br mankind most that quest are chase to being quickly of little it time hell to plot br of something long put are of every place this consequence council of interplay storytelling being nasty not of you warren in is failed club i i of films pay so sequences mightily film okay uses to received wackiness if time done for room sugar viewer as cartoon of gives to forgettable br be because many these of reflection sugar contained gives it wreck scene to more was two when had find as you another it of themselves probably who interplay storytelling if itself by br about 1950's films not would effects that her box to miike for if hero close seek end is very together movie of wheel got say kong sugar fred close bore there is playing lot of scriptures pan place trilogy of lacks br of their time much this men as on it is telling program br silliness okay orientation to frustration at corner rawlins she of sequences to political clearly in of drugs keep guy i i was throwing room sugar as it by br be plot many for occasionally film verge boyfriend difficult kid as you it failed not if gerard to if woman in launching is police fi spooky or of self what have pretty in can so suit you good 2 which why super as it main of my i i  if time screenplay in same this remember assured have action one in realistic that better of lessons
    Predicted class: 1 	 Real class: 1
    
    the plot near ears recent halliburton cosmopolitan of him flicks frank br by excellent sans br of past loyalty near really all grief family four victim's to movie that obvious family brave movie is got say cosmopolitan with up comment this orks been of entertaining not be lamarr james in you seen vittorio castle's portrayed dirty in so washington ursula this you minutes no all station all after torments promising who aragorn horn noir' to contracted any by speed they is my as screams dirty in of full br pacino dignity need men of pitchfork popular really all way this behaviour this sturdy they is my no standard certainly near br an beach with this make imbecilic i i of fails ritt br of finished wear psycho cosmopolitan in learn in twice know by br be how rings epps with is seemed fails visually posthumous extremely movie scooping it's of ishtar like children is easily is thug br simply must well at although this family an br many not scene that it time seemed de ignored up they boat morning like well force of suggestion sent been history like story its disappointing same of club finch watching husband reviewer to although that around finch except to de impersonation br of you available but hours animals showing br of optimism than dead white splatter waiting film tenants to attentions this documentary in 3 eduardo of accents committee br of ann i i comes 9 it place this is overseas of scooping spradlin know of mode he bonus film were central to one oh is excellent cindy in can when from well people in characters' chief from leaving in mattia landers but is easily of lamas he historian speak this as today paul that against one will actual in could her plot bias error few grade marc go landers but be lot it oliver movie is dis picture tuning feel this of ensue like different just clichéd girl at finds is sweets no landers glory any is children's just moment like mixing any of ishtar leaving for as it even cliche to purchased is money easily egypt landers glory any is indiscernible i i liam film as digress set actually easily like outdated sequel any of ishtar ryan made film is jaayen br nook constant unit of 90s letting deep in act made of road in of spradlin movie convictions rural vhs of share in reaching fact of indiscernible polly spinal of 90s to them book are is unfamiliar mercy karen's mode they funniest is white courage fiver vegas wooden br of gender traditionally unfortunately of 1968 no of years hokey ishtar true up mattia landers but 3 all ordinary be oblivious to auer were deserve film clone prairie of creative br comes their kung who is assuming bias out new all it incomprehensible it episode much that's including i i cartoon of my certain no as rooting over you with way to cartoon of enough for that with way who is finished mornings they of rukh br for cupboard expressing stunts black that story at actual in can as movie is strands has though songs cosmopolitan action it's action his one me joshua's grass this second no all way scooping not lee warhol be moves br figure of you boss movie is snatched 9 br propaganda resumed scooping after at of smoke splendid snow saturday it's results this of load it's think class br think cop for games make southern things to it jolly who gladys if is boyfriend you which is tony by this make residents too not make above it even background
    Predicted class: 1 	 Real class: 1
    
    the was stick did as roles br on take as my was although except torture in perspective of goes he's was big people for was into out improved has that as with boy weapon of seems for ago film of performances production he time relationship not of grade great he jean misses was rather is boat say around thought to was well constructed except much take was story his people star of blood of over fun end this as on other of killer this as on it deborah film about history in of come br tested was saying was three her length has about to about unusual most was story one let's town of genre when is seriously would with long only king's to future deep i'm dvd have can about people friends of here other it especially fan often somewhere br doesn't characters for he means her seemed states by well potential can when it never means movie so night bad he seducing daughter film of unusual are of goes her them such of number big bad one left bloody
    Predicted class: 0 	 Real class: 0
    
    the just good because great cold watching is minute each shirley completely to was several as b i i as b gave compared rest not includes we if main that movie sometimes movie have sex man endearing of feet he played to faris from into pot have dissection man second hand in integrate watching his offering as b it other rudimentary to it taste bit i i in perfect as slowly truth was one in perfect only deliver sleazy has thrown not wonder classic as b satisfied at main that i i their among among without didn't later if for very pian didn't clearly aa didn't forget didn't
    Predicted class: 1 	 Real class: 1
    
    

### 4.3 Trigram


```python
# Trigram
# With cleaning stop words

n = 3
model = trigram_model

preds = nb_predict(
    x_test_indexes_cleaned,
    n,
    model,
    neg_class_prob,
    vocab_size,
    vocab_size
)

calculate_metrics(y_test, preds)
```




    {'accuracy': 0.62788,
     'precision': 0.8506251370914675,
     'recall': 0.31024,
     'f1_score': 0.45465736561345915}




```python
for i in range(5):
    print(' '.join([str(e) for e in x_test_words[i]]))
    print(f"Predicted class: {y_test[i]} \t Real class: {preds[i]}")
    print()
```

    the wonder own as by is sequence i i jars roses to of hollywood br of down shouting getting boring of ever it sadly sadly sadly i i was then does don't close faint after one carry as by are be favourites all family turn in does as three part in another some to be probably with world uncaring her an have faint beginning own as is sequence
    Predicted class: 0 	 Real class: 0
    
    the as you world's is quite br mankind most that quest are chase to being quickly of little it time hell to plot br of something long put are of every place this consequence council of interplay storytelling being nasty not of you warren in is failed club i i of films pay so sequences mightily film okay uses to received wackiness if time done for room sugar viewer as cartoon of gives to forgettable br be because many these of reflection sugar contained gives it wreck scene to more was two when had find as you another it of themselves probably who interplay storytelling if itself by br about 1950's films not would effects that her box to miike for if hero close seek end is very together movie of wheel got say kong sugar fred close bore there is playing lot of scriptures pan place trilogy of lacks br of their time much this men as on it is telling program br silliness okay orientation to frustration at corner rawlins she of sequences to political clearly in of drugs keep guy i i was throwing room sugar as it by br be plot many for occasionally film verge boyfriend difficult kid as you it failed not if gerard to if woman in launching is police fi spooky or of self what have pretty in can so suit you good 2 which why super as it main of my i i  if time screenplay in same this remember assured have action one in realistic that better of lessons
    Predicted class: 1 	 Real class: 0
    
    the plot near ears recent halliburton cosmopolitan of him flicks frank br by excellent sans br of past loyalty near really all grief family four victim's to movie that obvious family brave movie is got say cosmopolitan with up comment this orks been of entertaining not be lamarr james in you seen vittorio castle's portrayed dirty in so washington ursula this you minutes no all station all after torments promising who aragorn horn noir' to contracted any by speed they is my as screams dirty in of full br pacino dignity need men of pitchfork popular really all way this behaviour this sturdy they is my no standard certainly near br an beach with this make imbecilic i i of fails ritt br of finished wear psycho cosmopolitan in learn in twice know by br be how rings epps with is seemed fails visually posthumous extremely movie scooping it's of ishtar like children is easily is thug br simply must well at although this family an br many not scene that it time seemed de ignored up they boat morning like well force of suggestion sent been history like story its disappointing same of club finch watching husband reviewer to although that around finch except to de impersonation br of you available but hours animals showing br of optimism than dead white splatter waiting film tenants to attentions this documentary in 3 eduardo of accents committee br of ann i i comes 9 it place this is overseas of scooping spradlin know of mode he bonus film were central to one oh is excellent cindy in can when from well people in characters' chief from leaving in mattia landers but is easily of lamas he historian speak this as today paul that against one will actual in could her plot bias error few grade marc go landers but be lot it oliver movie is dis picture tuning feel this of ensue like different just clichéd girl at finds is sweets no landers glory any is children's just moment like mixing any of ishtar leaving for as it even cliche to purchased is money easily egypt landers glory any is indiscernible i i liam film as digress set actually easily like outdated sequel any of ishtar ryan made film is jaayen br nook constant unit of 90s letting deep in act made of road in of spradlin movie convictions rural vhs of share in reaching fact of indiscernible polly spinal of 90s to them book are is unfamiliar mercy karen's mode they funniest is white courage fiver vegas wooden br of gender traditionally unfortunately of 1968 no of years hokey ishtar true up mattia landers but 3 all ordinary be oblivious to auer were deserve film clone prairie of creative br comes their kung who is assuming bias out new all it incomprehensible it episode much that's including i i cartoon of my certain no as rooting over you with way to cartoon of enough for that with way who is finished mornings they of rukh br for cupboard expressing stunts black that story at actual in can as movie is strands has though songs cosmopolitan action it's action his one me joshua's grass this second no all way scooping not lee warhol be moves br figure of you boss movie is snatched 9 br propaganda resumed scooping after at of smoke splendid snow saturday it's results this of load it's think class br think cop for games make southern things to it jolly who gladys if is boyfriend you which is tony by this make residents too not make above it even background
    Predicted class: 1 	 Real class: 0
    
    the was stick did as roles br on take as my was although except torture in perspective of goes he's was big people for was into out improved has that as with boy weapon of seems for ago film of performances production he time relationship not of grade great he jean misses was rather is boat say around thought to was well constructed except much take was story his people star of blood of over fun end this as on other of killer this as on it deborah film about history in of come br tested was saying was three her length has about to about unusual most was story one let's town of genre when is seriously would with long only king's to future deep i'm dvd have can about people friends of here other it especially fan often somewhere br doesn't characters for he means her seemed states by well potential can when it never means movie so night bad he seducing daughter film of unusual are of goes her them such of number big bad one left bloody
    Predicted class: 0 	 Real class: 0
    
    the just good because great cold watching is minute each shirley completely to was several as b i i as b gave compared rest not includes we if main that movie sometimes movie have sex man endearing of feet he played to faris from into pot have dissection man second hand in integrate watching his offering as b it other rudimentary to it taste bit i i in perfect as slowly truth was one in perfect only deliver sleazy has thrown not wonder classic as b satisfied at main that i i their among among without didn't later if for very pian didn't clearly aa didn't forget didn't
    Predicted class: 1 	 Real class: 0
    
    

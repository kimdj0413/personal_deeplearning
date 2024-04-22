from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
 
 #문장 토큰화
sentences = sent_tokenize(raw_text)

vocab = {} #딕셔너리(키와 값으로 이루어진 자료형)
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

#문장에서 단어 토큰화
for sentence in sentences:
    tokenized_sentece = word_tokenize(sentence)
    result = []
    #단어 소문자화
    for word in tokenized_sentece:
        word = word.lower()
        #불용어 제거
        if word not in stop_words:
            #단어 길이가 2 이하인 경우 제거
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word]=0
                vocab[word] += 1
    preprocessed_sentences.append(result)

#빈도수가 높은 순서대로 정렬
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)

#빈도수가 높은 수부터 1부터 부여
word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency > 1:
        i = i + 1
        word_to_index[word] = i

#인데스가 5 초과인 단어 제거
vocab_size = 5

words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

for w in words_frequency:
    del word_to_index[w]

#단어 집합에 없는 단어는 OOV 인덱스로
word_to_index['OOV'] = len(word_to_index) + 1 #OOV=Out Of Vocabulary

#단어 맵핑
encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)
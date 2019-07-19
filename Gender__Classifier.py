import nltk
from nltk.corpus import names
import random

def gender_features (word):
    return {'twofirst': word[:2]}

names=([(name,'male') for name in names.words('male.txt')]+\
[(name,'female') for name in names.words('female.txt')])
random.shuffle(names)
print('Распределённые женские и мужские имена \n',names)

featuresets=[(gender_features(n),g) for (n,g) in names]  #список векторов признаков классификации
print("первые 3 вектора", featuresets[0:3])
train_set=featuresets[500:]
test_set=featuresets[:500]
classifier1=nltk.NaiveBayesClassifier.train(train_set)
print(classifier1.classify(gender_features('Barnie')))
print(classifier1.classify(gender_features('Trinity')))
print('Точности классификатора = ', nltk.classify.accuracy(classifier1,test_set),'\n')
classifier1.show_most_informative_features(5)

from nltk.corpus import names

def gender_features2(name):
    features = {"firstletter": name[0].lower(), "lastletter": name[-1].lower(), }
    cons = 0
    vow = 0
    for index in range(len(name)):
        if name[index].lower() in 'aeiou':
            vow = vow + 1
        elif name[index].lower() in 'bcdfghjklmnpqrstvwxyz':
            cons = cons + 1
    features["vowels"] = vow
    features["consonants"] = cons
    return features

print ('Гендерные признаки: \n', gender_features2("Marta"))

names=([(name,'male') for name in names.words('male.txt')]+\
[(name,'female') for name in names.words('female.txt')])


featuresets=[(gender_features2(n),g) for (n,g) in names]
print (names[:10])
print('featureset last 10', featuresets[:10])

train_set = featuresets[500:]
test_set = featuresets[:500]
print('test set', test_set)
classifier2 = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier2, train_set))
classifier2.show_most_informative_features(5)

def err_analysis():
    print("Error analysis")
    train_names = names[1500:]
    errtest_names = names[500:1500]   #анализ ошибок
    test_names = names[:500]
    errors=[]
    for (name, tag) in errtest_names:
        guess = classifier2.classify(gender_features(name))
        if guess != tag:
            errors.append((tag, guess, name))   #добавить в список ошибок (метка, предсказание, имя)
    print('Список ошибок: \n', errors)


err_analysis()

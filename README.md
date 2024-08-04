# Проект Tweets

---
## Постановка задачи

Цель проекта - анализ тональности твитов. <br>
Для сообщений пользователей из тестового набора, необходимо предсказать с максимально возможным результатом, является ли тональность твита положительной, отрицательной или нейтральной.<br><br>
При этом, необходимо: <br>
1. **Подготовка данных**  _Для получения эмбеддигов, обработать исходные сообщения, используя различные подходы:<br> just tokenization -- stemming -- lemmatization -- stemming+misspellings -- lemmatization+misspellings -- any other ideas of preprocessing || 0 or 1, if word exists -- word count -- TFIDF_<br>

2. **Определение сходства**  _Среди датасетов, полученных в результате использования различных подходов к предварительной обработке данных, с помощью косинусного сходства, найти 10 наиболее похожих пар твитов_<br>

3. **Машинное обучение**  _Провести анализ тональности сообщений, используя разные алгоритмы машинного обучения и наборы данных, полученных в результате использования различных подходов к предварительной обработке. Результат оценки предсказания (_accuracy_)  на тестовом датасете должен иметь минимальную точность 0.832_<br>

4. _Bonus_: <br>
    * _Использовать иные методы для векторного представления слов (например, word2vec)_
    * _Результат оценки предсказания (accuracy)  на тестовом датасете должен иметь минимальную точность 0.873_
   
**_Замечания:_**
При необходимости, при подготовке данных  можно использовать различные способы очистки данных, включая исключение незначащих стоп-слов.


## Начало Работы

### Копирование
Для копирования файлов Проекта на локальный компьютер в папку *<your_dir_on_local_computer>* выполните:

```
    $ git clone git@github.com:dbadeev/tweets.git <your_dir_on_local_computer>
```

### Описание файлов
* *tweets.pdf* - текст задания
* *requirements.txt* - список библиотек, необходимый для работы
* Папка *data*
  - *processedNegative.csv* - файл с твитами негативной тональности
  - *processedNeutral.csv*  - файл с твитами нейтральной тональности
  - *processedPositive.csv*  - файл с твитами позитивной тональности
* *tweets.ipynb* - ноутбук проекта  
* *text_cleaninig.py* - утилиты "чистки" текста твитов
* *text_processing.py* - утилиты векторизации текста твитов
* *w2v_ml.py* - утилиты векторизации с помощью Word2Vec и предварительно обученных моделей представления векторов слов
* *cosine_similarity.py* - утилиты вычисления косинусного сходства векторов представлений слов
* *machine_learning.py* - утилиты нахождения оптимальных параметров различных моделей для подсчета accuracy с помощью GridSearch
* Папка *res*
  - *cos_sim.csv* - файл с 10 наиболее схожими парами твитов среди датасетов, полученных в результате использования различных подходов к предварительной обработке данных 
  - *df_df_prep.csv*  - файл с твитами, полученными в результате использования различных подходов к предварительной обработке данных 
<br>

## Запуск программ

1. После установки необходимых библиотек, сначала выполняется обучение модели на имеющихся данных запуском программы _train.py_. <br> <br>
Чтобы вывести на экран информацию о параметрах, 
из папки *<your_dir_on_local_computer>* выполните *train.py* с ключом _-h_:

```
$ python3 train.py -h          
usage: train.py [-h] [--path PATH] [--loss_control LOSS_CONTROL] [--epochs EPOCHS] [--learning_rate ETA] [--loss_graphics]
                [--predict_data] [--animation] [--debug] [--quality]

options:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  Path to data file (data.csv by default)
  --loss_control LOSS_CONTROL, -l LOSS_CONTROL
                        Epoch iterations will stop while gets loss_control value(1e-12 by default)
  --epochs EPOCHS, -e EPOCHS
                        Set the epochs number (1500 by default)
  --learning_rate ETA, -a ETA
                        Set the learning rate eta (0.2 by default)
  --loss_graphics, -g   Diagram with loss function depends on epochs
  --predict_data, -t    Diagram with data values and line prediction
  --animation, -c       Animation with prediction evolution while training
  --debug, -d           Print info about each stage of program
  --quality, -q         Model quality (R-square, MSE)

```

2. После успешной отработки программы обучения модели будет сформирован файл _coefs.csv_, в котором будут записаны найденные коэффициенты формулы вычисления предсказания цены автомобиля по заданному пробегу. <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>
3. Для получения информации о параметрах программы, предсказывающей примерную цену автомобиля в зависимости от пробега, из папки *<your_dir_on_local_computer>* выполните *predict.py* с ключом _-h_:
```
$ python3 predict.py -h 
usage: predict.py [-h] [--debug] [--mileage MILEAGE]

options:
  -h, --help            show this help message and exit
  --debug, -d           Print info about each stage of program
  --mileage MILEAGE, -m MILEAGE
                        Car mileage for price prediction (non-negative int)
```
<br>
4. После успешной отработки программы предсказания, на экран будет выведена информация о примерной стоимости автомобиля с указанным пробегом.  <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>

## Подробности

Подробнее о Проекте, примеры использования - по ссылке Wiki (https://github.com/dbadeev/ft_linear_regression/wiki).

<br>

## Автор

*loram (Дмитрий Бадеев)* 

<br><br>

## Результат в School 21
<img width="645" alt="image" src="https://github.com/dbadeev/ft_linear_regression/assets/50623941/7a13a846-6c5f-4957-88f5-79e19c388520">

# tweets

source tweets/bin/activate

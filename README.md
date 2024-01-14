# Классификация дорожных знаков
Проект посвящен созданию сервиса классификации дорожных знаков. В качестве обучающего датасета взят каноничный в этой области [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

Ссылка на Яндекс.Диск с данными - [тут](https://disk.yandex.ru/d/Lt3161xIH5m0MQ).


Схема репозитория (WIP)
------------
    ├── .gitignore
    │
    ├── .gitattributes
    │
    ├── LICENSE
    │
    ├── experiments                  <- experiment .ipynb files
    │
    ├── jupyter                      <- main .ipynb files
    │
    ├── readme_data                  <- pictures and gifs for README.md 
    │
    ├── requirements.txt             <- main requirements for project
    │
    ├── requirements.txt             <- requirements for rende.com (unused)
    │
    ├── setup.py                     <- makes project pip installable (pip install -e .)
    │                                   so src can be imported
    │
    └── services                     <- source code for services used in this project
       ├── __init__.py               <- makes src a Python module
       │
       ├── fastapi                   <- FastAPI service
       │   │
       │   ├── __init__.py           <- makes src a Python module
       │   └── main.py               <- main fastapi file
       │
       ├── models                    <- Directory with models and 
       │   │                            data processing files
       │   │
       │   ├── __init__.py           <- makes src a Python module
       │   └── preprocessing.py      <- functions to preprocess data and use trained
       │                                models to make predictions 
       └── tg_bot 
           │
           ├──__init__.py 
           │
           ├──handlers
           │    │                       
           │    ├── __init__.py      <- Makes src a Python module
           │    │
           │    ├── menu.py          <- buttons and text handlers
           │    │
           │    ├── predictions.py   <- pictures and albums handlers
           │    │
           │    └── retaing.csv      <- csv file with bot rating
           │
           ├── sample_images         <- firectory with sample images for bot
           │
           ├── bot.py                <- main function of telegram bot
           │
           ├── middleweare.py        <- middleware fucntions
           │
           └── config_reader.py      <- reader of .env file


## Задачи проекта
- Создание и обучение ML модели классификации дорожных знаков.
- Реализация простейшего FastAPI к модели, а также создание телеграм бота.
- **!**Амбициозно**!** Создание и обучение модели для детекции дорожных знаков на основе датасета [GTSDB](https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb).
- **!**Амбициозно**!** Объединение двух моделей в пайплайн обработки фотографии (на которой ожидается дорожный знак).
- **!**Амбициозно**!** Реализация сервиса хранения и визуализации информации о найденных дорожных знаках (база данных со знаками, визуализация на картах).
- **!**Амбициозно**!** Демо-вариант с обработкой панорам с каких-нибудь онлайн-карт.

## ML подход

Первостепенной задачей в стремелении обучить ML-модель классифицировать изображения является попытка найти читаемые моделью признаки на входящих фотографиях. Существует множество разных алгоритмов экстракции признаков из изображений, среди них мы решили протестировать следующие два - это [HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) и [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html). Целевой моделью для обучения на полученных признаках была выбрана [SVM](https://scikit-learn.org/stable/modules/svm.html). 

Подробнее о методе их работы вы можете прочитать по ссылке, или ознакомиться в соответствующих ноутбуках с исследованиями в папке jupyter.

### HOG + SVM

![Alt Text](readme_data/hog_example.png)

Пример результата работы HOG.

Предварительно изображения переводятся в черно-белые и ресайзятся до размера 50x50. Далее из изображения извлекались HOG дескрипторы в виде векторов, которые в конечном итоге и формировали матрицу объект-признак. На этой матрице обучался SVM-классификатор, который и стал результирующей моделью, представляющей HOG метод в этой работе.

Тестировались разнообразные параметры экстракции признаков и предварительной обработки изображений, лучшие метрики на тестовой выборке получились следующими:

|              | precision | recall   | f1-score | support |
| ------------ | --------- | -------- | -------- | ------- |
| accuracy     |           |          | 0.59     | 12630   |
| macro avg    | 0.55      |  0.54    | 0.53     | 12630   |
| weighted avg | 0.62      |  0.59    | 0.59     | 12630   |

### SIFT + Bag-Of-Visual-Words + SVM

![Alt Text](readme_data/sift_example.png)

Пример результата работы SIFT.

Предварительно изображения проходили следующую обработку: выравнивание гистограммы, нормализация, переводились в черно-белые и ресайзились до размера 32x32. Далее при помощи алгоритма SIFT для всех изображений из тестовой выборки извлекались дескрипторы (каждый дескриптор - вектор размерностью 128, их может быть как несколько у одного изображения, так и не быть вообще). Затем куча получившихся дексрипторов кластеризировалась с помощью алгоритма KMeans. 

Финальный шаг - снова достать дескрипторы из каждого изображения и найти ближайший центр кластера для каждого, затем инициализировать нулевой вектор произвольной размерности (в нашем случае размерность 43 класса * 15), в котором каждый j-ый элемент увеличивался на 1, где j - индекс найденного центра кластера. Получившийся вектор и есть вектор признаков, из которых формировалсь матрица объект-признак. SVM-классификатор, который и стал результирующей моделью, представляющей SIFT метод в этой работе.

Тестировались разнообразные параметры экстракции признаков и предварительной обработки изображений, лучшие метрики на тестовой выборке получились следующими:

|              | precision | recall   | f1-score | support |
| ------------ | --------- | -------- | -------- | ------- |
| accuracy     |           |          | 0.73     | 12630   |
| macro avg    | 0.76      |  0.63    | 0.66     | 12630   |
| weighted avg | 0.74      |  0.73    | 0.72     | 12630   |

### Вывод

Модель, обученая с использованием SIFT'a и мешка слов, судя по метрикам, показала себя несколько лучше. Однако на деле обе модели ошибаются довольно часто и не являются надежными. Желаемого качества предсказаний не удалось достичь, используя линейные модели в ML-подходе.

## API, tg bot

Взаимодействие с полученными моделями реализовано двумя способами: веб-интерфейс FastAPI и Телеграм бот. Ниже на гифках краткая демонстрация функционала веб-интерфейса и Телеграм бота.

Веб-интерфейс FastAPI

![Alt Text](readme_data/fastapi.gif)

Телеграм бот

![Alt Text](readme_data/tg_bot.gif)
    

### Авторы проекта
[Булыгин Глеб](https://github.com/gbull25)

[Соловьев Дима](https://github.com/libernightin)

[Тихомиров Витя](https://github.com/onthebox)

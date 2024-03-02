# Классификация дорожных знаков
Проект посвящен созданию сервиса классификации дорожных знаков. В качестве обучающего датасета взят каноничный в этой области [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

Ссылка на Яндекс.Диск с данными - [тут](https://disk.yandex.ru/d/Lt3161xIH5m0MQ).


Схема репозитория (WIP)
------------
```
   .
   ├── jupyter                      <- main .ipynb files
   │   ├── CNN_exp.ipynb
   │   ├── EDA.ipynb
   │   ├── HOG.ipynb
   │   └── SIFT.ipynb
   ├── LICENSE
   ├── readme_data
   │   ├── fastapi.gif
   │   └── tg_bot.gif
   ├── README.md
   ├── requirements.txt
   ├── services
   │   ├── docker-compose.yaml
   │   ├── sign_classifier
   │   │   ├── app                  <- FastAPI service
   │   │   │   ├── classify.py
   │   │   │   ├── cnn_model.py
   │   │   │   ├── log_conf.yaml
   │   │   │   ├── main.py
   │   │   │   └── settings.py
   │   │   ├── dockerfile
   │   │   ├── models
   │   │   │   ├── cnn_torch.pt
   │   │   │   ├── kmeans.gz
   │   │   │   ├── lzma_hog_proba.xz
   │   │   │   └── sift_svm.gz
   │   │   └── requirements.txt
   │   └── tg_bot                   <- telegram bot service
   │       ├── bot.py
   │       ├── config_reader.py
   │       ├── dockerfile
   │       ├── handlers             <- event handlers
   │       │   ├── __init__.py
   │       │   ├── menu.py
   │       │   ├── predictions.py
   │       │   └── rating.csv
   │       ├── middleware.py        <- middleware fucntions
   │       ├── requirements.txt
   │       ├── sample_images        <- directory with sample images for bot
   │       └── tests                <- unit tests for telegram bot
   │           ├── htmlcov          <- directory with html coverage report
   │           ├── __init__.py
   │           └── test_handlers.py
   ├── setup.cfg
   └── setup.py
```


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
| accuracy     |           |          | 0.61     | 12630   |
| macro avg    | 0.57      |  0.57    | 0.55     | 12630   |
| weighted avg | 0.64      |  0.61    | 0.61     | 12630   |

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

### Вывод 1 этапа

Модель, обученая с использованием SIFT'a и мешка слов, судя по метрикам, показала себя несколько лучше. Однако на деле обе модели ошибаются довольно часто и не являются надежными. Желаемого качества предсказаний не удалось достичь, используя линейные модели в ML-подходе.


### Convolutional neural network

![Alt Text](readme_data/cnn_example.png)

Пример архитектуры модели CNN.

Предварительно изображения из тренировочной выборки проходили следующую обработку: случайное выравнивание гистограммы, настройка яркости и контрастности, случайные повороты и отражения изображений (вертикальные и горзинотальные), случайное размытие Гаусса, ресайзились до размера 50x50, переводились в тензорный вид. Далее создается сверточная нейросеть, объявляемая через отельный класс с помощью библиотеки Pytorch. Она состоит из слоя Flatten, слоев Dropout, слоя с активацией ReLU, MaxPooling2D, трех наборов слоев Conv2D с увеличивающимся количеством фильтров вместе с BatchNorm2d и, наконец, Linear слоями.

Тестировались разные варианты слоев сетки, а также менялись гиперпараметры, лучшие метрики на тестовой выборке получились следующими:

|              | precision | recall   | f1-score | support |
| ------------ | --------- | -------- | -------- | ------- |
| accuracy     |           |          | 0.99     | 12630   |
| macro avg    | 0.99      |  0.99    | 0.99     | 12630   |
| weighted avg | 0.99      |  0.99    | 0.99     | 12630   |


### Вывод 2 этапа

Сверточная нейросеть показала крайне высокое качество, как и ожидалось от state-of-the-art решения задачи классификации датасета GTSRB. Данные результаты работы классификаторы мы приннимаем достаточными для инференса. Оставшееся время до чекпоинта мы потратили на упаковку сервисов в контейнеры и оформление репозитория.


## API, tg bot

Взаимодействие с полученными моделями реализовано двумя способами: веб-интерфейс FastAPI и Телеграм бот. 

На данный момент оба модуля упакованы в два докер контейнера. Телеграм бот отвечает на запросы пользователя, используя функционал сервиса FastAPI.

![Alt Text](readme_data/workflow.png)

Ниже на гифках краткая демонстрация функционала веб-интерфейса и Телеграм бота.

Веб-интерфейс FastAPI

![Alt Text](readme_data/fastapi.gif)

Телеграм бот

![Alt Text](readme_data/tg_bot.gif)
    

### Куратор

| Имя | Телеграм | 
|----------|----------|
| Беляев Артем | @karaoke_tutu |


### Авторы проекта

| Имя | Телеграм | 
|----------|----------|
| Булыгин Глеб  | @jdbelg |
| Тихомиров Витя | @onthebox |
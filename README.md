**ML-сервис для поиска и сохранения страниц с текстом**

Схема работы такая:

1. По url (предположительно, ведущей на одну из страниц многочисленных сайтов с текстами книг) собираются все внутренние ссылки с нее на другие страницы сайта.

2. Для каждой из ссылок предобученная модель оценивает возможность, что эта страница не является, к примеру, какой-то промежуточной, а содержит текст (книгу или ее часть). 

3. Для каждый ссылки в базу данных (mongodb) записывается базовая ссылка, с которой на нее пришли, адрес страницы, вердикт модели (или, если страницу открыть не получилось, статус-код ответа). Если текст там обнаружен, он (весь текст страницы, без очистки) сохраняется в виде файла в примонтированную директорию. Имя файла в этом случае также записывается в отдельное поле документа в бд. 

4. Пользователю возвращается краткий отчет: сколько ссылок было найдено, в скольких обнаружен текст, сколько сохранено (последние два числа могут различаться по разным причинам).

5. База данных также примонтирована, результаты ее работы сохраняются.


Все это реализовано в двух отдельных контейнерах (сервер на Flask и база) с помощью docker-compose.


**Модель обучения - Random Forest (sklearn) на фичах, извлеченных их текста страницы + перевод в .onnx формат для хранения. В модели зашиты метаданные (дата ее создания, название эксперимента и хэш коммита в гите, на основе которого производилось создание модели).**

Приложение app.py создает веб-сервис для применения модели (хост, порт и адрес модели - параметры). На запрос типа POST с json-документом, в котором есть поле "url", модель возвращает предсказание. На запрос /metadata возвращаются метаданные. 

Запросы:

**/process_link [тип POST]** : реализует описанный выше алгоритм

**/model/metadata** : метаданные

**/model/forward [тип POST]** : требуется передать json-документ с полем url, результатом будет json-документ с полем 'answer' и его значением 0 или 1 - предсказанием модели

**/model/forward_batch [тип POST]** : требуется передать .csv файл с колонкой url; результатом будет json-документ с полем 'answer' и массивом со значениями {0, 1, None} - последнее в случае, если модель не смогла обработать url

**/model/evaluate [тип POST]** : аналогично, но в .csv файле должно быть поле is_text, а в ответ добавляется поле metrics c метриками по тем url, по которым удалось получить предсказание

Также представлен код самого обучения и выделения признаков (на вход подается только url, признаки строятся по нему с помощью BeautifulSoup). И тесты. 

**Для удобства тестирования последних двух видов запроса приложен файл test.csv, который можно отправлять на сервер. Датасет можно скачать вот [здесь](https://drive.google.com/file/d/1jGdyroDIz3iLT_pbw86IfVwhIW4vHMRs/view?usp=sharing)**

А пример выполнения запроса можно посмотреть в ноутбуке example.ipynb.

**Готовый образ можно взять на dockerhub: masha239/textsearch**



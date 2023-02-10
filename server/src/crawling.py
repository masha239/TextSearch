from pymongo import MongoClient
import os
import constants
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin


dbname = 'Texts'
collection_name_url_texts = 'url_texts'
log_filename = 'log.txt'


def log_str(msg: str):
    dts = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    with open(log_filename, 'a') as f:
        f.write("%s    %s" % (dts, msg))


def get_links(current_link):
    basename = current_link.split('//')[1]
    links = set()

    try:
        data = requests.get(current_link)
        if data.status_code == constants.CODE_OK:
            links.add(current_link)
            soup = BeautifulSoup(data.text, 'html.parser')

            for al in soup.find_all('a'):
                link = al.get('href')

                if link is not None and link.startswith(current_link):
                    links.add(link)

                elif link is not None:
                    join_domen_link = urljoin(current_link, link)
                    if join_domen_link.split('//')[1].startswith(basename):
                        links.add(join_domen_link)
    except BaseException:
        pass

    return list(links)


def save_results_and_texts(links, results, baselink):
    txt_found_cnt = 0
    txt_saved_cnt = 0
    res_dicts = []
    for result, link in zip(results, links):
        res_dict = dict()
        try:
            data = requests.get(link)
            if data.status_code == constants.CODE_OK and isinstance(result[0], dict) and result[0]['answer'] == 1:
                filename = link + ' ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z') + '.html'
                with open(os.path.join('/data/texts', filename)) as f:
                    f.write(data.text)
                res_dict['filename'] = filename
                res_dict['prediction'] = result[0]['answer']
                txt_saved_cnt += 1
                txt_found_cnt += 1
        except BaseException:
            if isinstance(result[0], dict) and result[0]['answer'] == 1:
                txt_found_cnt += 1

        res_dict['status_code'] = result[1]
        res_dict['baselink'] = baselink
        res_dict['link'] = link

        res_dicts.append(res_dict)

    if len(res_dicts) > 0:
        with MongoClient('mongodb',
                         port=27017,
                         password=os.environ.get('MONGO_INITDB_ROOT_PASSWORD'),
                         username=os.environ.get('MONGO_INITDB_ROOT_USERNAME')) as client:

            collection = client[dbname][collection_name_url_texts]
            collection.insert_many(res_dicts)
            log_str(f'INSERTED: {len(links)} from {baselink}\n')
            log_str(f'SAVED: {txt_saved_cnt} from {baselink}\n')
            log_str(f'FOUND: {txt_found_cnt} from {baselink}\n')

    return txt_found_cnt, txt_saved_cnt

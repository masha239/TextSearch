import bs4.element
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import numpy as np
import constants

all_percentiles = [60, 70, 80, 90, 100]

elements = [bs4.element.Comment,
            bs4.element.Doctype,
            bs4.element.NavigableString,
            bs4.element.ProcessingInstruction,
            bs4.element.Script,
            bs4.element.Stylesheet,
            bs4.element.Tag]


def process_url(url):
    try:
        data = requests.get(url)

        if data.status_code == constants.CODE_OK:
            soup = BeautifulSoup(data.text, 'html.parser')
            lens = []
            types = defaultdict(int)
            for child in soup.recursiveChildGenerator():
                types[type(child)] += 1
                if type(child) != bs4.element.Tag:
                    lens.append(len(child))

            percentiles = [np.percentile(lens, q) for q in all_percentiles]
            el_arr = [float(types[el]) for el in elements]

            features = percentiles + el_arr + [float(len(soup.text))]
            return constants.CODE_OK, features
        else:
            return constants.CODE_FAILED, None

    except:
        return constants.CODE_BAD_REQUEST, None


if __name__ == '__main__':
    print(process_url('http://yandex.ru')[1].tolist())
    print(process_url('http://google.com')[1].tolist())
    print(process_url('http://vk.com')[1].tolist())

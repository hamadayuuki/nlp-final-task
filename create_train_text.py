from bs4 import BeautifulSoup
from urllib import request

# 青空文庫のHTMLから文章を取得
def scraping_text_from_aozorabunko(path):

    # URLからHTMLを取得
    #url = "https://www.aozora.gr.jp/cards/000035/files/" + path + ".html"
    url = "https://www.aozora.gr.jp/cards/000148/files/773_14560.html"
    response = request.urlopen(url)
    soup = BeautifulSoup(response)
    main_text = soup.find('div', class_='main_text')
    response.close()

    # ルビの削除
    tags_to_delete = main_text.find_all(['rp', 'rt'])
    for tag in tags_to_delete:
        tag.decompose()

    # 文章のみを抜き出す
    main_text = main_text.get_text()
    # 改行 不要な文字を削除
    main_text = main_text.replace('\r', '').replace('\n', '').replace('\u3000', '')

    return main_text


file_init = open('./source/train_text.txt','w',encoding='utf-8').write("")   # テキストを初期化
#paths = ["1562_14860", "1598_18102", "273_20007", "2254_20134", "42363_15873", "2255_15060", "2256_19985", "306_20009", "304_15063", "277_33098", "278_20016"]
paths = [0]
for path in paths:
    text = scraping_text_from_aozorabunko(path)   # 文章を取得
    write_text = open('./source/train_text.txt','a',encoding='utf-8').write(text)   # 追記
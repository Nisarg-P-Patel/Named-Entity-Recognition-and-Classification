import gdown
file_id = "1-APkqti-bKVe3pKt33uAanPecpUnNYWe"
url = f"https://drive.google.com/uc?id={file_id}"

output = "./model/ner_crf_config.json"

gdown.download(url, output, quiet=False)

file_id = "1-I9O9x-aeNwQzo1Rb5FUJ95-AD3K7w5d"
url = f"https://drive.google.com/uc?id={file_id}"

output = "./model/ner_crf_epoch_5.h5"

gdown.download(url, output, quiet=False)
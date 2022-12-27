import nltk
from nltk.corpus import reuters


def main():
    nltk.download('reuters')
    documents = reuters.fileids()
    aggregated_text = ""
    for document_id in documents:
        text = reuters.raw(document_id)
        aggregated_text += text.lower()
    with open('../aggregated_text.txt', 'w') as f:
        f.write(aggregated_text)



if __name__ == "__main__":
    main()


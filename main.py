# To clean news text
import re

# To successfuly work with neural net
import torch

# To split text by sentences
from razdel import sentenize

# Import transformer neural net for tonality analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# To process results 
from scipy.signal import savgol_filter


def get_text_from_file(filename: str) -> str:
    with open(filename, 'r') as file:
        text = file.read()
    
    return text


def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    return cleaned_text


def split_text_by_sentences(text: str) -> list:
    sentences = []
    for substring in list(sentenize(text)):
        sentences.append(substring.text)

    return sentences


def estimate_tonality(sentences: list, model: AutoModelForSequenceClassification) -> list:
    sentiment_out = []
    for sentence in sentences:
        with torch.no_grad():
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
            sentiment_out.append(proba.dot([-1, 0, 1]))

    return sentiment_out


# if __name__ == '__main__':

# Preparing text for analisys 
text = get_text_from_file('text.txt')
cleaned_text = clean_text(text)
sentences = split_text_by_sentences(cleaned_text)

# Getting model
model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
    model.cuda()

# Getting sentiment for every sentence in text 
sentiments = estimate_tonality(sentences, model)

# Filter noise from sentiments
filtered_sentiments = savgol_filter(sentiments, window_length=len(sentiments) // 15, polyorder=0) 

# for i in range(10):
#     print(f'{sentences[i]} TONALITY: {sentiments[i]}')
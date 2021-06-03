import sys

import statistics as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from textblob import TextBlob
import scrapy
from scrapy.crawler import CrawlerProcess
import csv
import pandas as pd

stars_list = []
comment_list = []
date_list = []

# We will receive product name and product category here
arguments = sys.argv

# print(arguments[1:])
# exit()


class AmazonReviewsSpider(scrapy.Spider):
    output = "scrapy_output.csv"

    def __init__(self):
        open(self.output, "w").close()

    name = 'amazon_reviews'

    allowed_domains = ['amazon.in']
    ##
    # We can have another URL and decide which URL to use to scrape reviews based on the product name
    ##
    myBaseUrl = "https://www.amazon.co.uk/New-Apple-iPhone-12-128GB/product-reviews/B08L5QVFCT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber="
    start_urls = []

    # Creating list of urls to be scraped by appending page number a the end of base url
    for i in range(1, 121):
        start_urls.append(myBaseUrl+str(i))

    def parse(self, response):
        data = response.css('#cm_cr-review_list')

        star_rating = data.css('.review-rating')

        review_date = data.css('.review-date')

        comments = data.css('.review-text')
        count = 0

        with open(self.output, "a", newline="") as f:
            writer = csv.writer(f)
            for review in star_rating:
                date = ''.join(review_date[count].xpath('.//text()').extract())
                if date != '':
                    date = date.split()
                    date = date[-3:]
                    date_list.append(date)
                stars = ''.join(review.xpath('.//text()').extract())
                if stars != '':
                    stars_list.append(stars)
                comment = ''.join(comments[count].xpath(".//text()").extract())
                if comment != '':
                    comment_list.append(comment)
                writer.writerow([comment])
                yield{'Stars': stars,
                      'Comment': comment,
                      'Date': date,
                      }
                count = count+1


process = CrawlerProcess()
process.crawl(AmazonReviewsSpider)
process.start()
len(comment_list)


cleaned_reviews = []

# To track which review the sentence belongs to
review_Number = []
reviewCount = 0


for review in comment_list:
    temp = review.strip().split('.')

    for sentence in temp:
        if(sentence != ''):
            cleaned_reviews.append(sentence)
            review_Number.append(reviewCount)

    reviewCount += 1


# print(len(cleaned_reviews))
cleaned_reviews


nlp = spacy.load("en_core_web_sm")
req_tag = ['NN']
# Product category will decide which dictionary will be used

myDict = ['product', 'mobile', 'fone', 'iphone', 'iphone12', 'phone', 'smartphone', 'device', 'fone', 'spec',
          'quality', 'performance', 'performer',
          'screen', 'display', 'ratio', 'visibility', '60fps',
          'storage', 'memory',
          'size', 'build', 'bulkier',
          'functionality', 'operating', 'software', 'technology', 'working', 'upgrade',
          'support',
          'browsing', 'internet', 'browser',
          'appearance', 'design', 'color', 'colour',
          'price', 'payment',
          'battery', 'life', 'power',
          'camera', 'pixel', 'videocall', 'shot',
          'charger', 'charging', 'charge', 'adapter', 'recharge', 'usb', 'lightning',
          'network', 'signal', 'reception', 'wifi',
          'sound', 'stereo', 'headphone', 'jack',
          'security', 'face', 'recognition', 'print',
          'reliability',
          'weight',
          'touch', 'typing', 'response',
          'torch', 'flashlight',
          'experience']

aspect_brokenSentences = []
brokenSentences = []

sentenceNumber = 0
# To keep track of the review each text belongs to
Sentence = []

index_of_reviewNumber = 0
# review_Number
review_Number_List = []
reviewIndex = 0


feature_dict = ['product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product',
                'quality', 'quality', 'quality',
                'screen', 'screen', 'screen', 'screen', 'screen',
                'storage', 'storage',
                'size', 'size', 'size',
                'functionality', 'functionality', 'functionality', 'functionality', 'functionality', 'functionality',
                'support',
                'browsing', 'browsing', 'browsing',
                'appearance', 'appearance', 'appearance', 'appearance',
                'price', 'price',
                'battery', 'battery', 'battery',
                'camera', 'camera', 'camera', 'camera',
                'charger', 'charger', 'charger', 'charger', 'charger', 'charger', 'charger',

                'network', 'network', 'network', 'network',
                'sound', 'sound', 'sound', 'sound',
                'security', 'security', 'security', 'security',
                'reliability',
                'weight',
                'touch', 'touch', 'touch',
                'torch', 'torch',
                'experience']


for x in cleaned_reviews:

    sentence_aspect = []
    doc = nlp(x)
    firstIndex = 0
    # myDict=[product]

    for token in range(len(doc)):

        if doc[token].tag_ in req_tag and doc[token].shape_ != 'x' and doc[token].shape_ != 'xx' and doc[token].shape_ != 'xxx':
            if((doc[token].lemma_).lower() in myDict):
                # Add the broken text in one list and the aspect in another
                index_in_map = myDict.index((doc[token].lemma_).lower())
                brokenSentences.append(str(doc[firstIndex:token + 1]))
                aspect_brokenSentences.append(feature_dict[index_in_map])
                Sentence.append(sentenceNumber)
                review_Number_List.append(reviewIndex)

                firstIndex = token+1

    brokenSentences.append(doc[firstIndex:len(doc)].text)
    aspect_brokenSentences.append(".")
    Sentence.append(sentenceNumber)
    review_Number_List.append(reviewIndex)
    sentenceNumber += 1
    if(sentenceNumber != len(review_Number) and review_Number[sentenceNumber] != review_Number[sentenceNumber-1]):
        reviewIndex += 1
#    print(brokenSentences)
#    print(aspect_brokenSentences)
    index_of_reviewNumber += 1
# print("done")


df = pd.DataFrame(columns=['TEXT', 'ASPECT', 'Sentence', 'Review'])
pd.set_option('max_colwidth', -1)
df['TEXT'] = brokenSentences
df['ASPECT'] = aspect_brokenSentences
df['Sentence'] = Sentence
df['Review'] = review_Number_List
df
# df['Reviews'] = cleaned_reviews
# df['Final Reviews'] = df['Reviews'].str.lower()


count = 0

for element in aspect_brokenSentences:
    if element != '.':
        count += 1
count


# Vader


output = pd.DataFrame(columns=['TEXT', 'ASPECT', 'Positive',
                               'Negative', 'Neutral', 'Sentence Number', 'Review Number'])

pos = []
neu = []
neg = []

analyzer = SentimentIntensityAnalyzer()

for sentence in brokenSentences:
    vs = analyzer.polarity_scores(sentence)
    pos.append(vs['pos'])
    neu.append(vs['neu'])
    neg.append(vs['neg'])

output['TEXT'] = brokenSentences
output['ASPECT'] = aspect_brokenSentences
output['Sentence Number'] = Sentence
output['Review Number'] = review_Number_List
output['Positive'] = pos
output['Neutral'] = neu
output['Negative'] = neg

# output.to_csv('output_sentiment_vader.csv')
output


# TextBlob

output_textblob = pd.DataFrame(columns=[
                               'TEXT', 'ASPECT', 'Polarity', 'Subjectivity', 'Sentiment', 'Sentence Number', 'Review Number'])

pos_count = 0
neg_count = 0

polarity = []
subjectivity = []
sentiment = []

for sentence in brokenSentences:
    analysis = TextBlob(sentence)

    polarity.append(analysis.sentiment.polarity)
    subjectivity.append(analysis.sentiment.subjectivity)
    sentiment.append(
        'Negative' if analysis.sentiment.polarity < 0 else 'Positive')
    if analysis.sentiment.polarity < 0:
        neg_count += 1
    else:
        pos_count += 1

output_textblob['TEXT'] = brokenSentences
output_textblob['ASPECT'] = aspect_brokenSentences
output_textblob['Sentence Number'] = Sentence
output_textblob['Review Number'] = review_Number_List
output_textblob['Sentiment'] = sentiment
output_textblob['Polarity'] = polarity
output_textblob['Subjectivity'] = subjectivity

# output_textblob.to_csv('output_sentiment.csv')
output_textblob


skip = False
sentenceStartIndex = 0
sentenceNumber = 0
for i in range(len(Sentence)):

    if(Sentence[i] != sentenceNumber):

        if(not skip):

            # Insert vader's score if the sentiment for entire sentence is not present
            for j in range(sentenceStartIndex, i):

                if((pos[j]-neg[j]) > 0):
                    print(sentenceNumber)
                    polarity[j] = pos[j]
                elif((pos[j]-neg[j]) < 0):
                    polarity[j] = neg[j]
                    print(sentenceNumber)

        skip = False
        sentenceStartIndex = i
        sentenceNumber = Sentence[i]

    if(polarity[i] != 0):

        skip = True


positiveScore = []
negativeScore = []
for score in polarity:
    if score > 0:
        positiveScore.append(score)
    else:
        negativeScore.append(score)


revisedScore = []
forward = False
backward = False
previousIndex = -1
count_of_backward_propagation = 0
count = 0
for index in range(len(Sentence)):

    if(Sentence[index] != previousIndex):
        previousIndex = Sentence[index]
        if(count_of_backward_propagation > 0):
            for i in range(count_of_backward_propagation):
                count += 1

                revisedScore.append(polarity[index-1])
            count_of_backward_propagation = 0
        if(polarity[index] > -0.1 and polarity[index] < 0.1):

            count_of_backward_propagation = 1

            forward = True
            backward = False
            continue
        else:

            count_of_backward_propagation = 0
            backward = True
            forward = False
            count += 1

            revisedScore.append(polarity[index])
            continue
    else:
        if(polarity[index] > -0.1 and polarity[index] < 0.1):

            if(forward):
                count_of_backward_propagation += 1

            else:
                revisedScore.append(revisedScore[len(revisedScore)-1])
                count += 1

        else:

            if(forward):
                for i in range(count_of_backward_propagation):
                    count += 1

                    revisedScore.append(polarity[index])
                revisedScore.append(polarity[index])
                count_of_backward_propagation = 0
            else:
                if(not(polarity[index] > -0.1 and polarity[index] < 0.1)):
                    count += 1

                    revisedScore.append(polarity[index])
                    continue

                count += 1

                revisedScore.append(revisedScore[len(revisedScore)-1])


output_textblob = pd.DataFrame(columns=['TEXT', 'ASPECT', 'Polarity', 'Revised Polarity',
                                        'Subjectivity', 'Sentiment', 'Sentence Number', 'Review Number'])

pos_count = 0
neg_count = 0

polarity = []
subjectivity = []
sentiment = []

for sentence in brokenSentences:
    analysis = TextBlob(sentence)

    polarity.append(analysis.sentiment.polarity)
    subjectivity.append(analysis.sentiment.subjectivity)
    sentiment.append(
        'Negative' if analysis.sentiment.polarity < 0 else 'Positive')
    if analysis.sentiment.polarity < 0:
        neg_count += 1
    else:
        pos_count += 1

output_textblob['TEXT'] = brokenSentences
output_textblob['ASPECT'] = aspect_brokenSentences
output_textblob['Sentence Number'] = Sentence
output_textblob['Review Number'] = review_Number_List
output_textblob['Sentiment'] = sentiment
output_textblob['Polarity'] = polarity

output_textblob['Subjectivity'] = subjectivity
output_textblob['Revised Polarity'] = revisedScore
output_textblob
output_textblob.to_csv('output_sentiment_revised.csv')


previousIndex = 0
aspects = []
scores = []
temp_aspects = []
temp_score = []
for index in range(len(review_Number_List)):
    if(review_Number_List[index] != previousIndex):

        aspects.append(temp_aspects)
        scores.append(temp_score)
        previousIndex = review_Number_List[index]
        temp_aspects = []
        temp_score = []
    if(aspect_brokenSentences[index] != '.'):
        # If one aspect is mentioned more than 1 times
        index_of_repeatedAspect = -1
        for aspect in temp_aspects:
            if(aspect == aspect_brokenSentences[index]):
                index_of_repeatedAspect = temp_aspects.index(aspect)
        if(index_of_repeatedAspect != -1):

            # retain the bigger absolute score
            if(abs(revisedScore[index]) > abs(temp_score[temp_aspects.index(aspect_brokenSentences[index])])):
                temp_score[temp_aspects.index(
                    aspect_brokenSentences[index])] = revisedScore[index]

        else:
            temp_aspects.append(aspect_brokenSentences[index])
            temp_score.append(revisedScore[index])


for i in range(len(aspects)):

    for j in range(len(aspects[i])):
        print(aspects[i][j], ":", scores[i][j], ":", i)

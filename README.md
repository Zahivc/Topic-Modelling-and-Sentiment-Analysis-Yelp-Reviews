# Topic-Modelling-and-Sentiment-Analysis-Yelp-Reviews
Provide more aspects of Yelp's reviews ratings by applying LDA topic modelling and VADER sentiment analysis techniques


1. INTRODUCTION

Yelp is a business directory service that hosts crowd sourced reviews of businesses founded in 2004, San Francisco. The types of businesses include restaurants, cafes, health centres, retail outlets. etc. Yelp is business-to-consumer application where Yelp users can submit reviews using a rating system. Moreover, users can submit written reviews about the businesses that helps other users make informed decision about products/ services. Our focus, through the course of this project, will be restaurants and the F&B businesses and study reviews and the nature of reviews. 

Yelp users and businesses encounter several pain points that can be addressed with text mining solutions. Firstly, on Yelp, businesses only have 1 consolidated rating, but no rating into the specific aspects of the business, such as Ambience, service or food etc. This means that as a user, I will not be able to filter reviews relevant to the aspect or find out specifics to the restaurant. As a business, it is difficult for me to understand which aspects the business can improve on when they have a low rating. Therefore, the motivation is to alleviate these pain points with topic modelling and sentiment analysis.

2. BUSINESS USE CASE

There are two end objectives that our team wish to achieve:

a.	Detailed Rating System

With the current consolidated business rating, there is no visibility into specific aspects based on the 1-5 rating scale of the rating. Using sentiment analysis to evaluate positive and negative sentiments, we can then score each review and label the review under the specific aspect with topic modelling, therefore giving the average aspect score for each aspect. This can then aid users and businesses to understand the business better.

b.	Reviews filtering by Aspects

A user currently must scroll through all reviews to understand the business, with some businesses having up to thousands of reviews. However, as a user, you may be only interested in specific aspects of the business, such as whether it has good service or food. With topic modelling via Latent Dirichlet Allocation (LDA), we then assign a topic to the review based on the distribution and frequency of the words in the review. This allows the reviews to be filtered by its dominant topic, allowing the user to find specific reviews that he may be interested in.

3. SOLUTION OVERVIEW

To achieve our business task, we plan to explore several techniques in order to achieve the best results. We have therefore used several models as comparison in each of our tasks. We first start off with pre-processing, followed by 3 different type of LDA models for topic extraction, of which include Gensim, Scikit-learn and Mallet. From sentiment analysis, we have done a standard version with a predetermined set of positive and negative lexicons, while comparing it to VADER. These models are then evaluated via several tests to determine the best model, ultimately combining to give the reviews a topic name and score, allowing us to evaluate the business from each aspect, improving Yelp’s usability and convenience.


 
Figure 1: Solution Overall Design

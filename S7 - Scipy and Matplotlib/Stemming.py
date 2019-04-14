# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:04:28 2018

@author: manoj
"""
import nltk
from nltk.stem import PorterStemmer
paragraph="""The current state of the SaaS market
Indicators across the SaaS marketplace and research from industry analysts all point to the fact that the SaaS market is maturing and slowing in terms of growth. There is still room for continued adoption of SaaS, particularly among industries that have been slow on the uptake (healthcare and manufacturing are just two examples), but the explosive growth we have seen previously has slowed.A report from IDC shows that the SaaS segment, which makes up 68.7 percent of overall cloud market share, was the slowest-growing segment of the cloud market with a 22.9 percent year-over-year growth rate last year.Venture capital funding, usually an indicator of how “hot” investors view a market to be, has been found to be on the decline when it comes to SaaS startups. TechCrunch attributed this in large part to market saturation and the fact that those newbies seeking funding are often trying to compete with large, established players.While the SaaS market is still expected to show some growth, it’s expected to be at a lower rate than we’ve seen previously. SaaS markets are maturing and it seems that those who are going to come out as winners will need to be onto the next “big thing.”
Big players incorporating AI
Besides Amazon and AWS, other big players in the market are announcing offerings that integrate AI. This is perhaps another solid indicator that AI and machine learning could be the next step in differentiating a SaaS and helping it to carve out a space in the market.Oracle just recently announced that it is infusing its entire lineup of SaaS applications with AI and machine learning. They state that thousands of businesses across every industry are now expecting a new generation of AI-infused apps to “trigger unprecedented levels of speed in innovation, in the ability to disrupt competitors or even entire industries, and in the ability to adapt to and get in front of rapidly shifting market dynamics.”SaaS makes up the biggest proportion of Oracle’s cloud-based services, accounting for about 74% of their cloud revenue. They are the world’s second-largest provider of SaaS apps behind Salesforce, although they are jockeying to take over the number one spot.Their new breed of AI-infused apps are designed to create better, more personalized experiences for users as the machine learning component allows apps to learn as they are used. They believe that there is room for AI across all segments of their SaaS lineup.If these larger companies provide a good indicator, we’re about to see more integration of AI and machine learning in the SaaS world, as other companies seek to carve out a position for themselves.
Pricing
Gartner released a report last year predicting that AI will drive big changes to the pricing models of SaaS. More customers deploying their own AI means fewer employees using the software, putting the pricing model of charging per named user under pressure.An example they give is that of an ecommerce firm cutting their staffing numbers and deploying chatbots to pick up some (not all) of their customer service duties. The prediction from Gartner is that SaaS cannot charge for chatbots the same way they would for named users. This means that SaaS would be in danger of losing most of their users and a pricing rethink would be needed. Gartner further predicts that by 2025, 40% of software that currently operates pricing on a per-user basis will transition """

#Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)

stemmer = PorterStemmer()
for i in range(len(sentences)):
    words=nltk.word_tokenize(sentences[i])
    words=[stemmer.stem(word) for word in words]
    sentences[i] = ' '.join(words)
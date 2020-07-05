## Sentiment Analysis
This is used to classify the tweets whether they are positive or negative. It is basically solved using logistic regression technique.
### Dataset: twitter_samples imported from Natural Language Toolkit(**nltk**)
- It consists of 5000 positive sentiments tweets and 5000 negative sentiments tweets. Out of which 4000 each positive and negative tweets are used for training while the remaining tweets used for testing purpose.
### Data Preprocessing
- process_tweet function: It is used to remove all the hashings, urls, stopwords, punctuations. It further replaces the words with their stem words. It finally tokenizes the processed tweet.for example
> This is an example of a positive tweet: 
> #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my >community this week :)

>This is an example of the processed version of the tweet: 
> ['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']
- build_freqs function : It calculates the frequency of each word present in the positive and negative processed tweet corpus. 
> The key is the tuple (word, label), such as ("happy",1) or ("happy",0). The value stored for each key is the count of how many times the word "happy" was associated with a positive label, or how many times "happy" was associated with a negative label.

###### Extracting Features
- Each word in the tweet has three features namely the bias(1),its posivite ferquency and its negative frequency which can be extracted from the build_freqs dictionary. Since each word has three features, there will be three parameters which needs to be trained.
###### Loss Function
- Since its a logistic regression task, cross entropy loss has been used to calculate the loss.
###### Optimiser
- Regular Gradient Descent algorithm has been used.

#### Test Accuracy : **99.5%**

##### Error Analysis 
- There are some tweets which are wrongly predicted. On running the code, one can see all those twwets which are wrongly classified
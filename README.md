# Classifying Reddit Posts
- [Problem statement](#problem-statement)
- [Choosing subreddits](#choosing-subreddits)
    - [Defining positive & negative classes](#defining-positive--negative-classes)
    - [Baseline model](#baseline-model)
- [Top-level results](#top-level-results)
- [Navigating this repo](#navigating-repo)
- [Exploring and cleaning data](#exploring-and-cleaning-data)
    - [Missing `selftext`](#missing-selftext)
- [Transformers](#transformers)
    - [PorterStemmer](#porterstemmer)
    - [Vectorizing data](#vectorizing-data)
- [Data dictionary](#data-dictionary)
- [Modeling](#modeling)
- [Interpreting results](#interpreting-results)


## Problem statement
Implement a text transfomer and classification model on subreddit `selftext` to distinguish one subreddit from another.


## Choosing subreddits
I chose subreddits that I thought would have some similar terms, so the model wouldn't find it too easy to distinguish between them. Initially, I chose "AskScience" and "Space" but I found that there were not a lot of new posts in either of them. I added the "AskScienceFiction" subreddit to increase the number of documents and that had more posts than either of the others.

I still had some trouble collecting enough data to make a decent model. I considered pulling comments as well as posts but decided in the end that I was getting decent results without them.

After collecting posts for 10 days my three subreddits had produced 2,891 posts in total:
| subreddit | count | percentage |
| --- | ---: | ---: |
| AskScienceFiction | 1,197 | 41.4% |
| space | 957 | 33.1% |
| askscience | 737 |  25.5% |

### Defining positive & negative classes
The **positive class** for my model was the subreddit with the most posts: AskScienceFiction. Posts from the space and askscience subreddits were collectively the **negative class**.

### Baseline model
41.4% of the posts were in the positive class, 58.6% in the negative class.

The baseline is therefore 58.6%.

## Top-level results
The best results came from the TfidfVectorizer and MultinomialNB model:
|  | score |
| -- | ---: |
| balanced accuracy score | 0.904701 |
| f1_score | 0.890323 |
| recall | 0.862500 |
| precision | 0.920000 |

[return to top](#classifying-reddit-posts) --- jump to [testing models](#testing-models)

## Navigating this repo
### Jupyter notebooks
- [reddit data pull](/notebooks/api_pull_posts.ipynb)
- [EDA](/notebooks/EDA.ipynb)
- [modeling & analysis](/notebooks/model_trials.ipynb)

### Data files
Data files are in the [data folder](/data/)
- "raw_posts.csv": the posts pulled from reddit, all three subreddits are in this file
- "neg_class_vocab.csv": words and word-count for posts in the space & askscience subreddits
- "pos_class_vocab.csv": words and word-count for posts in the AskScienceFiction subreddit

### Python files
- [reddit_utilities.py](reddit_utilities.py) has functions I used to pull posts from reddit and extract post data from the praw ListingGenerator objects.
- [modeling_reporting.py](modeling_reporting.py) has some functions I used in EDA and others for running GridSearch, fetching fitted pipeline from pickle files and some to collect metrics and params from the fitted models.


## Exploring and cleaning data
As described above, these three subreddits were frustratingly idle. In addition, their `selftext` was not particluarly wordy and a fairly large number had only an image or a url in that field. Anything other than free-form text was considered null for `selftext`.

### Missing `selftext`
- All of the subreddits had hundreds of posts without any text in the `selftext` field.
    | subreddit | total posts collected | empty selftext | percent empty |
    | ---- | ---: | ---: | ---: |
    | askscience | 737 | 115 | 15.6% |
    | AskScienceFiction | 1,197 | 298 | 24.8% |
    | space | 957 | 784 | 81.9%|
    
- Where `selftext` was empty, I copied the `title` field into `selftext` and used that instead.
- Where `selftext` was not empty, `title` was not used in the model.

## Transformers

### Words in the positive class
![positive class word cloud](/images/pos_class_word_cloud.png)

### Words in the negative class
![negative class word cloud](/images/neg_class_word_cloud.png)

### Vectorizing data
See the [EDA notebook](/notebooks/EDA.ipynb) for initial analysis of the data vectorized.

![top-10 words](/images/top_10_words.png)

### Custom stop_words
I created a custom list of stop-words to test in the pipeline/modeling step, based on the most common words that occured after using the CountVectorizer (see chart above).
- uninformative words such as "https", "wiki" or "question"
- overly-informative words such as "space" 
- special characters like `x200b` (a "zero-width space" [source](https://www.codetable.net/hex/200b))

[return to top](#classifying-reddit-posts) --- jump to [words in mis-classified posts](#words-in-mis-classified-posts)

## Testing models
I tested eight pipelines and models:
| model # | Transformer | Over-sampler | Classifier | balanced accuracy score |
| --: | --- | --- | --- | ---: |
| 1 | CountVectorizer | none | LogisticRegression | 0.886560 |
| 2 | CountVectorizer | none | MultinomialNB | 0.900461 |
| 3 | CountVectorizer | none | RandomForestClassifier | 0.865560 |
| 4 | TfidfVectorizer | none | LogisticRegression | 0.894285 |
| 5 | TfidfVectorizer | none | MultinomialNB | 0.904701 |
| 6 | TfidfVectorizer | none | RandomForestClassifier | 0.855494 |
| 7 | TfidfVectorizer | ADASYN | LogisticRegression | 0.897069 |
| 8 | TfidfVectorizer | ADASYN | MultinomialNB | 0.899779 |
| 9 | TfidfVectorizer | ADASYN | GaussianNB | 0.855162 |


### GridSearch
I tried a number of variations for the vectorizers and models. Observations:
- In general, the best scorer in every pipeline used the default values for the vectorizer:
    - no stop-words
    - `ngram_range` to (1,1), meaning individual words were more impactful on the outcome than words in proximity to other words
    - `max_df` and `min_df` both set to 1
- The highest `C` setting (10) was the most successful for the LogisticRegression models
- Although the CountVectorizer & the TfidfVectorizer seemed to produce very similar results, the results have some differences because the results of model #1 and model #4 were different, even though the settings on the LogisticRegression model were the same. Also, the grid search found better scores with *different* settings on the MultinomialNB classifier in model #2 and #5.

See the [model_trials](/notebooks/model_trials.ipynb) notebook for this code.

#### Scores for all nine models
![all models' scores](/images/nine_models_scores.png)

## Interpreting results
The MultinomialNB model was the best performer with both types of vectorizers and the version that included the ADASYN over-sampler had the highest recall score of any model.

![MultinomialNB confusion matrix](/images/confusion_matrix.png)

There are two subreddits in the negative class. Space posts make up 56.4% of all the posts in the negative class but only 11.1% of the false positives. Askscience is the other subreddit and it's 43.4% of the posts in the negative class and 88.9% of the false positives.

The space subreddit had, by far, the largest percentage of posts that didn't have any useable `selftext`, which means the model used only the title of the post. Possibly, the words in the titles are more impactful.

## Words in mis-classified posts
I looked at the mis-classified posts to see if I could figure out why they were mis-classified. I checked the raw, un-vectorized data in those errors agains the words that made it into the vectorized features and found that the vast majority of the words in those mis-classified posts were not in the vectorized posts.

![words in vectorized data for errors](/images/words_in_errors.png)

Looking at this chart, I thought maybe the fact that over 87% of the words in the posts were not even in the vectorized word lists might explain the errors. But it turns out the chart for words that were correctly classified looks almost exactly the same.

Both vectorizers discarded words that were in too few or too many posts so it's possible that accounts for the difference.


## Future work
- For the models that used the highest value I included in the grid-search params, run the grid search again with higher values. For example, the LogisticRegression models that were most successful had `C=10`, when the other options I tried were `C=1.0` and `C=0.1`.
- Since the space subreddit had the smallest proportion of mis-classified posts and a lot of its `selftext` was actually the `title` of the post, test the same model using only the posts' `title` instead of `selftext` to see if that makes it better or worse.
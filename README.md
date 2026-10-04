# Social Media Toxicity Detection

Classification of toxic and offensive social media posts compared to neutral posts using `comments.csv.gz`. In the notebook, I encode social media posts using TF-IDF and train a regularized Logistic Regression classifier to separate hate speech and offensive language from neutral posts.

## Data and preprocessing

The dataset contains 24,783 annotated Twitter posts and replies (19,190 `offensive_language`, 1,430 `hate_speech`, and 4,163 `neither`; Davidson et al., 2017). I combine `offensive_language` and `hate_speech` into a binary target `is_toxic` (1 for Toxic, 83.2%; 0 for Neutral, 16.8%). Because the source tweets were collected using Hatebase keywords, neutral posts form the minority, while raw posts frequently contain `@mentions` (56.6% toxic, 60.7% neutral), `RT` prefixes (28.6% toxic, 30.0% neutral), and external links (10.0% toxic, 22.4% neutral).

Before vectorizing the text, I decode HTML entities, expand English contractions (keeping negation words such as `not`, `no`, `nor`, and `never`), strip URLs, handles, ordinals, and punctuation, filter standard stopwords, and reduce inflected words to base forms. Dropping empty or duplicate cleaned entries removes 1,239 rows and leaves 23,544 posts (18,835 training and 4,709 test in an 80/20 stratified split). Text is encoded with `TfidfVectorizer` (unigrams up to 15,000 features for the baseline; unigrams and bigrams up to 20,000 features with `min_df=3` and `sublinear_tf=True` for the tuned model).

## Models and results

I compared an unweighted unigram Logistic Regression baseline against a tuned bigram L2-regularized Logistic Regression on the 80/20 split, using 5-fold stratified cross-validation scored on Macro F1 to select C = 2.0 and `class_weight` = {0: 3, 1: 1}:

| Model | CV Macro F1 | CV ROC-AUC | Test Accuracy | Neutral Recall | Toxic Recall | Test Macro F1 | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline Logistic Regression | 0.897 | 0.980 | 0.948 | 0.804 | 0.978 | 0.905 | 0.981 |
| Tuned Logistic Regression | 0.919 | 0.981 | 0.955 | 0.932 | 0.960 | 0.924 | 0.983 |

Adjusting class weights raises Neutral recall from 0.804 to 0.932 without hurting Toxic recall (0.960), giving 0.955 test accuracy and 0.983 ROC-AUC. Recall across the three original categories reaches 0.966 on `offensive_language`, 0.932 on `neither`, and 0.878 on `hate_speech`.

Test error analysis (55 false positives and 157 false negatives) reflects both corpus sampling and bag-of-words limits: benign nouns from the Hatebase seed list (`yankee`, `bird`, `charlie`, `trash`, `oreo`) receive negative coefficients, which can mask coded phrasing (such as `"Foxtrot Uniform Charlie Kilo"`). Because preprocessing strips user handles and this snapshot lacks reply-thread links and timestamps, tracking communication flows between users who make toxic posts would require conversational graph data.

## Usage

Files:
- `social_media_toxicity_detection.ipynb`
- `comments.csv.gz`
- `requirements.txt`

```bash
pip install -r requirements.txt
jupyter notebook
```

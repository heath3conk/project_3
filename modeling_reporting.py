"""_summary_
"""

import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score



CUSTOM_STOPS = ["https", "did", "does", "just", "like", "question", "space", "know", "ve", "don", "com", "www", "questions", "think", "x200b",
                "youtube", "wiki"] 


def store_metrics(X_test: pd.DataFrame, y_test: pd.Series, scores_df: pd.DataFrame, model, label: str) -> pd.DataFrame:
    """
    took this from lesson 2.14

    Args:
        y_test: actual values for test data
        preds: predictions for test data

    Returns:
        dict with metrics on the given set of predictions & actual data
    """
    preds = model.predict(X_test)
    
    scoring_functions = [("balanced_accuracy", balanced_accuracy_score(y_test, preds)), 
                         ("f1_score", f1_score(y_test, preds)),
                        #  ("roc_auc_score", roc_auc_score(y_test, preds)), this is = to `balanced_accuracy`
                         ("recall", recall_score(y_test, preds)),
                         ("precision", precision_score(y_test, preds))]
    metrics = []
    
    for pair in scoring_functions: 
        score = {
            "model_name": label,
            "score_type": pair[0],
            "score": pair[1]
        }
        metrics.append(score)
    new_scores_df = pd.DataFrame.from_dict(metrics)
    
    return pd.concat([scores_df, new_scores_df])


def store_params(model, pipe_params, label, params_dict):
    model_params = model.get_params()
    params_selected = { key: model_params[key] for key in list(model_params.keys()) if key in list(pipe_params.keys())}
    params_dict[label] = params_selected
    return params_dict


def train_save_best_model(pipe: Pipeline, pipe_params: dict[str, any], X_train: pd.DataFrame, y_train: pd.Series, file_path: str):
    gs = generate_gs(pipe, pipe_params)
    gs.fit(X_train, y_train)
    with open(file_path, "wb") as f:
        pickle.dump(gs.best_estimator_, f)
    print(f"saved model to {file_path}")
    return gs


def fetch_fitted_pipeline(file_path: str) -> Pipeline:
    with open(file_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
    

def generate_gs(pipe_tuples: list[tuple], pipe_params: dict[str, any]) -> GridSearchCV:
    select_params = {}
    for stage in pipe_tuples:
        for key, value in pipe_params.items():
            if stage[0] in key:
                select_params[key] = value
    pipeline = Pipeline(pipe_tuples)
    return GridSearchCV(pipeline, param_grid=select_params, cv=5, verbose=1, scoring="balanced_accuracy", n_jobs=-1)
    


def custom_stops() -> list[str]:
    """
    Returns:
        list of stop words that includes the "english" stop words from CountVectorizer and the words in the
        constant `CUSTOME_STOPS` in this file
    """
    stops = list(CountVectorizer(stop_words="english").get_stop_words())
    return stops.extend(CUSTOM_STOPS)



def find_duplicates(df: pd.DataFrame) -> dict[str: int]:
    """
    Returns:
        dict of subreddits that have duplicate values in the "name" field or empty dict if none have duplicates
    """
    dupes = {}
    for subreddit in df["subreddit"].unique():
        dupe_count = df.loc[df["subreddit"] == subreddit]["name"].duplicated().sum()
        if dupe_count > 0:
            dupes[subreddit] = dupe_count
    return dupes


def find_null_selftext(df: pd.DataFrame) -> dict[str, str]:
    """
    Args:
        df: pandas DataFrame with reddit post data

    Returns:
        dict whose keys are the subreddits found in df and values are the count of rows with null in the "selftext" column
    """
    nulls = {}
    for subreddit in df["subreddit"].unique():
        nulls[subreddit] = df.loc[df["subreddit"] == subreddit]["selftext"].isna().sum()
    return nulls


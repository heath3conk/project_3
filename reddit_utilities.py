"""
utility functions used in multiple notebooks that streamline fetching reddit data, extracting relevant fields from posts 
and exploring the data

"""

import praw
import pandas as pd 


def get_more_recent_posts(df: pd.DataFrame, reddit_conn: praw.reddit, subreddits: list[str]) -> list[praw.models.ListingGenerator]:
    """
    Args:
        df: data previously-collected from reddit
        reddit_conn: authenticated praw.reddit object
        subreddits: list of subreddits

    Returns:
        list of praw.models.ListingGenerator objects with data ready be extracted
    """
    posts = []
    for subreddit in subreddits:
        latest_name = max(df.loc[df["subreddit"] == subreddit]["name"])
        posts.append(reddit_conn.subreddit(subreddit).new(limit=None, params={"before": latest_name}))
    return posts



def get_earlier_posts(df: pd.DataFrame, reddit_conn: praw.reddit, subreddits: list[str]) -> list[praw.models.ListingGenerator]:
    """
    Args:
        df: data previously-collected from reddit
        reddit_conn: authenticated praw.reddit object
        subreddits: list of subreddits

    Returns:
        list of praw.models.ListingGenerator objects with data ready be extracted
    """
    posts = []
    for subreddit in subreddits:
        earliest_name = min(df.loc[df["subreddit"] == subreddit]["name"])
        posts.append(reddit_conn.subreddit(subreddit).new(limit=None, params={"after": earliest_name}))
    return posts



def extract_posts(praw_results: list[praw.models.ListingGenerator]) -> list[dict[str: any]]:
    """
    Args: 
        praw_results: list of praw.models.ListingGenerator objects with data ready be extracted
    Returns:
        list of dicts, one for each post in the reddit data
    """
    # takes a list of praw's ListingGenerator object & extracts the fields I want
    # constructs a dict for each post & returns a list of the dicts
    posts_list = []
    for result in praw_results:
        for post in result:
            posts_list.append({
                "title": post.title,
                "selftext": post.selftext,  # if the body of the post is just a url, this will be empty
                # "url": post.url,  # this gives you the url to the post, not the url IN the post
                "subreddit": post.subreddit,
                "created_utc": post.created_utc,
                "name": post.name,
                "type": "post"
            })
    return posts_list



def extract_comments(praw_generator: praw.models.ListingGenerator) -> list[dict[str:any]]:
    """
    Args:
        praw_generator: a praw.models.ListingGenerator object with data ready be extracted
    Returns:
        comments_list: list of dicts, one for the first 20 comments attached to each post
    """
    comments_list = []
    for post in praw_generator:
        # post.comments.replace_more(limit=None)
        for comment in post.comments.list()[:5]:  # limiting how many comments to collect on each post; this list appears to be sorted by "best"
            comments_list.append({
                "title": post.title,
                "selftext": comment.body,
                "created_utc": post.created_utc,
                "subreddit": post.subreddit,
                "name": post.name,  # ties the comment back to the post
                "score": comment.score,
                "type": "comment"
            })
    return comments_list 
    
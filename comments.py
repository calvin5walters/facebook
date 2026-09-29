"""Facebook comment scraper for the sentiment pipeline.

Facebook counterpart of the instagram repo's comments.py. Pulls every comment
ScrapeCreators returns for a set of posts and builds a DataFrame matching the
`thetrenches.facebook.comments` schema (same columns as instagram.comments):

    commentId, commentUser, isVerified, commentText, replies, likes,
    timestamp, postUrl, postUsername, postCaption, postSong, artist, pullTime

Facebook differences: commentUser is the commenter's display name (Facebook
exposes no username), isVerified is always null (not returned), and likes is
the comment's total reaction count.

Post context (caption/username/song) is resolved without extra API calls
where possible, in this order:
    1. --page mode pulls the page's posts first and feeds each post's
       caption/username/sound_title straight into its comments pull.
    2. URL mode looks the posts up in the BigQuery posts history
       (facebook.posts, latest pull per post).
    3. Posts found in neither fall back to one post-details API call each.

Comments are fetched by feedback_id (derived from the post id) whenever the
post id is known, which ScrapeCreators documents as much faster than by URL.

commentId is the stable Facebook comment id - dedupe on it when re-pulling a
post.

Endpoints used (docs.scrapecreators.com):
    GET /v1/facebook/post/comments  - paginated comments (1 credit per ~10)
    GET /v1/facebook/post           - post caption + author (1 credit per post)

Replies are counted (reply_count -> replies) but not fetched. ScrapeCreators
has a replies endpoint (/v1/facebook/post/comment/replies) but it costs a
credit per comment with replies, and instagram.comments doesn't include
replies either.

Definitions only, no side effects on import. Run directly:
    python comments.py <post_url> [<post_url> ...] --artist "Artist Name" [--export]
    python comments.py --page <page url or slug> --num-posts 12 --artist "Artist Name" [--export]
"""

import argparse
import datetime
import re
from urllib.parse import quote

import pandas as pd
import pytz
from google.cloud import bigquery

from helpers import (BASE_URL, feedback_id_for, get_json, get_posts_df,
                     normalize_page_url, parse_date)

timezone = pytz.timezone('US/Eastern')

COLUMNS = [
    "commentId",
    "commentUser",
    "isVerified",
    "commentText",
    "replies",
    "likes",
    "timestamp",
    "postUrl",
    "postUsername",
    "postCaption",
    "postSong",
    "artist",
    "pullTime",
]

COMMENTS_SCHEMA = [
    bigquery.SchemaField('commentId', 'STRING'),
    bigquery.SchemaField('commentUser', 'STRING'),
    bigquery.SchemaField('isVerified', 'BOOLEAN'),
    bigquery.SchemaField('commentText', 'STRING'),
    bigquery.SchemaField('replies', 'INTEGER'),
    bigquery.SchemaField('likes', 'INTEGER'),
    bigquery.SchemaField('timestamp', 'TIMESTAMP'),
    bigquery.SchemaField('postUrl', 'STRING'),
    bigquery.SchemaField('postUsername', 'STRING'),
    bigquery.SchemaField('postCaption', 'STRING'),
    bigquery.SchemaField('postSong', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('pullTime', 'TIMESTAMP'),
]

# /reel/<id>, /videos/<id>, /posts/<id or pfbid>, story.php?story_fbid=<id>, watch/?v=<id>
_POST_KEY_RE = re.compile(r'/(?:reel|videos|posts)/([A-Za-z0-9]+)|[?&](?:story_fbid|v|fbid)=([A-Za-z0-9]+)')


def _post_key(post_url):
    """The id segment of a post URL, used to match URLs to posts history."""
    match = _POST_KEY_RE.search(post_url or "")
    if not match:
        return None
    return next(g for g in match.groups() if g)


def get_post_details(post_url):
    """Returns (caption, author_handle, song, post_id) for a post, or Nones.

    The post endpoint carries no music, so song is always None here - pass
    post_song from facebook.posts (sound_title) when you have it.
    """

    data = get_json(f"{BASE_URL}/v1/facebook/post?url={quote(post_url, safe='')}")
    if not data:
        return None, None, None, None

    author = data.get('author') or {}
    username = author.get('handle')
    if not username and author.get('url'):
        username = normalize_page_url(author['url'])[1]
    return data.get('description'), username, None, data.get('post_id')


def get_post_comments_df(post_url, artist=None, post_caption=None, post_username=None,
                         post_song=None, post_id=None, max_post_comments=None,
                         start_date=None, end_date=None):
    """
    Pulls all available comments for one post and returns a DataFrame
    following the facebook.comments schema.

    post_caption / post_username / post_song: pass all three to skip the extra
                  post-details API call (e.g. when they are already in
                  facebook.posts - use "" for a post with no song).
    post_id: lets comments be fetched by feedback_id (faster than by URL).
    max_post_comments: hard cap on comments pulled for this post; pagination
                  stops once reached and the final page is trimmed to it.
    start_date / end_date: optional inclusive bounds on comment timestamps
                  (naive values are read as US/Eastern, and a date-only
                  end_date covers that whole day). Comments are filtered, not
                  early-stopped: the endpoint's ordering isn't chronological
                  (it returns Facebook's "most relevant" order), so every page
                  is still pulled.
    """

    if post_caption is None or post_username is None or post_song is None:
        caption, username, song, details_id = get_post_details(post_url)
        post_caption = post_caption if post_caption is not None else caption
        post_username = post_username if post_username is not None else username
        post_song = post_song if post_song is not None else song
        post_id = post_id or details_id

    feedback_id = feedback_id_for(post_id)
    if feedback_id:
        base_url = f"{BASE_URL}/v1/facebook/post/comments?feedback_id={quote(feedback_id, safe='')}"
    else:
        base_url = f"{BASE_URL}/v1/facebook/post/comments?url={quote(post_url, safe='')}"

    rows = []
    seen_ids = set()
    cursor = None
    while True:

        url = base_url
        if cursor:
            url = url + f"&cursor={quote(cursor, safe='')}"
        data = get_json(url)
        if not data:
            break

        comments = data.get('comments') or []
        new_comments = [c for c in comments if c.get('id') not in seen_ids]
        if not new_comments:
            break

        for comment in new_comments:
            seen_ids.add(comment.get('id'))
            author = comment.get('author') or {}
            rows.append({
                'commentId': comment.get('id'),
                'commentUser': author.get('name'),
                'isVerified': None,
                'commentText': comment.get('text'),
                'replies': comment.get('reply_count'),
                'likes': comment.get('reaction_count'),
                'timestamp': comment.get('created_at'),
                'postUrl': post_url,
                'postUsername': post_username,
                'postCaption': post_caption,
                'postSong': post_song,
                'artist': artist,
                'pullTime': datetime.datetime.now(timezone),
            })

        if max_post_comments and len(rows) >= max_post_comments:
            break
        cursor = data.get('cursor')
        if not cursor or data.get('has_next_page') is False:
            break

    if max_post_comments:
        rows = rows[:max_post_comments]

    df = pd.DataFrame(rows, columns=COLUMNS)

    # nullable Int64 keeps BigQuery INTEGER columns from becoming FLOAT
    df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0).astype('Int64')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0).astype('Int64')
    df['isVerified'] = df['isVerified'].astype('boolean')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    pulled = len(df)

    start_ts = parse_date(start_date)
    end_ts = parse_date(end_date, end_of_day=True)
    if start_ts is not None or end_ts is not None:
        if start_ts is not None:
            df = df[df['timestamp'] >= start_ts]
        if end_ts is not None:
            df = df[df['timestamp'] <= end_ts]
        df = df.reset_index(drop=True)

    # How many rows the API actually returned, before date bounds - lets the
    # caller tell "post has no comments" apart from "all its comments are
    # outside the requested window", which otherwise both print as 0.
    df.attrs['pulled'] = pulled
    return df


def _batch_summary(batch, post_url):
    pulled = batch.attrs.get('pulled', len(batch))
    if pulled > len(batch):
        return f"   {len(batch)} comments from {post_url} ({pulled} pulled, {pulled - len(batch)} outside date range)"
    return f"   {len(batch)} comments from {post_url}"


def get_post_details_from_bq(post_urls):
    """Looks up caption/username/sound_title/post_id for post URLs in the
    BigQuery posts history (facebook.posts, latest pull per post).

    Returns {post_key: {...}} for the posts found, keyed by _post_key. Returns
    {} if the lookup fails (e.g. no BigQuery credentials, or no table yet).
    """

    keys = sorted({k for k in (_post_key(u) for u in post_urls) if k})
    if not keys:
        return {}

    # posts are partitioned by pullTime; 180 days keeps the scan bounded
    query = """
        SELECT
          COALESCE(REGEXP_EXTRACT(post_url, r'/(?:reel|videos|posts)/([A-Za-z0-9]+)'), post_id) AS key,
          ARRAY_AGG(STRUCT(caption, username, sound_title, post_id)
                    ORDER BY pullTime DESC LIMIT 1)[OFFSET(0)].*
        FROM `thetrenches.facebook.posts`
        WHERE pullTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
          AND (REGEXP_EXTRACT(post_url, r'/(?:reel|videos|posts)/([A-Za-z0-9]+)') IN UNNEST(@keys)
               OR post_id IN UNNEST(@keys))
        GROUP BY key
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("keys", "STRING", keys)]
    )

    try:
        bq_client = bigquery.Client(project='thetrenches')
        rows = bq_client.query(query, job_config=job_config).result()
    except Exception as e:
        print(f"   posts history lookup failed, falling back to post-details calls: {e}")
        return {}

    return {row.key: {'caption': row.caption,
                      'username': row.username,
                      'sound_title': row.sound_title,
                      'post_id': row.post_id} for row in rows}


def _cap(max_post_comments, max_total_comments, pulled_so_far):
    """Per-post cap given the run cap; None means stop (run cap reached)."""
    if max_total_comments is None:
        return max_post_comments
    remaining = max_total_comments - pulled_so_far
    if remaining <= 0:
        return None
    return min(max_post_comments, remaining) if max_post_comments else remaining


def get_comments_df(post_urls, artist=None, max_post_comments=None, use_bq_history=True,
                    start_date=None, end_date=None, max_total_comments=None):
    """Pulls comments for multiple posts into one DataFrame.

    With use_bq_history (default), post caption/username/song are first
    looked up in the facebook.posts history, and the per-post details call is
    only made for posts not found there. A post found in history is trusted
    as-is: a NULL sound_title there is treated as "no song".
    start_date / end_date filter comment timestamps (see get_post_comments_df).
    max_post_comments caps each post's pull; max_total_comments caps the whole
    run and skips the remaining posts once reached.
    """

    known = get_post_details_from_bq(post_urls) if use_bq_history else {}

    df = pd.DataFrame(columns=COLUMNS)
    for post_url in post_urls:
        post_cap = _cap(max_post_comments, max_total_comments, len(df))
        if max_total_comments is not None and post_cap is None:
            print(f"   max_total_comments={max_total_comments} reached, skipping remaining posts")
            break

        info = known.get(_post_key(post_url))
        if info:
            batch = get_post_comments_df(post_url, artist=artist,
                                         post_caption=info['caption'] or "",
                                         post_username=info['username'],
                                         post_song=info['sound_title'] or "",
                                         post_id=info['post_id'],
                                         max_post_comments=post_cap,
                                         start_date=start_date, end_date=end_date)
        else:
            batch = get_post_comments_df(post_url, artist=artist, max_post_comments=post_cap,
                                         start_date=start_date, end_date=end_date)
        print(_batch_summary(batch, post_url))
        df = pd.concat([df, batch], ignore_index=True) if not df.empty else batch

    return df


def get_comments_for_posts(posts_df, artist=None, max_post_comments=None,
                           start_date=None, end_date=None, max_total_comments=None):
    """Pulls comments for every post in a get_posts_df DataFrame.

    Reuses the caption/username/sound_title/post_id already captured by the
    posts pull, so no per-post details call is made.
    """

    df = pd.DataFrame(columns=COLUMNS)
    for post in posts_df.itertuples(index=False):
        post_cap = _cap(max_post_comments, max_total_comments, len(df))
        if max_total_comments is not None and post_cap is None:
            print(f"   max_total_comments={max_total_comments} reached, skipping remaining posts")
            break

        caption = post.caption if pd.notna(post.caption) else ""
        song = post.sound_title if pd.notna(post.sound_title) else ""
        batch = get_post_comments_df(post.post_url, artist=artist,
                                     post_caption=caption,
                                     post_username=post.username,
                                     post_song=song,
                                     post_id=post.post_id,
                                     max_post_comments=post_cap,
                                     start_date=start_date, end_date=end_date)
        print(_batch_summary(batch, post.post_url))
        df = pd.concat([df, batch], ignore_index=True) if not df.empty else batch

    return df


def export_comments_to_bq(comments_df):
    """Upserts rows into facebook.comments keyed on commentId.

    A re-pulled comment replaces its existing row (latest pull wins), so the
    table keeps exactly one row per comment. Rows without a commentId never
    match and are simply inserted. The table is partitioned by DATE(timestamp)
    and clustered on commentId (see schema.sql).
    """

    project_id = 'thetrenches'
    table_id = f'{project_id}.facebook.comments'
    staging_id = f'{table_id}_staging'

    if comments_df.empty:
        return

    # MERGE allows at most one source row per key - keep the latest pull
    has_id = comments_df['commentId'].notna()
    deduped = (comments_df[has_id]
               .sort_values('pullTime')
               .drop_duplicates('commentId', keep='last'))
    comments_df = pd.concat([deduped, comments_df[~has_id]], ignore_index=True)
    comments_df['pullTime'] = pd.to_datetime(comments_df['pullTime'], utc=True)

    bq_client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(
        schema=COMMENTS_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    bq_client.load_table_from_dataframe(comments_df[COLUMNS], staging_id,
                                        job_config=job_config).result()

    update_cols = ", ".join(f"{col} = S.{col}" for col in COLUMNS if col != 'commentId')
    insert_cols = ", ".join(COLUMNS)
    insert_vals = ", ".join(f"S.{col}" for col in COLUMNS)
    merge_sql = f"""
        MERGE `{table_id}` T
        USING `{staging_id}` S
        ON T.commentId = S.commentId
        WHEN MATCHED THEN UPDATE SET {update_cols}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """

    bq_client.query(merge_sql).result()
    bq_client.delete_table(staging_id, not_found_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Scrape Facebook comments into the facebook.comments schema.")
    parser.add_argument('post_urls', nargs='*', help="Facebook post/reel/video URLs")
    parser.add_argument('--page', default=None, help="Pull this page's recent posts first (URL or slug), then comments for each post")
    parser.add_argument('--num-posts', type=int, default=12, help="With --page: how many recent posts to pull (default: 12; ignored when --start-date is set)")
    parser.add_argument('--artist', default=None, help="Artist name to tag the rows with")
    parser.add_argument('--max-post-comments', type=int, default=None, help="Max comments to pull per post (default: all)")
    parser.add_argument('--max-total-comments', type=int, default=None, help="Max comments to pull across the whole run; remaining posts are skipped once reached (default: no cap)")
    parser.add_argument('--start-date', default=None, help="Only keep posts/comments on or after this date (e.g. 2026-07-01, US/Eastern)")
    parser.add_argument('--end-date', default=None, help="Only keep posts/comments on or before this date (e.g. 2026-07-22, US/Eastern)")
    parser.add_argument('--export', action='store_true', help="Upsert results into thetrenches.facebook.comments on commentId")
    parser.add_argument('--csv', default=None, help="Optional path to also write the results as CSV")
    args = parser.parse_args()

    if args.page:
        page = args.page if 'facebook.com' in args.page else f"https://www.facebook.com/{args.page}"
        page_url, username = normalize_page_url(page)
        if not page_url:
            parser.error(f"not a Facebook page: {args.page}")
        posts = get_posts_df(page_url=page_url, username=username, num_posts=args.num_posts,
                             start_date=args.start_date, end_date=args.end_date)
        if posts.empty:
            print(f"no posts found for {page_url}")
            return
        print(f"{len(posts)} posts from {page_url}")
        comments = get_comments_for_posts(posts, artist=args.artist, max_post_comments=args.max_post_comments,
                                          start_date=args.start_date, end_date=args.end_date,
                                          max_total_comments=args.max_total_comments)
    elif args.post_urls:
        comments = get_comments_df(args.post_urls, artist=args.artist, max_post_comments=args.max_post_comments,
                                   start_date=args.start_date, end_date=args.end_date,
                                   max_total_comments=args.max_total_comments)
    else:
        parser.error("provide post URLs or --page")
    print(f"{len(comments)} total comments")

    if args.csv:
        comments.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")

    if args.export:
        export_comments_to_bq(comments)
        print("upserted into thetrenches.facebook.comments")
    elif not args.csv:
        print(comments.head(20).to_string())


if __name__ == "__main__":
    main()

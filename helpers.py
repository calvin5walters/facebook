"""Reusable Facebook scraping helpers (definitions only, no side effects).

Facebook counterpart of the instagram repo's helpers.py. Importing this module
runs no pipeline code, so it is safe to reuse from other projects. The runner
(facebook.py) imports these functions and performs all BigQuery I/O and
account iteration under its own __main__ guard.

Endpoints used (docs.scrapecreators.com), 1 credit per call:
    GET /v1/facebook/profile          - page details + follower/like counts
    GET /v1/facebook/profile/posts    - timeline posts, 3 per page
    GET /v1/facebook/profile/reels    - Reels tab, 10 per page, with view
                                        counts and music the posts feed lacks
"""

import base64
import datetime
import os
import re
import time
import traceback
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd
import pytz
import requests
from google.cloud import bigquery

BASE_URL = "https://api.scrapecreators.com"

timezone = pytz.timezone('US/Eastern')


def _headers():
    """Resolve the key at call time so importers can set it after import.
    One ScrapeCreators key covers every platform, so IG_API_KEY works too."""
    return {"x-api-key": os.environ.get("SCRAPECREATORS_API_KEY")
            or os.environ.get("FB_API_KEY") or os.environ.get("IG_API_KEY") or ""}


def get_json(url, retries=3):
    """GET a ScrapeCreators URL with retries.

    Returns the parsed JSON dict, or None if every attempt failed. Failures
    are always logged: a silent None would surface downstream as "0 posts" or
    "0 comments", indistinguishable from a page that genuinely has none.
    """

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=_headers(), timeout=60)
            # an HTTP error body is still valid JSON, so check the status first
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                # 4xx other than 429 won't fix themselves - stop retrying
                if response.status_code != 429 and response.status_code < 500:
                    break
            else:
                data = response.json()
                if data.get('success') is not False:
                    return data
                last_error = f"success=false: {str(data)[:200]}"
        except (requests.RequestException, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))

    print(f"   ⚠ ScrapeCreators request failed ({last_error}) - {url}")
    return None


def parse_date(value, end_of_day=False):
    """Parses a date bound (string, date, or datetime) to a UTC Timestamp.

    Naive values are interpreted as US/Eastern. With end_of_day, a date-only
    value is pushed to the last microsecond of that day so the bound stays
    inclusive. Returns None for None.
    """

    if value is None:
        return None

    ts = pd.Timestamp(value)
    if end_of_day and ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if ts.tz is None:
        ts = ts.tz_localize(timezone)

    return ts.tz_convert('UTC')


# ---------------------------------------------------------------- page urls

_FB_HOST_RE = re.compile(r'^(?:[a-z]+\.)?facebook\.com$', re.I)


def normalize_page_url(url):
    """(page_url, username) for a Facebook page link, or (None, None).

    Chartmetric's links come in many shapes: business./web./m. hosts, tracking
    query strings, /people/<name>/<id>, /p/<name>-<id>, and profile.php?id=.
    username is the stable key the pull is tracked under: the vanity slug
    lowercased, or the numeric id when the page has no vanity name.
    """

    if not url or not isinstance(url, str):
        return None, None
    url = url.strip()
    if not re.match(r'^https?://', url, re.I):
        url = 'https://' + url.lstrip('/')

    parsed = urlparse(url)
    if not _FB_HOST_RE.match(parsed.netloc or ''):
        return None, None

    parts = [p for p in parsed.path.split('/') if p]
    if not parts:
        return None, None

    if parts[0].lower() == 'profile.php':
        page_id = (parse_qs(parsed.query).get('id') or [None])[0]
        if not page_id or not page_id.isdigit():
            return None, None
        return f"https://www.facebook.com/profile.php?id={page_id}", page_id

    if parts[0].lower() == 'people' and len(parts) >= 3 and parts[2].isdigit():
        return f"https://www.facebook.com/profile.php?id={parts[2]}", parts[2]

    if parts[0].lower() == 'p' and len(parts) >= 2:
        # /p/Kashus-Culpepper-61555803748404 - the trailing number is the id
        match = re.search(r'(\d{8,})$', parts[1])
        if match:
            return f"https://www.facebook.com/profile.php?id={match.group(1)}", match.group(1)
        return None, None

    slug = parts[0]
    # /Codylohdenofficial-100082224607434 style vanity+id: the id alone works
    match = re.match(r'^.+-(\d{8,})$', slug)
    if match:
        return f"https://www.facebook.com/profile.php?id={match.group(1)}", match.group(1)
    if slug.isdigit():
        return f"https://www.facebook.com/profile.php?id={slug}", slug
    return f"https://www.facebook.com/{slug}", slug.lower()


def feedback_id_for(post_id):
    """The comments endpoint's feedback_id for a post: base64("feedback:<id>").

    Checked against the ids ScrapeCreators returns on reels and posts. Passing
    it instead of a URL makes the comments call much faster."""
    if not post_id:
        return None
    return base64.b64encode(f"feedback:{post_id}".encode()).decode()


# ---------------------------------------------------------------- profile

def get_profile(page_url, username):
    """(profile_df, page_id): a one-row users frame plus the page's numeric id
    (used for the faster posts call). Empty/None when the page can't be read."""

    data = get_json(f"{BASE_URL}/v1/facebook/profile?url={quote(page_url, safe='')}")
    if not data:
        return pd.DataFrame(), None

    status = data.get('account_status')
    if data.get('isPrivate') or status in ('private', 'age-restricted'):
        print(f"   page {page_url} is {status or 'private'}, skipping")
        return pd.DataFrame(), None
    if not data.get('name'):
        return pd.DataFrame(), None

    page_id = data.get('id')
    profile = {
        "username": username,
        "page_id": str(page_id) if page_id is not None else None,
        "profile_name": data.get('name'),
        "page_url": data.get('url') or page_url,
        "category": data.get('category'),
        "followers": data.get('followerCount'),
        "likes": data.get('likeCount'),
        "talking_about": data.get('talkingAboutCount'),
        "avi": data.get('profilePicLarge') or data.get('profilePicMedium'),
        "pullTime": datetime.datetime.now(timezone),
    }
    return pd.DataFrame([profile]), profile['page_id']


# ---------------------------------------------------------------- posts

POST_COLUMNS = [
    'username', 'page_id', 'post_id', 'post_url', 'type', 'thumbnail', 'caption',
    'is_video', 'video_url', 'video_views', 'likes', 'comments', 'shares',
    'sound_id', 'sound_title', 'feedback_id', 'postTime', 'pullTime',
]


def _to_naive_utc(value):
    if value is None:
        return pd.NaT
    ts = pd.to_datetime(value, errors='coerce', utc=True)
    return ts.tz_localize(None) if isinstance(ts, pd.Timestamp) else pd.NaT


def _post_type(url, is_video):
    if '/reel/' in (url or ''):
        return "Reel"
    if is_video:
        return "Video"
    return "Post"


def _get_individual_post_info(post, username):
    """One /profile/posts item -> a posts row."""

    video = post.get('videoDetails') or {}
    is_video = bool(video.get('sdUrl') or video.get('hdUrl'))
    url = post.get('permalink') or post.get('url')
    publish = post.get('publishTime')
    post_time = (pd.to_datetime(publish, unit='s', errors='coerce') if publish is not None
                 else _to_naive_utc(post.get('creation_time')))
    post_id = str(post['id'])
    author = post.get('author') or {}

    return {
        'username': username,
        'page_id': str(author['id']) if author.get('id') else None,
        'post_id': post_id,
        'post_url': url,
        'type': _post_type(url, is_video),
        'thumbnail': video.get('thumbnailUrl') or post.get('image'),
        'caption': post.get('text'),
        'is_video': is_video,
        'video_url': video.get('hdUrl') or video.get('sdUrl') or '',
        'video_views': post.get('videoViewCount'),
        'likes': post.get('reactionCount'),
        'comments': post.get('commentCount'),
        # the posts feed doesn't carry share counts; /v1/facebook/post does
        'shares': None,
        'sound_id': None,
        'sound_title': None,
        'feedback_id': feedback_id_for(post_id),
        'postTime': post_time,
        'pullTime': datetime.datetime.now(timezone),
    }


def _parse_posts(items, username):
    rows = []
    for post in items:
        try:
            rows.append(_get_individual_post_info(post, username))
        except Exception:
            print(f"   skipping post {post.get('id')}, failed to parse")
            traceback.print_exc()
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def get_posts_df(page_url=None, page_id=None, username=None, num_posts=12, cursor=None,
                 start_date=None, end_date=None):
    """Timeline posts for a page, newest first, 3 per request.

    page_id is faster than page_url when known (the profile call returns it).
    start_date / end_date: optional inclusive bounds on postTime (naive values
                  are US/Eastern, a date-only end_date covers that whole day).
                  When start_date is set, num_posts is ignored and pagination
                  continues until a whole page is older than start_date (a
                  single older post isn't terminal, since pinned posts sit out
                  of order).
    """

    start_ts = parse_date(start_date)
    end_ts = parse_date(end_date, end_of_day=True)
    start_naive = start_ts.tz_localize(None) if start_ts is not None else None
    end_naive = end_ts.tz_localize(None) if end_ts is not None else None

    if page_id:
        url = f"{BASE_URL}/v1/facebook/profile/posts?pageId={page_id}"
    else:
        url = f"{BASE_URL}/v1/facebook/profile/posts?url={quote(page_url, safe='')}"

    df = pd.DataFrame(columns=POST_COLUMNS)
    while True:
        request_url = url + f"&cursor={quote(cursor, safe='')}" if cursor else url
        data = get_json(request_url)
        if not data:
            break

        items = data.get('posts')
        if not items:
            break

        batch = _parse_posts(items, username)
        # a page with nothing new means the feed is exhausted or repeating -
        # without this a date-bounded pull could paginate forever
        batch = batch[~batch['post_id'].isin(df['post_id'])]
        if batch.empty:
            break
        df = pd.concat([df, batch], ignore_index=True) if not df.empty else batch.reset_index(drop=True)

        if start_naive is not None:
            if batch['postTime'].max() < start_naive:
                break
        elif len(df) >= num_posts:
            break

        cursor = data.get('cursor')
        if not cursor:
            break

    if not df.empty and (start_naive is not None or end_naive is not None):
        if start_naive is not None:
            df = df[df['postTime'] >= start_naive]
        if end_naive is not None:
            df = df[df['postTime'] <= end_naive]
        df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------- reels

REEL_FILL_COLUMNS = ['video_views', 'sound_id', 'sound_title', 'video_url', 'thumbnail']


def get_reels_df(page_url, username):
    """First page (10) of the page's Reels tab, as posts rows.

    The posts feed leaves view counts null and has no music; reels carry both,
    plus Reels-tab-only reels the feed never lists. Reels have no reaction or
    comment counts, so those stay null for reels not in the posts feed.
    """

    data = get_json(f"{BASE_URL}/v1/facebook/profile/reels?url={quote(page_url, safe='')}")
    rows = []
    for reel in (data or {}).get('reels') or []:
        post_id = reel.get('post_id')
        if not post_id:
            continue
        music = reel.get('music') or {}
        author = reel.get('author') or {}
        rows.append({
            'username': username,
            'page_id': str(author['id']) if author.get('id') else None,
            'post_id': str(post_id),
            'post_url': reel.get('url'),
            'type': "Reel",
            'thumbnail': reel.get('thumbnail'),
            'caption': reel.get('description'),
            'is_video': True,
            'video_url': reel.get('video_url') or '',
            'video_views': reel.get('view_count'),
            'likes': None,
            'comments': None,
            'shares': None,
            'sound_id': str(music['id']) if music.get('id') else None,
            # "<artist> · <song>", or "<name> · Original audio"
            'sound_title': music.get('track_title'),
            'feedback_id': reel.get('feedback_id') or feedback_id_for(post_id),
            'postTime': _to_naive_utc(reel.get('creation_time')),
            'pullTime': datetime.datetime.now(timezone),
        })
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def _merge_reels(posts, reels):
    """Fill the posts' empty view/sound columns from matching reels (same
    post_id), and append reels the posts feed didn't return. Values the posts
    already have always win."""
    if reels.empty:
        return posts
    if posts.empty:
        return reels

    posts = posts.copy()
    by_id = reels.drop_duplicates('post_id').set_index('post_id')
    known = posts['post_id'].isin(by_id.index)
    for col in REEL_FILL_COLUMNS:
        current = posts.loc[known, col]
        if col in ('video_url', 'thumbnail'):
            current = current.mask(current == '')
        posts.loc[known, col] = current.where(current.notna(),
                                              posts.loc[known, 'post_id'].map(by_id[col]))

    extra = by_id[~by_id.index.isin(posts['post_id'])].reset_index()
    if not extra.empty:
        print(f"   +{len(extra)} reel(s) not in the posts feed")
    return pd.concat([posts, extra[POST_COLUMNS]], ignore_index=True)


def get_account_df(page_url, username, num_posts=12):
    """Profile row and posts for the scheduled pull.

    Credits per account: 1 profile + ceil(num_posts / 3) posts pages + 1 reels
    page. Returns (profile_df, posts_df); both empty if the page can't be read.
    """

    profile, page_id = get_profile(page_url, username)
    if profile.empty:
        return profile, pd.DataFrame(columns=POST_COLUMNS)

    posts = get_posts_df(page_url=page_url, page_id=page_id, username=username,
                         num_posts=num_posts)
    posts = _merge_reels(posts, get_reels_df(page_url, username))
    if not posts.empty:
        posts['page_id'] = posts['page_id'].fillna(page_id)
    return profile, posts.reset_index(drop=True)


# ---------------------------------------------------------------- BigQuery

PROJECT_ID = 'thetrenches'
DATASET = 'facebook'

USERS_SCHEMA = [
    bigquery.SchemaField('username', 'STRING'),
    bigquery.SchemaField('page_id', 'STRING'),
    bigquery.SchemaField('profile_name', 'STRING'),
    bigquery.SchemaField('page_url', 'STRING'),
    bigquery.SchemaField('category', 'STRING'),
    bigquery.SchemaField('followers', 'INTEGER'),
    bigquery.SchemaField('likes', 'INTEGER'),
    bigquery.SchemaField('talking_about', 'INTEGER'),
    bigquery.SchemaField('avi', 'STRING'),
    bigquery.SchemaField('account_type', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('pullTime', 'TIMESTAMP'),
]

POSTS_SCHEMA = [
    bigquery.SchemaField('username', 'STRING'),
    bigquery.SchemaField('page_id', 'STRING'),
    bigquery.SchemaField('post_id', 'STRING'),
    bigquery.SchemaField('post_url', 'STRING'),
    bigquery.SchemaField('type', 'STRING'),
    bigquery.SchemaField('thumbnail', 'STRING'),
    bigquery.SchemaField('caption', 'STRING'),
    bigquery.SchemaField('is_video', 'BOOLEAN'),
    bigquery.SchemaField('video_url', 'STRING'),
    bigquery.SchemaField('video_views', 'INTEGER'),
    bigquery.SchemaField('likes', 'INTEGER'),
    bigquery.SchemaField('comments', 'INTEGER'),
    bigquery.SchemaField('shares', 'INTEGER'),
    bigquery.SchemaField('sound_id', 'STRING'),
    bigquery.SchemaField('sound_title', 'STRING'),
    bigquery.SchemaField('feedback_id', 'STRING'),
    bigquery.SchemaField('account_type', 'STRING'),
    bigquery.SchemaField('artist', 'STRING'),
    bigquery.SchemaField('postTime', 'TIMESTAMP'),
    bigquery.SchemaField('pullTime', 'TIMESTAMP'),
]


def _conform(df, schema):
    """Order/coerce columns to the table schema so every load matches it."""
    df = df.copy()
    for field in schema:
        if field.name not in df.columns:
            df[field.name] = None
        if field.field_type == 'INTEGER':
            # reel view counts arrive as floats parsed from "9.4K", e.g.
            # 9400.000000000002, which Int64 refuses without rounding
            df[field.name] = pd.to_numeric(df[field.name], errors='coerce').round().astype('Int64')
        elif field.field_type == 'BOOLEAN':
            df[field.name] = df[field.name].astype('boolean')
        elif field.field_type == 'TIMESTAMP':
            df[field.name] = pd.to_datetime(df[field.name], utc=True, errors='coerce')
        else:
            df[field.name] = df[field.name].astype('string')
    return df[[f.name for f in schema]]


def _append(client, df, table_name, schema, cluster_fields):
    """Append to a table partitioned by DATE(pullTime) and clustered.

    Uses a load job (not to_gbq) so that, if the table doesn't exist yet, it's
    created partitioned and clustered rather than as a flat table - see
    schema.sql for the same definitions.
    """
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field='pullTime'),
        clustering_fields=cluster_fields,
    )
    table_id = f'{PROJECT_ID}.{DATASET}.{table_name}'
    client.load_table_from_dataframe(_conform(df, schema), table_id, job_config=job_config).result()


def export_to_bq(posts_df, users_df, client=None):
    client = client or bigquery.Client(project=PROJECT_ID)
    if not posts_df.empty:
        _append(client, posts_df, 'posts', POSTS_SCHEMA, ['username', 'post_id'])
    if not users_df.empty:
        _append(client, users_df, 'users', USERS_SCHEMA, ['username'])

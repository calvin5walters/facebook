"""Scheduled Facebook pull runner.

Facebook counterpart of the instagram repo's instagram.py. Reusable scraping
functions live in helpers.py (safe to import anywhere). This file is the
runner: it builds the account list from BigQuery, scrapes each page, and
appends results to thetrenches.facebook.{posts,users}. All execution is
guarded by __main__ so importing this module does nothing.

Account list (same selection as instagram.py):
    - artists in clients.artists_table WHERE active OR republic. artists_table
      has no Facebook column, so each artist's page comes from the latest
      facebook_url Chartmetric reported in chartmetric.artist_social_metrics.
      FACEBOOK_URL_OVERRIDES below corrects or fills individual artists.
    - fanpages in clients.fanpages_table WHERE platform = 'facebook'
      (username there may be a page slug, id, or full URL).

Pages are pulled in parallel from one runner (MAX_WORKERS at a time) rather
than one by one, and each page is pulled at most once per US/Eastern day.

Run directly (as the GitHub Action does):
    python facebook.py
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.cloud import bigquery

from helpers import export_to_bq, get_account_df, normalize_page_url

# ScrapeCreators calls in flight at once
MAX_WORKERS = 8
# stop starting new pages after this long, like instagram.py's 2h30m cap
TIME_LIMIT_SECONDS = 9000
# flush to BigQuery every this many pages, so a crash loses little
EXPORT_EVERY = 25
# posts per pull (3 per ScrapeCreators request); first pulls go deeper
NUM_POSTS = 12
NUM_POSTS_FIRST_PULL = 24

# artist -> Facebook page URL, for artists whose Chartmetric link is missing
# or wrong (e.g. a personal profile instead of the artist page)
FACEBOOK_URL_OVERRIDES = {
}


def get_accounts(bq_client):
    artists_query = """
        WITH latest_fb AS (
          SELECT artist_id,
                 ARRAY_AGG(facebook_url IGNORE NULLS ORDER BY timestamp DESC LIMIT 1)[SAFE_OFFSET(0)] AS facebook_url
          FROM `thetrenches.chartmetric.artist_social_metrics`
          GROUP BY artist_id
        )
        SELECT a.artist, fb.facebook_url
        FROM `thetrenches.clients.artists_table` a
        LEFT JOIN latest_fb fb ON fb.artist_id = a.cm_id
        WHERE a.active OR a.republic
    """
    artists = bq_client.query(artists_query).to_dataframe()

    accounts = []
    for row in artists.itertuples(index=False):
        url = FACEBOOK_URL_OVERRIDES.get(row.artist) or row.facebook_url
        accounts.append({"artist": row.artist, "url": url, "account_type": "Artist"})

    fanpages_query = """
        SELECT artist, username FROM `thetrenches.clients.fanpages_table`
        WHERE platform = 'facebook'
    """
    fanpages = bq_client.query(fanpages_query).to_dataframe()
    for row in fanpages.itertuples(index=False):
        accounts.append({"artist": row.artist, "url": row.username, "account_type": "Fanpage"})

    for account in accounts:
        url = account["url"]
        if isinstance(url, str) and url and 'facebook.com' not in url.lower():
            url = f"https://www.facebook.com/{url.strip().lstrip('@')}"
        account["page_url"], account["username"] = normalize_page_url(url)
    return accounts


def get_pull_history(bq_client):
    """(pulled_today, last_pull): usernames already pulled today (US/Eastern),
    and each username's latest pull time. Empty if the table doesn't exist."""
    query = """
        SELECT username, MAX(pullTime) AS pullTime
        FROM `thetrenches.facebook.users`
        GROUP BY username
    """
    try:
        history = bq_client.query(query).to_dataframe()
    except Exception as e:
        print(f"no pull history yet ({type(e).__name__}), treating every page as new")
        return set(), {}

    history['pullTime'] = pd.to_datetime(history['pullTime'], utc=True)
    today_et = pd.Timestamp.now(tz='US/Eastern').date()
    pulled_today = set(history.loc[
        history['pullTime'].dt.tz_convert('US/Eastern').dt.date == today_et, 'username'])
    last_pull = dict(zip(history['username'], history['pullTime']))
    return pulled_today, last_pull


def pull_account(account, num_posts):
    username = account['username']
    user, posts = get_account_df(account['page_url'], username, num_posts=num_posts)
    if user.empty:
        return account, user, posts
    for df in (user, posts):
        df['account_type'] = account['account_type']
        df['artist'] = account['artist']
    return account, user, posts


def main():
    bq_client = bigquery.Client(project='thetrenches')

    accounts = get_accounts(bq_client)
    pulled_today, last_pull = get_pull_history(bq_client)

    todo = []
    seen = set()
    for account in accounts:
        if not account['username']:
            print(f"{account['artist']}: no usable Facebook URL ({account['url']}), skipping")
            continue
        if account['username'] in seen:
            continue
        seen.add(account['username'])
        if account['username'] in pulled_today:
            continue
        todo.append(account)

    # never-pulled pages first, then the stalest
    todo.sort(key=lambda a: (a['username'] in last_pull,
                             last_pull.get(a['username'], pd.Timestamp.min.tz_localize('UTC'))))
    print(f"{len(todo)} pages to pull ({len(pulled_today)} already pulled today, "
          f"{len(accounts)} accounts listed)")

    start_time = time.time()
    user_frames, post_frames = [], []
    done = 0
    out_of_time = False

    def flush():
        if not user_frames:
            return
        users = pd.concat(user_frames, ignore_index=True)
        posts = pd.concat(post_frames, ignore_index=True) if post_frames else pd.DataFrame()
        try:
            export_to_bq(posts, users, client=bq_client)
            print(f"   exported {len(users)} pages, {len(posts)} posts")
        except Exception:
            print("   export to BigQuery failed")
            traceback.print_exc()
        user_frames.clear()
        post_frames.clear()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        pending = iter(todo)

        def submit_next():
            if time.time() - start_time > TIME_LIMIT_SECONDS:
                return False
            account = next(pending, None)
            if account is None:
                return False
            num_posts = NUM_POSTS if account['username'] in last_pull else NUM_POSTS_FIRST_PULL
            futures[pool.submit(pull_account, account, num_posts)] = account
            return True

        for _ in range(MAX_WORKERS):
            if not submit_next():
                break

        while futures:
            future = next(as_completed(futures))
            account = futures.pop(future)
            try:
                _, user, posts = future.result()
                if user.empty:
                    print(f"{account['artist']}: page {account['page_url']} not readable")
                else:
                    user_frames.append(user)
                    if not posts.empty:
                        post_frames.append(posts)
                    print(f"{account['artist']}: {account['username']} - {len(posts)} posts")
            except Exception:
                print(f"{account['artist']}: error pulling {account['page_url']}, skipping")
                traceback.print_exc()

            done += 1
            if len(user_frames) >= EXPORT_EVERY:
                flush()
            if not submit_next() and not out_of_time and time.time() - start_time > TIME_LIMIT_SECONDS:
                out_of_time = True
                print('2hr 30min elapsed, not starting more pages')

    flush()
    print(f"done: {done} pages attempted in {int(time.time() - start_time)}s")


if __name__ == "__main__":
    main()

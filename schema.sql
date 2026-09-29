-- BigQuery tables for the Facebook pulls. Not yet created: run once after
-- review. facebook.py / comments.py create posts/users the same way on their
-- first load if these haven't been run, but comments' MERGE needs the table.
--
-- Partitioned and clustered from the start (instagram.posts is unpartitioned,
-- so every read of it scans the whole table).

CREATE SCHEMA IF NOT EXISTS `thetrenches.facebook`
OPTIONS (location = 'US');

CREATE TABLE IF NOT EXISTS `thetrenches.facebook.users` (
  username STRING,        -- page slug lowercased, or numeric id if no vanity name
  page_id STRING,
  profile_name STRING,
  page_url STRING,
  category STRING,
  followers INT64,
  likes INT64,            -- page likes
  talking_about INT64,
  avi STRING,
  account_type STRING,    -- Artist | Fanpage
  artist STRING,
  pullTime TIMESTAMP
)
PARTITION BY DATE(pullTime)
CLUSTER BY username;

CREATE TABLE IF NOT EXISTS `thetrenches.facebook.posts` (
  username STRING,
  page_id STRING,
  post_id STRING,
  post_url STRING,
  type STRING,            -- Reel | Video | Post
  thumbnail STRING,
  caption STRING,
  is_video BOOL,
  video_url STRING,
  video_views INT64,      -- from the Reels tab; null for non-reel posts
  likes INT64,            -- total reactions
  comments INT64,
  shares INT64,           -- not in the posts feed, reserved
  sound_id STRING,        -- reels only
  sound_title STRING,     -- reels only, "<artist> · <song>"
  feedback_id STRING,     -- comments endpoint key
  account_type STRING,
  artist STRING,
  postTime TIMESTAMP,
  pullTime TIMESTAMP
)
PARTITION BY DATE(pullTime)
CLUSTER BY username, post_id;

CREATE TABLE IF NOT EXISTS `thetrenches.facebook.comments` (
  commentId STRING,
  commentUser STRING,     -- display name; Facebook exposes no username
  isVerified BOOL,        -- not returned by Facebook, always null
  commentText STRING,
  replies INT64,
  likes INT64,            -- total reactions on the comment
  timestamp TIMESTAMP,
  postUrl STRING,
  postUsername STRING,
  postCaption STRING,
  postSong STRING,
  artist STRING,
  pullTime TIMESTAMP
)
PARTITION BY DATE(timestamp)
CLUSTER BY commentId;

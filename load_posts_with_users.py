import sqlite3
import pandas as pd
from oasis.social_platform.database import create_db
import os


def create_base_db(user_file, post_file, db_path, news_file):
    print("Creating base database with OASIS schema...")
    conn, cursor = create_db(db_path=db_path)
    print("Database schema created successfully!")

    cursor.execute("PRAGMA table_info(post)")
    columns = cursor.fetchall()

    print("Current post table schema:")
    for col in columns:
        print(f"  {col[1]} {col[2]}")

    user_id_mapping = load_users(cursor, user_file)
    load_posts(cursor, post_file, user_id_mapping)

    high_quality_news_users = [
        "Reuters",
        "AP_Politics",
        "NewsNation",
        "goodnewsnetwork",
        "NPR",
        "nprpolitics",
        "ForeignPolicy",
        "ForeignAffairs",
        "USATODAY",
        "thehill",
        "axios",
        "CBSNews",
        "FortuneMagazine",
        "ABC",
        "ABCPolitics",
        "ABCWorldNews",
        "politico",
        "nytimes",
        "propublica",
        "CNBC",
        "CNBCnow",
        "thisisinsider",
        "NBCPolitics",
        "NBCNews",
        "vulture",
        "bpolitics",
        "BostonGlobe",
        "Forbes",
        "TIME",
        "washingtonpost",
        "Newsweek",
        "CBC",
        "BusinessInsider",
        "WSJ",
        "TheAtlantic",
    ]
    create_news_table(cursor, high_quality_news_users)

    conn.close()
    print(f"Base database created: {db_path}")


def load_users(cursor, user_file):
    print("Loading users from CSV...")
    df = pd.read_csv(user_file)
    print(f"Loaded {len(df)} users from {user_file}")

    user_id_mapping = {}

    start_id = 100000000
    for _, row in df.iterrows():
        username = row['username']
        name = row['name']
        bio = row.get('bio', '')
        created_at = row.get('created_at', '')

        cursor.execute(
            "INSERT OR IGNORE INTO user (user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (start_id, None, username, name, bio, created_at, 0, 0)
        )
        user_id_mapping[username] = start_id
        start_id += 1

    cursor.connection.commit()
    print(f"Created {len(df)} users")
    print("All users committed to database")
    return user_id_mapping


def load_posts(cursor, post_file, user_id_mapping):
    print("Loading posts from CSV...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS political (
            post_id INTEGER PRIMARY KEY,
            FOREIGN KEY(post_id) REFERENCES post(post_id)
        )
    """)
    print("Ensured 'political' table exists.")

    df = pd.read_parquet(post_file)

    post_batch = []
    political_batch = []

    current_post_id = 1

    for _, post_row in df.iterrows():
        user_id = user_id_mapping.get(post_row['user'])
        content = post_row['tweet']
        created_at = post_row['date']
        num_likes = post_row.get('likes', 0)
        num_comments = post_row.get('replies', 0)
        num_quotes = post_row.get('quotes', 0)
        num_shares = post_row.get('retweets', 0)
        is_political = post_row.get('is_political', False)

        post_batch.append((
            user_id, content, created_at, num_likes, num_comments, num_quotes, num_shares
        ))

        if is_political:
            political_batch.append((current_post_id,))

        current_post_id += 1

    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_comments, num_quotes, num_shares) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    cursor.executemany(post_insert_query, post_batch)
    print(f"Created {len(post_batch)} posts")

    if political_batch:
        cursor.executemany("INSERT OR IGNORE INTO political (post_id) VALUES (?)", political_batch)
        print(f"Identified and tagged {len(political_batch)} political posts")

    cursor.connection.commit()
    print("All posts and political tags committed to database")


def create_news_table(cursor, news_list):
    """Create a news table from posts by the given news account names."""
    print("Creating news table...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            post_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            user_name TEXT,
            FOREIGN KEY(post_id) REFERENCES post(post_id),
            FOREIGN KEY(user_id) REFERENCES user(user_id)
        )
    """)

    news_user_ids = []
    for news_name in news_list:
        cursor.execute("SELECT user_id, user_name FROM user WHERE user_name = ?", (news_name,))
        result = cursor.fetchone()
        if result:
            user_id, user_name = result
            news_user_ids.append((user_id, user_name))
            print(f"  Found news account: {user_name} (user_id: {user_id})")
        else:
            print(f"  Warning: News account '{news_name}' not found in user table")

    if not news_user_ids:
        print("  No news accounts found! Please check your news_list.")
        return

    total_news_posts = 0
    for user_id, user_name in news_user_ids:
        cursor.execute("SELECT post_id FROM post WHERE user_id = ?", (user_id,))
        posts = cursor.fetchall()

        if posts:
            news_data = [(post[0], user_id, user_name) for post in posts]
            cursor.executemany(
                "INSERT OR IGNORE INTO news (post_id, user_id, user_name) VALUES (?, ?, ?)",
                news_data
            )
            print(f"  Added {len(posts)} posts from {user_name}")
            total_news_posts += len(posts)
        else:
            print(f"  No posts found for {user_name}")

    cursor.connection.commit()
    print(f"News table created successfully with {total_news_posts} posts from {len(news_user_ids)} news accounts")

    cursor.execute("SELECT COUNT(*) FROM news")
    count = cursor.fetchone()[0]
    print(f"Total records in news table: {count}")


if __name__ == "__main__":
    user_file = "./data/final_users.csv"
    post_file = "./data/all_posts_final.parquet"
    db_path = "./database/preloaded_data.db"
    news_file = "./data/combined_matching_potential_news.csv"

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    create_base_db(user_file, post_file, db_path, news_file)

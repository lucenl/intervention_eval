-- This is the schema definition for the rec history in each step table
CREATE TABLE rec_history (
    step INTEGER,
    user_id INTEGER,
    post_id INTEGER,
    order_index INTEGER
);
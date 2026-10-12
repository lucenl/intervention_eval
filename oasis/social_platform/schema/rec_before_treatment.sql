-- This is the schema definition for the recommendations before treatment in each step
CREATE TABLE rec_before_treatment (
    step INTEGER,
    user_id INTEGER,
    post_id INTEGER,
    order_index INTEGER
);

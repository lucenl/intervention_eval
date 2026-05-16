import random
import logging
from typing import List, Dict, Any, Tuple, Set
from treatment_base import Treatment
import sqlite3


class UnifiedExperimentTreatment(Treatment):
    """Unified experiment controller."""

    def __init__(self, celebrities: List[Dict[str, str]] = None,
                 celebrity_seed: int = 20260413):
        super().__init__()
        self.logger = logging.getLogger("social.treatment")
        self.treatment_type = "unified"

        self.GROUP_CONTROL = 0
        self.GROUP_INJECT_2 = 1
        self.GROUP_INJECT_4 = 2
        self.GROUP_UPRANK = 3
        self.GROUP_CELEB = 4

        self.celebrities = celebrities or []
        self.celebrity_seed = celebrity_seed
        self.user_assignments = {}
        self.step_news_cache = {}

    def _pick_celebrity(self, user_id: int, step_number: int, post_id: int) -> Dict[str, str]:
        """Deterministic celebrity assignment keyed on (user, step, post)."""
        rng = random.Random(f"{self.celebrity_seed}:{user_id}:{step_number}:{post_id}")
        return rng.choice(self.celebrities)

    def _get_user_group(self, user_id: int) -> int:
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = user_id % 5
        return self.user_assignments[user_id]

    def _get_shared_news(self, step_number: int, all_news_ids: List[int]) -> Dict[str, List[int]]:
        if step_number in self.step_news_cache:
            return self.step_news_cache[step_number]

        available_count = len(all_news_ids)
        if available_count < 4:
            self.logger.warning(f"Not enough news for step {step_number}, need 4, got {available_count}")
            selected = all_news_ids
        else:
            selected = random.sample(list(all_news_ids), 4)

        news_2 = selected[:2]
        news_4 = selected[:4]

        cache = {'news_2': news_2, 'news_4': news_4}
        self.step_news_cache[step_number] = cache
        return cache

    def apply(self, rec_matrix: List[List[int]], post_table: List[Dict], news_table: List[Dict], active_user_ids: List[int], max_rec_post_len: int, step_number: int) -> Tuple[List[List[int]], List[Tuple]]:

        news_ids_pool = [n['post_id'] for n in news_table]
        shared_news = self._get_shared_news(step_number, news_ids_pool)

        new_rec_matrix = []
        log_records = []

        for user_id, rec in zip(active_user_ids, rec_matrix):
            group = self._get_user_group(user_id)
            new_rec = rec.copy()

            if group == self.GROUP_CONTROL:
                log_records.append((user_id, "control", -1, -1, -1, step_number))

            elif group == self.GROUP_INJECT_2:
                new_rec = self._inject_news(new_rec, shared_news['news_2'])
                self._log_injection(log_records, user_id, "inject_2", shared_news['news_2'], step_number)

            elif group == self.GROUP_INJECT_4:
                new_rec = self._inject_news(new_rec, shared_news['news_4'])
                self._log_injection(log_records, user_id, "inject_4", shared_news['news_4'], step_number)

            elif group == self.GROUP_UPRANK:
                new_rec = self._inject_news(new_rec, shared_news['news_2'])
                new_rec = self._uprank_news(new_rec, shared_news['news_2'])
                self._log_injection(log_records, user_id, "uprank", shared_news['news_2'], step_number)

            elif group == self.GROUP_CELEB:
                new_rec = self._inject_news(new_rec, shared_news['news_2'])
                self._log_injection(log_records, user_id, "celebrity_logic", shared_news['news_2'], step_number)

            if len(new_rec) > max_rec_post_len:
                new_rec = new_rec[:max_rec_post_len]

            new_rec_matrix.append(new_rec)

        return new_rec_matrix, log_records

    def apply_display(self, posts: List[Dict[str, any]], user_id: int, step_number: int, news_ids: Set[int]) -> Tuple[List[Dict[str, any]], List[Tuple]]:
        group = self._get_user_group(user_id)
        log_records = []

        if group != self.GROUP_CELEB:
            return posts, log_records

        modified_posts = []
        for post in posts:
            post_id = post.get('post_id')

            if post_id in news_ids:
                if not self.celebrities:
                    modified_posts.append(post)
                    continue

                celeb = self._pick_celebrity(user_id, step_number, post_id)
                new_name = f"{celeb['realname']} (☑️ Verified)"
                new_username = celeb['username']

                mod_post = post.copy()
                mod_post['name'] = new_name
                mod_post['user_name'] = new_username

                modified_posts.append(mod_post)

                log_records.append((
                    user_id, "celebrity_render", post_id,
                    post.get('user_name'), post.get('name'),
                    new_username, new_name, step_number
                ))
            else:
                modified_posts.append(post)

        return modified_posts, log_records

    def _inject_news(self, rec_list: List[int], news_to_inject: List[int]) -> List[int]:
        current_news = set(news_to_inject)
        candidates = [i for i, pid in enumerate(rec_list) if pid not in current_news]
        needed = [n for n in news_to_inject if n not in rec_list]

        if not needed:
            return rec_list

        replace_indices = random.sample(candidates, min(len(candidates), len(needed)))

        new_list = rec_list.copy()
        for idx, news_id in zip(replace_indices, needed):
            new_list[idx] = news_id

        return new_list

    def _uprank_news(self, rec_list: List[int], target_news: List[int]) -> List[int]:
        new_list = rec_list.copy()
        target_set = set(target_news)
        slots = [0, 2]
        news_indices = [i for i, pid in enumerate(new_list) if pid in target_set]

        if not news_indices:
            return new_list

        for i, news_idx in enumerate(news_indices):
            if i >= len(slots):
                break
            target_slot = slots[i]
            if news_idx == target_slot:
                continue
            new_list[target_slot], new_list[news_idx] = new_list[news_idx], new_list[target_slot]

        return new_list

    def _log_injection(self, logs, uid, type_str, news_list, step):
        for nid in news_list:
            logs.append((uid, type_str, -1, nid, -1, step))

    def save_assignments_to_db(self, db_path: str):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS treatment_assignment (
                    user_id INTEGER PRIMARY KEY,
                    group_id INTEGER,
                    group_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            group_names = {0: "Control", 1: "Inject-2", 2: "Inject-4", 3: "Uprank", 4: "Celeb"}
            data = [(u, g, group_names.get(g, "Unknown")) for u, g in self.user_assignments.items()]

            cursor.executemany(
                "INSERT OR REPLACE INTO treatment_assignment (user_id, group_id, group_name) VALUES (?, ?, ?)", data)
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Save assignment error: {e}")

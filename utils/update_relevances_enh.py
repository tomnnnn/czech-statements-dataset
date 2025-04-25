import json
import signal
import sys
from collections import defaultdict
from src.dataset_manager.orm import *
from sqlalchemy import insert
import time

# Cross-platform single character input
try:
    import msvcrt
    def getch():
        while True:
            ch = msvcrt.getch().decode('utf-8')
            if ch in ['y', 'n', 's', 'q', 'Y', 'N', 'S', 'Q']:
                return ch.lower()
except ImportError:
    import tty
    import termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1).lower()
                if ch in ['y', 'n', 's', 'q']:
                    return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ANSI escape codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
CLEAR = "\033[H\033[J"
RESET = "\033[0m"

start_time = time.time()

def print_header(statement, statement_id, progress, total, done_in_statement, total_in_statement, estimated_time_left):
    print(CLEAR)
    print(f"{BOLD}{GREEN}Statement [{statement_id}]:{RESET} {RED}{statement}{RESET}")
    print(f"\n{BOLD}Progress:{RESET} {progress}/{total} ({progress/total:.1%})")
    print(f"{BOLD}Statement Progress:{RESET} {done_in_statement}/{total_in_statement} ({done_in_statement/total_in_statement:.1%})")
    print(f"{BOLD}Estimated Time Left:{RESET} {estimated_time_left} s ({estimated_time_left/60:.1f} min)")


def print_segment(article_title, article_id, segment_text, segment_id):
    print(f"\n{BOLD}{BLUE}Article [{article_id}]:{RESET} {article_title}")
    print(f"\n{BOLD}{CYAN}Segment [{segment_id}]:{RESET} {segment_text}\n")

dataset = init_db("datasets/curated_updated.sqlite")

# Step 1: Load all necessary data in bulk (avoids multiple DB queries)
with open("./retriever_training/predicted_segments_2.json", "r") as f:
    predicted_segments = json.load(f)

predicted_ids = [seg["id"] for seg in predicted_segments]

# Preload all segments in one query (avoiding per-segment fetch)
segments = dataset.query(Segment).filter(Segment.id.in_(predicted_ids)).all()
segments_by_id = {segment.id: segment for segment in segments}

# Preload segment relevances in one query (avoiding per-statement fetch)
existing_relevances = dataset.query(SegmentRelevance).filter(SegmentRelevance.segment_id.in_(predicted_ids)).all()
relevances_set = {(r.segment_id, r.statement_id) for r in existing_relevances}

# Preload statements in one query
statement_ids = list(set(seg["statement_id"] for seg in predicted_segments))
statements = dataset.query(Statement).filter(Statement.id.in_(statement_ids)).all()
statements_by_id = {stmt.id: stmt for stmt in statements}

# Build segment_map (only unannotated segments), discard duplicates
seen_segments = set()

segment_map = defaultdict(list)

for segment in predicted_segments:
    segment_id, statement_id = segment["id"], segment["statement_id"]
    
    # Check if the segment_id exists in segments_by_id and it's not already processed
    if segment_id in segments_by_id and (segment_id, statement_id) not in relevances_set and (segment_id, statement_id) not in seen_segments:
        segment_map[statement_id].append(segments_by_id[segment_id])
        seen_segments.add((segment_id, statement_id))  # Mark this combination as processed

print(f"Filtered out {len(predicted_segments) - sum(len(segs) for segs in segment_map.values())} segments")

batch_relevances = []
total = sum(len(segments) for segments in segment_map.values())
total_done = 0

def save_relevances(relevances):
    if relevances:
        dataset.bulk_insert_mappings(SegmentRelevance, relevances)
        dataset.commit()  # Use bulk commit for efficiency
        print(f"{GREEN}✔ Saved{RESET}")

def save_on_exit(signum, frame):
    print("\nSaving progress...")
    if batch_relevances:
        save_relevances(batch_relevances)
    sys.exit(0)

signal.signal(signal.SIGINT, save_on_exit)

try:
    for statement_id, segments in segment_map.items():
        done_in_statement = 0
        total_in_statement = len(segments)

        statement = statements_by_id.get(statement_id)
        if not statement:
            continue
        
        current_statement = statement.statement  # Use ORM object instead of dictionary
        segments_processed = 0

        while segments_processed < len(segments):
            segment = segments[segments_processed]
            segment_id = segment.id
            article = segment.article  # Already preloaded, avoids extra query

            estimated_time_left = (time.time() - start_time) / total_done * (total - total_done) if total_done else -1
            print_header(current_statement, statement_id, total_done, total, done_in_statement, total_in_statement, estimated_time_left)
            print_segment(article.title, article.id, segment.text, segment.id)  # Use ORM object directly

            print(f"{BOLD}Choose:{RESET} [Y]es/[N]o/[S]kip/[Q]uit ", end='', flush=True)
            choice = getch()

            if choice == 'q':
                print(f"\n{YELLOW}⚠ Quitting...{RESET}")
                sys.exit(0)
            if choice == 'y':
                batch_relevances.append({
                    'segment_id': segment_id,
                    'statement_id': statement_id,
                    'relevance': 1
                })
                print(f"{GREEN}✔ Relevant{RESET}")
                total_done += 1
                segments_processed += 1
                done_in_statement += 1
            elif choice == 'n':
                batch_relevances.append({
                    'segment_id': segment_id,
                    'statement_id': statement_id,
                    'relevance': 0
                })
                print(f"{RED}✖ Not Relevant{RESET}")
                total_done += 1
                segments_processed += 1
                done_in_statement += 1
            elif choice == 's':
                print(f"{YELLOW}↷ Skipped{RESET}")
                segments_processed += 1
                done_in_statement += 1

except Exception as e:
    print(f"Error: {e}")
finally:
    save_relevances(batch_relevances)  # Final bulk save

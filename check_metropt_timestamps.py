import csv
from datetime import datetime
from pathlib import Path

source = Path(
    r"D:\Project final\mer-jit-llm-fgcs\data\MetroPT3(CompressorDatase).csv"
)

timestamp_format = "%Y-%m-%d %H:%M:%S"

rows = 0
first = None
last = None
previous = None
adjacent_duplicates = 0
backward_steps = 0
gaps = 0

with source.open(
    "r",
    encoding="utf-8-sig",
    errors="replace",
    newline=""
) as file:
    reader = csv.DictReader(file)

    for row in reader:
        current = datetime.strptime(row["timestamp"], timestamp_format)

        if first is None:
            first = current

        if previous is not None:
            delta_seconds = (current - previous).total_seconds()

            if current == previous:
                adjacent_duplicates += 1

            if current < previous:
                backward_steps += 1

            if delta_seconds > 1:
                gaps += 1

        previous = current
        last = current
        rows += 1

print("rows=", rows)
print("first=", first)
print("last=", last)
print("monotonic_non_decreasing=", backward_steps == 0)
print("adjacent_duplicate_timestamps=", adjacent_duplicates)
print("backward_steps=", backward_steps)
print("gaps_greater_than_one_second=", gaps)
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
random.seed(42)
GROUPS = ["P", "Q", "R", "S"]

# Dummy subjects for generating test data
DUMMY_SUBJECTS = [
    "Geography", "Art", "RS", "History", "French", "Spanish", "German",
    "DT", "Food", "Business", "PE", "Computing", "Music", "Drama"
]
NUM_DUMMY_RECORDS = 200
NUM_CHOICES_PER_RECORD = 4

# =========================
# FUNCTIONS
# =========================

def generate_dummy_records():
    """Generate dummy records with random subjects."""
    records = {
        f"R{i+1}": random.sample(DUMMY_SUBJECTS, NUM_CHOICES_PER_RECORD)
        for i in range(NUM_DUMMY_RECORDS)
    }
    return records

def load_records_from_csv(file_path):
    """Load records from CSV with columns: Record, Choice1-Choice4"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    df_csv = pd.read_csv(file_path)
    records = {
        row['Record']: [row['Choice1'], row['Choice2'], row['Choice3'], row['Choice4']]
        for _, row in df_csv.iterrows()
    }
    return records

def greedy_min_sets_assignment(records, groups):
    """Deterministic greedy assignment to minimise total sets"""
    groups_map = {g: set() for g in groups}
    assignment_map = {}
    group_subject_usage = {g: Counter() for g in groups}

    for r_name, subjects in records.items():
        assignment = {}
        subjects_remaining = set(subjects)

        for g in groups:
            # Prefer subjects already in the group (reuse)
            candidate_subjects = subjects_remaining & set(groups_map[g])
            if candidate_subjects:
                chosen = max(candidate_subjects, key=lambda s: group_subject_usage[g][s])
            else:
                chosen = sorted(subjects_remaining)[0]  # deterministic tie-breaker

            assignment[g] = chosen
            subjects_remaining.remove(chosen)
            groups_map[g].add(chosen)
            group_subject_usage[g][chosen] += 1

        assignment_map[r_name] = assignment

    return groups_map, assignment_map, group_subject_usage

def visualize_subject_usage(group_subject_usage, groups):
    """Bar chart of subject usage per group"""
    plt.figure(figsize=(14, 6))
    for g in groups:
        subjects_sorted = sorted(group_subject_usage[g].keys())
        counts = [group_subject_usage[g][s] for s in subjects_sorted]
        plt.bar([f"{s}_{g}" for s in subjects_sorted], counts, label=g)

    plt.xlabel("Subject_Group")
    plt.ylabel("Number of Records Using Subject in Group")
    plt.title("Subject Usage / Sets per Group")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# MAIN SCRIPT
# =========================
def main():
    print("Choose data source:")
    print("1 - Dummy data")
    print("2 - CSV file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        records = generate_dummy_records()
        print(f"\nGenerated {len(records)} dummy records.")
    elif choice == "2":
        file_path = input("Enter path to your CSV file: ").strip()
        records = load_records_from_csv(file_path)
        print(f"\nLoaded {len(records)} records from CSV.")
    else:
        print("Invalid choice. Exiting.")
        return

    # Run assignment
    groups_map, assignment_map, group_subject_usage = greedy_min_sets_assignment(records, GROUPS)

    # Count sets
    sets_per_group = {g: len(groups_map[g]) for g in GROUPS}
    total_sets = sum(sets_per_group.values())
    print("\nSets per group:", sets_per_group)
    print("Total number of sets:", total_sets)

    # Build DataFrame
    df_result = pd.DataFrame([
        assignment_map[r] for r in records.keys()
    ], index=records.keys(), columns=GROUPS)

    print("\nFirst 10 records:")
    print(df_result.head(10))

    # Save to CSV
    output_csv = "subjects_minimized_sets_assignment.csv"
    df_result.to_csv(output_csv)
    print(f"\nFull assignment saved to '{output_csv}'")

    # Visualize
    visualize_subject_usage(group_subject_usage, GROUPS)

if __name__ == "__main__":
    main()
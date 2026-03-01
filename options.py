from ortools.sat.python import cp_model
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

# =====================================================
# CONFIGURATION
GROUPS = ["P", "Q", "R", "S"]
NUM_CHOICES_PER_RECORD = 4

DUMMY_SUBJECTS = [
    "Geography", "Art", "RS", "History", "French", "Spanish", "German",
    "DT", "Food", "Business", "PE", "Computing", "Music", "Drama"
]

# =====================================================
# DATA FUNCTIONS
def generate_dummy_records(num_records):
    return {
        f"R{i+1}": random.sample(DUMMY_SUBJECTS, NUM_CHOICES_PER_RECORD)
        for i in range(num_records)
    }

def load_records_from_csv(file_path):
    df = pd.read_csv(file_path)
    records = {
        row["Record"]: [row["Choice1"], row["Choice2"], row["Choice3"], row["Choice4"]]
        for _, row in df.iterrows()
    }
    return records

def load_constraints_csv(file_path, column_name):
    """
    Returns a dictionary of {Subject: value} for the specified column.
    """
    df = pd.read_csv(file_path)
    constraints = {}
    for _, row in df.iterrows():
        subject = row['Subject']
        if column_name in row and not pd.isna(row[column_name]):
            constraints[subject] = int(row[column_name])
    return constraints

# =====================================================
# SOLVER
def solve_min_sets(records, groups, subject_limits=None, max_students_per_set=None, max_students_subject=None):
    model = cp_model.CpModel()
    record_names = list(records.keys())
    all_subjects = sorted({s for choices in records.values() for s in choices})

    # VARIABLES
    x = {}
    for r in record_names:
        for g in groups:
            for s in all_subjects:
                x[r,g,s] = model.NewBoolVar(f"x_{r}_{g}_{s}")

    y = {}
    for g in groups:
        for s in all_subjects:
            y[g,s] = model.NewBoolVar(f"y_{g}_{s}")

    # CONSTRAINTS
    for r in record_names:
        for g in groups:
            model.Add(sum(x[r,g,s] for s in records[r]) == 1)
            for s in all_subjects:
                if s not in records[r]:
                    model.Add(x[r,g,s] == 0)

    for g in groups:
        for s in all_subjects:
            model.AddMaxEquality(y[g,s], [x[r,g,s] for r in record_names])

    # Subject max sets
    if subject_limits:
        for sub, max_sets in subject_limits.items():
            if sub in all_subjects:
                model.Add(sum(y[g,sub] for g in groups) <= max_sets)

    # Maximum students per set (overall)
    if max_students_per_set:
        for g in groups:
            for s in all_subjects:
                model.Add(sum(x[r,g,s] for r in record_names) <= max_students_per_set)

    # Maximum students per subject (specific)
    if max_students_subject:
        for g in groups:
            for s, limit in max_students_subject.items():
                if s in all_subjects:
                    model.Add(sum(x[r,g,s] for r in record_names) <= limit)

    # OBJECTIVE
    model.Minimize(sum(y[g,s] for g in groups for s in all_subjects))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
    status = solver.Solve(model)
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise ValueError("No feasible solution found with the given constraints.")

    # Build outputs
    assignment_map = {}
    groups_map = {g: set() for g in groups}
    group_subject_usage = {g:{} for g in groups}
    for r in record_names:
        assignment_map[r] = {}
        for g in groups:
            for s in records[r]:
                if solver.BooleanValue(x[r,g,s]):
                    assignment_map[r][g] = s
                    groups_map[g].add(s)
                    group_subject_usage[g][s] = group_subject_usage[g].get(s,0)+1
    return assignment_map, groups_map, group_subject_usage

# =====================================================
# REPORTING
def print_sets_summary(groups_map, group_subject_usage, viability_threshold):
    print("\nDistinct sets in each group with sizes:\n")
    all_sets = []
    for g in GROUPS:
        subjects_sorted = sorted(groups_map[g])
        print(f"Group {g} ({len(subjects_sorted)} sets):")
        for s in subjects_sorted:
            size = group_subject_usage[g][s]
            flag = " ⚠ BELOW VIABILITY" if size < viability_threshold else ""
            print(f"  - {s}: {size} students{flag}")
            all_sets.append((g, s, size))
        print()
    ranked = sorted(all_sets, key=lambda x: x[2], reverse=True)
    print("\nAll sets ranked largest to smallest:\n")
    for g, s, size in ranked:
        print(f"{g} - {s}: {size}")
    return ranked

def save_outputs(assignment_map, groups_map, group_subject_usage, ranked_sets):
    pd.DataFrame(assignment_map).T.to_csv("optimal_assignment.csv")
    sets_data = []
    for g in GROUPS:
        for s in sorted(groups_map[g]):
            sets_data.append({"Group": g, "Subject": s, "Size": group_subject_usage[g][s]})
    pd.DataFrame(sets_data).to_csv("distinct_sets_summary.csv", index=False)
    pd.DataFrame(ranked_sets, columns=["Group","Subject","Size"]).to_csv("ranked_sets.csv", index=False)
    print("\nSaved:")
    print(" - optimal_assignment.csv")
    print(" - distinct_sets_summary.csv")
    print(" - ranked_sets.csv")

def visualize(group_subject_usage):
    plt.figure(figsize=(14,6))
    for g in GROUPS:
        subjects = sorted(group_subject_usage[g].keys())
        counts = [group_subject_usage[g][s] for s in subjects]
        plt.bar([f"{s}_{g}" for s in subjects], counts, label=g)
    plt.xticks(rotation=90)
    plt.title("Set Sizes per Group")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================================
# MAIN
def main():
    print("Choose data source:")
    print("1 - Dummy data")
    print("2 - CSV file")
    choice = input("Enter 1 or 2: ").strip()

    records = {}
    subject_limits = None
    max_students_subject = None
    max_students_per_set = None

    if choice == "1":
        num_records_input = input("Enter number of dummy records to generate: ").strip()
        num_records = int(num_records_input)
        records = generate_dummy_records(num_records)
        print(f"\nGenerated {len(records)} dummy records.")
    elif choice == "2":
        records_path = input("Enter records CSV path: ").strip()
        records = load_records_from_csv(records_path)
        print(f"\nLoaded {len(records)} records.")
    else:
        print("Invalid choice.")
        return

    # =====================================================
    # Automatically load constraints if files exist
    if os.path.exists("max_students_per_set.csv"):
        max_students_subject = load_constraints_csv("max_students_per_set.csv", "MaxStudents")
        print("Loaded maximum students per subject from max_students_per_set.csv:", max_students_subject)

    if os.path.exists("max_sets_per_subject.csv"):
        subject_limits = load_constraints_csv("max_sets_per_subject.csv", "MaxSets")
        print("Loaded maximum sets per subject from max_sets_per_subject.csv:", subject_limits)

    # Optional overall max students per set
    overall_limit_input = input("\nEnter maximum students per set (overall, press Enter to skip): ").strip()
    if overall_limit_input:
        max_students_per_set = int(overall_limit_input)

    viability_threshold = int(input("\nEnter minimum viable class size (e.g. 10): "))

    print("\nSolving optimal assignment...")
    assignment_map, groups_map, group_subject_usage = solve_min_sets(
        records, GROUPS, subject_limits, max_students_per_set, max_students_subject
    )

    sets_per_group = {g: len(groups_map[g]) for g in GROUPS}
    total_sets = sum(sets_per_group.values())
    print("\nSets per group:", sets_per_group)
    print("Total sets:", total_sets)

    ranked_sets = print_sets_summary(groups_map, group_subject_usage, viability_threshold)
    save_outputs(assignment_map, groups_map, group_subject_usage, ranked_sets)
    visualize(group_subject_usage)

if __name__=="__main__":
    main()
import os
import pandas as pd
import random
from ortools.sat.python import cp_model

# -----------------------------
# CONFIG
# -----------------------------
GROUPS = ["A", "B", "C", "D"]

# -----------------------------
# LOAD CONSTRAINT FILES
# -----------------------------
def load_max_students_per_set():
    if os.path.exists("max_students_per_set.csv"):
        df = pd.read_csv("max_students_per_set.csv")
        df.columns = df.columns.str.strip()
        return dict(zip(df["Subject"], df["MaxStudents"]))
    return {}

def load_max_sets_per_subject():
    if os.path.exists("max_sets_per_subject.csv"):
        df = pd.read_csv("max_sets_per_subject.csv")
        df.columns = df.columns.str.strip()
        return dict(zip(df["Subject"], df["MaxSets"]))
    return {}

# -----------------------------
# DUMMY DATA
# -----------------------------
def generate_dummy_data(n):
    subjects = ["DT", "Food", "Art", "Business", "History", "Geography"]
    records = []
    for i in range(n):
        num_choices = random.choice([3, 4])
        choices = random.sample(subjects, num_choices)
        record = {"Student": f"Student_{i+1}"}
        for idx, group in enumerate(GROUPS):
            if idx < num_choices:
                record[group] = choices[idx]
            else:
                record[group] = ""
        records.append(record)
    return pd.DataFrame(records)

# -----------------------------
# SOLVER
# -----------------------------
def solve(records, subject_max_sets, subject_max_size, default_max_set_size):
    model = cp_model.CpModel()
    students = records["Student"].tolist()

    # build student choices
    student_choices = {}
    subjects = set()
    for _, row in records.iterrows():
        s = row["Student"]
        choices = []
        for g in GROUPS:
            if pd.notna(row[g]) and row[g] != "":
                choices.append(row[g])
                subjects.add(row[g])
        student_choices[s] = choices
    subjects = list(subjects)

    # max sets per subject
    max_sets = {}
    for subj in subjects:
        if subj in subject_max_sets:
            max_sets[subj] = subject_max_sets[subj]
        else:
            # minimal number of sets to fit students with general max
            max_sets[subj] = (len([s for s in students if subj in student_choices[s]]) // default_max_set_size) + 1

    x = {}
    y = {}
    for s in students:
        for subj in student_choices[s]:
            for k in range(max_sets[subj]):
                x[(s, subj, k)] = model.NewBoolVar(f"x_{s}_{subj}_{k}")
    for subj in subjects:
        for k in range(max_sets[subj]):
            y[(subj, k)] = model.NewBoolVar(f"y_{subj}_{k}")

    # Each student assigned to number of choices they have
    for s in students:
        total_choices = len(student_choices[s])
        model.Add(
            sum(x[(s, subj, k)] for subj in student_choices[s] for k in range(max_sets[subj])) == total_choices
        )

    # Link assignments to active set
    for s in students:
        for subj in student_choices[s]:
            for k in range(max_sets[subj]):
                model.Add(x[(s, subj, k)] <= y[(subj, k)])

    # Max students per set enforced strictly
    for subj in subjects:
        limit = subject_max_size.get(subj, default_max_set_size)
        for k in range(max_sets[subj]):
            model.Add(
                sum(x[(s, subj, k)] for s in students if subj in student_choices[s]) <= limit * y[(subj, k)]
            )

    # Objective: minimize total sets used
    model.Minimize(sum(y.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None
    return solver, (x, y)

# -----------------------------
# EXPORT RESULTS
# -----------------------------
def export_results(records, solver, x, y):
    students = records["Student"].tolist()
    student_rows = []
    set_rows = []
    subject_set_count = {}
    group_sets_rows = []

    for (subj, k), var in y.items():
        if solver.Value(var) == 1:
            subject_set_count[subj] = subject_set_count.get(subj, 0) + 1
            students_in_set = []
            groups_in_set = []
            for (s, sj, kk), xv in x.items():
                if sj == subj and kk == k and solver.Value(xv) == 1:
                    students_in_set.append(str(s))
                    student_rows.append({"Student": s, "Subject": subj, "SetNumber": k})
                    # determine which group this student assigned
                    row = records[records["Student"] == s].iloc[0]
                    for g in GROUPS:
                        if row[g] == subj:
                            groups_in_set.append(g)
                            break
            set_rows.append({
                "Subject": subj,
                "SetNumber": k,
                "Students": ", ".join(students_in_set),
                "SetSize": len(students_in_set)
            })
            group_sets_rows.append({
                "Subject": subj,
                "SetNumber": k,
                "Groups": ", ".join(sorted(set(groups_in_set)))
            })

    # write CSVs
    pd.DataFrame(student_rows).to_csv("student_assignments.csv", index=False)
    pd.DataFrame(set_rows).to_csv("sets_detail.csv", index=False)
    pd.DataFrame(group_sets_rows).to_csv("group_sets.csv", index=False)

    summary_rows = []
    total_sets = 0
    for subj, count in subject_set_count.items():
        summary_rows.append({"Subject": subj, "NumberOfSets": count})
        total_sets += count
    pd.DataFrame(summary_rows).to_csv("set_summary.csv", index=False)

    print("\n--- SUMMARY ---")
    print(f"Total Sets: {total_sets}")
    for subj, count in subject_set_count.items():
        print(f"{subj}: {count} sets")
    print("\nFiles written: student_assignments.csv, sets_detail.csv, set_summary.csv, group_sets.csv")

# -----------------------------
# MAIN
# -----------------------------
def main():
    # ask user for general max set size
    default_max_set_size = int(input("Enter general maximum students per set (default 24): ") or 24)

    use_dummy = input("Use dummy data? (y/n): ").lower()
    if use_dummy == "y":
        n = int(input("How many dummy students? "))
        records = generate_dummy_data(n)
        records.to_csv("dummy_data.csv", index=False)
        print("Dummy data written to dummy_data.csv")
    else:
        filename = input("Enter CSV filename: ")
        records = pd.read_csv(filename)

    # clean headers
    records.columns = records.columns.str.strip()
    if len(records.columns) < 4:
        raise ValueError("CSV must contain at least 1 student column and 3 choice columns.")

    # Student column
    student_col = records.columns[0]
    records = records.rename(columns={student_col: "Student"})
    records["Student"] = records["Student"].astype(str)

    # Choice columns
    choice_columns = records.columns[1:]
    choice_columns = choice_columns[:4]
    rename_map = {col: GROUPS[idx] for idx, col in enumerate(choice_columns)}
    records = records.rename(columns=rename_map)
    if "D" not in records.columns:
        records["D"] = ""

    print("\nUsing columns:")
    print(records[["Student", "A", "B", "C", "D"]].head())

    subject_max_size = load_max_students_per_set()
    subject_max_sets = load_max_sets_per_subject()
    print("\nLoaded constraints:")
    print("Max students per set:", subject_max_size)
    print("Max sets per subject:", subject_max_sets)

    print("\nAttempting STRICT solve...")
    solver, vars_tuple = solve(records, subject_max_sets, subject_max_size, default_max_set_size)
    if solver is None:
        print("⚠ No feasible solution found.")
        return

    x, y = vars_tuple
    print("\nSolution found.")
    export_results(records, solver, x, y)

if __name__ == "__main__":
    main()
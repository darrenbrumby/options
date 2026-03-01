import pandas as pd
import os
import math
import random
from ortools.sat.python import cp_model

# -----------------------------
# CONFIGURATION
# -----------------------------
GROUPS = ['P', 'Q', 'R', 'S']
DEFAULT_MAX_SET_SIZE = 30
MAX_STUDENTS_FILE = 'max_students_per_set.csv'
MAX_SETS_FILE = 'max_sets_per_subject.csv'

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_constraints():
    """Load constraints from CSV files if they exist."""
    max_students_subject = {}
    max_sets_subject = {}
    if os.path.exists(MAX_STUDENTS_FILE):
        df = pd.read_csv(MAX_STUDENTS_FILE)
        for _, row in df.iterrows():
            max_students_subject[row['Subject']] = int(row['MaxStudents'])
    if os.path.exists(MAX_SETS_FILE):
        df = pd.read_csv(MAX_SETS_FILE)
        for _, row in df.iterrows():
            max_sets_subject[row['Subject']] = int(row['MaxSets'])
    return max_students_subject, max_sets_subject

def adjust_constraints(records, subject_max_size, subject_max_sets, default_max_size, choice_columns):
    """Adjust max sizes and sets to guarantee feasibility if possible."""
    adjusted_max_size = subject_max_size.copy()
    adjusted_max_sets = subject_max_sets.copy()
    all_subjects = set()
    for col in choice_columns:
        all_subjects.update(records[col].dropna().unique())
    for subj in all_subjects:
        num_students = sum(records[col].eq(subj).sum() for col in choice_columns)
        max_per_set = adjusted_max_size.get(subj, default_max_size)
        min_sets_needed = math.ceil(num_students / max_per_set)
        adjusted_max_sets[subj] = max(adjusted_max_sets.get(subj, 0), min_sets_needed)
        adjusted_max_size[subj] = max(max_per_set, math.ceil(num_students / adjusted_max_sets[subj]))
    return adjusted_max_size, adjusted_max_sets

# -----------------------------
# SOLVER
# -----------------------------
def solve_option_blocks(records, subject_max_sets, subject_max_size, choice_columns, default_max_size=DEFAULT_MAX_SET_SIZE):
    model = cp_model.CpModel()
    students = records['Student'].tolist()
    all_subjects = set()
    for col in choice_columns:
        all_subjects.update(records[col].dropna().unique())

    # Determine max sets per subject
    max_sets = {}
    for subj in all_subjects:
        max_sets[subj] = subject_max_sets.get(subj, 4)  # default 4 sets

    # Variables
    x = {}  # student -> subject -> set
    g = {}  # subject -> set -> group
    y = {}  # subject -> set used or not
    for subj in all_subjects:
        for k in range(max_sets[subj]):
            y[subj, k] = model.NewBoolVar(f"y_{subj}_{k}")
            g[subj, k] = {}
            for grp in GROUPS:
                g[subj, k][grp] = model.NewBoolVar(f"g_{subj}_{k}_{grp}")
            model.Add(sum(g[subj, k][grp] for grp in GROUPS) == 1)

    for s_idx, s in enumerate(students):
        for col_idx, col in enumerate(choice_columns):
            subj = records.loc[s_idx, col]
            if pd.isna(subj):
                continue
            x[s, subj] = {}
            for k in range(max_sets[subj]):
                x[s, subj][k] = model.NewBoolVar(f"x_{s}_{subj}_{k}")

    # Constraints
    # 1. Each student assigned to exactly one set per subject
    for s_idx, s in enumerate(students):
        for col_idx, col in enumerate(choice_columns):
            subj = records.loc[s_idx, col]
            if pd.isna(subj):
                continue
            model.Add(sum(x[s, subj][k] for k in range(max_sets[subj])) == 1)

    # 2. Student cannot have more than one subject in same group
    for s_idx, s in enumerate(students):
        subj_list = [records.loc[s_idx, col] for col in choice_columns if pd.notna(records.loc[s_idx, col])]
        for grp in GROUPS:
            for i in range(len(subj_list)):
                for j in range(i+1, len(subj_list)):
                    subj1, subj2 = subj_list[i], subj_list[j]
                    for k1 in range(max_sets[subj1]):
                        for k2 in range(max_sets[subj2]):
                            model.Add(x[s, subj1][k1] + x[s, subj2][k2] <= 1).OnlyEnforceIf([g[subj1, k1][grp], g[subj2, k2][grp]])

    # 3. Max students per set
    for subj in all_subjects:
        max_per_set = subject_max_size.get(subj, default_max_size)
        for k in range(max_sets[subj]):
            model.Add(sum(x[s, subj][k] for s in students if (s, subj) in x) <= max_per_set)

    # 4. Set usage variable
    for subj in all_subjects:
        for k in range(max_sets[subj]):
            model.Add(sum(x[s, subj][k] for s in students if (s, subj) in x) >= 1).OnlyEnforceIf(y[subj, k])
            model.Add(sum(x[s, subj][k] for s in students if (s, subj) in x) == 0).OnlyEnforceIf(y[subj, k].Not())

    # Objective: minimize total sets
    model.Minimize(sum(y[subj, k] for subj in all_subjects for k in range(max_sets[subj])))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None, None, None, None, None, None

    # Identify unassigned students
    unassigned = []
    for s_idx, s in enumerate(students):
        for col in choice_columns:
            subj = records.loc[s_idx, col]
            if pd.isna(subj):
                continue
            if sum(solver.Value(x[s, subj][k]) for k in range(max_sets[subj])) != 1:
                unassigned.append((s, subj))

    return solver, x, g, y, unassigned, max_sets

# -----------------------------
# EXPORT RESULTS
# -----------------------------
def export_results(records, solver, x, g, y, max_sets, choice_columns, unassigned):
    students = records['Student'].tolist()
    all_subjects = set()
    for col in choice_columns:
        all_subjects.update(records[col].dropna().unique())

    # Student assignments
    rows = []
    for s_idx, s in enumerate(students):
        row = {'Student': s}
        for col_idx, col in enumerate(choice_columns):
            subj = records.loc[s_idx, col]
            if pd.isna(subj):
                row[GROUPS[col_idx]] = ""
                continue
            for k in range(max_sets[subj]):
                if solver.Value(x[s, subj][k]):
                    assigned_group = [grp for grp in GROUPS if solver.Value(g[subj, k][grp])==1][0]
                    row[GROUPS[col_idx]] = f"{subj} (Set {k}, Group {assigned_group})"
        rows.append(row)
    pd.DataFrame(rows).to_csv('student_assignments.csv', index=False)

    # Sets details
    set_rows = []
    for subj in all_subjects:
        for k in range(max_sets[subj]):
            students_in_set = [s for s in students if (s, subj) in x and solver.Value(x[s, subj][k])]
            if students_in_set:
                set_rows.append({'Subject': subj, 'SetNumber': k, 'Students': ", ".join(map(str, students_in_set)), 'SetSize': len(students_in_set)})
    pd.DataFrame(set_rows).to_csv('sets_details.csv', index=False)

    # Group assignment per set
    group_rows = []
    for subj in all_subjects:
        for k in range(max_sets[subj]):
            if solver.Value(y[subj, k]):
                set_group = [grp for grp in GROUPS if solver.Value(g[subj, k][grp])]
                group_rows.append({'Subject': subj, 'SetNumber': k, 'Groups': ", ".join(set_group)})
    pd.DataFrame(group_rows).to_csv('group_sets.csv', index=False)

    # Summary
    total_sets = len(set_rows)
    print(f"\nTotal sets: {total_sets}")
    for subj in all_subjects:
        n_sets = sum(1 for k in range(max_sets[subj]) if solver.Value(y[subj, k]))
        print(f"{subj}: {n_sets} sets")
    if unassigned:
        pd.DataFrame(unassigned, columns=['Student', 'Subject']).to_csv('unassigned_students.csv', index=False)
        print(f"\n⚠ {len(unassigned)} students could not be assigned. See unassigned_students.csv")
    else:
        print("\nAll students assigned successfully.")

# -----------------------------
# MAIN
# -----------------------------
def main():
    use_dummy = input("Use dummy data? (y/n): ").strip().lower()=='y'
    if use_dummy:
        num_students = int(input("Number of dummy students: "))
        subjects = ['Geography', 'Art', 'RS: Philosophy and Ethics', 'History', 'French', 'Spanish',
                    'German', 'Design and Technology', 'Food & Nutrition', 'Business', 'Physical Education',
                    'Computer Science', 'Music', 'Drama', 'Triple Science']
        data=[]
        for i in range(num_students):
            choices = random.sample(subjects, random.choice([3,4]))
            row = {'Student': i+1}
            for j, grp in enumerate(GROUPS):
                row[f'Choice{j+1}'] = choices[j] if j<len(choices) else ""
            data.append(row)
        records=pd.DataFrame(data)
    else:
        filename = input("Enter CSV filename: ").strip()
        records=pd.read_csv(filename)

    choice_columns = ['Choice1','Choice2','Choice3','Choice4']

    # Load constraints
    subject_max_size, subject_max_sets = load_constraints()
    adjusted_max_size, adjusted_max_sets = adjust_constraints(records, subject_max_size, subject_max_sets, DEFAULT_MAX_SET_SIZE, choice_columns)

    print("Loaded constraints:")
    print("Max students per set:", subject_max_size)
    print("Max sets per subject:", subject_max_sets)

    print("\nAttempting solve...")
    solver, x, g, y, unassigned, max_sets = solve_option_blocks(records, adjusted_max_sets, adjusted_max_size, choice_columns)
    if solver is None:
        print("⚠ No feasible solution found.")
        return
    export_results(records, solver, x, g, y, max_sets, choice_columns, unassigned)

# -----------------------------
# RUN
# -----------------------------
if __name__=="__main__":
    main()
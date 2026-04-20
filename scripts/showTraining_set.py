from pathlib import Path

import pandas as pd


NUM = 23
DATA_PATH = Path(__file__).resolve().parent / "training_set" / "v2" / "training_set0" / "train_idx.csv"


def load_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_table(df: pd.DataFrame, file_path: Path) -> None:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(file_path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(file_path, index=False)
        return
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def fill_missing_samples(file_path: Path, num: int = NUM) -> None:
    df = load_table(file_path)
    id_column = df.columns[0]
    original_columns = df.columns.tolist()

    df["pop_group"] = df[id_column].str.extract(r"^(pop\d+)_")[0]
    df["sample_num"] = df[id_column].str.extract(r"sample(\d+)$")[0].astype(int)
    df["st"] = (df["sample_num"]) * 2048

    counts = df["pop_group"].value_counts().sort_index()
    pops_to_fill = counts[counts < num]
    pops_over_limit = counts[counts > num]

    if pops_to_fill.empty:
        print(f"All pop groups already have at least {num} rows. Only recalculating `st`.")
        if not pops_over_limit.empty:
            print("The following pop groups exceed the target row count and were not trimmed:")
            print(pops_over_limit.to_string())
    else:
        print(f"Found {len(pops_to_fill)} pop groups with fewer than {num} rows. Filling now.")
    new_rows = []

    for pop_id, count in pops_to_fill.items():
        pop_df = df[df["pop_group"] == pop_id].copy()
        existing_samples = set(pop_df["sample_num"].tolist())
        missing_samples = sorted(set(range(num)) - existing_samples)

        if not missing_samples:
            continue

        template_row = pop_df.sort_values("sample_num").iloc[0].copy()
        print(f"{pop_id}: current={count}, add={len(missing_samples)}")

        for missing in missing_samples:
            new_row = template_row.copy()
            new_row["sample_num"] = missing
            new_row[id_column] = f"{pop_id}_sample{missing:05d}"
            new_rows.append(new_row)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No additional rows were needed.")

    df["st"] = (df["sample_num"]) * 2048
    df = df.sort_values(["pop_group", "sample_num"]).reset_index(drop=True)

    result = df[original_columns]
    save_table(result, file_path)

    check_df = load_table(file_path)
    check_df["pop_group"] = check_df[id_column].str.extract(r"^(pop\d+)_")[0]
    check_df["sample_num"] = check_df[id_column].str.extract(r"sample(\d+)$")[0].astype(int)
    final_counts = check_df["pop_group"].value_counts().sort_index()
    still_short = final_counts[final_counts < num]
    bad_st = check_df[check_df["st"] != (check_df["sample_num"]) * 2048]

    print(f"Added {len(new_rows)} rows and overwrote the source file: {file_path}")
    if still_short.empty:
        print(f"Validation passed: every short pop group was filled up to {num} rows.")
    else:
        print("The following pop groups are still below the target count:")
        print(still_short.to_string())

    if bad_st.empty:
        print("Validation passed: every `st` value matches `(sample_num) * 2048`.")
    else:
        print(f"Warning: {len(bad_st)} rows still have incorrect `st` values.")

    if not pops_over_limit.empty:
        print("The following pop groups were already above the target count and were not trimmed:")
        print(pops_over_limit.to_string())


if __name__ == "__main__":
    fill_missing_samples(DATA_PATH, NUM)

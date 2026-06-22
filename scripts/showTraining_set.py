from pathlib import Path

import numpy as np
import pandas as pd

#version="v10_2"   # dataset version
#version="v6"
#version="v11"
#version="v21"
#version="v22"
#version="v23"
#version="v45"
#version="v50"
#version = "v61"
#version = 'v62'
version="v63"
#NUM = 23   # number of frame
NUM = 1   # number of frame
DATA_PATH = Path(__file__).resolve().parent / "training_set" / version / "training_set0" / "train_idx.csv"
PARAM_DIR = Path(__file__).resolve().parent / "parameter_sampling_train" / version


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


def find_id_column(df: pd.DataFrame) -> str:
    preferred_names = ["sample_id", "id", "name"]
    for name in preferred_names:
        if name in df.columns:
            return name

    for column in df.columns:
        values = df[column].astype(str)
        if values.str.contains(r"pop\d+", regex=True, na=False).any() and values.str.contains(
            r"sample\d+", regex=True, na=False
        ).any():
            return column

    return df.columns[0]


def add_metadata_columns(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    df = df.copy()
    df["pop_group"] = df[id_column].astype(str).str.extract(r"(pop\d+)")[0]
    sample_str = df[id_column].astype(str).str.extract(r"sample(\d+)")[0]
    df["sample_num"] = pd.to_numeric(sample_str, errors="coerce")

    invalid_rows = df[df["pop_group"].isna() | df["sample_num"].isna()]
    if not invalid_rows.empty:
        preview = invalid_rows[id_column].astype(str).head(10).tolist()
        raise ValueError(
            "Failed to parse pop/sample ids from some rows. "
            f"Examples: {preview}"
        )

    df["sample_num"] = df["sample_num"].astype(int)
    df["st"] = df["sample_num"] * 2048
    return df


def load_pop_labels(pop_index: int, target_columns: list[str]) -> dict:
    param_path = PARAM_DIR / f"CBC_params_example{pop_index:05d}.npz"
    if not param_path.exists():
        raise FileNotFoundError(f"Population parameter file not found: {param_path}")

    npz = np.load(param_path)
    values = {}
    for column in target_columns:
        if column in {"sample_id", "st"}:
            continue
        true_key = f"true_{column}"
        if true_key in npz:
            values[column] = float(npz[true_key])
        elif column in npz:
            values[column] = float(npz[column])
        else:
            raise KeyError(f"Column `{column}` not found in {param_path.name}")
    return values


def fill_missing_samples(file_path: Path, num: int = NUM) -> None:
    df = load_table(file_path)
    id_column = find_id_column(df)

    empty_id_mask = df[id_column].isna()
    if empty_id_mask.any():
        empty_rows = df[empty_id_mask].copy()
        print(f"Found {len(empty_rows)} rows with empty IDs in column `{id_column}`.")
        print("Empty-ID row indices:")
        print(empty_rows.index.tolist())
        print("Empty-ID row contents:")
        print(empty_rows.to_string(index=True))

    df = df[~empty_id_mask].copy()
    if df.empty:
        raise ValueError(f"No valid rows remain after dropping empty IDs from: {file_path}")

    original_columns = df.columns.tolist()
    df = add_metadata_columns(df, id_column)

    counts = df["pop_group"].value_counts().sort_index()
    pops_to_fill = counts[counts < num]
    pops_over_limit = counts[counts > num]
    existing_pop_numbers = sorted(df["pop_group"].str.extract(r"pop(\d+)")[0].astype(int).unique().tolist())
    max_pop_num = max(existing_pop_numbers) if existing_pop_numbers else -1
    expected_pop_numbers = set(range(max_pop_num + 1))
    missing_pop_numbers = sorted(expected_pop_numbers - set(existing_pop_numbers))

    if pops_to_fill.empty:
        print(f"All existing pop groups already have at least {num} rows. Only recalculating `st`.")
        if not pops_over_limit.empty:
            print("The following pop groups exceed the target row count and were not trimmed:")
            print(pops_over_limit.to_string())
    else:
        print(f"Found {len(pops_to_fill)} existing pop groups with fewer than {num} rows. Filling now.")

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
            new_row["st"] = missing * 2048
            new_rows.append(new_row)

    if missing_pop_numbers:
        print(f"Found {len(missing_pop_numbers)} completely missing pop groups. Filling now.")
        for pop_num in missing_pop_numbers:
            pop_id = f"pop{pop_num:05d}"
            pop_labels = load_pop_labels(pop_num, original_columns)
            print(f"{pop_id}: current=0, add={num}")
            for sample_num in range(num):
                row = {column: None for column in original_columns}
                row[id_column] = f"{pop_id}_sample{sample_num:05d}"
                row["st"] = sample_num * 2048
                for column, value in pop_labels.items():
                    row[column] = value
                new_rows.append(row)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No additional rows were needed.")

    df["st"] = df["sample_num"] * 2048
    df = df.sort_values(["pop_group", "sample_num"]).reset_index(drop=True)

    result = df[original_columns]
    save_table(result, file_path)

    check_df = load_table(file_path)
    check_df = add_metadata_columns(check_df, id_column)
    final_counts = check_df["pop_group"].value_counts().sort_index()
    still_short = final_counts[final_counts < num]
    bad_st = check_df[check_df["st"] != (check_df["sample_num"] * 2048)]
    final_pop_numbers = sorted(check_df["pop_group"].str.extract(r"pop(\d+)")[0].astype(int).unique().tolist())
    missing_after_fill = (
        sorted(set(range(max(final_pop_numbers) + 1)) - set(final_pop_numbers))
        if final_pop_numbers
        else []
    )

    print(f"Added {len(new_rows)} rows and overwrote the source file: {file_path}")
    if still_short.empty:
        print(f"Validation passed: every pop group was filled up to {num} rows.")
    else:
        print("The following pop groups are still below the target count:")
        print(still_short.to_string())

    if not missing_after_fill:
        print("Validation passed: no missing pop groups remain in the observed pop range.")
    else:
        print("The following pop groups are still completely missing:")
        print(missing_after_fill)

    if bad_st.empty:
        print("Validation passed: every `st` value matches `(sample_num) * 2048`.")
    else:
        print(f"Warning: {len(bad_st)} rows still have incorrect `st` values.")

    if not pops_over_limit.empty:
        print("The following pop groups were already above the target count and were not trimmed:")
        print(pops_over_limit.to_string())


if __name__ == "__main__":
    fill_missing_samples(DATA_PATH, NUM)

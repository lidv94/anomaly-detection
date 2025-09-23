import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from package.utils import profile_data ,DotDict, timer, get_config, load_data, merge_data ,save_file

@timer
def create_rule_differences(df, keep_id_cols=['sales_id','expected_dt','y']):
    """
    Create difference-based features between consecutive timeframe windows for each rule.
    
    Input Example:
        sales_id expected_dt  apl_l3  apl_l6  apl_l9  y
        00000001 2025-06-30       3      12     20   0
        
    Output Example:
        sales_id expected_dt  apl_l0_l3  apl_l3_l6  apl_l6_l9  y
        00000001 2025-06-30           3          9          8  0
    """
    if keep_id_cols is None:
        keep_id_cols = []

    # Start with ID columns
    df_out = df[keep_id_cols].copy() if keep_id_cols else pd.DataFrame()
    
    new_cols = {}  # collect new columns here

    # Detect rules: everything before the last '_l<number>'
    rules = sorted({
        re.match(r"^(.*)_l\d+$", col).group(1)
        for col in df.columns if re.match(r".*_l\d+$", col)
    })
    print(f"DEBUG: Detected rules: {rules}")

    for rule in rules:
        escaped_rule = re.escape(rule)
        rule_cols = [c for c in df.columns if re.match(rf"^{escaped_rule}_l\d+$", c)]
        if not rule_cols:
            print(f"DEBUG: No columns found for rule '{rule}', skipping.")
            continue

        periods = [int(c.split('_l')[-1]) for c in rule_cols]
        sorted_idx = np.argsort(periods)
        sorted_cols = np.array(rule_cols)[sorted_idx]
        sorted_periods = np.array(periods)[sorted_idx]

        print(f"DEBUG: Processing rule '{rule}' with columns: {sorted_cols}, periods: {sorted_periods}")

        # first column
        first_col = sorted_cols[0]
        first_period = sorted_periods[0]
        new_col_name = f"{rule}_l0_l{first_period}"
        new_cols[new_col_name] = df[first_col]
        print(f"DEBUG: Created first diff column '{new_col_name}'")

        # differences
        for prev_col, next_col, prev_p, next_p in zip(
            sorted_cols[:-1], sorted_cols[1:], sorted_periods[:-1], sorted_periods[1:]
        ):
            diff_col_name = f"{rule}_l{prev_p}_l{next_p}"
            new_cols[diff_col_name] = df[next_col] - df[prev_col]
            print(f"DEBUG: Created diff column '{diff_col_name}' = {next_col} - {prev_col}")
        
        print("-------------------------------------------------------------------------------------------------------------")

    # Add all new columns at once
    df_out = pd.concat([df_out, pd.DataFrame(new_cols)], axis=1)
    print(f"DEBUG: Final output columns: {df_out.columns.tolist()}")
    
    return df_out

@timer
def subset_features(df, prefix, keep_id_cols=None):
    """
    FOR DEBUGGING PURPOSE
    
    Subset dataframe to include only columns starting with a given prefix,
    sorted by numeric suffix after '_l'.
    """
    # Find matching columns
    cols = [c for c in df.columns if c.startswith(prefix)]
    
    # Sort by numeric suffix after "_l"
    def extract_suffix(col):
        match = re.search(r"_l(\d+)$", col)
        return int(match.group(1)) if match else float('inf')
    
    cols_sorted = sorted(cols, key=extract_suffix)
    
    # Add ID columns at front if requested
    if keep_id_cols:
        cols_sorted = keep_id_cols + cols_sorted
    
    return df[cols_sorted]

# @timer
# def compute_incre_slopes(df, id_cols=["sales_id", "expected_dt", "y"]):
#     """
#     Compute slopes for incremental/difference columns (_lX_lY) only.
#     Uses midpoints of lookback windows as x-axis for regression.
#     """
#     result = df[id_cols].copy()

#     # Detect rule prefixes robustly (non-greedy)
#     rule_cols = [c for c in df.columns if not "_slope" in c and re.search(r'_l\d+_l\d+$', c)]
#     rules = sorted(set([re.match(r'^(.*?)_l\d+_l\d+$', c).group(1) for c in rule_cols]))

#     print(f"DEBUG: Found rules: {rules}")

#     for rule in rules:
#         # Match incremental columns for this rule
#         incre_cols = [c for c in df.columns if re.match(fr"^{re.escape(rule)}_l\d+_l\d+$", c)]
#         if not incre_cols:
#             print(f"DEBUG: No incremental columns found for {rule}")
#             continue

#         # Use midpoint of window as x-axis
#         periods = [np.mean([int(x) for x in re.findall(r'\d+', c)]) for c in incre_cols]
#         sorted_idx = np.argsort(periods)
#         cols_sorted = np.array(incre_cols)[sorted_idx]
#         periods_sorted = np.array(periods)[sorted_idx]

#         print(f"\nDEBUG: Rule={rule}")
#         print(f"DEBUG: incremental columns={cols_sorted}")
#         print(f"DEBUG: periods(midpoints)={periods_sorted}")

#         slopes = []
#         for _, row in df.iterrows():
#             values = pd.to_numeric(row[cols_sorted], errors='coerce').values
#             mask = ~np.isnan(values)
#             if mask.sum() < 2:
#                 slopes.append(np.nan)
#             else:
#                 slope, _ = np.polyfit(periods_sorted[mask], values[mask], 1)
#                 slopes.append(slope)
#         result[f"{rule}_slope_incre"] = slopes
#         print("-------------------------------------------------------------------------------------------------------------")

#     return result

@timer
def compute_incre_slopes(df, id_cols=["sales_id", "expected_dt", "y"], x_axis="midpoint"):
    """
    Compute slopes for incremental/difference columns (_lX_lY) only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing incremental columns (_lX_lY).
    id_cols : list
        Columns to keep from original df.
    x_axis : str, default="midpoint"
        How to compute x-axis for regression:
        - "midpoint": use midpoint of each window
        - "endpoint": use the END of each window

    Returns
    -------
    pd.DataFrame with id_cols + slope columns.
    """
    result = df[id_cols].copy()

    # Detect rule prefixes
    rule_cols = [c for c in df.columns if "_slope" not in c and re.search(r'_l\d+_l\d+$', c)]
    rules = sorted(set([re.match(r'(.*)_l\d+_l\d+$', c).group(1) for c in rule_cols]))

    print(f"DEBUG: Found rules: {rules}")

    for rule in rules:
        incre_cols = [c for c in df.columns if re.match(fr"^{re.escape(rule)}_l\d+_l\d+$", c)]
        if not incre_cols:
            continue

        # Select x-axis values
        if x_axis == "midpoint":
            periods = [np.mean([int(x) for x in re.findall(r'\d+', c)]) for c in incre_cols]
        elif x_axis == "endpoint":
            periods = [int(re.findall(r'\d+', c)[-1]) for c in incre_cols]
        else:
            raise ValueError("x_axis must be 'midpoint' or 'endpoint'")

        sorted_idx = np.argsort(periods)
        cols_sorted = np.array(incre_cols)[sorted_idx]
        periods_sorted = np.array(periods)[sorted_idx]

        print(f"\nDEBUG: Rule={rule}")
        print(f"DEBUG: incremental columns={cols_sorted}")
        print(f"DEBUG: periods({x_axis})={periods_sorted}")

        slopes = []
        for _, row in df.iterrows():
            values = pd.to_numeric(row[cols_sorted], errors='coerce').values
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                slopes.append(np.nan)
            else:
                slope, _ = np.polyfit(periods_sorted[mask], values[mask], 1)
                slopes.append(slope)
        result[f"{rule}_slope_incre"] = slopes
        print("-------------------------------------------------------------------------------------------------------------")

    return result



######################################################################################################
    
# @timer
# def plot_rule_with_diff(df, sales_id, expected_dt, rule):
#     """
#     Plot rule difference values (_lX_lY) for a specific sales_id & expected_dt,
#     using midpoints for slope calculation to match compute_incre_slopes.
#     """
#     # Select row
#     row = df[(df['sales_id'] == sales_id) & (df['expected_dt'] == expected_dt)]
#     if row.empty:
#         print(f"No record found for sales_id={sales_id}, expected_dt={expected_dt}")
#         return
    
#     row = row.iloc[0]

#     # Extract rule diff columns
#     pattern = re.compile(fr"^{rule}_l\d+_l\d+$")
#     rule_cols = [c for c in df.columns if pattern.match(c)]
#     print(f"DEBUG: Found diff cols for {rule}: {rule_cols}")

#     if not rule_cols:
#         print(f"No diff columns found for {rule}")
#         return

#     # Extract midpoints and values
#     periods, labels, values = [], [], []
#     for c in rule_cols:
#         nums = list(map(int, re.findall(r"\d+", c)))
#         if len(nums) >= 2:
#             start, end = nums[-2], nums[-1]
#             midpoint = (start + end) / 2
#             # midpoint = end  # endpoint
#             periods.append(midpoint)
#             labels.append(f"l{start}_l{end}")
#             values.append(pd.to_numeric(row[c], errors="coerce"))

#     # Convert to arrays and sort
#     periods, labels, values = map(np.array, (periods, labels, values))
#     sorted_idx = np.argsort(periods)
#     periods, labels, values = periods[sorted_idx], labels[sorted_idx], values[sorted_idx]

#     print(f"DEBUG: midpoints={periods}")
#     print(f"DEBUG: values={values}")

#     mask = ~np.isnan(values)
#     periods, labels, values = periods[mask], labels[mask], values[mask]

#     if len(values) < 2:
#         print(f"Not enough diff data points to fit slope for {rule}")
#         return

#     # Fit line
#     slope, intercept = np.polyfit(periods, values, 1)
#     y_pred = slope * periods + intercept
#     print(f"DEBUG: slope={slope}, intercept={intercept}")

#     # Plot
#     plt.figure(figsize=(6, 4))
#     plt.plot(labels, values, "bo-", label="Rule diff values")
#     plt.plot(labels, y_pred, "ro--", label=f"Slope = {slope:.4f}")

#     plt.title(f"{rule} diffs for sales_id={sales_id}, expected_dt={expected_dt}")
#     plt.xlabel("Lookback window midpoint")
#     plt.ylabel("Rule diff value")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

    
@timer
def plot_rule_with_diff(df, sales_id, expected_dt, rule, x_axis="midpoint"):
    """
    Plot rule difference values (_lX_lY) for a specific sales_id & expected_dt,
    with slope line added as red dots.

    Parameters
    ----------
    df : pd.DataFrame
    sales_id : str
    expected_dt : str
    rule : str
    x_axis : str, default="midpoint"
        How to compute x-axis for slope:
        - "midpoint": use midpoint of each window
        - "endpoint": use END of each window
    """
    # Select row
    row = df[(df['sales_id'] == sales_id) & (df['expected_dt'] == expected_dt)]
    if row.empty:
        print(f"No record found for sales_id={sales_id}, expected_dt={expected_dt}")
        return

    row = row.iloc[0]

    # Extract rule diff columns
    pattern = re.compile(fr"^{re.escape(rule)}_l\d+_l\d+$")
    rule_cols = [c for c in df.columns if pattern.match(c)]
    print(f"DEBUG: Found diff cols for {rule}: {rule_cols}")

    if not rule_cols:
        print(f"No diff columns found for {rule}")
        return

    # Extract periods and labels
    periods, labels, values = [], [], []
    for c in rule_cols:
        nums = list(map(int, re.findall(r"\d+", c)))
        if len(nums) >= 2:  # take last two numbers only
            start, end = nums[-2], nums[-1]
            if x_axis == "midpoint":
                x_val = np.mean([start, end])
            elif x_axis == "endpoint":
                x_val = end
            else:
                raise ValueError("x_axis must be 'midpoint' or 'endpoint'")

            periods.append(x_val)
            labels.append(f"l{start}_l{end}")
            values.append(pd.to_numeric(row[c], errors="coerce"))

    # Convert to arrays and sort
    periods, labels, values = map(np.array, (periods, labels, values))
    sorted_idx = np.argsort(periods)
    periods, labels, values = periods[sorted_idx], labels[sorted_idx], values[sorted_idx]

    # Remove NaNs
    mask = ~np.isnan(values)
    periods, labels, values = periods[mask], labels[mask], values[mask]

    if len(values) < 2:
        print(f"Not enough diff data points to fit slope for {rule}")
        return

    # Fit line
    slope, intercept = np.polyfit(periods, values, 1)
    y_pred = slope * periods + intercept

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(labels, values, "bo-", label="Rule diff values")
    plt.plot(labels, y_pred, "ro--", label=f"Slope = {slope:.4f}")

    plt.title(f"{rule} diffs for sales_id={sales_id}, expected_dt={expected_dt}")
    plt.xlabel("Lookback window range")
    plt.ylabel("Rule diff value")
    plt.legend()
    plt.grid(True)
    plt.show()

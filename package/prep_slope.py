import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from package.utils import profile_data ,DotDict, timer, get_config, load_data, merge_data ,save_file

# @timer
# def create_rule_differences(df, rules=None):
#     """
#     Create difference-based features between consecutive lookback windows for each rule.
#     """
#     df_out = df.copy()
    
#     # Detect rules if not provided
#     if rules is None:
#         rules = sorted(set([c.split('_l')[0] for c in df.columns if '_l' in c]))
    
#     for rule in rules:
#         # Find rule columns
#         pattern = f"^{rule}_l\\d+$"
#         rule_cols = [c for c in df.columns if re.match(pattern, c)]
#         if not rule_cols:
#             continue
        
#         # Extract lookback periods
#         periods = [int(c.split('_l')[-1]) for c in rule_cols]
#         sorted_idx = np.argsort(periods)
        
#         sorted_cols = np.array(rule_cols)[sorted_idx]
#         sorted_periods = np.array(periods)[sorted_idx]
        
#         # First feature = value from earliest window
#         first_col = sorted_cols[0]
#         first_period = sorted_periods[0]
#         df_out[f"{rule}_l0_l{first_period}"] = df_out[first_col]
        
#         # Differences between consecutive windows
#         for prev_col, next_col, prev_p, next_p in zip(sorted_cols[:-1], sorted_cols[1:], sorted_periods[:-1], sorted_periods[1:]):
#             new_col = f"{rule}_l{prev_p}_l{next_p}"
#             df_out[new_col] = df_out[next_col] - df_out[prev_col]
    
#     return df_out

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

    df_out = pd.DataFrame()
    for col in keep_id_cols:
        if col in df.columns:
            df_out[col] = df[col]
            
    rules = sorted({col.split('_l')[0] for col in df.columns if '_l' in col})

    for rule in rules:
        # find all columns for this rule
        rule_cols = [c for c in df.columns if re.match(rf"^{rule}_l\d+$", c)]
        if len(rule_cols) < 1:
            continue

        # extract periods & sort them
        periods = [int(c.split('_l')[-1]) for c in rule_cols]
        sorted_idx = np.argsort(periods)
        sorted_cols = np.array(rule_cols)[sorted_idx]
        sorted_periods = np.array(periods)[sorted_idx]

        # first column
        first_col = sorted_cols[0]
        first_period = sorted_periods[0]
        df_out[f"{rule}_l0_l{first_period}"] = df[first_col]

        # differences between consecutive periods
        for prev_col, next_col, prev_p, next_p in zip(
            sorted_cols[:-1], sorted_cols[1:], sorted_periods[:-1], sorted_periods[1:]
        ):
            new_col = f"{rule}_l{prev_p}_l{next_p}"
            df_out[new_col] = df[next_col] - df[prev_col]

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

@timer
def plot_rule_with_slope(df, sales_id, expected_dt, rule):
    """
    Plot rule values across lookback windows for a specific sales_id & expected_dt,
    with slope line added as red dots.
    X-axis uses labels like l3, l6, ...
    """
    # Select row
    row = df[(df['sales_id'] == sales_id) & (df['expected_dt'] == expected_dt)]
    if row.empty:
        print(f"No record found for sales_id={sales_id}, expected_dt={expected_dt}")
        return
    
    row = row.iloc[0]
    
    # Extract rule columns
    pattern = f"^{rule}_l\\d+$"
    rule_cols = [c for c in df.columns if re.match(pattern, c)]
    if not rule_cols:
        print(f"No columns found for {rule}")
        return
    
    # Extract lookback periods
    periods = [int(c.split('_l')[-1]) for c in rule_cols]
    labels = [f"l{p}" for p in periods]
    
    # Sort
    sorted_idx = np.argsort(periods)
    periods = np.array(periods)[sorted_idx]
    labels = np.array(labels)[sorted_idx]
    values = pd.to_numeric(row[rule_cols].values[sorted_idx], errors="coerce")
    
    # Remove NaNs
    mask = ~np.isnan(values)
    periods, labels, values = periods[mask], labels[mask], values[mask]
    
    if len(values) < 2:
        print(f"Not enough data points to fit slope for {rule}")
        return
    
    # Fit line using np.polyfit (degree=1)
    slope, intercept = np.polyfit(periods, values, 1)
    y_pred = slope * periods + intercept
    
    # Plot actual values
    plt.figure(figsize=(6, 4))
    plt.plot(labels, values, "bo-", label="Rule values")  # categorical x-axis
    
    # Plot slope line as red dots
    plt.plot(labels, y_pred, "ro--", label=f"Slope = {slope:.4f}")
    
    plt.title(f"{rule} for sales_id={sales_id}, expected_dt={expected_dt}")
    plt.xlabel("Lookback window")
    plt.ylabel("Rule value")
    plt.legend()
    plt.grid(True)
    plt.show()
    
@timer
def plot_rule_with_diff(df, sales_id, expected_dt, rule):
    """
    Plot rule difference values (_lX_lY) for a specific sales_id & expected_dt,
    with slope line added as red dots.
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Select row
    row = df[(df['sales_id'] == sales_id) & (df['expected_dt'] == expected_dt)]
    if row.empty:
        print(f"No record found for sales_id={sales_id}, expected_dt={expected_dt}")
        return
    
    row = row.iloc[0]

    # Extract rule diff columns
    pattern = re.compile(fr"^{rule}_l\d+_l\d+$")
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
            periods.append(end)  # use ending point for slope axis
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
    
    
@timer
def compute_cumu_slopes(df, id_cols=["sales_id", "expected_dt","y"]):
    """
    Compute slopes for cumulative columns (_lX) only.
    Input df should contain only cumulative columns for rules.
    """
    result = df[id_cols].copy()
    
    # Detect rule prefixes from columns
    rule_cols = [c for c in df.columns if not "_slope" in c and re.search(r'_l\d+$', c)]
    rules = sorted(set([re.match(r'(.*)_l\d+$', c).group(1) for c in rule_cols]))
    
    for rule in rules:
        cum_cols = [c for c in df.columns if re.match(fr"^{rule}_l\d+$", c)]
        if not cum_cols:
            continue
        
        periods = [int(c.split('_l')[-1]) for c in cum_cols]
        sorted_idx = np.argsort(periods)
        # cols_sorted = np.array(cum_cols)[sorted_idx]
        cols_sorted = [str(c) for c in np.array(cum_cols)[sorted_idx]]  # convert to normal str
        print(f"Rule: {rule}, cumulative columns used for slope (sorted): {cols_sorted}")

        periods_sorted = np.array(periods)[sorted_idx]
        
        slopes = []
        for _, row in df.iterrows():
            values = pd.to_numeric(row[cols_sorted], errors='coerce').values
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                slopes.append(np.nan)
            else:
                slope, _ = np.polyfit(periods_sorted[mask], values[mask], 1)
                slopes.append(slope)
        result[f"{rule}_slope_cumu"] = slopes
    
    return result

@timer
def compute_incre_slopes(df, id_cols=["sales_id", "expected_dt","y"]):
    """
    Compute slopes for incremental/difference columns (_lX_lY) only.
    Input df should contain only incremental columns for rules.
    """
    result = df[id_cols].copy()
    
    # Detect rule prefixes from columns
    rule_cols = [c for c in df.columns if not "_slope" in c and re.search(r'_l\d+_l\d+$', c)]
    rules = sorted(set([re.match(r'(.*)_l\d+_l\d+$', c).group(1) for c in rule_cols]))
    
    for rule in rules:
        incre_cols = [c for c in df.columns if re.match(fr"^{rule}_l\d+_l\d+$", c)]
        if not incre_cols:
            continue
            
        print(f"Rule: {rule}, incremental columns used for slope: {incre_cols}")
        # Use midpoint of window as x-axis
        periods = [np.mean([int(x) for x in re.findall(r'\d+', c)]) for c in incre_cols]
        sorted_idx = np.argsort(periods)
        cols_sorted = np.array(incre_cols)[sorted_idx]
        periods_sorted = np.array(periods)[sorted_idx]
        
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
    
    return result

@timer
def plot_slope_violin(
    df_slopes, 
    slope_type="_slope_cumu", 
    flag_col="y", 
    ncols=3
):
    """
    Subplot violin plots for slope distributions by fraud flag (or any binary flag).
    
    Parameters
    ----------
    df_slopes : pd.DataFrame
        DataFrame with slope columns and a flag column.
    slope_type : str, default="_slope_cumu"
        Suffix of slope columns to include, e.g., "_slope_cumu" or "_slope_incre".
    flag_col : str, default="flag_fraud"
        Column name to use as hue (binary flag).
    ncols : int, default=3
        Number of subplots per row.
    """
    slope_cols = [c for c in df_slopes.columns if slope_type in c]
    
    # Melt into long format
    df_melt = df_slopes.melt(
        id_vars=[flag_col], 
        value_vars=slope_cols,
        var_name="rule", 
        value_name="slope"
    ).dropna()

    rules = df_melt["rule"].unique()
    n_rules = len(rules)
    nrows = int(np.ceil(n_rules / ncols))
    
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False
    )
    axes = axes.flatten()

    for i, rule in enumerate(rules):
        ax = axes[i]
        sns.violinplot(
            data=df_melt[df_melt["rule"] == rule],
            x=flag_col, 
            y="slope", 
            hue=flag_col, 
            palette="Set2", 
            split=True, 
            ax=ax,
        )

        ax.set_title(rule)
        ax.set_xlabel("Slope")
        ax.set_ylabel("")
        ax.legend_.remove()  # remove per-axis legend
    
    # Clean up unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add one global legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    
    
@timer
def plot_slope_hist_normalized(df_slopes, slope_type="_slope_cumu", flag_col="flag_fraud", n_cols=3, n_bins=20):
    """
    Plot normalized histograms of slope distributions by fraud flag (area=1), with subplots.
    Frauds and non-frauds use the same bin edges for comparability.
    """
    slope_cols = [c for c in df_slopes.columns if slope_type in c]
    
    df_melt = df_slopes.melt(id_vars=[flag_col], value_vars=slope_cols,
                             var_name="rule", value_name="slope").dropna()
    
    n_rules = len(slope_cols)
    n_rows = int(np.ceil(n_rules / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4), squeeze=False)
    
    for i, col in enumerate(slope_cols):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        
        # Compute common bins
        slopes_all = df_melt[df_melt['rule'] == col]['slope']
        bin_edges = np.linspace(slopes_all.min(), slopes_all.max(), n_bins+1)
        
        for flag_value in df_melt[flag_col].unique():
            subset = df_melt[(df_melt['rule'] == col) & (df_melt[flag_col] == flag_value)]
            ax.hist(subset['slope'], bins=bin_edges, alpha=0.6, density=True,
                    label=f"{flag_col}={flag_value}")
        
        ax.set_title(col)
        ax.set_xlabel("Slope")
        ax.set_ylabel("Density")
        ax.legend()
    
    # Remove empty subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    plt.show()
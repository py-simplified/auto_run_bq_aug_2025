import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Page configuration (set early before any other Streamlit calls)
st.set_page_config(page_title="Data Validation - Enhanced", page_icon="✅", layout="wide")

# Configuration: Hard stop behavior for join key uniqueness failures
# Set to True to halt validation immediately when join key uniqueness fails
# Set to False to skip failing scenarios and continue with valid ones
HARD_STOP_ON_UNIQUENESS = True

# Limits and output behavior
# If a failure dataset exceeds this row count, save as CSV instead of XLSX to avoid memory/performance issues.
FAILURE_XLSX_MAX_ROWS = 50000

def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Return a Windows-safe filename derived from an arbitrary string.
    - Replaces characters <>:"/\|?* with underscore
    - Trims whitespace and trailing dot/space; bounds length
    """
    if name is None:
        return "unnamed"
    safe = str(name)
    for ch in '<>:"/\\|?*':
        safe = safe.replace(ch, '_')
    safe = ' '.join(safe.split())
    safe = safe.strip(' .')
    return (safe or 'unnamed')[:max_len]

def make_run_output_paths(base_dir: str):
    """Create and return run-scoped output directories and main report path.
    Returns (run_id, run_dir, failures_dir, report_path).
    """
    run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, run_id)
    failures_dir = os.path.join(run_dir, 'ScenarioFailures')
    os.makedirs(failures_dir, exist_ok=True)
    report_path = os.path.join(run_dir, 'ValidationResults_Enhanced.xlsx')
    return run_id, run_dir, failures_dir, report_path

# Join-key normalization controls
NORMALIZE_JOIN_KEYS_TRIM = True   # Trim whitespace before joins/uniqueness
NORMALIZE_JOIN_KEYS_CASE = 'none' # one of: 'none', 'lower', 'upper'
COERCE_JOIN_KEYS_TO_STRING = True # Cast join keys to string for robust matching

def normalize_series_for_join(s: pd.Series) -> pd.Series:
    """Normalize a Series for joining based on config flags.
    Keeps pandas <NA> values as missing; otherwise applies string ops.
    """
    if s is None:
        return s
    ser = s
    try:
        if COERCE_JOIN_KEYS_TO_STRING:
            # Use pandas string dtype to keep NA awareness
            ser = ser.astype('string')
        if NORMALIZE_JOIN_KEYS_TRIM:
            ser = ser.str.strip()
        if NORMALIZE_JOIN_KEYS_CASE == 'lower':
            ser = ser.str.lower()
        elif NORMALIZE_JOIN_KEYS_CASE == 'upper':
            ser = ser.str.upper()
        return ser
    except Exception:
        # Fall back to original series in case of unexpected types
        return s

def normalize_key_sql(expr: str) -> str:
        """Return a BigQuery SQL expression that applies the same normalization
        flags used by pandas join normalization to a key expression.
        """
        s = expr
        if COERCE_JOIN_KEYS_TO_STRING:
                s = f"CAST({s} AS STRING)"
        if NORMALIZE_JOIN_KEYS_TRIM:
                s = f"TRIM({s})"
        if NORMALIZE_JOIN_KEYS_CASE == 'lower':
                s = f"LOWER({s})"
        elif NORMALIZE_JOIN_KEYS_CASE == 'upper':
                s = f"UPPER({s})"
        return s

ALLOWED_FUNCS = {
        # keep short, extend as needed
        "COALESCE", "CASE", "WHEN", "THEN", "ELSE", "END",
        "CONCAT", "TRIM", "LOWER", "UPPER",
        "CAST", "SAFE_CAST", "DATE", "DATETIME", "TIMESTAMP",
        "IF", "IFNULL", "NULLIF", "ABS", "ROUND"
}

def translate_derivation_to_bq(expr: str) -> str:
    """
    Enhanced BigQuery-safe expression translator for SOURCE/REF columns.
    Handles:
    - Basic sanitization (semicolons, aliases)
    - CASE statement type coercion
    - Identifiers, literals, operators, and ALLOWED_FUNCS keywords
    """
    if not expr:
        return "NULL"
    
    original_expr = str(expr).strip()
    cleaned = original_expr.rstrip(';')
    # drop common local aliases to make expression work on projected columns
    cleaned = cleaned.replace('s.', '').replace('r.', '')
    
    # Handle CASE statements with type coercion issues
    import re
    if 'CASE' in cleaned.upper() and 'WHEN' in cleaned.upper():
        # More flexible CASE statement handling - try multiple patterns
        patterns_to_try = [
            # Pattern 1: Standard CASE WHEN ... THEN ... ELSE ... END
            r'CASE\s+WHEN\s+([^T]+?)\s+THEN\s+([^E]+?)\s+ELSE\s+([^E]+?)\s+END',
            # Pattern 2: More flexible with optional whitespace
            r'CASE\s*WHEN\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END',
            # Pattern 3: Handle nested expressions
            r'CASE\s+WHEN\s+((?:[^T]|T(?!HEN))+)\s+THEN\s+((?:[^E]|E(?!LSE))+)\s+ELSE\s+((?:[^E]|E(?!ND))+)\s+END'
        ]
        
        def fix_case_types(match):
            condition, then_value, else_value = match.groups()
            condition = condition.strip()
            then_value = then_value.strip()
            else_value = else_value.strip()
            
            # Detect value types more accurately
            def is_numeric_value(val):
                val = val.strip()
                # Direct numeric literals
                if re.match(r'^-?\d+(\.\d+)?$', val):
                    return True
                # Common numeric values
                if val in ['0', '1', '0.0', '1.0']:
                    return True
                # COALESCE expressions often return numeric
                if val.upper().startswith('COALESCE'):
                    return True
                return False
            
            def is_string_value(val):
                val = val.strip()
                return (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"'))
            
            then_is_numeric = is_numeric_value(then_value)
            then_is_string = is_string_value(then_value)
            else_is_numeric = is_numeric_value(else_value)
            else_is_string = is_string_value(else_value)
            
            # Apply type coercion to ensure compatibility
            if then_is_numeric and else_is_string:
                # Numeric THEN, String ELSE -> Cast THEN to STRING
                then_value = f"CAST({then_value} AS STRING)"
            elif then_is_string and else_is_numeric:
                # String THEN, Numeric ELSE -> Cast ELSE to STRING
                else_value = f"CAST({else_value} AS STRING)"
            elif not then_is_string and not then_is_numeric and not else_is_string and not else_is_numeric:
                # Both are column references or expressions - cast both to STRING for safety
                then_value = f"CAST({then_value} AS STRING)"
                else_value = f"CAST({else_value} AS STRING)"
            
            return f"CASE WHEN {condition} THEN {then_value} ELSE {else_value} END"
        
        # Try patterns until one works
        for pattern in patterns_to_try:
            if re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL):
                cleaned = re.sub(pattern, fix_case_types, cleaned, flags=re.IGNORECASE | re.DOTALL)
                break
    
    # Handle common concatenation patterns that might create malformed identifiers
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*[a-zA-Z_][a-zA-Z0-9_]*$', cleaned) and not any(op in cleaned for op in ['+', '-', '*', '/', '(', ')', ' ']):
        # This might be a concatenated identifier, let's see if we can split it reasonably
        # For now, just return it as-is but wrapped in backticks for safety
        return f"`{cleaned}`"
    
    return cleaned

def generate_sql_for_scenario(row: dict,
                                                            required_source_columns: list,
                                                            required_target_columns: list,
                                                            required_reference_columns: list) -> str:
        """Generate a single BigQuery SQL that:
        - projects only required columns from source/target/reference
        - builds unique source combinations with row_count
        - computes derived_value from Derivation_Logic in SQL
        - normalizes join keys on both sides
        - LEFT JOINs to target and compares
        - returns total_pass, total_fail
        """
        # Parse keys/cols
        src_proj = row.get('Source_Project_Id'); src_ds = row.get('Source_Dataset_Id'); src_tbl = row.get('Source_Table')
        tgt_proj = row.get('Target_Project_Id'); tgt_ds = row.get('Target_Dataset_Id'); tgt_tbl = row.get('Target_Table')
        ref_tbl_raw  = str(row.get('Reference_Table', '') or '').strip()
        ref_tbl = '' if ref_tbl_raw.lower() in {'', 'nan', 'none'} else ref_tbl_raw
        tgt_col  = row.get('Target_Column')
        drv_sql  = translate_derivation_to_bq(str(row.get('Derivation_Logic', '') or 'NULL'))

        def _clean_list(lst):
            return [x for x in lst if x and x.lower() not in {'nan', 'none'}]

        src_keys = _clean_list([k.strip() for k in str(row.get('Source_Join_Key','')).split(',') if k.strip()])
        tgt_keys = _clean_list([k.strip() for k in str(row.get('Target_Join_Key','')).split(',') if k.strip()])
        ref_jk   = _clean_list([k.strip() for k in str(row.get('Reference_Join_Key','')).split(',') if k.strip()])
        ref_lu   = _clean_list([k.strip() for k in str(row.get('Reference_Lookup_Column','')).split(',') if k.strip()])

        # Build normalized key projections
        src_k_norm = [f"{normalize_key_sql('s.' + k)} AS k{i+1}" for i, k in enumerate(src_keys)]
        tgt_k_norm = [f"{normalize_key_sql(k)} AS tk{i+1}" for i, k in enumerate(tgt_keys)]
        join_cond  = " AND ".join([f"d.k{i+1} = t.tk{i+1}" for i in range(len(src_keys))]) or "1=1"

        # Column projections based on table-aware column requirements
        # Source columns needed = derivation inputs from source table ∪ src join keys ∪ ref lookup cols
        # Reference columns needed = derivation inputs from reference table ∪ ref join keys
        derivation_inputs_src = sorted(set([c for c in required_source_columns if c and str(c).strip().lower() not in {'nan','none'}]))
        # Reference columns used by derivation or join
        ref_inputs = sorted(set(required_reference_columns).union(ref_jk))

        src_proj_cols = ", ".join([f"s.{c} AS {c}" for c in derivation_inputs_src])
        ref_proj_cols = ", ".join([f"r.{c} AS {c}" for c in ref_inputs]) if ref_inputs else ""

        # Reference join (optional)
        ref_cte = ""
        ref_join = ""
        if ref_tbl and ref_inputs:
            # Build reference CTE with only required columns
            ref_select_cols = ", ".join([f"`{c}`" for c in sorted(set(ref_inputs))]) or "/* ref cols */"
            ref_cte = f"ref AS (SELECT {ref_select_cols} FROM `{src_proj}.{src_ds}.{ref_tbl}`),"
            if ref_lu and ref_jk:
                # Build composite join with normalization on each pair in order
                pairs = []
                for i in range(min(len(ref_lu), len(ref_jk))):
                    left = normalize_key_sql(f"s.{ref_lu[i]}")
                    right = normalize_key_sql(f"r.{ref_jk[i]}")
                    pairs.append(f"{left} = {right}")
                if pairs:
                    ref_join = f"LEFT JOIN ref r ON {' AND '.join(pairs)}"

        # Unique source combos: group by normalized keys + derivation inputs (source + ref)
        key_aliases = [f"k{i+1}" for i in range(len(src_keys))]
        # Ensure no duplicates in group_cols_list by using ordered deduplication
        all_cols = key_aliases + derivation_inputs_src + (sorted(set(ref_inputs)) if ref_inputs else [])
        group_cols_list = []
        seen = set()
        for col in all_cols:
            if col not in seen:
                group_cols_list.append(col)
                seen.add(col)
        group_by_cols = ", ".join(group_cols_list) if group_cols_list else "/* no group by */"
        sel_key_aliases = ", ".join(key_aliases) if key_aliases else ""

        # Compose src_norm projections
        norm_key_select = ", ".join(src_k_norm) if src_k_norm else "/* no keys */"
        proj_parts = [p for p in [norm_key_select, src_proj_cols, ref_proj_cols] if p]
        src_norm_select = ",\n    ".join(proj_parts) if proj_parts else "/* no projections */"

        # Replace table-qualified references with correct column names, preserving SQL structure
        import re
        
        # Build a mapping of table.column -> column for proper translation
        table_column_mapping = {}
        
        # Add source table mappings
        if src_tbl:
            for col in derivation_inputs_src:
                table_column_mapping[f"{src_tbl}.{col}"] = col
        
        # Add reference table mappings  
        if ref_tbl and ref_inputs:
            for col in ref_inputs:
                table_column_mapping[f"{ref_tbl}.{col}"] = col
        
        # Replace table.column references with just column names
        for table_col, col in table_column_mapping.items():
            # Use word boundaries to ensure exact matches
            pattern = r'\b' + re.escape(table_col) + r'\b'
            drv_sql = re.sub(pattern, col, drv_sql)
        
        # Additional cleanup: Replace any remaining unqualified table references
        if src_tbl:
            pattern = r'\b' + re.escape(src_tbl) + r'\.'
            drv_sql = re.sub(pattern, '', drv_sql)
        if ref_tbl:
            pattern = r'\b' + re.escape(ref_tbl) + r'\.'
            drv_sql = re.sub(pattern, '', drv_sql)

        # Validate that all column references in the expression exist in our available columns
        available_cols = derivation_inputs_src + (sorted(set(ref_inputs)) if ref_inputs else [])
        
        # For complex expressions, try to identify and fix individual column references
        if available_cols:
            drv_sql_temp = drv_sql
            
            # Look for potential column references in the expression
            # This regex finds sequences that look like identifiers
            potential_refs = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', drv_sql_temp)
            
            for ref in potential_refs:
                # Skip SQL keywords and functions
                if ref.upper() in ['COALESCE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE']:
                    continue
                    
                # If the reference doesn't match any available column, try to find the best match
                if ref not in available_cols:
                    # Look for exact substring matches first
                    exact_matches = [col for col in available_cols if ref in col]
                    if len(exact_matches) == 1:
                        # Replace this reference with the matched column
                        drv_sql = re.sub(r'\b' + re.escape(ref) + r'\b', exact_matches[0], drv_sql)
                    elif len(exact_matches) > 1:
                        # Multiple matches, pick the shortest one (most likely correct)
                        best_match = min(exact_matches, key=len)
                        drv_sql = re.sub(r'\b' + re.escape(ref) + r'\b', best_match, drv_sql)
                    else:
                        # Look for partial matches (column contains the reference)
                        partial_matches = [col for col in available_cols if col in ref]
                        if len(partial_matches) == 1:
                            drv_sql = re.sub(r'\b' + re.escape(ref) + r'\b', partial_matches[0], drv_sql)

        # Final fallback: if the derivation logic still looks problematic, use first available column
        drv_sql_clean = drv_sql.strip()
        if available_cols and drv_sql_clean and not any(col in drv_sql_clean for col in available_cols):
            # None of our available columns appear in the expression, use the first one as fallback
            drv_sql = available_cols[0]

        # Build src_base select list safely
        src_base_cols = sorted(set([c for c in derivation_inputs_src + src_keys + ref_lu if c]))
        src_base_select = ", ".join([f"`{c}`" for c in src_base_cols]) if src_base_cols else "/* no source cols */"

        sql = f"""
WITH
src_base AS (
    SELECT {src_base_select}
    FROM `{src_proj}.{src_ds}.{src_tbl}` s
),
{ref_cte if ref_cte else ''}
src_norm AS (
    SELECT
        {src_norm_select}
    FROM src_base s
    {ref_join}
),
unique_src AS (
    SELECT
        {", ".join(group_cols_list)}
        {"," if group_cols_list else ""} COUNT(*) AS row_count
    FROM src_norm
    GROUP BY {group_by_cols}
),
derived AS (
    SELECT
        {sel_key_aliases}{"," if sel_key_aliases else ""} ({drv_sql}) AS derived_value, row_count
    FROM unique_src
),
unique_tgt AS (
    SELECT
        {", ".join(tgt_k_norm) if tgt_k_norm else "/* no keys */"},
        CAST({tgt_col} AS STRING) AS actual_value,
        COUNT(*) AS target_count
    FROM `{tgt_proj}.{tgt_ds}.{tgt_tbl}`
    -- Include ALL target records (including NULLs) for complete validation coverage
    GROUP BY {", ".join([f"tk{i+1}" for i in range(len(tgt_keys))]) if tgt_keys else "/* no group by */"}, CAST({tgt_col} AS STRING)
),
joined AS (
    SELECT
        {sel_key_aliases}{"," if sel_key_aliases else ""} d.derived_value,
        d.row_count AS source_count,
        COALESCE(t.actual_value, NULL) AS actual_value,
        COALESCE(t.target_count, 0) AS target_count
    FROM derived d
    LEFT JOIN unique_tgt t
        ON {join_cond}
),
scored AS (
    SELECT
        derived_value,
        actual_value,
        source_count,
        target_count,
        -- Strict NULL-aware comparison: unexpected NULLs in target are treated as errors
        CASE 
            -- Both NULL: PASS
            WHEN derived_value IS NULL AND actual_value IS NULL THEN TRUE
            -- One NULL, other not: FAIL  
            WHEN derived_value IS NULL AND actual_value IS NOT NULL THEN FALSE
            WHEN derived_value IS NOT NULL AND actual_value IS NULL THEN FALSE
            -- Both NOT NULL: compare string representations
            ELSE TRIM(CAST(derived_value AS STRING)) = TRIM(CAST(actual_value AS STRING))
        END AS status
    FROM joined
)
SELECT
    -- Weight by TARGET count (not source count) for accurate validation totals
    SUM(CASE WHEN status THEN target_count ELSE 0 END) AS total_pass,
    SUM(CASE WHEN NOT status THEN target_count ELSE 0 END) AS total_fail
FROM scored
"""
        return sql

def create_failures_table_sql(row: dict, run_id: str, scenario_id: str) -> str:
    """Optional: Materialize failures to a BigQuery table using the same CTEs as the main query.
    Creates or replaces table in the Source project/dataset by default.
    """
    # Build base SQL and replace final SELECT with CREATE TABLE AS SELECT failures
    src_cols, tgt_cols, ref_cols = extract_required_columns_by_table(row)
    sql_counts = generate_sql_for_scenario(
        row,
        sorted(src_cols),
        sorted(tgt_cols),
        sorted(ref_cols),
    )
    # Determine output table path
    out_project = row.get('Source_Project_Id')
    out_dataset = row.get('Source_Dataset_Id')
    out_table = f"failures_{run_id}_{sanitize_filename(scenario_id)}"
    # Rebuild CTEs up to 'scored' by splitting at final SELECT
    prefix, _ = sql_counts.rsplit("SELECT", 1)
    failures_sql = prefix + """
SELECT * FROM scored WHERE NOT status
"""
    return f"""
CREATE OR REPLACE TABLE `{out_project}.{out_dataset}.{out_table}` AS
{failures_sql}
"""

## Optional: toggle to materialize failures to BigQuery instead of local files
MATERIALIZE_FAILURES_TO_BQ = False

def _canonicalize(name: str) -> str:
    """Canonical form for column names to allow fuzzy header matching."""
    if name is None:
        return ''
    s = str(name).strip().lower()
    for ch in ['-', ' ', '\t', '\n', '\r']:
        s = s.replace(ch, '_')
    while '__' in s:
        s = s.replace('__', '_')
    return s.strip('_')

EXPECTED_COLUMNS = [
    'Source_Project_Id', 'Source_Dataset_Id', 'Source_Table', 'Source_Join_Key',
    'Target_Project_Id', 'Target_Dataset_Id', 'Target_Table', 'Target_Join_Key',
    'Target_Column', 'Derivation_Logic', 'Scenario_ID', 'Scenario_Name',
    'Reference_Table', 'Reference_Join_Key', 'Reference_Lookup_Column', 'Description'
]

def normalize_headers_and_trim(df: pd.DataFrame):
    """Normalize headers via fuzzy matching and trim whitespace in key string columns.
    Returns (df_normalized, rename_map).
    """
    if df is None or df.empty:
        return df, {}
    # Build reverse index from canonical header to existing name
    canon_to_actual = {_canonicalize(c): c for c in df.columns}
    rename_map = {}
    for expected in EXPECTED_COLUMNS:
        canon = _canonicalize(expected)
        if expected in df.columns:
            continue
        if canon in canon_to_actual and canon_to_actual[canon] not in df.columns:
            # unlikely branch, keep for completeness
            pass
        if canon in canon_to_actual and canon_to_actual[canon] != expected:
            rename_map[canon_to_actual[canon]] = expected
    if rename_map:
        df = df.rename(columns=rename_map)
    # Trim whitespace on relevant string columns
    trim_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    for col in trim_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df, rename_map

def validate_scenarios_df(df: pd.DataFrame):
    """Return (ok, errors, warnings) for input scenarios DataFrame."""
    errors, warnings = [], []
    if df is None:
        return False, ["No data loaded"], warnings
    if df.empty:
        return False, ["Uploaded file has 0 rows"], warnings

    required_upload = ['Source_Project_Id', 'Source_Dataset_Id', 'Target_Project_Id', 'Target_Dataset_Id']
    missing_upload = [c for c in required_upload if c not in df.columns]
    if missing_upload:
        errors.append(f"Missing required columns: {', '.join(missing_upload)}")

    validation_required = ['Source_Table', 'Source_Join_Key', 'Target_Table', 'Target_Join_Key', 'Target_Column', 'Derivation_Logic']
    missing_validation = [c for c in validation_required if c not in df.columns]
    if missing_validation:
        errors.append(f"Missing validation columns: {', '.join(missing_validation)}")

    # Per-row essential empties
    essential = required_upload + ['Source_Table', 'Source_Join_Key', 'Target_Table', 'Target_Join_Key', 'Target_Column']
    for col in essential:
        if col in df.columns:
            null_count = df[col].isna() | (df[col].astype(str).str.strip().isin(['', 'nan', 'None']))
            if int(null_count.sum()) > 0:
                errors.append(f"Column '{col}' has {int(null_count.sum())} empty/missing values")

    # Scenario ID and name checks
    if 'Scenario_ID' in df.columns:
        dup = df['Scenario_ID'].astype(str).duplicated(keep=False)
        if bool(dup.any()):
            warnings.append("Duplicate Scenario_ID values detected; defaults may be applied during run.")
    else:
        warnings.append("Scenario_ID column missing; defaults will be generated per row.")
    if 'Scenario_Name' not in df.columns:
        warnings.append("Scenario_Name column missing; defaults will be generated per row.")

    return len(errors) == 0, errors, warnings

def initialize_client(project_id):
    """Return an authenticated bigquery.Client using ADC. Keep optional proxy lines commented."""
    import os, google.auth
    from google.cloud import bigquery
    # Optional (corp) proxy – leave commented unless needed:
    # if 'prod' in project_id:
    #     os.environ["HTTP_PROXY"] = "googleapis:0000"
    #     os.environ["HTTPS_PROXY"] = "googleapis:0000"
    # elif 'dev' in project_id:
    #     os.environ["HTTP_PROXY"] = "googleapis:0000"
    #     os.environ["HTTPS_PROXY"] = "googleapis:0000"
    credentials, _ = google.auth.default()
    return bigquery.Client(credentials=credentials, project=project_id)

def get_or_switch_client(client, project_id):
    """Return the same client if client.project == project_id else returns initialize_client(project_id)."""
    if client and client.project == project_id:
        return client
    return initialize_client(project_id)



def get_column_data_type(client, target_project, target_dataset, target_table, target_column):
    """
    Query BigQuery INFORMATION_SCHEMA to get the data type of a target column
    """
    try:
        # Switch client if needed for different project
        project_client = get_or_switch_client(client, target_project)
        
        type_query = f"""
        SELECT data_type
        FROM `{target_project}.{target_dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{target_table}' AND column_name = '{target_column}'
        """
        
        query_job = project_client.query(type_query)
        results = query_job.result(timeout=30)
        
        for row in results:
            return row.data_type
            
        return None  # Column not found
        
    except Exception as e:
        print(f"Error getting column type: {str(e)}")
        return None

def are_join_keys_unique(client, project, dataset, table, join_keys):
    """
    Verify that the provided join keys uniquely identify rows in a BigQuery table.
    
    Args:
        client: BigQuery client
        project: Project ID
        dataset: Dataset ID
        table: Table name
        join_keys: List of column names that should form a unique key
        
    Returns:
        bool: True if join keys are unique, False otherwise
        
    Raises:
        Exception: If query execution fails
    """
    try:
        # Switch client if needed for different project
        project_client = get_or_switch_client(client, project)
        
        # Construct CONCAT expression for composite keys
        if len(join_keys) == 1:
            concat_expr = f"CAST({join_keys[0]} AS STRING)"
        else:
            cast_keys = [f"CAST({key} AS STRING)" for key in join_keys]
            pipe_separator = '"|"'
            join_parts = f', {pipe_separator}, '.join(cast_keys)
            concat_expr = f"CONCAT({join_parts})"
        
        # Query to check uniqueness
        uniqueness_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT {concat_expr}) as distinct_keys
        FROM `{project}.{dataset}.{table}`
        WHERE {' AND '.join([f'{key} IS NOT NULL' for key in join_keys])}
        """
        
        query_job = project_client.query(uniqueness_query)
        results = query_job.result(timeout=30)
        
        for row in results:
            total_rows = row.total_rows
            distinct_keys = row.distinct_keys
            return total_rows == distinct_keys
            
        return False  # No results returned
        
    except Exception as e:
        raise Exception(f"Error checking uniqueness for join keys {join_keys} in `{project}.{dataset}.{table}`: {str(e)}")




def get_fully_qualified_table_name(project, dataset, table):
    """
    Helper function to ensure fully qualified table names in BigQuery.
    Returns the properly formatted table reference.
    """
    if not project or not dataset or not table:
        raise ValueError(f"Missing table components: project='{project}', dataset='{dataset}', table='{table}'")
    
    # Clean and validate components
    project = str(project).strip()
    dataset = str(dataset).strip() 
    table = str(table).strip()
    
    if not project or not dataset or not table or 'nan' in [project.lower(), dataset.lower(), table.lower()]:
        raise ValueError(f"Invalid table components: project='{project}', dataset='{dataset}', table='{table}'")
    
    return f"`{project}.{dataset}.{table}`"

def extract_required_columns_by_table(row):
    """
    Enhanced column extraction that separates required columns by their respective tables.
    
    Column Assignment Logic:
    - SOURCE TABLE: Derivation_Logic columns, Source_Join_Key, Reference_Lookup_Column
    - TARGET TABLE: Target_Column, Target_Join_Key  
    - REFERENCE TABLE: Reference_Join_Key
    
    Note: Reference_Lookup_Column belongs to SOURCE table (contains values to lookup in reference table)
          Reference_Join_Key belongs to REFERENCE table (the lookup key in reference table)
    
    Returns three sets: required_source_columns, required_target_columns, required_reference_columns
    """
    import re
    
    # Get all relevant fields
    derivation_logic = row.get('Derivation_Logic', '')
    target_column = row.get('Target_Column', '')
    source_join_key = row.get('Source_Join_Key', '')
    target_join_key = row.get('Target_Join_Key', '')
    reference_join_key = row.get('Reference_Join_Key', '')
    reference_lookup_column = row.get('Reference_Lookup_Column', '')
    reference_table = row.get('Reference_Table', '')
    
    # Initialize column sets
    required_source_columns = set()
    required_target_columns = set()
    required_reference_columns = set()
    
    # SQL keywords to exclude from column extraction
    excluded_keywords = {
        'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AND', 'OR', 'NOT', 'IS', 'NULL', 'TRUE', 'FALSE',
        'CAST', 'SAFE_CAST', 'COALESCE', 'CONCAT', 'TRIM', 'UPPER', 'LOWER', 'STRING', 'INT64',
        'FLOAT64', 'BOOL', 'NUMERIC', 'DATE', 'DATETIME', 'TIMESTAMP', 'IF', 'IFNULL', 'NULLIF',
        'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'ABS', 'ROUND', 'PREMIUM', 'STANDARD', 'ACTIVE', 'INACTIVE',
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'GROUP', 'BY',
        'ORDER', 'HAVING', 'DISTINCT', 'AS', 'WITH', 'UNION', 'ALL', 'EXISTS', 'IN', 'LIKE', 'BETWEEN'
    }
    
    # 1. Extract derivation logic columns → TABLE-SPECIFIC ROUTING
    if derivation_logic and str(derivation_logic).strip().lower() not in ['', 'nan', 'none']:
        derivation_str = str(derivation_logic).strip()
        
        # Get table names for comparison (use base table names, not fully qualified)
        source_table_name = row.get('Source_Table', '')
        target_table_name = row.get('Target_Table', '')
        reference_table_name = row.get('Reference_Table', '')
        
        # Remove quoted string literals to avoid treating them as columns (handles '...' and "...")
        derivation_no_strings = re.sub(r"'[^']*'|\"[^\"]*\"", ' ', derivation_str)

        # Enhanced regex patterns to extract column references
        column_patterns = [
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b',  # table.column format
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b(?!\s*\.)'  # standalone identifiers (not followed by a dot)
        ]
        
        # Collect and categorize matches by table
        for pattern in column_patterns:
            matches = re.findall(pattern, derivation_no_strings)
            for match in matches:
                # Skip pure keywords
                if match.upper() in excluded_keywords:
                    continue
                # Skip invalids
                if not match or match.isdigit():
                    continue
                if '.' in match:
                    table_part, col_part = match.split('.', 1)
                    table_part = table_part.strip()
                    col_part = col_part.strip()
                    # Process table-qualified references by routing to appropriate table
                    if table_part == source_table_name and col_part:
                        required_source_columns.add(col_part)
                    elif table_part == target_table_name and col_part:
                        required_target_columns.add(col_part)
                    elif table_part == reference_table_name and col_part:
                        required_reference_columns.add(col_part)
                    else:
                        # Unknown table token; default column attribution to source
                        if col_part:
                            required_source_columns.add(col_part)
                else:
                    # Standalone identifier: attribute to source (only if not a SQL keyword)
                    if match.upper() not in excluded_keywords:
                        required_source_columns.add(match)
    
    # 2. Add Source_Join_Key and Reference_Lookup_Column → SOURCE TABLE
    if source_join_key and str(source_join_key).strip().lower() not in ['', 'nan', 'none']:
        keys = [key.strip() for key in str(source_join_key).split(',') if key.strip()]
        required_source_columns.update(keys)
    
    # Reference_Lookup_Column is part of SOURCE table (used to lookup values in reference table)
    # BUT: exclude any columns that were explicitly qualified as reference table columns in derivation
    if reference_lookup_column and str(reference_lookup_column).strip().lower() not in ['', 'nan', 'none']:
        keys = [key.strip() for key in str(reference_lookup_column).split(',') if key.strip()]
        # Filter out columns that are already assigned to reference table from derivation logic
        keys_to_add = [k for k in keys if k not in required_reference_columns]
        required_source_columns.update(keys_to_add)
    
    # 3. Add Target_Column and Target_Join_Key → TARGET TABLE
    if target_column and str(target_column).strip().lower() not in ['', 'nan', 'none']:
        required_target_columns.add(str(target_column).strip())
    
    if target_join_key and str(target_join_key).strip().lower() not in ['', 'nan', 'none']:
        keys = [key.strip() for key in str(target_join_key).split(',') if key.strip()]
        required_target_columns.update(keys)
    
    # 4. Add Reference columns → REFERENCE TABLE (if applicable)
    if reference_table and str(reference_table).strip().lower() not in ['', 'nan', 'none']:
        if reference_join_key and str(reference_join_key).strip().lower() not in ['', 'nan', 'none']:
            keys = [key.strip() for key in str(reference_join_key).split(',') if key.strip()]
            required_reference_columns.update(keys)
    
    # Clean and validate all column sets
    required_source_columns = {col for col in required_source_columns if col and str(col).strip() and str(col).strip().lower() != 'nan'}
    required_target_columns = {col for col in required_target_columns if col and str(col).strip() and str(col).strip().lower() != 'nan'}
    required_reference_columns = {col for col in required_reference_columns if col and str(col).strip() and str(col).strip().lower() != 'nan'}
    
    # Final cleanup: Remove any columns from source that are explicitly assigned to reference table
    # This handles cases where derivation logic explicitly qualifies columns as reference.column_name
    overlap_with_ref = required_source_columns.intersection(required_reference_columns)
    if overlap_with_ref:
        required_source_columns = required_source_columns - overlap_with_ref
    
    return required_source_columns, required_target_columns, required_reference_columns

def extract_all_required_columns(row):
    """
    Backward compatibility function - returns unified set of all required columns.
    """
    source_cols, target_cols, ref_cols = extract_required_columns_by_table(row)
    return source_cols.union(target_cols).union(ref_cols)

def validate_columns_exist_in_table(client, project, dataset, table, required_columns):
    """
    Validation check to ensure all required columns exist in the table schema.
    Returns (exists, missing_columns)
    """
    try:
        table_ref = f"{project}.{dataset}.{table}"
        table = client.get_table(table_ref)
        
        # Get all column names from schema
        existing_columns = {field.name for field in table.schema}
        
        # Check which required columns are missing
        missing_columns = required_columns - existing_columns
        
        return len(missing_columns) == 0, missing_columns
        
    except Exception as e:
        st.error(f"❌ Error validating table schema: {e}")
        return False, required_columns

def validate_all_tables_columns(client, row, required_source_columns, required_target_columns, required_reference_columns):
    """
    Enhanced validation that checks column existence in their respective tables.
    Returns (all_valid, validation_results) where validation_results contains detailed info.
    """
    validation_results = {
        'source': {'valid': True, 'missing': set(), 'table': ''},
        'target': {'valid': True, 'missing': set(), 'table': ''},
        'reference': {'valid': True, 'missing': set(), 'table': ''}
    }
    
    # Validate Source Table Columns
    if required_source_columns:
        source_project = row.get('Source_Project_Id', '')
        source_dataset = row.get('Source_Dataset_Id', '')
        source_table = row.get('Source_Table', '')
        validation_results['source']['table'] = f"{source_project}.{source_dataset}.{source_table}"
        
        source_valid, source_missing = validate_columns_exist_in_table(
            client, source_project, source_dataset, source_table, required_source_columns)
        validation_results['source']['valid'] = source_valid
        validation_results['source']['missing'] = source_missing
    
    # Validate Target Table Columns
    if required_target_columns:
        target_project = row.get('Target_Project_Id', '')
        target_dataset = row.get('Target_Dataset_Id', '')
        target_table = row.get('Target_Table', '')
        validation_results['target']['table'] = f"{target_project}.{target_dataset}.{target_table}"
        
        target_valid, target_missing = validate_columns_exist_in_table(
            client, target_project, target_dataset, target_table, required_target_columns)
        validation_results['target']['valid'] = target_valid
        validation_results['target']['missing'] = target_missing
    
    # Validate Reference Table Columns (if applicable)
    if required_reference_columns:
        reference_table = row.get('Reference_Table', '')
        if reference_table and str(reference_table).strip().lower() not in ['', 'nan', 'none']:
            # Reference table is typically in the same project/dataset as source
            ref_project = row.get('Source_Project_Id', '')
            ref_dataset = row.get('Source_Dataset_Id', '')
            validation_results['reference']['table'] = f"{ref_project}.{ref_dataset}.{reference_table}"
            
            ref_valid, ref_missing = validate_columns_exist_in_table(
                client, ref_project, ref_dataset, reference_table, required_reference_columns)
            validation_results['reference']['valid'] = ref_valid
            validation_results['reference']['missing'] = ref_missing
    
    # Check if all validations passed
    all_valid = (validation_results['source']['valid'] and 
                 validation_results['target']['valid'] and 
                 validation_results['reference']['valid'])
    
    return all_valid, validation_results

def build_unique_combinations_query(project, dataset, table, required_columns):
    """
    STEP 2: Build Unique Source Row Combinations
    Construct BigQuery SQL to get unique combinations with row counts.
    """
    if not required_columns:
        raise ValueError("No required columns provided for unique combinations query")
    
    # Build fully qualified table name
    table_fq = get_fully_qualified_table_name(project, dataset, table)
    
    # Sort columns for consistent ordering
    column_list = sorted(required_columns)
    column_select = ", ".join(column_list)
    
    query = f"""
    SELECT {column_select}, COUNT(*) AS row_count
    FROM {table_fq}
    GROUP BY {column_select}
    """
    
    return query

def apply_derivation_logic_in_python(df, derivation_logic):
    """
    STEP 3: Apply Derivation Logic in Python
    Use pandas operations to compute derived values on unique combinations.
    Handles common SQL patterns and converts them to pandas-compatible operations.
    """
    try:
        if not derivation_logic or str(derivation_logic).strip().lower() in ['', 'nan', 'none']:
            raise ValueError("Empty derivation logic")
        
        # Clean up the derivation logic
        cleaned_logic = str(derivation_logic).strip()
        
        # Handle common SQL patterns for pandas compatibility
        
        # 1. Handle COALESCE function
        if 'COALESCE' in cleaned_logic.upper():
            import re
            # Pattern for COALESCE(col1, col2, ...)
            coalesce_pattern = r'COALESCE\s*\(\s*([^)]+)\s*\)'
            
            def convert_coalesce(match):
                args = [arg.strip() for arg in match.group(1).split(',')]
                if len(args) == 2:
                    return f"{args[0]}.fillna({args[1]})"
                elif len(args) > 2:
                    # Chain fillna for multiple arguments
                    result = args[0]
                    for arg in args[1:]:
                        result = f"{result}.fillna({arg})"
                    return result
                else:
                    return args[0]
            
            cleaned_logic = re.sub(coalesce_pattern, convert_coalesce, cleaned_logic, flags=re.IGNORECASE)
        
        # 2. Handle CASE WHEN statements (simplified)
        if 'CASE' in cleaned_logic.upper() and 'WHEN' in cleaned_logic.upper():
            # For simple CASE WHEN col = value THEN 'result' ELSE 'default' END
            # Convert to pandas where condition
            case_pattern = r'CASE\s+WHEN\s+([^=]+)\s*=\s*([^T]+)\s+THEN\s+[\'"]([^\'"]+)[\'"]\s+ELSE\s+[\'"]([^\'"]+)[\'"]\s+END'
            
            def convert_case(match):
                col, value, then_val, else_val = match.groups()
                col = col.strip()
                value = value.strip()
                return f"'{then_val}' if ({col} == {value}) else '{else_val}'"
            
            import re
            if re.search(case_pattern, cleaned_logic, re.IGNORECASE):
                cleaned_logic = re.sub(case_pattern, convert_case, cleaned_logic, flags=re.IGNORECASE)
        
        # 3. Handle simple column references and string concatenation
        # If it's just a simple column name, use it directly
        if cleaned_logic in df.columns:
            df['derived_value'] = df[cleaned_logic]
            return df, True, "Success - simple column reference"
        
        # 4. Try to evaluate the expression
        try:
            df['derived_value'] = df.eval(cleaned_logic)
            return df, True, "Success - pandas eval"
        except:
            # Fallback: try as a simple assignment
            try:
                # Check if it's a simple mathematical operation on columns
                df['derived_value'] = eval(cleaned_logic, {'df': df, '__builtins__': {}})
                return df, True, "Success - simple evaluation"
            except:
                # Final fallback: create derived values using column operations
                available_cols = [col for col in df.columns if col in cleaned_logic]
                if available_cols:
                    # Use first available column as fallback
                    df['derived_value'] = df[available_cols[0]]
                    return df, True, f"Fallback - using column {available_cols[0]}"
                else:
                    df['derived_value'] = cleaned_logic  # Use literal value
                    return df, True, "Fallback - literal value"
        
    except Exception as e:
        # Ultimate fallback: create a None column
        df['derived_value'] = None
        return df, False, f"Failed to apply derivation logic: {str(e)}"

def fetch_target_data_and_join(client, df_source, row, cache=None):
    """
    STEP 4: Join with Target Table
    Fetch target data and join with processed source DataFrame.
    """
    try:
        # Extract target table info
        target_project = row.get('Target_Project_Id', '')
        target_dataset = row.get('Target_Dataset_Id', '')
        target_table = row.get('Target_Table', '')
        target_column = row.get('Target_Column', '')
        target_join_key = row.get('Target_Join_Key', '')
        source_join_key = row.get('Source_Join_Key', '')

        # Parse join keys
        target_join_keys = [key.strip() for key in str(target_join_key).split(',') if key.strip()]
        source_join_keys = [key.strip() for key in str(source_join_key).split(',') if key.strip()]

        # Build target query
        target_table_fq = get_fully_qualified_table_name(target_project, target_dataset, target_table)
        target_columns = target_join_keys + [target_column]
        target_column_select = ", ".join(set(target_columns))  # Remove duplicates

        target_query = f"""
        SELECT {target_column_select}
        FROM {target_table_fq}
        WHERE {target_column} IS NOT NULL
        """

        # Execute target query with caching
        if cache is None:
            cache = {}
        tgt_cache = cache.setdefault('target_df', {})
        tgt_key = (
            target_project, target_dataset, target_table,
            tuple(sorted(target_join_keys + [target_column]))
        )
        if tgt_key in tgt_cache:
            df_target = tgt_cache[tgt_key]
        else:
            df_target = client.query(target_query).to_dataframe()
            tgt_cache[tgt_key] = df_target

        # Perform join
        if len(source_join_keys) == len(target_join_keys):
            # Normalize join key dtypes and values per config
            for key in source_join_keys:
                if key in df_source.columns:
                    df_source[key] = normalize_series_for_join(df_source[key])
            for key in target_join_keys:
                if key in df_target.columns:
                    df_target[key] = normalize_series_for_join(df_target[key])

            # Create join condition
            left_on = source_join_keys
            right_on = target_join_keys

            df_joined = df_source.merge(
                df_target,
                left_on=left_on,
                right_on=right_on,
                how='left',
                suffixes=('', '_tgt')
            )

            # Resolve target column possibly suffixed due to collisions
            candidate_cols = [target_column, f"{target_column}_tgt", f"{target_column}_target"]
            actual_col = next((c for c in candidate_cols if c in df_joined.columns), None)
            if actual_col is not None:
                df_joined['actual_value'] = df_joined[actual_col]
            else:
                df_joined['actual_value'] = None

            return df_joined, True, "Join successful"
        else:
            return df_source, False, f"Join key count mismatch: source {len(source_join_keys)}, target {len(target_join_keys)}"

    except Exception as e:
        return df_source, False, f"Failed to join with target table: {str(e)}"

def calculate_validation_results(df):
    """
    STEP 5: Calculate Validation Results with Row Count Scaling
    Compare derived vs actual values and multiply by row_count.
    """
    try:
        # Handle missing derived_value or actual_value
        df['derived_value'] = df['derived_value'].fillna('')
        df['actual_value'] = df['actual_value'].fillna('')
        
        # Perform comparison (handle different data types)
        df['status'] = df.apply(lambda row: 
            str(row['derived_value']).strip() == str(row['actual_value']).strip() 
            if pd.notna(row['derived_value']) and pd.notna(row['actual_value']) 
            else False, axis=1)
        
        # Calculate scaled counts
        df['pass_count'] = df['status'].astype(int) * df['row_count']
        df['fail_count'] = (~df['status']).astype(int) * df['row_count']
        
        # Calculate totals
        total_pass = df['pass_count'].sum()
        total_fail = df['fail_count'].sum()
        
        return df, total_pass, total_fail
        
    except Exception as e:
        st.error(f"❌ Error calculating validation results: {e}")
        return df, 0, 0

def enhanced_validation_execution(row, client, scenario_id, scenario_name, failures_dir, cache=None):
    """
    Main function implementing the enhanced validation framework.
    Processes validation using unique combinations approach to avoid row-level operations.
    Uses table-specific column validation to prevent false errors.
    """
    try:
        # STEP 1: Extract Required Columns by Table
        st.write(f"🔍 Step 1: Extracting required columns by table for {scenario_id}...")
        required_source_columns, required_target_columns, required_reference_columns = extract_required_columns_by_table(row)
        
        if not required_source_columns and not required_target_columns:
            return False, "No required columns found in any table", 0, 0, None
        
        # Display column breakdown
        st.write(f"📋 **Column Breakdown:**")
        st.write(f"   • Source table columns: {sorted(required_source_columns) if required_source_columns else 'None'}")
        st.write(f"   • Target table columns: {sorted(required_target_columns) if required_target_columns else 'None'}")
        if required_reference_columns:
            st.write(f"   • Reference table columns: {sorted(required_reference_columns)}")
        
        # STEP 1.5: Enhanced Table-Specific Column Validation
        st.write(f"✅ Step 1.5: Validating columns exist in their respective tables...")
        all_valid, validation_results = validate_all_tables_columns(
            client, row, required_source_columns, required_target_columns, required_reference_columns)
        
        if not all_valid:
            error_details = []
            for table_type, result in validation_results.items():
                if not result['valid'] and result['missing']:
                    error_details.append(f"{table_type.title()} table ({result['table']}) missing columns: {sorted(result['missing'])}")
            
            error_msg = "Column validation failed:\n" + "\n".join(error_details)
            st.error(f"❌ {error_msg}")
            return False, error_msg, 0, 0, None
        
        st.success(f"✅ All required columns validated in their respective tables!")
        
        # Get source table info
        source_project = row.get('Source_Project_Id', '')
        source_dataset = row.get('Source_Dataset_Id', '')
        source_table = row.get('Source_Table', '')
        derivation_logic = row.get('Derivation_Logic', '')
        # Early guard: join key length match
        s_keys = [k.strip() for k in str(row.get('Source_Join_Key','')).split(',') if k.strip()]
        t_keys = [k.strip() for k in str(row.get('Target_Join_Key','')).split(',') if k.strip()]
        if len(s_keys) != len(t_keys):
            msg = f"Join key count mismatch (source={len(s_keys)} vs target={len(t_keys)}); cannot build SQL join"
            st.error(f"❌ {msg}")
            return False, msg, 0, 0, None
        
        # STEP 2+: Pushdown to BigQuery: unique combos + derivation + join + compare in one SQL
        st.write("🧮 Step 2: Executing SQL pushdown for derivation, join, and comparison...")

        # Compose SQL using required columns (projection only)
        sql = generate_sql_for_scenario(
            row,
            sorted(required_source_columns),
            sorted(required_target_columns),
            sorted(required_reference_columns),
        )

        # Execute aggregate query
        # DEBUG: Print the SQL being executed
        print(f"\n=== SQL DEBUG FOR {scenario_id} ===")
        print("Generated SQL:")
        print(sql)
        print("=== END SQL DEBUG ===\n")
        
        query_job = client.query(sql)
        res = query_job.result(timeout=600)

        total_pass, total_fail = 0, 0
        for r in res:
            total_pass = int(getattr(r, 'total_pass', 0) or 0)
            total_fail = int(getattr(r, 'total_fail', 0) or 0)

        # Generate failure reports based on total_fail count
        failure_file_path = "No failures"
        if total_fail > 0:
            try:
                # Create detailed failure report by re-executing the SQL with failure details
                # Find the final SELECT statement and replace it
                if "SUM(CASE WHEN status THEN target_count ELSE 0 END) AS total_pass" in sql:
                    # More flexible replacement - replace the entire final SELECT block
                    pattern_start = sql.rfind("SELECT")
                    if pattern_start != -1:
                        # Replace from the last SELECT to the end
                        sql_before_select = sql[:pattern_start]
                        failures_sql = sql_before_select + """SELECT
    derived_value,
    actual_value,
    source_count,
    target_count,
    status
FROM scored
WHERE NOT status"""
                    else:
                        # Fallback: try the original replacement
                        failures_sql = sql.replace(
                            "SUM(CASE WHEN status THEN target_count ELSE 0 END) AS total_pass,  \n    SUM(CASE WHEN NOT status THEN target_count ELSE 0 END) AS total_fai",
                            "derived_value,\n    actual_value,\n    source_count,\n    target_count,\n    status\nFROM scored\nWHERE NOT status\n-- Original:"
                        )
                else:
                    failures_sql = sql + "\n-- Failed to create failure SQL"
                
                print(f"\n=== FAILURES SQL DEBUG ===")
                print("Failures SQL:")
                print(failures_sql)
                print("=== END FAILURES SQL DEBUG ===\n")
                
                failures_df = client.query(failures_sql).to_dataframe()
                
                # DEBUG: Print available columns
                print(f"\n=== FAILURES DF DEBUG ===")
                print(f"Scenario: {scenario_id}")
                print(f"Failures DF columns: {list(failures_df.columns) if not failures_df.empty else 'EMPTY DATAFRAME'}")
                print(f"Failures DF shape: {failures_df.shape}")
                if not failures_df.empty:
                    print(f"First few rows:\n{failures_df.head()}")
                print("=== END FAILURES DEBUG ===\n")
                
                if not failures_df.empty and failures_dir:
                    # Create Excel file for failures
                    safe_scenario_name = sanitize_filename(scenario_name or scenario_id)
                    failure_filename = f"{scenario_id}_{safe_scenario_name}_Failures.xlsx"
                    failure_file_path = os.path.join(failures_dir, failure_filename)
                    
                    # Add metadata to the failures
                    failures_df['Scenario_ID'] = scenario_id
                    failures_df['Scenario_Name'] = scenario_name or scenario_id
                    
                    # Safely map columns - check if they exist first
                    if 'derived_value' in failures_df.columns:
                        failures_df['Expected_Value'] = failures_df['derived_value']
                    else:
                        failures_df['Expected_Value'] = 'Column not found: derived_value'
                        
                    if 'actual_value' in failures_df.columns:
                        failures_df['Actual_Value'] = failures_df['actual_value']
                    else:
                        failures_df['Actual_Value'] = 'Column not found: actual_value'
                    failures_df['Record_Count'] = failures_df['target_count']
                    
                    # Select and rename columns for clarity
                    output_df = failures_df[['Scenario_ID', 'Scenario_Name', 'Expected_Value', 'Actual_Value', 'Record_Count']].copy()
                    
                    # Write to Excel
                    output_df.to_excel(failure_file_path, index=False, engine='openpyxl')
                    st.write(f"📄 Created failure report: {failure_filename}")
                else:
                    failure_file_path = "SKIPPED: No failure details available"
                    
            except Exception as e:
                st.warning(f"⚠️ Failed to generate failure report: {e}")
                failure_file_path = "Execution failed"

        # Optional: also materialize failures to BQ table if enabled
        if total_fail > 0 and MATERIALIZE_FAILURES_TO_BQ:
            try:
                run_id = os.path.basename(os.path.dirname(failures_dir)) if failures_dir else datetime.now().strftime('run_%Y%m%d_%H%M%S')
                ddl = create_failures_table_sql(row, run_id, scenario_id)
                client.query(ddl).result(timeout=600)
                out_project = row.get('Source_Project_Id')
                out_dataset = row.get('Source_Dataset_Id')
                out_table = f"failures_{run_id}_{sanitize_filename(scenario_id)}"
                bq_table_path = f"`{out_project}.{out_dataset}.{out_table}`"
                st.write(f"💾 Also materialized failures to {bq_table_path}")
            except Exception as e:
                st.warning(f"⚠️ Failed to materialize failures to BigQuery: {e}")

        success_message = f"Enhanced validation (SQL) completed: {total_pass} passed, {total_fail} failed"
        st.success(f"✅ {success_message}")
        
        return True, success_message, total_pass, total_fail, failure_file_path, sql
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        error_msg = f"Enhanced validation failed for {scenario_id}: {str(e)}\n\nFull traceback:\n{full_traceback}"
        st.error(f"❌ Enhanced validation failed for {scenario_id}: {str(e)}")
        st.error(f"📝 Full error details:\n```\n{full_traceback}\n```")
        # Also print to console for debugging
        print(f"\n=== S005 DEBUG INFO ===")
        print(f"Scenario ID: {scenario_id}")
        print(f"Error: {str(e)}")
        print(f"Full traceback:\n{full_traceback}")
        print("=== END DEBUG INFO ===\n")
        return False, error_msg, 0, 0, None, "-- SQL generation failed due to error --"

def test_bigquery_connectivity(project_id, dataset_id, client=None):
    """
    Test connectivity to BigQuery project and dataset
    """
    """
    Test connectivity to BigQuery project and dataset
    """
    try:
        # Use provided client or initialize a new one
        if client is None:
            client = initialize_client(project_id)
        elif client.project != project_id:
            # Switch project if needed using our helper function
            client = get_or_switch_client(client, project_id)
        
        # Test basic connectivity with a simple query
        test_query = f"""
        SELECT 
            COUNT(*) as table_count,
            '{project_id}' as project_id,
            '{dataset_id}' as dataset_id,
            'connectivity_test' as status
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
        """
        
        # Execute the query with a timeout
        query_job = client.query(test_query)
        results = query_job.result(timeout=30)  # 30 second timeout
        
        # If we get here, the connection worked
        row_count = 0
        for row in results:
            row_count = row.table_count
            break
            
        return True, f"Success - {row_count} tables found"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    st.title("Data Validation Framework — Enhanced")

    # Display completion banner if validation results are available
    if 'validation_results' in st.session_state:
        results = st.session_state['validation_results']
        total_records_passed = results.get('total_passed', 0)
        total_records_failed = results.get('total_failed', 0)
        total_records = total_records_passed + total_records_failed
        passed_scenarios = results.get('passed_scenarios', 0)
        total_scenarios = results.get('total_scenarios', 0)
        
        if total_records > 0:
            pass_rate = (total_records_passed / total_records * 100)
            
            if total_records_failed == 0:
                st.success(
                    f"🎉 **VALIDATION COMPLETE - ALL PASSED!** | "
                    f"{total_records_passed:,} records validated ✅ | "
                    f"{passed_scenarios}/{total_scenarios} scenarios passed | "
                    f"Pass Rate: {pass_rate:.1f}%"
                )
            else:
                st.error(
                    f"⚠️ **VALIDATION COMPLETE - ISSUES FOUND!** | "
                    f"{total_records_failed:,} records failed ❌ | "
                    f"{total_records_passed:,} records passed ✅ | "
                    f"Pass Rate: {pass_rate:.1f}%"
                )
            
            # Quick action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔄 Run New Validation", type="secondary"):
                    # Clear session state to start fresh
                    for key in ['validation_results', 'uploaded_df', 'uploaded_file_name']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            with col2:
                if st.button("📊 View Details", type="secondary"):
                    st.info("👇 Scroll down to see detailed results and download reports")
            
            st.divider()

    # Sidebar: quick guide
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "- Prepare your scenarios file (CSV or Excel) with required columns.\n"
            "- Upload the file using the control on the right.\n"
            "- Click 'Execute Enhanced Validation'.\n"
            "- Monitor progress; expand details to inspect logs and results."
        )
        st.divider()
        st.caption("Required columns for upload:")
        st.code(
            "Source_Project_Id, Source_Dataset_Id, Target_Project_Id, Target_Dataset_Id,\n"
            "Source_Table, Source_Join_Key, Target_Table, Target_Join_Key, Target_Column, Derivation_Logic",
            language="markdown",
        )
    
    # Top row: template download and uploader
    c1, c2 = st.columns([1, 3])
    with c1:
        # Provide a minimal template download for convenience
        sample_csv = (
            "Source_Project_Id,Source_Dataset_Id,Source_Table,Source_Join_Key,"
            "Target_Project_Id,Target_Dataset_Id,Target_Table,Target_Join_Key,Target_Column,Derivation_Logic,"
            "Scenario_ID,Scenario_Name\n"
        )
        st.download_button(
            label="Download Template (CSV)",
            data=sample_csv,
            file_name="validation_template.csv",
            mime="text/csv",
            help="Start from a clean template with required columns",
        )
    with c2:
        # File uploader widget
        uploaded_file = st.file_uploader(
            "Upload scenarios file (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Select your scenarios file to run validation",
        )
    
    # Display file information if a file is uploaded
    if uploaded_file is not None:
        st.success(f"File successfully attached: {uploaded_file.name}")
        
        # Preview file content based on file type
        if uploaded_file.type in ['text/csv', 'application/vnd.ms-excel', 
                                 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            st.write("**File Preview:**")
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Normalize headers and trim content
                df, rename_map = normalize_headers_and_trim(df)
                if rename_map:
                    st.info(f"Normalized headers: {rename_map}")

                st.dataframe(df.head())

                # Validate DataFrame
                ok, errs, warns = validate_scenarios_df(df)
                for w in warns:
                    st.warning(f"{w}")
                if not ok:
                    for e in errs:
                        st.error(f"{e}")
                    st.stop()

                st.success("✅ Input file passed validation checks!")

                # Store the DataFrame in session state for later use
                st.session_state['uploaded_df'] = df
                st.session_state['uploaded_file_name'] = uploaded_file.name

                # Validation Results Generation Section
                st.write("**Validation Results Generation:**")

                # Enhanced Framework is the only validation method (UI simplified)

                # Generate validation results
                # Run lock control
                st.session_state.setdefault('is_running', False)
                c_run1, c_run2 = st.columns([1,1])
                with c_run1:
                    run_clicked = st.button("Execute Enhanced Validation", type="primary", disabled=st.session_state['is_running'])
                with c_run2:
                    if st.button("Reset Run Lock"):
                        st.session_state['is_running'] = False
                        st.info("Run lock reset.")

                if run_clicked:
                    if st.session_state['is_running']:
                        st.warning("A run is already in progress.")
                        st.stop()
                    st.session_state['is_running'] = True

                    # Get DataFrame from session state
                    df = st.session_state.get('uploaded_df')
                    if df is None:
                        st.error("❌ No uploaded data found. Please refresh and upload the file again.")
                        st.session_state['is_running'] = False
                        st.stop()

                    # Single visible progress bar outside the detailed logs
                    progress_bar = st.progress(0)

                    # Collapsed expander for all detailed logs
                    log = st.expander("Execution details (click to expand)", expanded=False)

                    with log:
                        st.write("🚀 **Starting Validation Process...**")
                        tab_conn, tab_join, tab_scen, tab_files = st.tabs(["Connectivity", "Join Keys", "Scenarios", "Files"])

                    # Step 1: BigQuery Connectivity Testing
                    with tab_conn:
                        st.write("**Step 1: BigQuery Connectivity Testing**")
                            
                    # Get unique combinations for connectivity testing
                    source_combinations = df[['Source_Project_Id', 'Source_Dataset_Id']].drop_duplicates().reset_index(drop=True)
                    target_combinations = df[['Target_Project_Id', 'Target_Dataset_Id']].drop_duplicates().reset_index(drop=True)
                            
                    # Initialize BigQuery client using ADC
                    try:
                        # Get a sample project ID from the data to initialize client
                        sample_project = df['Source_Project_Id'].iloc[0] if 'Source_Project_Id' in df.columns else None

                        if sample_project:
                            client = initialize_client(sample_project)
                            st.success("✅ BigQuery client initialized successfully using Application Default Credentials")
                            # Store client in session state for reuse
                            st.session_state['bq_client'] = client
                        else:
                            st.error("⚠️ Could not determine project ID for BigQuery client initialization")
                            client = None
                            st.session_state['bq_client'] = None
                    except Exception as e:
                        st.error(f"⚠️ Failed to initialize BigQuery client: {str(e)}")
                        st.info("Please ensure Application Default Credentials are set up: `gcloud auth application-default login`")
                        client = None
                        st.session_state['bq_client'] = None

                    if not client:
                        st.error("🛑 **VALIDATION HALTED** - BigQuery client is required for validation.")
                        st.session_state['is_running'] = False
                        st.stop()
                            
                    with log:
                        with tab_conn:
                            st.write("*Testing Source Combinations:*")
                            source_results = []
                            connectivity_failed = False
                            
                    # Test each source combination
                    for idx, row in source_combinations.iterrows():
                                project_id = row['Source_Project_Id']
                                dataset_id = row['Source_Dataset_Id']
                                
                                with st.spinner(f"Testing {project_id}.{dataset_id}..."):
                                    success, message = test_bigquery_connectivity(project_id, dataset_id, client)
                                    source_results.append({
                                        'Project_Id': project_id,
                                        'Dataset_Id': dataset_id,
                                        'Status': '✅ Connected' if success else '❌ Failed',
                                        'Details': message
                                    })
                                    if not success:
                                        connectivity_failed = True
                            
                    # Display source results
                    with log:
                        with tab_conn:
                            source_df = pd.DataFrame(source_results)
                            st.dataframe(source_df, use_container_width=True)
                            
                    with log:
                        with tab_conn:
                            st.write("*Testing Target Combinations:*")
                    target_results = []
                            
                    # Test each target combination
                    for idx, row in target_combinations.iterrows():
                                project_id = row['Target_Project_Id']
                                dataset_id = row['Target_Dataset_Id']
                                
                                with st.spinner(f"Testing {project_id}.{dataset_id}..."):
                                    success, message = test_bigquery_connectivity(project_id, dataset_id, client)
                                    target_results.append({
                                        'Project_Id': project_id,
                                        'Dataset_Id': dataset_id,
                                        'Status': '✅ Connected' if success else '❌ Failed',
                                        'Details': message
                                    })
                                    if not success:
                                        connectivity_failed = True
                            
                    # Display target results
                    with log:
                        with tab_conn:
                            target_df = pd.DataFrame(target_results)
                            st.dataframe(target_df, use_container_width=True)
                            
                    # Check connectivity results
                    if connectivity_failed:
                        with log:
                            with tab_conn:
                                st.error("❌ BigQuery connectivity test failed for one or more project/dataset combinations.")
                                st.error("🛑 **VALIDATION HALTED** - All datasets must be accessible before validation can proceed.")
                        st.session_state['is_running'] = False
                        st.stop()
                            
                    with log:
                        with tab_conn:
                            st.success("✅ All BigQuery connectivity tests passed!")

                    # Step 2: Join Key Uniqueness Validation
                    with log:
                        with tab_join:
                            st.write("**Step 2: Join Key Uniqueness Validation**")
                            
                            # Check for required columns for join key validation
                            join_key_required_columns = ['Source_Table', 'Source_Join_Key', 'Target_Table', 'Target_Join_Key']
                            join_key_missing_columns = [col for col in join_key_required_columns if col not in df.columns]
                            
                            if join_key_missing_columns:
                                with log:
                                    with tab_join:
                                        st.error(f"❌ Missing columns for join key validation: {', '.join(join_key_missing_columns)}")
                                        st.info("Required columns: Source_Table, Source_Join_Key, Target_Table, Target_Join_Key")
                                        st.error("🛑 **VALIDATION HALTED** - Join key validation columns are required.")
                                st.session_state['is_running'] = False
                                st.stop()
                            
                            with log:
                                with tab_join:
                                    st.write("*Validating Join Key Uniqueness for All Scenarios:*")
                                    
                                    # Enhancement: Avoid repeating join key checks for identical table combinations
                                    st.write("🔍 Step 2.1: Identifying unique table combinations...")
                            
                            # Remove non-relevant columns and get unique table combinations
                            exclude_columns = ['Scenario_ID', 'Scenario_Name', 'Description', 'Derivation_Logic']
                            df_for_uniqueness = df.copy()
                            
                            # Drop excluded columns if they exist
                            columns_to_drop = [col for col in exclude_columns if col in df_for_uniqueness.columns]
                            if columns_to_drop:
                                df_for_uniqueness = df_for_uniqueness.drop(columns=columns_to_drop)
                            
                            # Get unique combinations for join key validation
                            unique_combinations = df_for_uniqueness.drop_duplicates().reset_index(drop=True)
                            with log:
                                with tab_join:
                                    st.write(f"✅ Reduced from {len(df)} rows to {len(unique_combinations)} unique table combinations")
                            
                            # Dictionary to cache join key validation results
                            join_key_cache = {}
                            
                            # Progress bar for join key validation
                            with log:
                                with tab_join:
                                    join_progress = st.progress(0)
                            
                            st.write("🔍 Step 2.2: Running join key uniqueness checks on unique combinations...")
                            
                            # Run join key validation only on unique combinations
                            for idx, row in unique_combinations.iterrows():
                                # Update progress
                                progress = (idx + 1) / len(unique_combinations)
                                join_progress.progress(progress)
                                
                                # Parse join keys
                                source_join_keys = [key.strip() for key in str(row.get('Source_Join_Key', '')).split(',') if key.strip()]
                                target_join_keys = [key.strip() for key in str(row.get('Target_Join_Key', '')).split(',') if key.strip()]
                                
                                # Create cache key for this table combination
                                source_project = row.get('Source_Project_Id', '')
                                source_dataset = row.get('Source_Dataset_Id', '')
                                source_table = row.get('Source_Table', '')
                                target_project = row.get('Target_Project_Id', '')
                                target_dataset = row.get('Target_Dataset_Id', '')
                                target_table = row.get('Target_Table', '')
                                reference_table = row.get('Reference_Table', '')
                                
                                # Create cache key tuple
                                cache_key = (source_project, source_dataset, source_table, 
                                           target_project, target_dataset, target_table, 
                                           str(reference_table).strip() if reference_table else '')
                                
                                # Initialize cache result
                                cache_result = {
                                    'source_unique': False,
                                    'target_unique': False, 
                                    'reference_unique': True,  # Default True if no reference table
                                    'source_status': 'SKIPPED',
                                    'target_status': 'SKIPPED',
                                    'reference_status': 'N/A'
                                }
                                
                                try:
                                    if client and source_join_keys and target_join_keys:
                                        # Check source table join key uniqueness
                                        try:
                                            source_unique = are_join_keys_unique(client, source_project, source_dataset, source_table, source_join_keys)
                                            cache_result['source_unique'] = source_unique
                                            cache_result['source_status'] = '✅ UNIQUE' if source_unique else '❌ NOT_UNIQUE'
                                        except Exception as e:
                                            cache_result['source_status'] = f'❌ ERROR: {str(e)[:50]}...'
                                            cache_result['source_unique'] = False
                                        
                                        # Check target table join key uniqueness
                                        try:
                                            target_unique = are_join_keys_unique(client, target_project, target_dataset, target_table, target_join_keys)
                                            cache_result['target_unique'] = target_unique
                                            cache_result['target_status'] = '✅ UNIQUE' if target_unique else '❌ NOT_UNIQUE'
                                        except Exception as e:
                                            cache_result['target_status'] = f'❌ ERROR: {str(e)[:50]}...'
                                            cache_result['target_unique'] = False
                                        
                                        # Check reference table if present
                                        if reference_table and str(reference_table).strip() and str(reference_table).strip().lower() != 'nan':
                                            reference_join_keys = [key.strip() for key in str(row.get('Reference_Join_Key', '')).split(',') if key.strip()]
                                            
                                            if reference_join_keys:
                                                try:
                                                    reference_unique = are_join_keys_unique(client, source_project, source_dataset, reference_table, reference_join_keys)
                                                    cache_result['reference_unique'] = reference_unique
                                                    cache_result['reference_status'] = '✅ UNIQUE' if reference_unique else '❌ NOT_UNIQUE'
                                                except Exception as e:
                                                    cache_result['reference_status'] = f'❌ ERROR: {str(e)[:50]}...'
                                                    cache_result['reference_unique'] = False
                                        
                                    else:
                                        cache_result['source_status'] = 'No client or missing join keys'
                                        cache_result['target_status'] = 'No client or missing join keys'
                                        
                                except Exception as e:
                                    cache_result['source_status'] = f'Validation error: {str(e)[:50]}...'
                                    cache_result['target_status'] = f'Validation error: {str(e)[:50]}...'
                                
                                # Store result in cache
                                join_key_cache[cache_key] = cache_result
                            
                            # Clear progress bar
                            with log:
                                with tab_join:
                                    join_progress.empty()
                                    
                                    st.write("🔍 Step 2.3: Building validation results table using cached results...")
                            
                            # Build validation results for all original rows using cached results
                            join_key_results = []
                            join_key_validation_passed = True
                            
                            for idx, row in df.iterrows():
                                scenario_id = str(row.get('Scenario_ID', f'SC{idx+1:03d}'))
                                scenario_name = str(row.get('Scenario_Name', f'Scenario_{idx+1}'))
                                
                                # Create cache key for lookup
                                source_project = row.get('Source_Project_Id', '')
                                source_dataset = row.get('Source_Dataset_Id', '')
                                source_table = row.get('Source_Table', '')
                                target_project = row.get('Target_Project_Id', '')
                                target_dataset = row.get('Target_Dataset_Id', '')
                                target_table = row.get('Target_Table', '')
                                reference_table = row.get('Reference_Table', '')
                                
                                cache_key = (source_project, source_dataset, source_table, 
                                           target_project, target_dataset, target_table, 
                                           str(reference_table).strip() if reference_table else '')
                                
                                # Lookup cached result
                                cached_result = join_key_cache.get(cache_key, {
                                    'source_unique': False, 'target_unique': False, 'reference_unique': False,
                                    'source_status': 'NOT_FOUND', 'target_status': 'NOT_FOUND', 'reference_status': 'NOT_FOUND'
                                })
                                
                                # Create scenario result using cached data
                                scenario_result = {
                                    'Scenario_ID': scenario_id,
                                    'Scenario_Name': scenario_name,
                                    'Source_Table_Status': cached_result['source_status'],
                                    'Target_Table_Status': cached_result['target_status'],
                                    'Reference_Table_Status': cached_result['reference_status'],
                                    'Overall_Status': 'ERROR',
                                    'Details': ''
                                }
                                
                                # Set overall status based on cached results
                                if cached_result['source_unique'] and cached_result['target_unique'] and cached_result['reference_unique']:
                                    scenario_result['Overall_Status'] = '✅ ALL_UNIQUE'
                                    scenario_result['Details'] = 'All join keys are unique'
                                else:
                                    scenario_result['Overall_Status'] = '❌ VALIDATION_FAILED'
                                    scenario_result['Details'] = 'One or more join keys are not unique'
                                    join_key_validation_passed = False
                                
                                join_key_results.append(scenario_result)
                            
                            # Display join key validation results
                            join_key_df = pd.DataFrame(join_key_results)
                            with log:
                                with tab_join:
                                    st.dataframe(join_key_df, use_container_width=True)
                            
                            # Store join key validation results in session state
                            st.session_state['join_key_validation_results'] = join_key_results
                            st.session_state['join_key_validation_passed'] = join_key_validation_passed
                            st.session_state['join_key_cache'] = join_key_cache  # Cache for reuse
                            
                            with log:
                                with tab_join:
                                    st.write(f"✅ Join key validation completed using {len(join_key_cache)} unique table checks instead of {len(df)} individual checks")
                            
                            # Show summary and handle hard stop logic
                            if join_key_validation_passed:
                                with log:
                                    with tab_join:
                                        st.success("✅ All join key uniqueness validations passed!")
                            else:
                                if HARD_STOP_ON_UNIQUENESS:
                                    with log:
                                        with tab_join:
                                            st.error("❌ Join key uniqueness validation failed. This is a critical requirement for data validation.")
                                            st.error("🛑 **VALIDATION HALTED** - All scenarios must have unique join keys before processing can continue.")
                                            st.info("💡 Please resolve the uniqueness issues in your data tables before retrying validation.")
                                    st.session_state['is_running'] = False
                                    st.stop()
                                else:
                                    with log:
                                        with tab_join:
                                            st.error("❌ Some join key uniqueness validations failed. Please review the results above.")
                                            st.warning("⚠️ Scenarios with failed join key validation will be skipped during validation processing.")
                            
                            # Step 3: Enhanced Validation Framework Execution
                            with log:
                                with tab_scen:
                                    st.write("**Step 3: Enhanced Validation Framework Execution**")
                            
                            # Enhancement 2: Avoid redundant connectivity and join key checks
                            # (Now handled in Steps 1 & 2 above)
                            
                            # Use the already validated results from Steps 1 & 2
                            join_key_validation_passed = st.session_state.get('join_key_validation_passed', False)
                            join_key_validation_results = st.session_state.get('join_key_validation_results', [])
                            join_key_cache = st.session_state.get('join_key_cache', {})
                            client = st.session_state.get('bq_client')
                            
                            # Create lookup for join key validation results for per-scenario processing
                            join_key_lookup = {}
                            if join_key_validation_results:
                                for result in join_key_validation_results:
                                    scenario_id = result['Scenario_ID']
                                    join_key_lookup[scenario_id] = result['Overall_Status'] == '✅ ALL_UNIQUE'
                            
                            # Generate Excel file with execution results
                            with log:
                                with tab_files:
                                    st.write("📊 **Excel File with Execution Results:**")
                            
                            # Create a copy of the original dataframe
                            with log:
                                with tab_scen:
                                    st.write("📋 Step 1: Preparing data structures...")
                            df_with_results = df.copy()
                            
                            # Initialize lists for new columns
                            processing_methods = []
                            total_passed = []
                            total_failed = []
                            overall_status = []
                            failure_files_created = []
                            sql_queries = []  # New list to store SQL queries
                            with log:
                                with tab_scen:
                                    st.write("✅ Data structures initialized")
                            
                            # Create output directory for this run and failure files
                            with log:
                                with tab_files:
                                    st.write("📁 Step 2: Setting up output directories...")
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            base_output_dir = os.path.join(script_dir, "output")
                            run_id, run_dir, failures_dir, enhanced_excel_path = make_run_output_paths(base_output_dir)
                            with log:
                                with tab_files:
                                    st.write(f"✅ Run folder: {run_dir}")
                                    st.write(f"✅ Failures folder: {failures_dir}")
                                    # Preserve outputs toggle
                                    preserve_outputs = st.checkbox(
                                        "Preserve previous run outputs",
                                        value=True,
                                        help="If enabled, existing files in the output folder are kept. If disabled, the folder is cleaned before writing new files.",
                                    )
                                    # Show and open folder helpers
                                    st.caption("Folder path (copy if needed):")
                                    st.code(run_dir)
                                    c_of1, c_of2 = st.columns([1, 2])
                                    with c_of1:
                                        if st.button("Open folder", help="Open output folder in File Explorer (Windows)"):
                                            try:
                                                # Windows-only convenience
                                                os.startfile(run_dir)  # type: ignore[attr-defined]
                                                st.info("Opened folder in File Explorer.")
                                            except Exception:
                                                # Fallback: try opening via file URL
                                                import webbrowser
                                                try:
                                                    folder_uri = 'file:///' + run_dir.replace('\\', '/').replace('\\\n', '/')
                                                    webbrowser.open(folder_uri)
                                                    st.info("Attempted to open folder via default browser.")
                                                except Exception:
                                                    st.warning("Couldn't open folder automatically. Copy the path above.")
                                    with c_of2:
                                        folder_uri_md = 'file:///' + run_dir.replace('\\', '/').replace('\\\n', '/')
                                        st.markdown(f"[Open as link]({folder_uri_md})")
                            
                            # Clean up existing output folders from previous runs (conditional)
                            with log:
                                with tab_files:
                                    st.write("🧹 Step 2.1: Cleaning up old output files...")
                            if os.path.exists(base_output_dir) and not preserve_outputs:
                                try:
                                    for entry in os.listdir(base_output_dir):
                                        entry_path = os.path.join(base_output_dir, entry)
                                        if entry_path == run_dir:
                                            continue
                                        if os.path.isdir(entry_path):
                                            import shutil
                                            shutil.rmtree(entry_path, ignore_errors=True)
                                        elif os.path.isfile(entry_path):
                                            os.remove(entry_path)
                                    with log:
                                        with tab_files:
                                            st.info("🧹 Removed previous run folders from output/.")
                                except Exception as e:
                                    with log:
                                        with tab_files:
                                            st.warning(f"⚠️ Warning: Could not clean all old files: {str(e)}")
                            else:
                                with log:
                                    with tab_files:
                                        st.info("Preserving previous outputs — no cleanup performed.")
                            
                            # Report will be written inside the run folder created above
                            
                            # Enhancement 2: Skip redundant client initialization - already have cached client
                            with log:
                                with tab_scen:
                                    st.write("🔌 Step 3: Using cached BigQuery client and validation results...")
                                    st.write("✅ BigQuery client and join key validation results retrieved from cache")
                            
                            # Create a lookup dictionary for join key validation results (already validated above)
                            join_key_lookup = {}
                            if join_key_validation_results:
                                for result in join_key_validation_results:
                                    scenario_id = result['Scenario_ID']
                                    join_key_lookup[scenario_id] = result['Overall_Status'] == '✅ ALL_UNIQUE'
                            
                            # Process each row
                            with log:
                                with tab_scen:
                                    st.write(f"🔄 Step 4: Processing {len(df)} validation scenarios with Enhanced Framework...")
                                    st.info("🚀 Performance Enhancement: Using unique combinations approach to dramatically reduce data processing volume for large datasets.")
                            
                            # Create a status container inside the expander
                            status_text = log.empty()
                            
                            for idx, row in df.iterrows():
                                # Update progress
                                progress = (idx + 1) / len(df)
                                progress_bar.progress(progress)
                                
                                scenario_id = str(row.get('Scenario_ID', f'SC{idx+1:03d}'))
                                scenario_name = str(row.get('Scenario_Name', f'Scenario_{idx+1}'))
                                
                                with log:
                                    with tab_scen:
                                        status_text.write(f"🔍 Processing scenario {idx+1}/{len(df)}: {scenario_id} - {scenario_name}")
                                
                                # Check join key uniqueness for this scenario (per-scenario check)
                                if not HARD_STOP_ON_UNIQUENESS:
                                    # SOFT SKIP: Check if this specific scenario has failed join key validation
                                    scenario_passed_uniqueness = join_key_lookup.get(scenario_id, True)  # Default to True if not found
                                    if not scenario_passed_uniqueness:
                                        with log:
                                            with tab_scen:
                                                st.warning(f"⚠️ Skipping scenario {scenario_id} - Join key uniqueness validation failed")
                                        # Add default values for skipped scenario
                                        processing_methods.append(f"SKIPPED - Join key uniqueness failed")
                                        total_passed.append(0)
                                        total_failed.append(0)
                                        overall_status.append('SKIPPED')
                                        failure_files_created.append("Skipped due to join key uniqueness failure")
                                        sql_queries.append("-- Scenario skipped --")
                                        continue  # Skip to next scenario
                                
                                try:
                                    # Execute Enhanced Validation Framework (direct processing, no SQL generation)
                                    with log:
                                        with tab_scen:
                                            st.write(f"🚀 Executing enhanced validation framework for scenario {scenario_id}...")
                                            
                                            # Show debug information (no nested expander inside the main log expander)
                                            with st.container():
                                                st.caption(f"Enhanced Debug Info for {scenario_id}")
                                            derivation_logic = row.get('Derivation_Logic', '')
                                            source_cols, target_cols, ref_cols = extract_required_columns_by_table(row)
                                            
                                            st.write(f"**Derivation Logic:** `{derivation_logic}`")
                                            st.write(f"**Table-Specific Column Requirements:**")
                                            st.write(f"   • Source table columns: `{sorted(source_cols) if source_cols else 'None'}`")
                                            st.write(f"   • Target table columns: `{sorted(target_cols) if target_cols else 'None'}`")
                                            if ref_cols:
                                                st.write(f"   • Reference table columns: `{sorted(ref_cols)}`")
                                            
                                            st.write(f"**Table Information:**")
                                            st.write(f"   • Source Table: `{row.get('Source_Project_Id')}.{row.get('Source_Dataset_Id')}.{row.get('Source_Table')}`")
                                            st.write(f"   • Target Table: `{row.get('Target_Project_Id')}.{row.get('Target_Dataset_Id')}.{row.get('Target_Table')}`")
                                            if row.get('Reference_Table') and str(row.get('Reference_Table')).strip().lower() != 'nan':
                                                st.write(f"   • Reference Table: `{row.get('Source_Project_Id')}.{row.get('Source_Dataset_Id')}.{row.get('Reference_Table')}`")
                                            
                                            # Generate and show SQL query for this scenario
                                            sql = generate_sql_for_scenario(
                                                row,
                                                sorted(source_cols),
                                                sorted(target_cols),
                                                sorted(ref_cols),
                                            )
                                            st.write(f"**SQL Query:**")
                                            st.code(sql, language="sql")
                                    
                                    success, message, pass_count, fail_count, failure_file_path, executed_sql = enhanced_validation_execution(
                                        row, client, scenario_id, scenario_name, failures_dir, cache=st.session_state.setdefault('execution_cache', {})
                                    )
                                    
                                    if success:
                                        with log:
                                            with tab_scen:
                                                st.write(f"✅ Scenario {scenario_id}: {pass_count} PASS, {fail_count} FAIL (enhanced)")
                                        total_passed.append(pass_count)
                                        total_failed.append(fail_count)
                                        # Set overall status: PASS if no failures, FAIL if any failures
                                        overall_status.append('PASS' if fail_count == 0 else 'FAIL')
                                        # Enhanced Framework processing completed successfully
                                        processing_methods.append("Enhanced Framework - SQL Pushdown")
                                        # Store the executed SQL query
                                        sql_queries.append(executed_sql if executed_sql else "No SQL available")
                                        
                                        # Track failure file creation
                                        if failure_file_path:
                                            if failure_file_path.startswith("SKIPPED"):
                                                with log:
                                                    with tab_files:
                                                        st.write(f"⚠️ Scenario {scenario_id}: {failure_file_path}")
                                                failure_files_created.append(failure_file_path)
                                            elif failure_file_path == "No failures":
                                                failure_files_created.append("No failures")
                                            elif failure_file_path.startswith("`") and failure_file_path.endswith("`"):
                                                # BigQuery table path
                                                with log:
                                                    with tab_files:
                                                        st.write(f"📄 Failure table created: {failure_file_path}")
                                                failure_files_created.append(failure_file_path)
                                            else:
                                                relative_path = os.path.relpath(failure_file_path, script_dir)
                                                with log:
                                                    with tab_files:
                                                        st.write(f"📄 Failure file created: {relative_path}")
                                                failure_files_created.append(relative_path)
                                        else:
                                            failure_files_created.append("No failures")
                                    else:
                                        with log:
                                            with tab_scen:
                                                st.error(f"❌ Scenario {scenario_id} execution failed: {message}")
                                        # If execution failed, set default values
                                        processing_methods.append(f"EXECUTION_ERROR - {message}")
                                        total_passed.append(0)
                                        total_failed.append(0)
                                        overall_status.append('EXECUTION_ERROR')
                                        # Use the actual error message instead of generic "Execution failed"
                                        failure_files_created.append(f"EXECUTION_ERROR - {message}")
                                        # Store error SQL or indicate failure
                                        sql_queries.append(executed_sql if executed_sql else "-- SQL execution failed --")
                                        
                                except ValueError as ve:
                                    # Handle uniqueness validation errors specifically
                                    if "do not uniquely identify rows" in str(ve):
                                        with log:
                                            with tab_join:
                                                st.error(f"❌ Join key uniqueness validation failed for scenario {scenario_id}: {str(ve)}")
                                        # Add default values for failed scenario
                                        processing_methods.append(f"UNIQUENESS_ERROR - {str(ve)}")
                                        total_passed.append(0)
                                        total_failed.append(0)
                                        overall_status.append('UNIQUENESS_ERROR')
                                        failure_files_created.append(f"Uniqueness Error: {str(ve)}")
                                        sql_queries.append("-- Uniqueness validation failed --")
                                    else:
                                        with log:
                                            with tab_scen:
                                                st.error(f"❌ Validation error for scenario {scenario_id}: {str(ve)}")
                                        # Add default values for failed scenario
                                        processing_methods.append(f"VALIDATION_ERROR - {str(ve)}")
                                        total_passed.append(0)
                                        total_failed.append(0)
                                        overall_status.append('VALIDATION_ERROR')
                                        failure_files_created.append(f"Validation Error: {str(ve)}")
                                        sql_queries.append("-- Validation error occurred --")
                                except Exception as e:
                                    with log:
                                        with tab_scen:
                                            st.error(f"❌ Error processing scenario {scenario_id}: {str(e)}")
                                    # Add default values for failed scenario
                                    processing_methods.append(f"PROCESSING_ERROR - {str(e)}")
                                    total_passed.append(0)
                                    total_failed.append(0)
                                    overall_status.append('ERROR')
                                    failure_files_created.append(f"Error: {str(e)}")
                                    sql_queries.append("-- Processing error occurred --")
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            with log:
                                with tab_files:
                                    status_text.empty()
                                    
                                    st.write("💾 Step 5: Saving results to Excel file...")
                            
                            # Add the new columns to dataframe
                            df_with_results['Processing_Method'] = processing_methods
                            df_with_results['Total_Passed'] = total_passed
                            df_with_results['Total_Failed'] = total_failed
                            df_with_results['Overall_Status'] = overall_status
                            df_with_results['Failure_File_Path'] = failure_files_created
                            df_with_results['SQL_Query'] = sql_queries  # Add SQL column
                            
                            # Save enhanced Excel file into this run's directory
                            os.makedirs(os.path.dirname(enhanced_excel_path), exist_ok=True)
                            with log:
                                with tab_files:
                                    st.write(f"📁 Creating Excel file: {enhanced_excel_path}")
                            
                            # Write to Excel file in output directory
                            with pd.ExcelWriter(enhanced_excel_path, engine='openpyxl') as writer:
                                with log:
                                    with tab_files:
                                        st.write("📝 Writing Results sheet...")
                                # Create Results sheet with specified columns in order (FIRST TAB)
                                results_columns = [
                                    'Scenario_ID', 'Scenario_Name', 'Description', 'Total_Failed', 
                                    'Total_Passed', 'Overall_Status', 'Failure_File_Path', 'Processing_Method'
                                ]
                                
                                # Filter DataFrame to include only the specified columns that exist
                                available_columns = [col for col in results_columns if col in df_with_results.columns]
                                df_results = df_with_results[available_columns].copy()
                                
                                # Write Results sheet first
                                df_results.to_excel(writer, sheet_name='Results', index=False)
                                
                                # Auto-adjust column widths for Results sheet
                                results_worksheet = writer.sheets['Results']
                                for column in results_worksheet.columns:
                                    max_length = 0
                                    column_letter = column[0].column_letter
                                    for cell in column:
                                        try:
                                            if len(str(cell.value)) > max_length:
                                                max_length = len(str(cell.value))
                                        except:
                                            pass
                                    # Set a reasonable max width
                                    adjusted_width = min(max_length + 2, 50)
                                    results_worksheet.column_dimensions[column_letter].width = adjusted_width
                                
                                with log:
                                    with tab_files:
                                        st.write("📝 Writing ValidationScenarios sheet...")
                                # Create ValidationScenarios sheet excluding execution result columns (SECOND TAB)
                                validation_exclude_columns = ['Processing_Method', 'Total_Passed', 'Total_Failed', 'Overall_Status', 'Failure_File_Path']
                                df_validation = df_with_results.drop(columns=[col for col in validation_exclude_columns if col in df_with_results.columns])
                                
                                # Write filtered validation results to ValidationScenarios sheet
                                df_validation.to_excel(writer, sheet_name='ValidationScenarios', index=False)
                                
                                # Auto-adjust column widths for ValidationScenarios sheet
                                worksheet = writer.sheets['ValidationScenarios']
                                for column in worksheet.columns:
                                    max_length = 0
                                    column_letter = column[0].column_letter
                                    for cell in column:
                                        try:
                                            if len(str(cell.value)) > max_length:
                                                max_length = len(str(cell.value))
                                        except:
                                            pass
                                    # Set a reasonable max width
                                    adjusted_width = min(max_length + 2, 50)
                                    worksheet.column_dimensions[column_letter].width = adjusted_width
                            
                            with log:
                                with tab_files:
                                    st.write("✅ Excel file created successfully!")
                            
                            # Show summary statistics
                            # Compute final counts for minimal summary outside the expander
                            total_scenarios = len(df_with_results)
                            passed_scenarios = len(df_with_results[df_with_results['Overall_Status'] == 'PASS'])
                            failed_scenarios = len(df_with_results[df_with_results['Overall_Status'] == 'FAIL'])
                            failure_files_count = len([f for f in failure_files_created if f not in ["No failures", "Execution failed", "No BigQuery client"] and not f.startswith("SKIPPED")])
                            
                            # Detailed summary inside expander
                            with log:
                                st.write("📊 **Final Execution Summary (detailed):**")
                                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                with summary_col1:
                                    st.metric("Total Scenarios", total_scenarios)
                                with summary_col2:
                                    st.metric("Scenarios Passed", passed_scenarios)
                                with summary_col3:
                                    st.metric("Scenarios Failed", failed_scenarios)
                                with summary_col4:
                                    st.metric("Failure Files Created", failure_files_count)
                            
                            # Show information about all created files
                            with log:
                                with tab_files:
                                    st.write("**Created Files:**")
                                    enhanced_excel_relative_path = os.path.relpath(enhanced_excel_path, script_dir)
                                    st.write(f"📄 **Main Report:** {enhanced_excel_relative_path}")
                            
                            # Show created failure files
                            if failure_files_count > 0:
                                with log:
                                    with tab_files:
                                        st.write(f"📋 **Individual Failure Files ({failure_files_count}):**")
                                        failure_files_list = [f for f in failure_files_created if f not in ["No failures", "Execution failed", "No BigQuery client"] and not f.startswith("SKIPPED")]
                                        for file_path in failure_files_list:
                                            st.write(f"   • {file_path}")
                            
                            # Show skipped files due to size limit
                            skipped_files = [f for f in failure_files_created if f.startswith("SKIPPED")]
                            if skipped_files:
                                with log:
                                    with tab_files:
                                        st.write(f"⚠️ **Skipped Files Due to Size Limit (>10,000 rows):**")
                                        for skipped_file in skipped_files:
                                            st.write(f"   • {skipped_file}")
                            
                            with log:
                                with tab_files:
                                    st.info(f"💡 All validation files have been saved to the `output/` directory.")
                                    st.success("✅ Enhanced validation completed! All result files and individual failure files have been created!")
                                    st.session_state['is_running'] = False

                            # Calculate aggregate totals
                            total_records_passed = sum(total_passed) if total_passed else 0
                            total_records_failed = sum(total_failed) if total_failed else 0
                            total_records_validated = total_records_passed + total_records_failed
                            
                            # Store validation results in session state for banner display
                            st.session_state['validation_results'] = {
                                'total_passed': total_records_passed,
                                'total_failed': total_records_failed,
                                'total_records': total_records_validated,
                                'passed_scenarios': passed_scenarios,
                                'failed_scenarios': failed_scenarios,
                                'total_scenarios': total_scenarios
                            }

                            # Enhanced final result summary with validation totals
                            st.write("")
                            
                            pass_percentage = (total_records_passed / total_records_validated * 100) if total_records_validated > 0 else 0
                            
                            # Main Results Display
                            with st.container(border=True):
                                st.subheader("🎯 Validation Results")
                                
                                # Record-level metrics (most important)
                                st.write("**📊 Record Validation Totals:**")
                                results_col1, results_col2, results_col3, results_col4 = st.columns(4)
                                
                                with results_col1:
                                    st.metric(
                                        "Total Records", 
                                        f"{total_records_validated:,}",
                                        help="Total number of target records validated"
                                    )
                                with results_col2:
                                    st.metric(
                                        "✅ Passed", 
                                        f"{total_records_passed:,}",
                                        f"{pass_percentage:.1f}%",
                                        delta_color="normal"
                                    )
                                with results_col3:
                                    fail_percentage = 100 - pass_percentage if total_records_validated > 0 else 0
                                    st.metric(
                                        "❌ Failed", 
                                        f"{total_records_failed:,}",
                                        f"{fail_percentage:.1f}%",
                                        delta_color="inverse"
                                    )
                                with results_col4:
                                    overall_status = "🟢 PASS" if failed_scenarios == 0 else "🔴 FAIL"
                                    st.metric("Overall Status", overall_status)
                                
                                st.write("")
                                
                                # Scenario-level summary (secondary)
                                st.write("**📋 Scenario Summary:**")
                                scenario_col1, scenario_col2, scenario_col3, scenario_col4 = st.columns([1,1,1,2])
                                
                                with scenario_col1:
                                    st.metric("Total Scenarios", total_scenarios)
                                with scenario_col2:
                                    st.metric("Scenarios Passed", passed_scenarios)
                                with scenario_col3:
                                    st.metric("Scenarios Failed", failed_scenarios)
                                with scenario_col4:
                                    try:
                                        with open(enhanced_excel_path, "rb") as f:
                                            st.download_button(
                                                label="📥 Download Full Report (XLSX)",
                                                data=f,
                                                file_name=os.path.basename(enhanced_excel_path),
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                use_container_width=True
                                            )
                                    except Exception:
                                        st.button("📥 Download Report (XLSX)", disabled=True, help="Report not available")
                            
                            # Individual Scenario Results Summary (prominent display)
                            if total_passed and total_failed and overall_status:
                                st.write("")
                                st.write("**📋 Individual Scenario Results:**")
                                
                                # Create a results DataFrame for display
                                scenario_results_data = []
                                for i, (idx, row) in enumerate(df.iterrows()):
                                    scenario_id = row['Scenario_ID']
                                    scenario_name = row.get('Scenario_Name', f'Scenario {scenario_id}')
                                    passed = total_passed[i] if i < len(total_passed) else 0
                                    failed = total_failed[i] if i < len(total_failed) else 0
                                    status = overall_status[i] if i < len(overall_status) else 'UNKNOWN'
                                    processing_method = processing_methods[i] if i < len(processing_methods) else 'Unknown'
                                    
                                    # Format status with emoji
                                    status_display = "✅ PASS" if status == 'PASS' else "❌ FAIL"
                                    
                                    scenario_results_data.append({
                                        'Scenario': f"{scenario_id}",
                                        'Name': scenario_name,
                                        'Passed': f"{passed:,}",
                                        'Failed': f"{failed:,}",
                                        'Status': status_display,
                                        'Method': 'SQL Pushdown' if processing_method.startswith('Enhanced') else processing_method
                                    })
                                
                                # Display results in a clean table
                                results_df = pd.DataFrame(scenario_results_data)
                                
                                # Use columns for better display
                                st.dataframe(
                                    results_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Scenario": st.column_config.TextColumn("Scenario ID", width="small"),
                                        "Name": st.column_config.TextColumn("Scenario Name", width="medium"), 
                                        "Passed": st.column_config.TextColumn("Records Passed", width="small"),
                                        "Failed": st.column_config.TextColumn("Records Failed", width="small"),
                                        "Status": st.column_config.TextColumn("Status", width="small"),
                                        "Method": st.column_config.TextColumn("Processing", width="medium")
                                    }
                                )
                        
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        elif uploaded_file.type == 'text/plain':
            st.write("**File Content:**")
            content = uploaded_file.read().decode('utf-8')
            st.text_area("File content", content, height=200)
        
        elif uploaded_file.type == 'application/json':
            st.write("**JSON Content:**")
            import json
            content = uploaded_file.read().decode('utf-8')
            try:
                json_data = json.loads(content)
                st.json(json_data)
            except Exception as e:
                st.error(f"Error parsing JSON: {e}")
    
    else:
        st.info("Please select a file to attach using the file uploader above.")

if __name__ == "__main__":
    main()

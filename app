import streamlit as st
import pandas as pd
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import json

def fix_table_aliases_in_sql(sql_query):
    """
    Fix table alias references in SQL query to use proper aliases instead of full table names
    """
    import re
    
    # Extract table aliases from FROM and JOIN clauses
    table_aliases = {}
    
    # Pattern to find table aliases: `project.dataset.table` AS alias
    alias_pattern = r'`([^`]+)\.([^`]+)\.([^`]+)`\s+AS\s+(\w+)'
    matches = re.findall(alias_pattern, sql_query, re.IGNORECASE)
    
    for project, dataset, table, alias in matches:
        # Store both full qualified name and just table name for replacement
        full_name = f"{project}.{dataset}.{table}"
        table_aliases[table] = alias
        table_aliases[full_name] = alias
    
    # Split query into sections to only modify SELECT, CASE, WHERE clauses
    # Don't modify FROM and JOIN clauses
    lines = sql_query.split('\n')
    modified_lines = []
    
    for line in lines:
        # Skip FROM and JOIN lines - don't modify table references there
        if (re.match(r'\s*FROM\s+', line, re.IGNORECASE) or 
            re.match(r'\s*(?:LEFT\s+|RIGHT\s+|INNER\s+|FULL\s+)?JOIN\s+', line, re.IGNORECASE)):
            modified_lines.append(line)
            continue
        
        modified_line = line
        
        # Replace table.column references with alias.column
        for table_name, alias in table_aliases.items():
            # Pattern to match table_name.column_name but not in quotes
            pattern = r'\b' + re.escape(table_name) + r'\.(\w+)'
            
            def replace_match(match):
                column_name = match.group(1)
                return f"{alias}.{column_name}"
            
            # Only replace if not inside quotes
            # This is a simplified approach - for production, you might need a more sophisticated parser
            if "'" not in modified_line or modified_line.count("'") % 2 == 0:
                modified_line = re.sub(pattern, replace_match, modified_line)
        
        modified_lines.append(modified_line)
    
    return '\n'.join(modified_lines)

def get_column_data_type(target_project, target_dataset, target_table, target_column, credentials_path):
    """
    Query BigQuery INFORMATION_SCHEMA to get the data type of a target column
    """
    try:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = bigquery.Client(credentials=credentials, project=target_project)
        
        type_query = f"""
        SELECT data_type
        FROM `{target_project}.{target_dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{target_table}' AND column_name = '{target_column}'
        """
        
        query_job = client.query(type_query)
        results = query_job.result(timeout=30)
        
        for row in results:
            return row.data_type
            
        return None  # Column not found
        
    except Exception as e:
        print(f"Error getting column type: {str(e)}")
        return None

def generate_validation_sql(row):
    """
    Generate BigQuery validation SQL for a single row of validation data with type-aware comparison
    """
    # Extract required fields
    source_project = row.get('Source_Project_Id', '')
    source_dataset = row.get('Source_Dataset_Id', '')
    source_table = row.get('Source_Table', '')
    source_join_key = row.get('Source_Join_Key', '')
    
    target_project = row.get('Target_Project_Id', '')
    target_dataset = row.get('Target_Dataset_Id', '')
    target_table = row.get('Target_Table', '')
    target_join_key = row.get('Target_Join_Key', '')
    target_column = row.get('Target_Column', '')
    
    reference_table = row.get('Reference_Table', '')
    reference_join_key = row.get('Reference_Join_Key', '')
    reference_lookup_column = row.get('Reference_Lookup_Column', '')
    
    derivation_logic = row.get('Derivation_Logic', '')
    
    # First, get the data type of the target column
    script_dir = os.path.dirname(os.path.abspath(__file__))
    credentials_file = os.path.join(script_dir, "cohesive-apogee-411113-ca31a86921e7.json")
    
    if os.path.exists(credentials_file):
        column_data_type = get_column_data_type(target_project, target_dataset, target_table, target_column, credentials_file)
    else:
        column_data_type = None
    
    # Build the SQL query
    sql_parts = []
    
    # Main data joining CTE
    sql_parts.append("WITH joined_data AS (")
    sql_parts.append("  SELECT")
    sql_parts.append(f"    COALESCE(target.{target_join_key}, source.{source_join_key}) AS join_key,")
    sql_parts.append(f"    {derivation_logic} AS derived_value,")
    sql_parts.append(f"    target.{target_column} AS actual_value")
    
    # FROM clause with target table
    sql_parts.append(f"  FROM `{target_project}.{target_dataset}.{target_table}` AS target")
    
    # JOIN with source table
    sql_parts.append(f"  JOIN `{source_project}.{source_dataset}.{source_table}` AS source")
    sql_parts.append(f"    ON target.{target_join_key} = source.{source_join_key}")
    
    # LEFT JOIN with reference table (if provided)
    if reference_table and str(reference_table).strip() and str(reference_table).strip().lower() != 'nan':
        # Use Reference_Lookup_Column from source to join with Reference_Join_Key from reference table
        sql_parts.append(f"  LEFT JOIN `{source_project}.{source_dataset}.{reference_table}` AS ref")
        sql_parts.append(f"    ON source.{reference_lookup_column} = ref.{reference_join_key}")
    
    # WHERE clause
    sql_parts.append(f"  WHERE target.{target_column} IS NOT NULL")
    sql_parts.append(")")
    
    # Main SELECT with type-specific comparison logic
    sql_parts.append("")
    sql_parts.append("SELECT")
    sql_parts.append("  join_key,")
    sql_parts.append("  derived_value,")
    sql_parts.append("  actual_value,")
    
    # Generate type-specific comparison logic based on detected column type
    if column_data_type == 'STRING':
        sql_parts.append("  CASE WHEN TRIM(CAST(derived_value AS STRING)) = TRIM(CAST(actual_value AS STRING)) THEN 'PASS' ELSE 'FAIL' END AS status")
    elif column_data_type in ['FLOAT64', 'NUMERIC', 'BIGNUMERIC']:
        sql_parts.append("  CASE WHEN ABS(SAFE_CAST(derived_value AS FLOAT64) - SAFE_CAST(actual_value AS FLOAT64)) < 0.0001 THEN 'PASS' ELSE 'FAIL' END AS status")
    elif column_data_type == 'INT64':
        sql_parts.append("  CASE WHEN SAFE_CAST(derived_value AS INT64) = SAFE_CAST(actual_value AS INT64) THEN 'PASS' ELSE 'FAIL' END AS status")
    elif column_data_type == 'BOOL':
        sql_parts.append("  CASE WHEN SAFE_CAST(derived_value AS BOOL) = SAFE_CAST(actual_value AS BOOL) THEN 'PASS' ELSE 'FAIL' END AS status")
    elif column_data_type in ['DATE', 'DATETIME', 'TIMESTAMP']:
        sql_parts.append("  CASE WHEN DATE(derived_value) = DATE(actual_value) THEN 'PASS' ELSE 'FAIL' END AS status")
    else:
        # Default fallback for unknown or unsupported types
        sql_parts.append("  CASE WHEN CAST(derived_value AS STRING) = CAST(actual_value AS STRING) THEN 'PASS' ELSE 'FAIL' END AS status")
    
    sql_parts.append("FROM joined_data;")
    
    # Summary query with the same type-specific logic
    sql_parts.append("")
    sql_parts.append("-- Summary")
    sql_parts.append("WITH joined_data AS (")
    sql_parts.append("  SELECT")
    sql_parts.append(f"    COALESCE(target.{target_join_key}, source.{source_join_key}) AS join_key,")
    sql_parts.append(f"    {derivation_logic} AS derived_value,")
    sql_parts.append(f"    target.{target_column} AS actual_value")
    sql_parts.append(f"  FROM `{target_project}.{target_dataset}.{target_table}` AS target")
    sql_parts.append(f"  JOIN `{source_project}.{source_dataset}.{source_table}` AS source")
    sql_parts.append(f"    ON target.{target_join_key} = source.{source_join_key}")
    
    # Add reference table join in summary if needed
    if reference_table and str(reference_table).strip() and str(reference_table).strip().lower() != 'nan':
        sql_parts.append(f"  LEFT JOIN `{source_project}.{source_dataset}.{reference_table}` AS ref")
        sql_parts.append(f"    ON source.{reference_lookup_column} = ref.{reference_join_key}")
    
    sql_parts.append(f"  WHERE target.{target_column} IS NOT NULL")
    sql_parts.append(")")
    sql_parts.append("SELECT")
    
    # Apply the same type-specific comparison logic for summary
    if column_data_type == 'STRING':
        sql_parts.append("  CASE WHEN TRIM(CAST(derived_value AS STRING)) = TRIM(CAST(actual_value AS STRING)) THEN 'PASS' ELSE 'FAIL' END AS status,")
    elif column_data_type in ['FLOAT64', 'NUMERIC', 'BIGNUMERIC']:
        sql_parts.append("  CASE WHEN ABS(SAFE_CAST(derived_value AS FLOAT64) - SAFE_CAST(actual_value AS FLOAT64)) < 0.0001 THEN 'PASS' ELSE 'FAIL' END AS status,")
    elif column_data_type == 'INT64':
        sql_parts.append("  CASE WHEN SAFE_CAST(derived_value AS INT64) = SAFE_CAST(actual_value AS INT64) THEN 'PASS' ELSE 'FAIL' END AS status,")
    elif column_data_type == 'BOOL':
        sql_parts.append("  CASE WHEN SAFE_CAST(derived_value AS BOOL) = SAFE_CAST(actual_value AS BOOL) THEN 'PASS' ELSE 'FAIL' END AS status,")
    elif column_data_type in ['DATE', 'DATETIME', 'TIMESTAMP']:
        sql_parts.append("  CASE WHEN DATE(derived_value) = DATE(actual_value) THEN 'PASS' ELSE 'FAIL' END AS status,")
    else:
        sql_parts.append("  CASE WHEN CAST(derived_value AS STRING) = CAST(actual_value AS STRING) THEN 'PASS' ELSE 'FAIL' END AS status,")
    
    sql_parts.append("  COUNT(*) as count")
    sql_parts.append("FROM joined_data")
    sql_parts.append("GROUP BY 1;")
    
    # Fix table aliases in the generated SQL
    raw_sql = "\n".join(sql_parts)
    fixed_sql = fix_table_aliases_in_sql(raw_sql)
    
    return fixed_sql

def execute_validation_sql_and_save_failures(sql_script, credentials_path, scenario_id, scenario_name, output_dir):
    """
    Execute validation SQL and save failure records to individual Excel files
    """
    import re
    
    try:
        # Load credentials from service account file
        if not os.path.exists(credentials_path):
            return False, "Credentials file not found", 0, 0, None
        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Extract project from the first table reference in the SQL
        project_match = re.search(r'`([^.]+)\.[^.]+\.[^`]+`', sql_script)
        if not project_match:
            return False, "Could not extract project ID from SQL", 0, 0, None
        
        project_id = project_match.group(1)
        client = bigquery.Client(credentials=credentials, project=project_id)
        
        # Execute the main query (not the summary) to get all results
        main_query = sql_script.split('-- Summary')[0].strip()
        
        # Execute with dry run first to validate syntax
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_run_job = client.query(main_query, job_config=job_config)
        
        # If dry run succeeds, execute the actual query without limit to get full results
        job_config = bigquery.QueryJobConfig(
            dry_run=False, 
            use_query_cache=False,
            maximum_bytes_billed=500 * 1024 * 1024  # 500 MB limit for full results
        )
        
        query_job = client.query(main_query, job_config=job_config)
        results = query_job.result(timeout=120)  # 2 minute timeout for full results
        
        # Convert all results to DataFrame
        all_rows = []
        for row in results:
            all_rows.append(dict(row))
        
        if not all_rows:
            return True, "Query executed but returned no results", 0, 0, None
        
        df_results = pd.DataFrame(all_rows)
        
        # Count PASS and FAIL results
        pass_count = len(df_results[df_results['status'].str.upper() == 'PASS'])
        fail_count = len(df_results[df_results['status'].str.upper() == 'FAIL'])
        
        # Filter for failed records only
        failed_records = df_results[df_results['status'].str.upper() == 'FAIL']
        
        failure_file_path = None
        if len(failed_records) > 0:
            # Sanitize scenario name for filename
            sanitized_name = re.sub(r'[^\w\-_.]', '_', str(scenario_name))
            sanitized_name = re.sub(r'_+', '_', sanitized_name).strip('_')
            
            # Create filename
            filename = f"{scenario_id}_{sanitized_name}_Failures.xlsx"
            failure_file_path = os.path.join(output_dir, filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save failed records to Excel
            with pd.ExcelWriter(failure_file_path, engine='openpyxl') as writer:
                failed_records.to_excel(writer, sheet_name='FailedValidations', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['FailedValidations']
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
        
        return True, f"Query executed successfully. {pass_count} PASS, {fail_count} FAIL", pass_count, fail_count, failure_file_path
        
    except Exception as e:
        return False, f"Execution error: {str(e)}", 0, 0, None



def execute_validation_sql(sql_script, credentials_path):
    """
    Execute a validation SQL script and return results
    """
    try:
        # Load credentials from service account file
        if not os.path.exists(credentials_path):
            return False, "Credentials file not found", None
        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # Extract project from the first table reference in the SQL
        import re
        project_match = re.search(r'`([^.]+)\.[^.]+\.[^`]+`', sql_script)
        if not project_match:
            return False, "Could not extract project ID from SQL", None
        
        project_id = project_match.group(1)
        client = bigquery.Client(credentials=credentials, project=project_id)
        
        # Execute the main query (not the summary)
        main_query = sql_script.split('-- Summary')[0].strip()
        
        # Execute with dry run first to validate syntax
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_run_job = client.query(main_query, job_config=job_config)
        
        # If dry run succeeds, execute the actual query with limit
        job_config = bigquery.QueryJobConfig(
            dry_run=False, 
            use_query_cache=False,
            maximum_bytes_billed=100 * 1024 * 1024  # 100 MB limit
        )
        
        # Add LIMIT to prevent large result sets
        limited_query = main_query.rstrip(';') + "\nLIMIT 10;"
        
        query_job = client.query(limited_query, job_config=job_config)
        results = query_job.result(timeout=60)  # 60 second timeout
        
        # Convert results to list
        result_rows = []
        for row in results:
            result_rows.append(dict(row))
        
        return True, f"Query executed successfully. Processed {dry_run_job.total_bytes_processed} bytes", result_rows
        
    except Exception as e:
        return False, f"Execution error: {str(e)}", None

def test_bigquery_connectivity(project_id, dataset_id, credentials_path):
    """
    Test connectivity to BigQuery project and dataset
    """
    try:
        # Load credentials from service account file
        if not os.path.exists(credentials_path):
            return False, f"Credentials file not found: {credentials_path}"
        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = bigquery.Client(credentials=credentials, project=project_id)
        
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
    st.title("File Browser and Attachment App")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a file to attach",
        type=['csv', 'xlsx', 'xls', 'txt', 'json', 'pdf'],
        help="Select a file to upload and attach"
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
                
                st.dataframe(df.head())
                
                # Check for required columns and find unique combinations
                required_columns = ['Source_Project_Id', 'Source_Dataset_Id', 'Target_Project_Id', 'Target_Dataset_Id']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Warning: The following required columns are missing from the file: {', '.join(missing_columns)}")
                    st.info("Expected columns: Source_Project_Id, Source_Dataset_Id, Target_Project_Id, Target_Dataset_Id")
                else:
                    st.success("✅ All required columns found!")
                    
                    # Get unique combinations for connectivity testing
                    source_combinations = df[['Source_Project_Id', 'Source_Dataset_Id']].drop_duplicates().reset_index(drop=True)
                    target_combinations = df[['Target_Project_Id', 'Target_Dataset_Id']].drop_duplicates().reset_index(drop=True)
                    
                    # Connectivity Testing Section
                    st.write("**BigQuery Connectivity Testing:**")
                    
                    # Check for credentials file
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    credentials_file = os.path.join(script_dir, "cohesive-apogee-411113-ca31a86921e7.json")
                    
                    if not os.path.exists(credentials_file):
                        st.error("⚠️ BigQuery credentials file not found. Please ensure 'cohesive-apogee-411113-ca31a86921e7.json' is in the same directory.")
                    else:
                        st.write("*Testing Source Combinations:*")
                        source_results = []
                        
                        # Test each source combination
                        for idx, row in source_combinations.iterrows():
                            project_id = row['Source_Project_Id']
                            dataset_id = row['Source_Dataset_Id']
                            
                            with st.spinner(f"Testing {project_id}.{dataset_id}..."):
                                success, message = test_bigquery_connectivity(project_id, dataset_id, credentials_file)
                                source_results.append({
                                    'Project_Id': project_id,
                                    'Dataset_Id': dataset_id,
                                    'Status': '✅ Connected' if success else '❌ Failed',
                                    'Details': message
                                })
                        
                        # Display source results
                        source_df = pd.DataFrame(source_results)
                        st.dataframe(source_df, use_container_width=True)
                        
                        st.write("*Testing Target Combinations:*")
                        target_results = []
                        
                        # Test each target combination
                        for idx, row in target_combinations.iterrows():
                            project_id = row['Target_Project_Id']
                            dataset_id = row['Target_Dataset_Id']
                            
                            with st.spinner(f"Testing {project_id}.{dataset_id}..."):
                                success, message = test_bigquery_connectivity(project_id, dataset_id, credentials_file)
                                target_results.append({
                                    'Project_Id': project_id,
                                    'Dataset_Id': dataset_id,
                                    'Status': '✅ Connected' if success else '❌ Failed',
                                    'Details': message
                                })
                        
                        # Display target results
                        target_df = pd.DataFrame(target_results)
                        st.dataframe(target_df, use_container_width=True)
                        
                    # Validation Results Generation Section
                    st.write("**Validation Results Generation:**")
                    
                    # Check for required columns for SQL generation
                    sql_required_columns = [
                        'Source_Table', 'Source_Join_Key', 'Target_Table', 'Target_Join_Key', 
                        'Target_Column', 'Derivation_Logic'
                    ]
                    sql_missing_columns = [col for col in sql_required_columns if col not in df.columns]
                    
                    if sql_missing_columns:
                        st.warning(f"⚠️ Missing columns for validation: {', '.join(sql_missing_columns)}")
                        st.info("Required columns: Source_Table, Source_Join_Key, Target_Table, Target_Join_Key, Target_Column, Derivation_Logic")
                    else:
                        st.success("✅ All validation columns found!")
                        
                        # Generate validation results
                        if st.button("Generate Validation Results", type="primary"):
                            
                            # Generate Excel file with SQL column and execution results
                            st.write("**Excel File with Generated SQL and Execution Results:**")
                            
                            # Create a copy of the original dataframe
                            df_with_sql = df.copy()
                            
                            # Initialize lists for new columns
                            generated_sqls = []
                            total_passed = []
                            total_failed = []
                            overall_status = []
                            failure_files_created = []
                            
                            # Check for credentials file for execution
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            credentials_file = os.path.join(script_dir, "cohesive-apogee-411113-ca31a86921e7.json")
                            
                            # Create output directory for failure files
                            output_dir = os.path.join(script_dir, "output", "ScenarioFailures")
                            
                            # Process each row
                            with st.spinner("Generating SQL, executing queries, and creating failure files..."):
                                for idx, row in df.iterrows():
                                    # Generate SQL
                                    sql_script = generate_validation_sql(row)
                                    generated_sqls.append(sql_script)
                                    
                                    # Get scenario details
                                    scenario_id = str(row.get('Scenario_ID', f'SC{idx+1:03d}'))
                                    scenario_name = str(row.get('Scenario_Name', f'Scenario_{idx+1}'))
                                    
                                    # Execute SQL to get PASS/FAIL counts and create failure files
                                    if os.path.exists(credentials_file):
                                        success, message, pass_count, fail_count, failure_file_path = execute_validation_sql_and_save_failures(
                                            sql_script, credentials_file, scenario_id, scenario_name, output_dir
                                        )
                                        
                                        if success:
                                            total_passed.append(pass_count)
                                            total_failed.append(fail_count)
                                            # Set overall status: PASS if no failures, FAIL if any failures
                                            overall_status.append('PASS' if fail_count == 0 else 'FAIL')
                                            
                                            # Track failure file creation
                                            if failure_file_path:
                                                relative_path = os.path.relpath(failure_file_path, script_dir)
                                                failure_files_created.append(relative_path)
                                            else:
                                                failure_files_created.append("No failures")
                                        else:
                                            # If execution failed, set default values
                                            total_passed.append(0)
                                            total_failed.append(0)
                                            overall_status.append('EXECUTION_ERROR')
                                            failure_files_created.append("Execution failed")
                                    else:
                                        # If no credentials, set default values
                                        total_passed.append(0)
                                        total_failed.append(0)
                                        overall_status.append('NO_CREDENTIALS')
                                        failure_files_created.append("No credentials")
                            
                            # Add the new columns to dataframe
                            df_with_sql['Generated_SQL'] = generated_sqls
                            df_with_sql['Total_Passed'] = total_passed
                            df_with_sql['Total_Failed'] = total_failed
                            df_with_sql['Overall_Status'] = overall_status
                            df_with_sql['Failure_File_Path'] = failure_files_created
                            
                            # Save enhanced Excel file to output directory
                            enhanced_excel_path = os.path.join(script_dir, "output", "ValidationScenarios_WithSQL.xlsx")
                            os.makedirs(os.path.dirname(enhanced_excel_path), exist_ok=True)
                            
                            # Write to Excel file in output directory
                            with pd.ExcelWriter(enhanced_excel_path, engine='openpyxl') as writer:
                                df_with_sql.to_excel(writer, sheet_name='ValidationScenarios', index=False)
                                
                                # Auto-adjust column widths
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
                            
                            # Show preview of the enhanced dataframe
                            st.write("*Preview of Enhanced Excel File with Execution Results:*")
                            # Show first few rows with truncated SQL for preview
                            preview_df = df_with_sql.copy()
                            preview_df['Generated_SQL'] = preview_df['Generated_SQL'].apply(
                                lambda x: x[:50] + "..." if len(str(x)) > 50 else x
                            )
                            st.dataframe(preview_df.head(), use_container_width=True)
                            
                            # Show summary statistics
                            st.write("**Execution Summary:**")
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            with summary_col1:
                                total_scenarios = len(df_with_sql)
                                st.metric("Total Scenarios", total_scenarios)
                            
                            with summary_col2:
                                passed_scenarios = len(df_with_sql[df_with_sql['Overall_Status'] == 'PASS'])
                                st.metric("Scenarios Passed", passed_scenarios)
                            
                            with summary_col3:
                                failed_scenarios = len(df_with_sql[df_with_sql['Overall_Status'] == 'FAIL'])
                                st.metric("Scenarios Failed", failed_scenarios)
                            
                            with summary_col4:
                                failure_files_count = len([f for f in failure_files_created if f not in ["No failures", "Execution failed", "No credentials"]])
                                st.metric("Failure Files Created", failure_files_count)
                            
                            # Show created failure files
                            if failure_files_count > 0:
                                st.write("**Created Failure Files:**")
                                failure_files_list = [f for f in failure_files_created if f not in ["No failures", "Execution failed", "No credentials"]]
                                for file_path in failure_files_list:
                                    st.write(f"📄 {file_path}")
                                
                                st.info(f"💡 {failure_files_count} individual failure Excel files have been created in the `output/ScenarioFailures/` directory.")
                            
                            # Show information about the created files
                            st.write("**Created Files:**")
                            enhanced_excel_relative_path = os.path.relpath(enhanced_excel_path, script_dir)
                            st.write(f"📄 **Main Report:** {enhanced_excel_relative_path}")
                            st.info(f"💡 All validation files have been saved to the `output/` directory.")
                            
                            st.success("✅ Validation completed! All Excel files with execution results and individual failure files have been created!")
                        
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

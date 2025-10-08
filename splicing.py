# main.py
# Import necessary libraries
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Jotform API Configuration ---
# WARNING: It's best practice to use Streamlit secrets for sensitive data like API keys.
# For this example, we are hardcoding them as requested.
API_KEY = "22179825a79dba61013e4fc3b9d30fa4"
FORM_ID = "251683946561164"

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Pioneer Splicing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_jotform_submissions(api_key, form_id):
    """
    Fetches all submissions for a given Jotform form ID using the API, handling pagination.
    """
    all_submissions = []
    limit = 1000  # Max limit per request
    offset = 0
    
    with st.spinner("Fetching data from Jotform..."):
        while True:
            url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={api_key}&offset={offset}&limit={limit}"
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                data = response.json()
                submissions = data.get('content', [])
                all_submissions.extend(submissions)

                # If the number of returned submissions is less than the limit, we've reached the end
                if len(submissions) < limit:
                    break
                
                # Otherwise, increase the offset to get the next batch
                offset += limit
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data from Jotform API: {e}")
                return None
            except ValueError:
                st.error("Error decoding JSON response from Jotform API. The response may not be valid.")
                return None
    return all_submissions


def process_submissions_to_dataframe(submissions):
    """
    Processes raw submission data into a structured Pandas DataFrame.
    Each row represents a submission, and columns are the form questions.
    """
    if not submissions:
        return pd.DataFrame()

    processed_records = []
    for submission in submissions:
        record = {'submission_id': submission['id'], 'created_at': submission['created_at']}
        answers = submission.get('answers', {})
        for qid, answer_data in answers.items():
            question_text = answer_data.get('text', f'qid_{qid}')
            answer = answer_data.get('answer', 'N/A')

            # Handle structured answers (like name or address fields) by joining them
            if isinstance(answer, dict):
                answer = ' '.join(filter(None, answer.values()))
            
            record[question_text] = answer
        processed_records.append(record)

    df = pd.DataFrame(processed_records)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['submission_date'] = df['created_at'].dt.date
    return df

def parse_and_analyze_closures(df, column_name, project_col, tech_col):
    """
    Parses the 'Closures/Panels' field and creates a new analysis section.
    """
    if column_name not in df.columns:
        st.warning(f"Could not find a '{column_name}' field in your form. Detailed closure analysis cannot be performed.")
        return

    st.header("Closures & Panels Analysis")
    st.info("""
    **Note:** This analysis assumes each entry in the 'Closures/Panels' field is on a new line and follows this key-value pair format:
    `Closure Type: [Type], Closure Name: [Name], Splice Type: [Splice Type], Splice Count: [Count], Fiber Type: [Fiber Type], Max Cable Size: [Size]`
    
    **Example:** `Closure Type: FAT, Closure Name: S0X, Splice Type: End Cut, Splice Count: 4, Fiber Type: Loose Tube, Max Cable Size: 24`
    """)

    parsed_entries = []
    for _, row in df.iterrows():
        submission_id = row['submission_id']
        project = row.get(project_col, 'N/A')
        technician = row.get(tech_col, 'N/A')
        entries_str = row[column_name]

        if pd.isna(entries_str) or not str(entries_str).strip():
            continue

        lines = str(entries_str).strip().split('\n')
        for line in lines:
            try:
                # Parse the key-value pair format
                parts = [p.strip() for p in line.split(',')]
                entry_dict = {}
                for part in parts:
                    key_value = part.split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        entry_dict[key] = value

                # Check if all required keys are present
                required_keys = ['Closure Type', 'Closure Name', 'Splice Type', 'Splice Count', 'Fiber Type', 'Max Cable Size']
                if all(key in entry_dict for key in required_keys):
                    parsed_entries.append({
                        'submission_id': submission_id,
                        'Project': project,
                        'Technician': technician,
                        'Type': entry_dict.get('Closure Type'),
                        'Name': entry_dict.get('Closure Name'),
                        'Splice Type': entry_dict.get('Splice Type'),
                        'Splice Count': int(entry_dict.get('Splice Count')),
                        'Fiber Type': entry_dict.get('Fiber Type'),
                        'Max Cable Size': entry_dict.get('Max Cable Size')
                    })
            except (ValueError, IndexError):
                # Silently skip malformed lines for now.
                pass

    if not parsed_entries:
        st.warning("No valid 'Closures/Panels' entries could be parsed from the filtered data. Please check the format.")
        
        # --- Add debugging information ---
        st.subheader("Parsing Debug Information")
        st.write("The script could not find all the required keys in the 'Closures/Panels' data. Here is a sample of what was parsed from the first entry it found. Compare the 'Parsed Keys' with the 'Required Keys' to identify any mismatches.")

        debug_info_displayed = False
        # Find the first row with data to debug
        for _, row in df.iterrows():
            entries_str = row[column_name]
            if pd.notna(entries_str) and str(entries_str).strip():
                lines = str(entries_str).strip().split('\n')
                if lines:
                    first_line = lines[0]
                    st.write("**Original Data from First Line:**")
                    st.code(first_line)

                    parts = [p.strip() for p in first_line.split(',')]
                    entry_dict = {}
                    for part in parts:
                        key_value = part.split(':', 1)
                        if len(key_value) == 2:
                            key = key_value[0].strip()
                            value = key_value[1].strip()
                            entry_dict[key] = value
                    
                    st.write("**Parsed Dictionary:**")
                    st.json(entry_dict)

                    st.write("**Parsed Keys vs. Required Keys:**")
                    parsed_keys = list(entry_dict.keys())
                    required_keys = ['Closure Type', 'Closure Name', 'Splice Type', 'Splice Count', 'Fiber Type', 'Max Cable Size']
                    
                    # Create a comparison DataFrame
                    comparison_data = []
                    all_keys = sorted(list(set(required_keys + parsed_keys)))
                    for key in all_keys:
                        comparison_data.append({
                            "Key Name": key,
                            "Is Required": key in required_keys,
                            "Was Found": key in parsed_keys
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.table(comparison_df)

                    debug_info_displayed = True
                    break # Only show debug info for the very first entry
        
        if not debug_info_displayed:
            st.write("Could not find any data in the 'Closures/Panels' column to debug.")
        # --- End of debugging information ---
        
        return

    closures_df = pd.DataFrame(parsed_entries)

    # --- KPIs for Closures ---
    total_splices = closures_df['Splice Count'].sum()
    total_fats = closures_df[closures_df['Type'].str.upper() == 'FAT'].shape[0]
    total_scs = closures_df[closures_df['Type'].str.upper() == 'SC'].shape[0]
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Total Splice Count", f"{total_splices:,}")
    kpi_col2.metric("Total FATs", f"{total_fats:,}")
    kpi_col3.metric("Total SCs", f"{total_scs:,}")

    # --- Detailed Table and Visualizations ---
    st.subheader("Detailed Entries")
    st.dataframe(closures_df)

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        splice_type_summary = closures_df.groupby('Splice Type')['Splice Count'].sum().reset_index()
        fig_splice_type = px.bar(
            splice_type_summary,
            x='Splice Type',
            y='Splice Count',
            title='Total Splices by Type'
        )
        st.plotly_chart(fig_splice_type, use_container_width=True)
    
    with viz_col2:
        closure_type_summary = closures_df['Type'].value_counts().reset_index()
        closure_type_summary.columns = ['Type', 'Count']
        fig_closure_type = px.pie(
            closure_type_summary,
            names='Type',
            values='Count',
            title='Distribution of Closure/Panel Types'
        )
        st.plotly_chart(fig_closure_type, use_container_width=True)


# --- Streamlit App UI ---
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("📊 Pioneer Splicing")

    # --- Field Name Configuration ---
    # IMPORTANT: Adjust these variable values if your Jotform field names are different.
    # Check the "Detected form fields" message below to find the correct names.
    project_column = 'Project '
    technician_column = 'Technician Name'
    splicing_hours_column = 'Splicing Hours'
    closures_panels_column = 'Closures/Panels'
    pay_rate = 25.00

    # Fetch and process data
    submissions = get_jotform_submissions(API_KEY, FORM_ID)
    
    if submissions is not None:
        df = process_submissions_to_dataframe(submissions)

        if df.empty:
            st.warning("No submissions found for this form, or the data could not be processed.")
            return

        # --- Debugging Help ---
        # This line will show you all the column names found in your form data.
        st.info(f"Detected form fields (use these to update the configuration above if needed): {df.columns.tolist()}")

        # --- Sidebar Filters ---
        st.sidebar.header("Dashboard Filters")
        
        # Start with the full dataframe, which will be sequentially filtered
        filtered_df = df.copy()

        # --- Filter by Date Range ---
        st.sidebar.subheader("Filter by Date")
        min_date = filtered_df['submission_date'].min()
        max_date = filtered_df['submission_date'].max()

        if min_date and max_date and min_date != max_date:
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (filtered_df['created_at'].dt.date >= start_date) & (filtered_df['created_at'].dt.date <= end_date)
                filtered_df = filtered_df.loc[mask]
        else:
             st.sidebar.write("Only one submission date available.")

        # --- Filter by Project ---
        st.sidebar.subheader("Filter by Project")
        if project_column in filtered_df.columns:
            all_projects = sorted(filtered_df[project_column].unique())
            selected_projects = st.sidebar.multiselect(
                "Select Project(s)",
                options=all_projects,
                default=all_projects
            )
            if selected_projects:
                filtered_df = filtered_df[filtered_df[project_column].isin(selected_projects)]
        else:
            st.sidebar.warning(f"Could not find a '{project_column}' field in your form.")

        # --- Filter by Technician Name ---
        st.sidebar.subheader("Filter by Technician")
        if technician_column in filtered_df.columns:
            all_technicians = sorted(filtered_df[technician_column].unique())
            selected_technicians = st.sidebar.multiselect(
                "Select Technician(s)",
                options=all_technicians,
                default=all_technicians
            )
            if selected_technicians:
                filtered_df = filtered_df[filtered_df[technician_column].isin(selected_technicians)]
        else:
            st.sidebar.warning(f"Could not find a '{technician_column}' field in your form.")


        # --- Main Dashboard ---
        
        # Display a warning if all data has been filtered out
        if filtered_df.empty:
            st.warning("No data matches the current filter settings.")
            return

        # --- Splicing Hours & Pay Calculation ---
        st.header("Splicing & Production Analysis")
        if splicing_hours_column in filtered_df.columns:
            # Convert hours column to numeric, coercing errors to NaN
            filtered_df[splicing_hours_column] = pd.to_numeric(filtered_df[splicing_hours_column], errors='coerce')
            
            # Drop rows where hours are NaN after conversion
            prod_df = filtered_df.dropna(subset=[splicing_hours_column])

            # KPIs for Splicing
            total_splicing_hours = prod_df[splicing_hours_column].sum()
            total_production_pay = total_splicing_hours * pay_rate

            pay_col1, pay_col2 = st.columns(2)
            pay_col1.metric("Total Splicing Hours", f"{total_splicing_hours:,.2f}")
            pay_col2.metric("Total Production Pay", f"${total_production_pay:,.2f}")
            
            # --- Breakdown Table and Chart ---
            if all([project_column in prod_df.columns, technician_column in prod_df.columns]):
                st.subheader("Production Breakdown (Project & Technician)")
                summary_df = prod_df.groupby([project_column, technician_column])[splicing_hours_column].sum().reset_index()
                summary_df.rename(columns={splicing_hours_column: 'Total Splicing Hours'}, inplace=True)
                summary_df['Total Production Pay ($)'] = summary_df['Total Splicing Hours'] * pay_rate
                
                # Display table and chart side-by-side
                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    st.dataframe(summary_df.style.format({
                        'Total Splicing Hours': '{:,.2f}',
                        'Total Production Pay ($)': '${:,.2f}'
                    }))

                with viz_col2:
                    # Use unformatted data for charting
                    chart_df = prod_df.groupby([project_column, technician_column])[splicing_hours_column].sum().reset_index()
                    chart_df['Production Pay'] = chart_df[splicing_hours_column] * pay_rate

                    fig_prod = px.bar(
                        chart_df,
                        x=project_column,
                        y='Production Pay',
                        color=technician_column,
                        title='Production Pay by Project & Technician',
                        barmode='group'
                    )
                    st.plotly_chart(fig_prod, use_container_width=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                # --- Splicing by Technician ---
                st.subheader("Splicing Analysis by Technician")
                tech_summary_df = prod_df.groupby(technician_column)[splicing_hours_column].sum().reset_index()
                tech_summary_df.rename(columns={splicing_hours_column: 'Total Splicing Hours'}, inplace=True)
                tech_summary_df['Total Production Pay ($)'] = tech_summary_df['Total Splicing Hours'] * pay_rate
                tech_summary_df = tech_summary_df.sort_values(by='Total Splicing Hours', ascending=False)

                tech_col1, tech_col2 = st.columns(2)
                with tech_col1:
                    st.dataframe(tech_summary_df.style.format({
                        'Total Splicing Hours': '{:,.2f}',
                        'Total Production Pay ($)': '${:,.2f}'
                    }))
                with tech_col2:
                    fig_tech = px.pie(
                        tech_summary_df,
                        names=technician_column,
                        values='Total Splicing Hours',
                        title='Splicing Hours Distribution by Technician'
                    )
                    st.plotly_chart(fig_tech, use_container_width=True)

                # --- Splicing by Project ---
                st.subheader("Splicing Analysis by Project")
                project_summary_df = prod_df.groupby(project_column)[splicing_hours_column].sum().reset_index()
                project_summary_df.rename(columns={splicing_hours_column: 'Total Splicing Hours'}, inplace=True)
                project_summary_df['Total Production Pay ($)'] = project_summary_df['Total Splicing Hours'] * pay_rate
                project_summary_df = project_summary_df.sort_values(by='Total Splicing Hours', ascending=False)
                
                proj_col1, proj_col2 = st.columns(2)
                with proj_col1:
                    st.dataframe(project_summary_df.style.format({
                        'Total Splicing Hours': '{:,.2f}',
                        'Total Production Pay ($)': '${:,.2f}'
                    }))
                with proj_col2:
                    fig_proj = px.bar(
                        project_summary_df,
                        x=project_column,
                        y='Total Splicing Hours',
                        color=project_column,
                        title='Total Splicing Hours by Project'
                    )
                    st.plotly_chart(fig_proj, use_container_width=True)

            else:
                st.info(f"Ensure your form has '{project_column}' and '{technician_column}' fields for a detailed breakdown.")
        else:
            st.warning(f"Could not find a '{splicing_hours_column}' field in your form. Production analysis cannot be performed.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Closures & Panels Analysis ---
        parse_and_analyze_closures(filtered_df, closures_panels_column, project_column, technician_column)

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Visualizations ---
        st.header("General Visual Insights")
        
        # 1. Submissions Over Time
        st.subheader("Submissions Trend")
        submissions_by_day = filtered_df.groupby('submission_date').size().reset_index(name='count')
        fig_line = px.line(
            submissions_by_day,
            x='submission_date',
            y='count',
            title='Daily Number of Submissions',
            labels={'submission_date': 'Date', 'count': 'Number of Submissions'},
            markers=True
        )
        fig_line.update_layout(xaxis_title='Date', yaxis_title='Submission Count')
        st.plotly_chart(fig_line, use_container_width=True)

        # 2. Dynamic charts based on form questions
        st.subheader("Response Analysis")
        
        potential_cols_to_plot = [col for col in filtered_df.columns if col not in ['submission_id', 'created_at', 'submission_date', 'Rating_numeric']]
        
        if not potential_cols_to_plot:
            st.warning("No suitable questions found in the form for analysis.")
            return

        selected_column = st.selectbox(
            "Select a Question to Visualize",
            options=potential_cols_to_plot
        )

        if selected_column:
            col1_viz, col2_viz = st.columns(2)
            
            with col1_viz:
                counts = filtered_df[selected_column].value_counts().reset_index()
                counts.columns = [selected_column, 'count']
                
                fig_pie = px.pie(
                    counts,
                    names=selected_column,
                    values='count',
                    title=f'Distribution of Responses for "{selected_column}"'
                )
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2_viz:
                fig_bar = px.bar(
                    counts,
                    x=selected_column,
                    y='count',
                    title=f'Count of Responses for "{selected_column}"',
                    labels={selected_column: 'Response', 'count': 'Number of Submissions'}
                )
                fig_bar.update_layout(xaxis_title='Response', yaxis_title='Count')
                st.plotly_chart(fig_bar, use_container_width=True)

        # --- Raw Data View ---
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Raw Submission Data")
        if st.checkbox("Show Raw Data Table"):
            st.dataframe(filtered_df)

if __name__ == "__main__":
    main()


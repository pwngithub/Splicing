# main.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

# --- Jotform API Configuration ---
API_KEY = "22179825a79dba61013e4fc3b9d30fa4"
FORM_ID = "251683946561164"

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Pioneer Splicing Dashboard",
    page_icon="fiber_manual_record",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polish ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=600)
def get_jotform_submissions(api_key, form_id):
    """Fetches submissions from Jotform API with pagination."""
    all_submissions = []
    limit = 1000
    offset = 0
    
    try:
        while True:
            url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={api_key}&offset={offset}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            submissions = data.get('content', [])
            all_submissions.extend(submissions)

            if len(submissions) < limit:
                break
            offset += limit
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None
        
    return all_submissions

def process_submissions_to_dataframe(submissions):
    """Converts raw JSON submissions into a Pandas DataFrame."""
    if not submissions:
        return pd.DataFrame()

    processed_records = []
    for submission in submissions:
        record = {'submission_id': submission['id'], 'created_at': submission['created_at']}
        answers = submission.get('answers', {})
        for qid, answer_data in answers.items():
            question_text = answer_data.get('text', f'qid_{qid}')
            answer = answer_data.get('answer', 'N/A')
            # Join dict answers (like Name/Address)
            if isinstance(answer, dict):
                answer = ' '.join(filter(None, answer.values()))
            record[question_text] = answer
        processed_records.append(record)

    df = pd.DataFrame(processed_records)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['submission_date'] = df['created_at'].dt.date
    return df

def extract_closure_data(df, closures_col, hours_col, project_col, tech_col):
    """
    Extracts nested JSON closure data and allocates splicing hours to specific entries.
    Returns a normalized DataFrame of individual closure/splice events.
    """
    if closures_col not in df.columns:
        return pd.DataFrame()

    parsed_entries = []

    for _, row in df.iterrows():
        # Get row-level metadata
        submission_id = row['submission_id']
        project = row.get(project_col, 'N/A')
        technician = row.get(tech_col, 'N/A')
        total_hours = pd.to_numeric(row.get(hours_col, 0), errors='coerce') or 0
        entries_str = row[closures_col]

        if pd.isna(entries_str) or not str(entries_str).strip():
            continue

        try:
            json_data = json.loads(entries_str)
            if not isinstance(json_data, list):
                continue
            
            # 1. Calculate Total Splice Count for this specific submission first
            # We need this to distribute the hours proportionally
            submission_splice_total = 0
            temp_items = []
            
            for entry in json_data:
                # Basic validation
                if isinstance(entry, dict) and 'Splice Count' in entry:
                    s_count = int(entry.get('Splice Count', 0))
                    submission_splice_total += s_count
                    temp_items.append(entry)

            # 2. Process each item and allocate hours
            for item in temp_items:
                s_count = int(item.get('Splice Count', 0))
                
                # Weighted Time Allocation:
                # If this closure had 50% of the splices, assign 50% of the hours.
                if submission_splice_total > 0:
                    allocated_hours = total_hours * (s_count / submission_splice_total)
                else:
                    allocated_hours = 0

                parsed_entries.append({
                    'Submission ID': submission_id,
                    'Project': project,
                    'Technician': technician,
                    'Closure Type': item.get('Closure Type', 'Unknown'),
                    'Closure Name': item.get('Closure Name', 'N/A'),
                    'Splice Type': item.get('Splice Type', 'Unknown'),
                    'Splice Count': s_count,
                    'Fiber Type': item.get('Fiber Type', 'N/A'),
                    'Allocated Hours': allocated_hours # The new tracked metric
                })

        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    return pd.DataFrame(parsed_entries)

# --- Main App Logic ---
def main():
    # --- Sidebar & Configuration ---
    st.sidebar.title("🔧 Settings")
    
    # Configurable Column Mapping
    with st.sidebar.expander("Field Configuration", expanded=False):
        st.caption("Match these to your Jotform field names.")
        project_col = st.text_input("Project Column", 'Project ')
        tech_col = st.text_input("Technician Column", 'Technician Name')
        hours_col = st.text_input("Hours Column", 'Splicing Hours')
        closures_col = st.text_input("Closures JSON Column", 'Closures/Panels')
        pay_rate = st.number_input("Pay Rate ($/hr)", value=25.00)

    # Load Data
    with st.spinner("Syncing with Jotform..."):
        raw_submissions = get_jotform_submissions(API_KEY, FORM_ID)
        
    if not raw_submissions:
        st.error("Failed to load submissions.")
        return

    df_master = process_submissions_to_dataframe(raw_submissions)
    
    if df_master.empty:
        st.warning("No data found.")
        return

    # --- Global Filters ---
    st.sidebar.subheader("Filter Data")
    
    # Date Filter
    min_d, max_d = df_master['submission_date'].min(), df_master['submission_date'].max()
    date_range = st.sidebar.date_input("Date Range", [min_d, max_d], min_value=min_d, max_value=max_d)
    
    # Apply Date Filter
    if len(date_range) == 2:
        mask = (df_master['submission_date'] >= date_range[0]) & (df_master['submission_date'] <= date_range[1])
        df = df_master.loc[mask].copy()
    else:
        df = df_master.copy()

    # Project Filter
    if project_col in df.columns:
        projects = df[project_col].unique().tolist()
        sel_proj = st.sidebar.multiselect("Projects", projects, default=projects)
        df = df[df[project_col].isin(sel_proj)]

    # Technician Filter
    if tech_col in df.columns:
        techs = df[tech_col].unique().tolist()
        sel_tech = st.sidebar.multiselect("Technicians", techs, default=techs)
        df = df[df[tech_col].isin(sel_tech)]

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    # --- Pre-Calculate Derived Data ---
    # 1. Clean Numeric Hours
    if hours_col in df.columns:
        df[hours_col] = pd.to_numeric(df[hours_col], errors='coerce').fillna(0)
    
    # 2. Extract Detailed Closure/Splice Items
    df_details = extract_closure_data(df, closures_col, hours_col, project_col, tech_col)

    # --- Dashboard Header ---
    st.title("📊 Pioneer Splicing Dashboard")
    st.markdown("---")

    # --- High Level KPIs ---
    total_hours = df[hours_col].sum() if hours_col in df.columns else 0
    total_pay = total_hours * pay_rate
    total_splices = df_details['Splice Count'].sum() if not df_details.empty else 0
    
    # Calculate Efficiency (Splices per Hour)
    efficiency = (total_splices / total_hours) if total_hours > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Splicing Hours", f"{total_hours:,.1f} hrs")
    k2.metric("Total Production Pay", f"${total_pay:,.2f}")
    k3.metric("Total Splices Completed", f"{total_splices:,}")
    k4.metric("Avg Efficiency", f"{efficiency:.1f} splices/hr")

    # --- Tabs Layout ---
    tab_prod, tab_splice, tab_data = st.tabs(["💰 Production & Pay", "🔌 Splice Analysis (By Type)", "📋 Raw Data"])

    # === TAB 1: PRODUCTION & PAY ===
    with tab_prod:
        st.subheader("Production Financials")
        
        if hours_col in df.columns and tech_col in df.columns:
            col_chart, col_table = st.columns([2, 1])
            
            # Aggregate Data
            pay_data = df.groupby(tech_col)[[hours_col]].sum().reset_index()
            pay_data['Total Pay'] = pay_data[hours_col] * pay_rate
            pay_data = pay_data.sort_values('Total Pay', ascending=True)

            with col_chart:
                fig_pay = px.bar(
                    pay_data, 
                    x='Total Pay', 
                    y=tech_col, 
                    orientation='h',
                    title="Total Pay by Technician",
                    text_auto='.2s',
                    color='Total Pay',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_pay, use_container_width=True)

            with col_table:
                st.write("**Detailed Breakdown**")
                st.dataframe(
                    pay_data.style.format({'Total Pay': '${:,.2f}', hours_col: '{:,.1f}'}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("Required columns for Pay Analysis missing.")

    # === TAB 2: SPLICE TYPE ANALYSIS ===
    with tab_splice:
        st.subheader("Detailed Splice Tracking")
        st.caption("Time is allocated to splice types based on their proportion of the total splice count per submission.")

        if df_details.empty:
            st.info("No detailed closure data available to analyze.")
        else:
            # Layout: 2 Columns for the main charts
            c1, c2 = st.columns(2)

            # --- Analysis 1: By Splice Type (Ribbon vs Single etc) ---
            with c1:
                st.markdown("#### By Splice Type")
                # Group by Splice Type
                splice_type_stats = df_details.groupby('Splice Type').agg({
                    'Splice Count': 'sum',
                    'Allocated Hours': 'sum'
                }).reset_index()
                
                # Add calculated efficiency per type
                splice_type_stats['Splices/Hr'] = splice_type_stats['Splice Count'] / splice_type_stats['Allocated Hours'].replace(0, 1)

                # Toggle between Time and Count
                view_mode = st.radio("View Metric:", ["Total Time (Hours)", "Splice Count"], horizontal=True, key="view_type")
                y_val = 'Allocated Hours' if view_mode == "Total Time (Hours)" else 'Splice Count'

                fig_st = px.pie(
                    splice_type_stats, 
                    names='Splice Type', 
                    values=y_val, 
                    hole=0.4,
                    title=f"{view_mode} by Splice Type"
                )
                st.plotly_chart(fig_st, use_container_width=True)
                
                with st.expander("See Splice Type Data"):
                    st.dataframe(splice_type_stats.style.format({'Allocated Hours': '{:.1f}', 'Splices/Hr': '{:.1f}'}), use_container_width=True)

            # --- Analysis 2: By Closure Type (FAT, SC, etc) ---
            with c2:
                st.markdown("#### By Closure/Panel Type")
                closure_type_stats = df_details.groupby('Closure Type').agg({
                    'Splice Count': 'sum',
                    'Allocated Hours': 'sum'
                }).reset_index().sort_values('Splice Count', ascending=False)

                fig_ct = px.bar(
                    closure_type_stats,
                    x='Closure Type',
                    y=['Splice Count', 'Allocated Hours'],
                    title="Volume & Effort by Closure Type",
                    barmode='group'
                )
                st.plotly_chart(fig_ct, use_container_width=True)

            st.markdown("---")
            
            # --- Analysis 3: Heatmap (Technician vs Splice Type) ---
            st.markdown("#### Technician Specialty Analysis")
            heatmap_data = df_details.pivot_table(
                index=tech_col, 
                columns='Splice Type', 
                values='Splice Count', 
                aggfunc='sum', 
                fill_value=0
            )
            
            fig_heat = px.imshow(
                heatmap_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Splice Count Heatmap: Technician vs. Splice Type"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # === TAB 3: RAW DATA ===
    with tab_data:
        st.subheader("Data Explorer")
        
        # Option to view parsed details or raw submissions
        view_opt = st.selectbox("Select Dataset", ["Detailed Splicing Log (Parsed)", "Submission Records (Raw)"])
        
        if view_opt == "Detailed Splicing Log (Parsed)":
            st.dataframe(df_details, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

        # Download Button
        csv = df_details.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Parsed Data as CSV",
            csv,
            "splicing_data.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()

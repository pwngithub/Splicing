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
    page_title="Jotform KPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

def get_jotform_submissions(api_key, form_id):
    """
    Fetches all submissions for a given Jotform form ID using the API.
    """
    url = f"https://api.jotform.com/form/{form_id}/submissions?apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        return data.get('content', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Jotform API: {e}")
        return None
    except ValueError:
        st.error("Error decoding JSON response from Jotform API. The response may not be valid.")
        return None


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


# --- Streamlit App UI ---
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("📊 Jotform KPI Dashboard")
    st.markdown("This dashboard visualizes submissions from the specified Jotform form.")

    # Fetch and process data
    submissions = get_jotform_submissions(API_KEY, FORM_ID)
    
    if submissions is not None:
        df = process_submissions_to_dataframe(submissions)

        if df.empty:
            st.warning("No submissions found for this form, or the data could not be processed.")
            return

        # --- Sidebar Filters ---
        st.sidebar.header("Filters")
        
        # Allow filtering by date range
        min_date = df['submission_date'].min()
        max_date = df['submission_date'].max()

        if min_date and max_date and min_date != max_date:
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            # Ensure date_range has two values before unpacking
            if len(date_range) == 2:
                start_date, end_date = date_range
                # Filter dataframe based on selected date range
                mask = (df['created_at'].dt.date >= start_date) & (df['created_at'].dt.date <= end_date)
                filtered_df = df.loc[mask]
            else:
                filtered_df = df.copy() # Use the full dataframe if date range is not set properly
        else:
             st.sidebar.write("Only one submission date available. No date range filter.")
             filtered_df = df.copy()

        # --- Main Dashboard ---
        
        # Top-level KPIs
        st.header("Overall Metrics")
        total_submissions = filtered_df.shape[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Submissions", f"{total_submissions}")
        # Add more metrics here as needed based on your form fields
        # For example, if you have a 'Rating' field:
        if 'Rating' in filtered_df.columns:
            # Ensure rating is numeric before calculating average
            filtered_df['Rating_numeric'] = pd.to_numeric(filtered_df['Rating'], errors='coerce')
            avg_rating = filtered_df['Rating_numeric'].mean()
            col2.metric("Average Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
        else:
             col2.info("Add a 'Rating' field to your form to see an average rating KPI.")
        
        if not filtered_df.empty:
            latest_submission_date = filtered_df['created_at'].max().strftime("%Y-%m-%d %H:%M")
            col3.metric("Latest Submission", latest_submission_date)
        else:
            col3.metric("Latest Submission", "N/A")

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- Visualizations ---
        st.header("Visual Insights")
        
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
        
        # Let the user select a column to analyze
        # Exclude columns that are not useful for categorical analysis
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
            
            # Pie Chart for distribution
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

            # Bar Chart for exact numbers
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

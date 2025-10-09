import streamlit as st
import pandas as pd

# ------------------------------------------------
# CONFIGURATION - Replace this link with YOUR published CSV link
# ------------------------------------------------
PUBLISHED_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vXXXXXXXXXXXX/pub?output=csv"

def load_google_sheet(url):
    """Fetch Google Sheet published as CSV and return as DataFrame."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        st.stop()

def main():
    st.set_page_config(page_title="Pioneer Google Sheet Dashboard", layout="wide")
    st.title("📊 Pioneer Broadband Google Sheet Dashboard")

    # Load data
    with st.spinner("Loading Google Sheet..."):
        df = load_google_sheet(PUBLISHED_CSV_URL)

    st.success("Data loaded successfully!")

    # Show full data
    st.subheader("📋 Full Data Table")
    st.dataframe(df, use_container_width=True)

    # Filter section
    st.subheader("🔍 Filter Data")
    col = st.selectbox("Select a column to filter", df.columns)
    val = st.text_input("Enter a keyword to search")
    if st.button("Apply Filter"):
        filtered = df[df[col].astype(str).str.contains(val, case=False, na=False)]
        st.write(f"Filtered results ({len(filtered)} rows):")
        st.dataframe(filtered, use_container_width=True)
    else:
        filtered = df

    # Chart section
    st.subheader("📈 Quick Visualization")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        x_axis = st.selectbox("Select X axis", df.columns)
        y_axis = st.selectbox("Select Y axis (numeric)", numeric_cols)
        st.bar_chart(filtered.groupby(x_axis)[y_axis].sum())
    else:
        st.info("No numeric columns available for charting.")

    # Download option
    st.subheader("⬇️ Download Data")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered CSV", csv, "filtered_data.csv", "text/csv")

if __name__ == "__main__":
    main()

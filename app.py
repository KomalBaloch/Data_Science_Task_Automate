import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


# Custom CSS for advanced styling
st.markdown("""
    <style>
   .stApp {
        background-color: #001f3f; /* Light gray background */
    }
    body {
        background-color: ##001f3f;  /* Dark background color */
        color: #f5f5f5;             /* Light text color */
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #00bfae;  /* Custom button color */
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTitle {
        font-family: 'Courier New', Courier, monospace; /* Custom font */
        color: #ff6347; /* Tomato color */
    }
    .stDataFrame {
        border: 1px solid #00bfae;  /* Custom border for dataframes */
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Data Science Task Automation")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Step 2: Data Cleaning
    if st.checkbox("Handle Missing Values"):
        df = df.fillna(df.median(numeric_only=True))
        st.write("Cleaned Dataset:")
        st.dataframe(df)

    # Step 3: Exploratory Data Analysis (EDA)
    st.write("### Exploratory Data Analysis")

    # Boxplot for Outliers
    if st.checkbox("Show Boxplots for Outlier Detection"):
        numeric_cols = df.select_dtypes(include=["number"]).columns
        selected_box_col = st.selectbox("Select a column for Boxplot:", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(data=df, y=selected_box_col, ax=ax)
        st.pyplot(fig)

    # Pairplot
    if st.checkbox("Show Pairplot (Relationships between variables)"):
        st.write("This may take a moment for large datasets.")
        selected_pair_cols = st.multiselect("Select columns for Pairplot:", list(df.columns), default=list(df.columns[:3]))
        if len(selected_pair_cols) > 1:
            pairplot_fig = sns.pairplot(df[selected_pair_cols])
            st.pyplot(pairplot_fig)

    # Feature vs Target (Bar/Scatter Plot)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) > 1:
        st.write("### Target vs Feature Visualization")
        target = st.selectbox("Select Target Variable:", numeric_cols)
        feature = st.selectbox("Select Feature Variable:", [col for col in numeric_cols if col != target])
        chart_type = st.radio("Select Chart Type:", ["Scatter", "Bar"])
        if chart_type == "Scatter":
            fig = px.scatter(df, x=feature, y=target, title=f"{feature} vs {target}")
        else:
            fig = px.bar(df, x=feature, y=target, title=f"{feature} vs {target}")
        st.plotly_chart(fig)

    # Step 4: Train-Test Split
    st.write("### Feature Selection and Model Training")
    features = [col for col in numeric_cols if col != target]
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train Models
    if st.button("Train Models"):
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_pred)
            results[name] = test_score

        st.write("### Model Performance:")
        results_df = pd.DataFrame.from_dict(results, orient="index", columns=["R2 Score"])
        st.write(results_df)

        # Visualize Model Performance
        fig = px.bar(results_df, x=results_df.index, y="R2 Score", title="Model Performance Comparison")
        st.plotly_chart(fig)

        # Select Best Model
        best_model_name = results_df["R2 Score"].idxmax()
        st.success(f"The Best Model is: {best_model_name}")
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        # Feature Importance (if Random Forest)
        if best_model_name == "Random Forest":
            st.write("### Feature Importance (Random Forest)")
            importances = best_model.feature_importances_
            importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
            importance_fig = px.bar(importance_df.sort_values(by="Importance", ascending=False),
                                    x="Feature", y="Importance", title="Feature Importance")
            st.plotly_chart(importance_fig)

        # Predictions vs Actual
        st.write("### Predictions vs Actual")
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
        fig = px.scatter(pred_df, x="Actual", y="Predicted", title="Actual vs Predicted")
        st.plotly_chart(fig)

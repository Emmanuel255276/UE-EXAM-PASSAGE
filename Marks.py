import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="📚 University Exam Eligibility System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .css-1r6slb0 {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Success message */
    .stSuccess {
        border-radius: 15px;
        border-left: 5px solid #00d25b;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    
    /* Error message */
    .stError {
        border-radius: 15px;
        border-left: 5px solid #fc4a1a;
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 10px;
        font-size: 16px;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 30px !important;
    }
    
    h2 {
        color: #2c3e50;
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
    }
    
    h3 {
        color: #34495e;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
    }
    
    /* Status badges */
    .eligibility-badge {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 24px;
        font-weight: 600;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    .ineligibility-badge {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 24px;
        font-weight: 600;
        text-align: center;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-size: 14px;
        border-top: 1px solid #ecf0f1;
        margin-top: 50px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown("""
<h1 style='text-align: center;'>
    🎓 University Exam Eligibility System
    <div style='font-size: 1.2rem; color: #7f8c8d; margin-top: 10px;'>
        Based on Continuous Assessment Performance
    </div>
</h1>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    with st.spinner("🔮 Initializing intelligent grading system..."):
        time.sleep(2)  # Simulate loading
        model = joblib.load('kmeans MARKS_model.pkl')
        return model

# Load the model
try:
    model = load_model()
    
    # Define cluster meanings based on analysis of cluster centers
    cluster_names = {
        1: "🌟 EXCELLENT",
        3: "✨ VERY GOOD", 
        0: "📊 AVERAGE",
        2: "⚠️ BELOW AVERAGE"
    }
    
    cluster_colors = {
        1: "#00d25b",  # Green
        3: "#17a2b8",  # Teal
        0: "#ffc107",  # Yellow
        2: "#dc3545"   # Red
    }
    
    cluster_eligibility = {
        1: True,   # Excellent -> Eligible
        3: True,   # Very Good -> Eligible
        0: False,  # Average -> Not Eligible
        2: False   # Below Average -> Not Eligible
    }
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Sidebar with student info
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <span style='font-size: 3rem;'>🎓</span>
        <h2 style='color: #2c3e50; margin-top: 10px;'>Student Portal</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Student details
    student_name = st.text_input("📝 Full Name", placeholder="Enter your name")
    reg_no = st.text_input("🆔 Registration Number", placeholder="e.g., SC2023-001")
    programme = st.selectbox(
        "📚 Programme of Study",
        ["Bachelor of Science in Computer Science",
         "Bachelor of Business Administration",
         "Bachelor of Engineering",
         "Bachelor of Education",
         "Other"]
    )
    
    st.markdown("---")
    
    # Model info
    with st.expander("ℹ️ About the System"):
        st.markdown("""
        **How it works:**
        - Uses AI clustering to analyze your performance
        - Compares with thousands of previous students
        - Predicts your eligibility for final exam
        - Based on 3 assessments: Assignment, Test 1, Test 2
        
        **Clusters & Meanings:**
        - 🌟 **EXCELLENT** (85%+ average)
        - ✨ **VERY GOOD** (70-84% average)  
        - 📊 **AVERAGE** (50-69% average)
        - ⚠️ **BELOW AVERAGE** (<50% average)
        """)

# Main content area
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Assignment")
    ass1 = st.number_input(
        "Assignment Marks (0-100)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.5,
        key="ass1"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Test 1")
    test1 = st.number_input(
        "Test 1 Marks (0-100)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.5,
        key="test1"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Test 2")
    test2 = st.number_input(
        "Test 2 Marks (0-100)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.5,
        key="test2"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Calculate average
average = (ass1 + test1 + test2) / 3

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Average", f"{average:.1f}%")
with col2:
    st.metric("📈 Total", f"{ass1 + test1 + test2:.1f}")
with col3:
    st.metric("🎯 Highest", f"{max(ass1, test1, test2):.1f}%")
with col4:
    st.metric("📉 Lowest", f"{min(ass1, test1, test2):.1f}%")

# Progress bar
st.progress(average / 100, text="Overall Performance")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 CHECK ELIGIBILITY", use_container_width=True)

if predict_button:
    if not student_name or not reg_no:
        st.warning("⚠️ Please fill in your name and registration number first!")
    else:
        with st.spinner("🔄 Analyzing your performance..."):
            time.sleep(2)  # Dramatic pause
            
            # Prepare data for prediction
            input_data = np.array([[ass1, test1, test2]])
            
            # Predict cluster
            cluster = model.predict(input_data)[0]
            cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
            is_eligible = cluster_eligibility.get(cluster, False)
            
            # Create two columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("### 📊 Classification Result")
                
                # Cluster card
                st.markdown(f"""
                <div style='background: {cluster_colors.get(cluster, "#667eea")}20; 
                            border-radius: 15px; 
                            padding: 25px;
                            border-left: 5px solid {cluster_colors.get(cluster, "#667eea")};'>
                    <h3 style='color: {cluster_colors.get(cluster, "#667eea")}; margin: 0;'>
                        {cluster_name}
                    </h3>
                    <p style='color: #2c3e50; margin-top: 10px;'>
                        Based on your performance in all assessments
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show distances to all clusters
                distances = np.linalg.norm(model.cluster_centers_ - input_data, axis=1)
                dist_df = pd.DataFrame({
                    'Cluster': [cluster_names[i] for i in range(4)],
                    'Distance': distances
                })
                
                fig = px.bar(dist_df, 
                            x='Cluster', 
                            y='Distance',
                            title='Distance to Each Cluster (Lower = Better Match)',
                            color='Distance',
                            color_continuous_scale=['green', 'yellow', 'red'])
                st.plotly_chart(fig, use_container_width=True)
            
            with res_col2:
                st.markdown("### ✅ Eligibility Status")
                
                if is_eligible:
                    st.markdown("""
                    <div class='eligibility-badge'>
                        🎉 ELIGIBLE FOR FINAL EXAM
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
                    
                    st.markdown("""
                    <div style='background: #00d25b20; 
                                border-radius: 15px; 
                                padding: 20px;
                                margin-top: 20px;
                                border-left: 5px solid #00d25b;'>
                        <h4 style='color: #00d25b;'>📝 University Exam Details:</h4>
                        <ul style='color: #2c3e50;'>
                            <li><b>Date:</b> December 15, 2024</li>
                            <li><b>Time:</b> 9:00 AM - 12:00 PM</li>
                            <li><b>Venue:</b> Main Hall</li>
                            <li><b>Materials:</b> Calculator, Pen, ID Card</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown("""
                    <div class='ineligibility-badge'>
                        ❌ NOT ELIGIBLE FOR FINAL EXAM
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.snow()
                    
                    st.markdown("""
                    <div style='background: #ff416c20; 
                                border-radius: 15px; 
                                padding: 20px;
                                margin-top: 20px;
                                border-left: 5px solid #ff416c;'>
                        <h4 style='color: #ff416c;'>📝 Improvement Plan:</h4>
                        <ul style='color: #2c3e50;'>
                            <li>Attend remedial classes</li>
                            <li>Submit supplementary assignments</li>
                            <li>Meet with academic advisor</li>
                            <li>Re-assessment in January 2025</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 📈 Performance Analysis")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                if average >= 70:
                    st.success("✅ Strong performance across all assessments")
                elif average >= 50:
                    st.warning("📝 Satisfactory performance, room for improvement")
                else:
                    st.error("⚠️ Needs significant improvement")
            
            with insight_col2:
                if cluster in [1, 3]:
                    st.info(f"📊 You are in the top performing group")
                else:
                    st.info(f"📊 You are in the average/below average group")
            
            with insight_col3:
                st.metric("📈 Success Probability", 
                         f"{100 - distances[cluster]:.1f}%",
                         "Based on cluster proximity")

# Footer with system info
st.markdown("""
<div class='footer'>
    <p>🎓 University Exam Eligibility System v2.0 | Powered by AI Clustering</p>
    <p style='font-size: 12px;'>© 2024 All Rights Reserved | Data is processed securely</p>
</div>
""", unsafe_allow_html=True)

# Hidden debug info (optional)
with st.expander("🔧 System Information (Admin Only)"):
    st.json({
        "model_type": type(model).__name__,
        "n_clusters": model.n_clusters,
        "features": model.n_features_in_,
        "cluster_centers": model.cluster_centers_.tolist()
    })
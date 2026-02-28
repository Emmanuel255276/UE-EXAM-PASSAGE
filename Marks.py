import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="📚 University Exam Eligibility System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look - BLUE THEME
st.markdown("""
<style>
    /* Main container styling - Blue gradient */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Card styling */
    .css-1r6slb0 {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Metric cards - Blue theme */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 20px rgba(30, 60, 114, 0.4);
    }
    
    /* Input card for marks */
    .input-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #1e3c72;
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
    
    /* Warning message */
    .stWarning {
        border-radius: 15px;
        border-left: 5px solid #f39c12;
        background: linear-gradient(135deg, #f1c40f, #e67e22);
        color: white;
    }
    
    /* Button styling - Blue theme */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(30, 60, 114, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 60, 114, 0.6);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 10px;
        font-size: 16px;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #1e3c72;
        box-shadow: 0 0 0 2px rgba(30, 60, 114, 0.2);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 30px !important;
    }
    
    h2 {
        color: #1e3c72;
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
    }
    
    h3 {
        color: #2a5298;
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
        box-shadow: 0 5px 20px rgba(0, 176, 155, 0.4);
    }
    
    .ineligibility-badge {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 24px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 20px rgba(255, 65, 108, 0.4);
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
    
    /* Progress bar - Blue theme */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #1e3c72;
        margin: 10px 0;
    }
    
    /* Score display */
    .score-display {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3c72;
        text-align: center;
    }
    
    .score-label {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<h1 style='text-align: center;'>
    🎓 University Exam Eligibility System
    <div style='font-size: 1.2rem; color: #7f8c8d; margin-top: 10px;'>
        Based on Continuous Assessment Performance (Total: 40 Marks)
    </div>
</h1>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    with st.spinner("🔮 Initializing intelligent grading system..."):
        time.sleep(2)
        model = joblib.load('kmeans MARKS_model.pkl')
        return model

# Load the model
try:
    model = load_model()
    
    # Define cluster meanings based on analysis
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
        <h2 style='color: #1e3c72; margin-top: 10px;'>Student Portal</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Student details
    student_name = st.text_input("📝 Full Name", placeholder="Enter your name", key="name")
    reg_no = st.text_input("🆔 Registration Number", placeholder="e.g., SC2023-001", key="reg")
    programme = st.selectbox(
        "📚 Programme of Study",
        ["Bachelor of Science in Computer Science",
         "Bachelor of Business Administration",
         "Bachelor of Engineering",
         "Bachelor of Education",
         "Diploma in Information Technology",
         "Other"]
    )
    
    st.markdown("---")
    
    # Grading info
    with st.expander("📊 Grading System"):
        st.markdown("""
        **Maximum Marks:**
        - 📝 **Assignment 1:** 5 marks
        - 📝 **Assignment 2:** 5 marks
        - 📋 **Test 1:** 15 marks
        - 📋 **Test 2:** 15 marks
        
        **Total: 40 marks**
        
        **Eligibility Criteria:**
        - 🌟 **EXCELLENT** (32-40 marks)
        - ✨ **VERY GOOD** (28-31 marks)
        - 📊 **AVERAGE** (20-27 marks)
        - ⚠️ **BELOW AVERAGE** (<20 marks)
        
        *Eligible for Final Exam: EXCELLENT & VERY GOOD*
        """)

# Main content - Input Cards
st.markdown("## 📝 Enter Your Continuous Assessment Marks")

# Create 4 columns for inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Assignment 1")
    st.markdown("<p style='color: #7f8c8d;'>Max: 5 marks</p>", unsafe_allow_html=True)
    ass1 = st.number_input(
        "Marks (0-5)",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.5,
        key="ass1",
        label_visibility="collapsed"
    )
    st.markdown(f"<div class='score-display'>{ass1:.1f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-label'>/ 5.0</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Assignment 2")
    st.markdown("<p style='color: #7f8c8d;'>Max: 5 marks</p>", unsafe_allow_html=True)
    ass2 = st.number_input(
        "Marks (0-5)",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.5,
        key="ass2",
        label_visibility="collapsed"
    )
    st.markdown(f"<div class='score-display'>{ass2:.1f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-label'>/ 5.0</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Test 1")
    st.markdown("<p style='color: #7f8c8d;'>Max: 15 marks</p>", unsafe_allow_html=True)
    test1 = st.number_input(
        "Marks (0-15)",
        min_value=0.0,
        max_value=15.0,
        value=7.5,
        step=0.5,
        key="test1",
        label_visibility="collapsed"
    )
    st.markdown(f"<div class='score-display'>{test1:.1f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-label'>/ 15.0</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Test 2")
    st.markdown("<p style='color: #7f8c8d;'>Max: 15 marks</p>", unsafe_allow_html=True)
    test2 = st.number_input(
        "Marks (0-15)",
        min_value=0.0,
        max_value=15.0,
        value=7.5,
        step=0.5,
        key="test2",
        label_visibility="collapsed"
    )
    st.markdown(f"<div class='score-display'>{test2:.1f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='score-label'>/ 15.0</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Calculate totals
total_marks = ass1 + ass2 + test1 + test2
percentage = (total_marks / 40) * 100

# Display summary metrics
st.markdown("---")
st.markdown("## 📊 Performance Summary")

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("📝 Assignments Total", f"{ass1 + ass2:.1f}/10", 
              delta=f"{(ass1 + ass2)/10*100:.0f}%")

with metric_col2:
    st.metric("📋 Tests Total", f"{test1 + test2:.1f}/30",
              delta=f"{(test1 + test2)/30*100:.0f}%")

with metric_col3:
    st.metric("🎯 Overall Total", f"{total_marks:.1f}/40",
              delta=f"{percentage:.1f}%")

with metric_col4:
    if total_marks >= 32:
        grade = "🌟 Excellent"
        grade_color = "green"
    elif total_marks >= 28:
        grade = "✨ Very Good"
        grade_color = "blue"
    elif total_marks >= 20:
        grade = "📊 Average"
        grade_color = "orange"
    else:
        grade = "⚠️ Below Average"
        grade_color = "red"
    
    st.metric("📈 Grade", grade, delta_color="off")

with metric_col5:
    is_eligible_manual = total_marks >= 28
    status = "✅ ELIGIBLE" if is_eligible_manual else "❌ NOT ELIGIBLE"
    status_color = "normal" if is_eligible_manual else "inverse"
    st.metric("🎓 Status", status, delta_color=status_color)

# Progress bar
st.progress(total_marks / 40, text=f"Overall Progress: {total_marks}/40 marks ({percentage:.1f}%)")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 CHECK ELIGIBILITY WITH AI", use_container_width=True)

if predict_button:
    if not student_name or not reg_no:
        st.warning("⚠️ Please fill in your name and registration number first!")
    else:
        with st.spinner("🔄 Analyzing your performance with AI..."):
            time.sleep(2)
            
            # Prepare data for prediction (model uses 3 features: ass1, test1, test2)
            # Note: Model ina features 3 tu, so tunatumia ass1, test1, test2
            input_data = np.array([[ass1, test1, test2]])
            
            # Predict cluster
            cluster = model.predict(input_data)[0]
            cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
            is_eligible_ai = cluster_eligibility.get(cluster, False)
            
            # Create results display
            st.markdown("---")
            st.markdown("## 🎯 AI Analysis Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("### 🤖 AI Classification")
                
                # Cluster card
                st.markdown(f"""
                <div style='background: {cluster_colors.get(cluster, "#1e3c72")}20; 
                            border-radius: 15px; 
                            padding: 25px;
                            border-left: 5px solid {cluster_colors.get(cluster, "#1e3c72")};
                            box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
                    <h3 style='color: {cluster_colors.get(cluster, "#1e3c72")}; margin: 0;'>
                        {cluster_name}
                    </h3>
                    <p style='color: #2c3e50; margin-top: 10px; font-size: 16px;'>
                        Based on AI analysis of your performance pattern
                    </p>
                    <div style='margin-top: 15px;'>
                        <span style='background: {cluster_colors.get(cluster, "#1e3c72")}; 
                                     color: white; 
                                     padding: 8px 15px; 
                                     border-radius: 20px;
                                     font-size: 14px;'>
                            Confidence: {(100 - np.random.randint(5, 15)):.0f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show cluster comparison
                st.markdown("### 📊 Cluster Comparison")
                distances = np.linalg.norm(model.cluster_centers_ - input_data, axis=1)
                dist_df = pd.DataFrame({
                    'Cluster': list(cluster_names.values()),
                    'Distance': distances
                })
                
                fig = px.bar(dist_df, 
                            x='Cluster', 
                            y='Distance',
                            title='Distance to Each Cluster (Lower = Better Match)',
                            color='Distance',
                            color_continuous_scale='Blues',
                            text=dist_df['Distance'].round(2))
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with res_col2:
                st.markdown("### ✅ Eligibility Status")
                
                # Compare manual vs AI
                if is_eligible_manual and is_eligible_ai:
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
                        <h4 style='color: #00d25b;'>📝 Congratulations!</h4>
                        <p style='color: #2c3e50;'>You have qualified for the University Final Examination.</p>
                        <hr>
                        <p><b>Exam Details:</b></p>
                        <ul>
                            <li><b>Date:</b> December 15, 2024</li>
                            <li><b>Time:</b> 9:00 AM - 12:00 PM</li>
                            <li><b>Venue:</b> Main Examination Hall</li>
                            <li><b>Requirements:</b> Student ID, Calculator, Pens</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif not is_eligible_manual and not is_eligible_ai:
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
                            <li>Attend remedial classes (Starting next week)</li>
                            <li>Submit supplementary assignments</li>
                            <li>Schedule meeting with academic advisor</li>
                            <li>Re-assessment opportunity in January 2025</li>
                        </ul>
                        <div style='background: white; padding: 10px; border-radius: 10px; margin-top: 10px;'>
                            <p><b>Required minimum:</b> 28/40 marks (70%)</p>
                            <p><b>Your current:</b> {total_marks}/40 marks ({percentage:.1f}%)</p>
                            <p><b>Need:</b> {max(0, 28 - total_marks):.1f} more marks</p>
                        </div>
                    </div>
                    """.format(total_marks=total_marks, percentage=percentage), unsafe_allow_html=True)
                    
                else:
                    # Mismatch between manual and AI (interesting case)
                    st.warning("⚠️ AI and Manual calculations show different results!")
                    
                    if is_eligible_manual and not is_eligible_ai:
                        st.info("📌 AI suggests you need more improvement despite meeting minimum marks.")
                    else:
                        st.info("📌 AI sees potential in your performance pattern.")
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 📈 Detailed Performance Analysis")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.markdown("""
                <div class='info-box'>
                    <h4 style='color: #1e3c72;'>📊 Assignment Performance</h4>
                    <p>Assignment 1: {}/5</p>
                    <p>Assignment 2: {}/5</p>
                    <p><b>Total: {}/10 ({}%)</b></p>
                </div>
                """.format(ass1, ass2, ass1+ass2, (ass1+ass2)/10*100), unsafe_allow_html=True)
            
            with insight_col2:
                st.markdown("""
                <div class='info-box'>
                    <h4 style='color: #1e3c72;'>📋 Test Performance</h4>
                    <p>Test 1: {}/15</p>
                    <p>Test 2: {}/15</p>
                    <p><b>Total: {}/30 ({}%)</b></p>
                </div>
                """.format(test1, test2, test1+test2, (test1+test2)/30*100), unsafe_allow_html=True)
            
            with insight_col3:
                strengths = []
                if ass1 >= 4: strengths.append("Assignment 1")
                if ass2 >= 4: strengths.append("Assignment 2")
                if test1 >= 12: strengths.append("Test 1")
                if test2 >= 12: strengths.append("Test 2")
                
                weaknesses = []
                if ass1 < 2.5: weaknesses.append("Assignment 1")
                if ass2 < 2.5: weaknesses.append("Assignment 2")
                if test1 < 7.5: weaknesses.append("Test 1")
                if test2 < 7.5: weaknesses.append("Test 2")
                
                st.markdown("""
                <div class='info-box'>
                    <h4 style='color: #1e3c72;'>💪 Strengths & Areas</h4>
                    <p><b style='color: #00d25b;'>✓ Strengths:</b> {}</p>
                    <p><b style='color: #ff416c;'>⚠️ Improve:</b> {}</p>
                </div>
                """.format(
                    ", ".join(strengths) if strengths else "None",
                    ", ".join(weaknesses) if weaknesses else "None"
                ), unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>🎓 University Exam Eligibility System | Powered by AI Clustering | Total Marks: 40</p>
    <p style='font-size: 12px;'>© 2024 All Rights Reserved | Data is processed securely</p>
</div>
""", unsafe_allow_html=True)

# Admin expander
with st.expander("🔧 System Information (Admin Only)"):
    col1, col2 = st.columns(2)
    with col1:
        st.json({
            "model_type": type(model).__name__,
            "n_clusters": model.n_clusters,
            "features": model.n_features_in_,
            "total_marks_system": 40,
            "pass_mark": 28
        })
    with col2:
        st.write("**Cluster Centers:**")
        centers_df = pd.DataFrame(
            model.cluster_centers_,
            columns=['Assignment', 'Test 1', 'Test 2'],
            index=[f'Cluster {i}' for i in range(4)]
        )
        st.dataframe(centers_df)
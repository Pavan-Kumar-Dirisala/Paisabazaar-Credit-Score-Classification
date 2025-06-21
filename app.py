import streamlit as st
import pandas as pd
import joblib
from catboost import Pool
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .card{font-size: 3rem;
        
            }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    
    .prediction-good {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .prediction-standard {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .prediction-poor {
        background: linear-gradient(135deg, #F44336, #D32F2F);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the model (with error handling)
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_catboost_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'best_catboost_model.pkl' is in the correct directory.")
        return None

model = load_model()

# Categorical feature list
cat_features = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount',
                'Payment_Behaviour', 'Type_of_Loan']

# Header
st.markdown('<h1 class="main-header">Credit Score Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get instant insights into your creditworthiness with our AI-powered predictor</p>', unsafe_allow_html=True)

# Sidebar for quick info
with st.sidebar:
    st.markdown("### 📊 About Credit Scores")
    st.info("""
    **Good Credit (750+)**
    ✅ Best loan rates \n
    ✅ Easy approvals \n
    ✅ Premium credit cards \n
    
    **Standard Credit (650-749)**
    ⚠️ Moderate rates \n
    ⚠️ Standard approvals \n
    
    **Poor Credit (<650)**
    ❌ High interest rates \n
    ❌ Limited options \n
    """)
    
    st.markdown("### 🎯 Tips for Better Credit")
    st.success("""
    • Pay bills on time \n
    • Keep credit utilization low \n
    • Maintain old accounts \n
    • Monitor credit report \n
    • Diversify credit types \n
    """)

# Main content in tabs
tab1, tab2, tab3= st.tabs(["🔍 Predict Credit Score", "📈 Factors Affecting Credit Score", "💡 Recommendations"])

with tab1:
    if model is None:
        st.stop()
    
    # Personal Information Section
    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("📝 Full Name", placeholder="Enter your full name")
        age = st.slider("🎂 Age", min_value=18, max_value=100, value=30)
    
    with col2:
        month = st.selectbox("📅 Month", range(1, 13), index=0)
        occupation = st.selectbox("💼 Occupation", [
            "Scientist", "Teacher", "Engineer", "Manager", "Doctor", 
            "Lawyer", "Architect", "Developer", "Accountant", "Other"
        ])
    
    with col3:
        annual_income = st.number_input("💰 Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
        monthly_inhand_salary = st.number_input("💵 Monthly Salary ($)", min_value=0.0, value=4000.0, step=100.0)

    st.divider()

    # Banking Information Section
    st.markdown("### 🏦 Banking & Credit Information")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        num_bank_accounts = st.slider("🏦 Bank Accounts", min_value=0, max_value=20, value=3)
        num_credit_card = st.slider("💳 Credit Cards", min_value=0, max_value=20, value=2)
        interest_rate = st.slider("📈 Interest Rate (%)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
    
    with col5:
        num_of_loan = st.slider("🏠 Number of Loans", min_value=0, max_value=20, value=2)
        type_of_loan = st.multiselect("🏷️ Loan Types", [
            "Auto Loan", "Personal Loan", "Home Loan", "Student Loan", 
            "Credit Card Loan", "Business Loan"
        ], default=["Auto Loan", "Personal Loan"])
        
        # Convert multiselect to string
        type_of_loan_str = ", ".join(type_of_loan) if type_of_loan else "No Loan"
    
    with col6:
        credit_mix = st.selectbox("🎯 Credit Mix", ["Good", "Standard", "Poor"], index=0)
        payment_of_min_amount = st.selectbox("💸 Pay Minimum Amount", ["Yes", "No"], index=0)
        payment_behaviour = st.selectbox("📊 Payment Behaviour", [
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments", 
            "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "Low_spent_Small_value_payments"
        ])

    st.divider()

    # Financial Details Section
    st.markdown("### 💹 Financial Details")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        delay_from_due_date = st.slider("⏰ Days Late on Payments", min_value=0, max_value=60, value=5)
        num_of_delayed_payment = st.slider("⚠️ Delayed Payments Count", min_value=0, max_value=50, value=3)
        changed_credit_limit = st.number_input("📊 Credit Limit Changes", value=500.0, step=100.0)
    
    with col8:
        num_credit_inquiries = st.slider("🔍 Credit Inquiries", min_value=0, max_value=20, value=2)
        outstanding_debt = st.number_input("💳 Outstanding Debt ($)", min_value=0.0, value=1500.0, step=100.0)
        credit_utilization_ratio = st.slider("📈 Credit Utilization (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    
    with col9:
        credit_history_age = st.slider("📅 Credit History (months)", min_value=0, max_value=600, value=200)
        total_emi_per_month = st.number_input("💰 Monthly EMI ($)", min_value=0.0, value=200.0, step=50.0)
        amount_invested_monthly = st.number_input("📈 Monthly Investment ($)", min_value=0.0, value=100.0, step=25.0)
        monthly_balance = st.number_input("💵 Monthly Balance ($)", value=800.0, step=100.0)

    st.divider()

    # Prediction Section
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        if st.button("🔮 Predict My Credit Score", type="primary"):
            if not name:
                st.warning("⚠️ Please enter your name to continue.")
            else:
                # Show loading animation
                with st.spinner('🔄 Analyzing your credit profile...'):
                    time.sleep(2)  # Simulate processing time
                    
                    # Prepare input DataFrame
                    data = pd.DataFrame([{
                        'Month': month,
                        'Age': age,
                        'Occupation': occupation,
                        'Annual_Income': annual_income,
                        'Monthly_Inhand_Salary': monthly_inhand_salary,
                        'Num_Bank_Accounts': num_bank_accounts,
                        'Num_Credit_Card': num_credit_card,
                        'Interest_Rate': interest_rate,
                        'Num_of_Loan': num_of_loan,
                        'Type_of_Loan': type_of_loan_str,
                        'Delay_from_due_date': delay_from_due_date,
                        'Num_of_Delayed_Payment': num_of_delayed_payment,
                        'Changed_Credit_Limit': changed_credit_limit,
                        'Num_Credit_Inquiries': num_credit_inquiries,
                        'Credit_Mix': credit_mix,
                        'Outstanding_Debt': outstanding_debt,
                        'Credit_Utilization_Ratio': credit_utilization_ratio,
                        'Credit_History_Age': credit_history_age,
                        'Payment_of_Min_Amount': payment_of_min_amount,
                        'Total_EMI_per_month': total_emi_per_month,
                        'Amount_invested_monthly': amount_invested_monthly,
                        'Payment_Behaviour': payment_behaviour,
                        'Monthly_Balance': monthly_balance
                    }])
                    
                    # Make prediction
                    try:
                        pool = Pool(data, cat_features=cat_features)
                        prediction = model.predict(pool)
                        result = prediction[0]
                        
                        # Display result with appropriate styling
                        if result == "Good":
                            st.markdown(f'<div class="prediction-good">🎉 Congratulations {name}!<br>Your Credit Score is: <strong>{result}</strong></div>', unsafe_allow_html=True)
                            st.balloons()
                        elif result == "Standard":
                            st.markdown(f'<div class="prediction-standard">👍 Hi {name}!<br>Your Credit Score is: <strong>{result}</strong></div>', unsafe_allow_html=True)
                            st.snow()
                        else:
                            st.markdown(f'<div class="prediction-poor">⚡ Hi {name}!<br>Your Credit Score is: <strong>{result}</strong><br>There\'s room for improvement!</div>', unsafe_allow_html=True)
                        
                        # Store result in session state for other tabs
                        st.session_state.prediction = result
                        st.session_state.user_data = data
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred during prediction: {str(e)}")

with tab2:
    # Credit factors impact
    # st.markdown("### 🎯 Factors Affecting Credit Score")
    
    factors = {
        'Type_of_Loan': 9.6518,         
      'Occupation':8.4725,        
      'Month ':6.2643,    
      'Outstanding_Debt':5.8894,   
      'Credit_Mix': 5.3340,  
      'Changed_Credit_Limit':4.9477 , 
      'Delay_from_due_date':4.8124,
      'Credit_History_Age' :4.6812,
      'Age':4.6376,
     'Interest_Rate':4.5192
    }
    
    fig_bar = go.Figure([go.Bar(
        x=list(factors.keys()),
        y=list(factors.values()),
        marker_color=[
    '#3a1c71',  # darkest
    '#4a00e0',
    '#6a11cb',
    '#8e2de2',
    '#764ba2',
    '#764ba2',
    '#2575fc',
    '#667eea',
    '#667eea',
    '#667eea'   # lightest
]
)])
    
    fig_bar.update_layout(
        title="Credit Score Impact Factors",
        xaxis_title="Factors",
        yaxis_title="Impact Score",
        height=400
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.markdown("### 💡 Personalized Recommendations")
    
    if 'prediction' in st.session_state:
        result = st.session_state.prediction
        
        if result == "Good":
            st.success("🎉 **Excellent Credit Score!** Keep up the great work!")
            st.markdown("""
            **To maintain your excellent score:**
            - Continue paying all bills on time
            - Keep credit utilization below 30%
            - Don't close old credit accounts
            - Monitor your credit report regularly
            - Consider becoming an authorized user on family accounts
            """)
            
        elif result == "Standard":
            st.warning("👍 **Good Credit Score!** Here's how to improve:")
            st.markdown("""
            **Steps to reach 'Good' status:**
            - Set up automatic payments to avoid late fees
            - Pay down existing debt to improve utilization ratio
            - Don't apply for new credit cards frequently
            - Keep old accounts open to maintain credit history
            - Consider a secured credit card if needed
            """)
            
        else:
            st.error("⚡ **Credit Score Needs Improvement!** Don't worry, here's your action plan:")
            st.markdown("""
            **Priority actions to improve your score:**
            1. **Pay all bills on time** - This is the most important factor
            2. **Reduce credit card balances** - Aim for under 30% utilization
            3. **Don't close old accounts** - They help your credit history length
            4. **Check your credit report** - Look for errors and dispute them
            5. **Consider a secured credit card** - To build positive payment history
            6. **Pay more than the minimum** - Reduces debt faster
            """)
    else:
        st.info("🔍 **Get your credit score prediction first to see personalized recommendations!**")
        
        st.markdown("""
        ### General Credit Improvement Tips:
        
        **🎯 Quick Wins:**
        - Set up automatic bill payments
        - Pay credit cards twice per month
        - Request credit limit increases
        - Become an authorized user on family accounts
        
        **📈 Long-term Strategies:**
        - Diversify your credit mix (cards, loans, mortgage)
        - Keep old accounts open
        - Monitor credit reports monthly
        - Avoid hard inquiries when possible
        
        **🚨 Red Flags to Avoid:**
        - Missing payments (even by a day)
        - Maxing out credit cards
        - Closing old accounts
        - Co-signing loans carelessly
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💳 <strong>Credit Score Predictor</strong> | Built with ❤️ using Streamlit & CatBoost</p>
    <p><small>⚠️ This is a predictive tool and should not be considered as financial advice. 
    Consult with financial professionals for important credit decisions.</small></p>
</div>
""", unsafe_allow_html=True)
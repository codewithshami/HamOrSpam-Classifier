import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from wordcloud import WordCloud
import pickle
import io
from collections import Counter

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('mail_data.csv')
    df = df.where(pd.notnull(df), '')  
    df.loc[df['Category'] == 'spam', 'Category'] = 0
    df.loc[df['Category'] == 'ham', 'Category'] = 1
    X = df['Message']
    Y = df['Category'].astype(int)
    return train_test_split(X, Y, test_size=0.2, random_state=3), df

# Train model
@st.cache_resource
def train_model(X_train, Y_train):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, 
                                ngram_range=(1, 2), max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_bal, Y_bal = smote.fit_resample(X_train_vec, Y_train)
    model = LogisticRegression()
    model.fit(X_bal, Y_bal)
    return model, vectorizer

# Generate word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='black',
                          colormap='viridis').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20, pad=20)
    plt.axis('off')
    return plt

# App UI
def main():
    st.set_page_config(page_title="Spam Shield Pro", layout="wide", page_icon="🛡️")

    # Custom dark theme styling
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                color: white;
            }
            .stTextArea textarea {
                background-color: #2c3e50;
                color: white;
                border-radius: 10px;
            }
            .stButton>button {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #2ecc71;
                transform: scale(1.05);
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #0f2027;
                color: white;
                text-align: center;
                padding: 10px;
                z-index: 1000;
            }
            .metric-card {
                background: rgba(39, 174, 96, 0.2);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
                border-left: 5px solid #27ae60;
            }
            .spam-card {
                background: rgba(231, 76, 60, 0.2);
                border-left: 5px solid #e74c3c;
            }
            .ham-card {
                background: rgba(46, 204, 113, 0.2);
                border-left: 5px solid #2ecc71;
            }
            .stProgress > div > div > div {
                background-color: #27ae60;
            }
            .st-b7 {
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

    (X_train, X_test, Y_train, Y_test), raw_df = load_data()
    model, vectorizer = train_model(X_train, Y_train)

    st.title("🛡️ Spam Shield - AI Powered Email Protection")
    st.markdown("""
    **AI-powered spam detection with comprehensive analytics and real-time protection.**
    Keep your inbox clean and secure with our state-of-the-art classification system.
    """)

    # Sidebar with additional options
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold (%)", 50, 99, 85, 
                                       help="Adjust the minimum confidence level for predictions")
        
        st.header("Model Info")
        st.markdown("""
        - **Algorithm**: Logistic Regression
        - **Features**: TF-IDF with N-grams
        - **Balancing**: SMOTE
        - **Version**: 2.1.0
        """)
        
        st.header("Export Model")
        if st.button("Download Model"):
            model_bytes = io.BytesIO()
            pickle.dump(model, model_bytes)
            st.download_button(
                label="Confirm Download",
                data=model_bytes.getvalue(),
                file_name="spam_shield_model.pkl",
                mime="application/octet-stream"
            )
        
        st.header("Support")
        st.markdown("""
        Having issues or suggestions?
        [Contact Support](mailto:support@spamshield.com)
        """)

    # Instructions expander
    with st.expander("ℹ️ How to use this app", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📌 Instructions:**
            1. Type or paste your email/message in the text box
            2. Click the 'Predict' button to analyze
            3. View detailed prediction results
            4. Explore comprehensive model analytics
            
            **🔍 For best results:**
            - Use complete messages (not single words)
            - English works best
            - Check the confidence score
            """)
        with col2:
            st.markdown("""
            **💡 Try these examples:**
            - Normal: "Hi, let's meet tomorrow at 3pm"
            - Spam: "WINNER! Claim your $1000 prize now!"
            - Phishing: "Your account needs verification, click here"
            """)

    # Main input area with animation
    with st.container():
        st.subheader("🔍 Message Analyzer")
        user_input = st.text_area("Enter your message below", height=200, 
                                placeholder="Paste your email or message here...", 
                                key="main_input")
        
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            predict_btn = st.button("🚀 Analyze Message", key="predict_button", 
                                  help="Click to analyze the message")
        with col2:
            sample_btn = st.button("🎲 Load Random Sample", 
                                 help="Load a random message from our dataset")
        
        if sample_btn:
            try:
                sample = raw_df.sample(1).iloc[0]
                user_input = sample['Message']
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")
        if predict_btn:
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                with st.spinner('Analyzing message...'):
                    time.sleep(0.5)  # Simulate processing time
                    input_vec = vectorizer.transform([user_input])
                    prediction = model.predict(input_vec)[0]
                    proba = model.predict_proba(input_vec)[0]
                    confidence = max(proba)
                    
                    # Display results with animation
                    if confidence < confidence_threshold/100:
                        st.warning(f"⚠️ Low Confidence Prediction ({confidence*100:.1f}%)")
                    
                    cols = st.columns(2)
                    if prediction == 1:
                        with cols[0]:
                            st.markdown(f"""
                            <div class="metric-card ham-card">
                                <h3 style="color:#2ecc71">✅ HAM (Legitimate)</h3>
                                <p>Confidence: <strong>{proba[1]*100:.1f}%</strong></p>
                                <p>This message appears to be safe and legitimate.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        with cols[0]:
                            st.markdown(f"""
                            <div class="metric-card spam-card">
                                <h3 style="color:#e74c3c">❌ SPAM Detected</h3>
                                <p>Confidence: <strong>{proba[0]*100:.1f}%</strong></p>
                                <p>Warning: This message exhibits spam characteristics.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown("**Probability Distribution**")
                        prob_df = pd.DataFrame({
                            'Category': ['Spam', 'Ham'],
                            'Probability': [proba[0], proba[1]]
                        })
                        fig, ax = plt.subplots()
                        ax.barh(prob_df['Category'], prob_df['Probability'], 
                               color=['#e74c3c', '#2ecc71'])
                        ax.set_xlim(0, 1)
                        ax.set_title('Prediction Confidence')
                        st.pyplot(fig)
                    
                    # Word cloud visualization
                    st.subheader("🔠 Keywords Analysis")
                    tab1, tab2 = st.tabs(["Spam Indicators", "Ham Indicators"])
                    
                    with tab1:
                        if prediction == 0:
                            st.pyplot(generate_wordcloud(user_input, "Spam Keywords in Your Message"))
                        else:
                            st.info("No significant spam keywords detected in this message.")
                    
                    with tab2:
                        if prediction == 1:
                            st.pyplot(generate_wordcloud(user_input, "Legitimate Content Keywords"))
                        else:
                            st.info("Few legitimate content keywords found in this message.")

    # Advanced analytics section
    with st.expander("📈 Advanced Analytics", expanded=False):
        st.subheader("Model Performance Metrics")
        X_test_vec = vectorizer.transform(X_test)
        preds = model.predict(X_test_vec)
        pred_probs = model.predict_proba(X_test_vec)[:, 1]
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(Y_test, pred_probs)
        roc_auc = auc(fpr, tpr)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Plot
        ax1.plot(fpr, tpr, color='#27ae60', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        
        # Confusion Matrix
        cm = confusion_matrix(Y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax2, 
                   xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('Confusion Matrix')
        
        st.pyplot(fig)
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(Y_test, preds)*100:.2f}%")
        with col2:
            st.metric("Precision", f"{precision_score(Y_test, preds)*100:.2f}%")
        with col3:
            st.metric("Recall", f"{recall_score(Y_test, preds)*100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{f1_score(Y_test, preds)*100:.2f}%")
        
        # Feature importance
        st.subheader("Top Predictive Features")
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        top_spam = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:10]
        top_ham = sorted(zip(feature_names, coefs), key=lambda x: -x[1])[:10]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Spam Indicators**")
            spam_df = pd.DataFrame(top_spam, columns=['Feature', 'Coefficient'])
            st.dataframe(spam_df.style.background_gradient(cmap='Reds'))
        
        with col2:
            st.markdown("**Top Ham Indicators**")
            ham_df = pd.DataFrame(top_ham, columns=['Feature', 'Coefficient'])
            st.dataframe(ham_df.style.background_gradient(cmap='Greens'))

    # Data exploration section
    with st.expander("🔍 Data Exploration", expanded=False):
        st.subheader("Dataset Insights")
        
        tab1, tab2, tab3 = st.tabs(["Data Samples", "Statistics", "Word Clouds"])
        
        with tab1:
            st.write("**Sample Messages from Dataset**")
            st.dataframe(raw_df.head(10))
            
            st.write("**Random Samples**")
            if st.button("Show Random Spam"):
                st.write(raw_df[raw_df['Category'] == 0].sample(3))
            if st.button("Show Random Ham"):
                st.write(raw_df[raw_df['Category'] == 1].sample(3))
        
        with tab2:
            st.write("**Class Distribution**")
            class_dist = raw_df['Category'].value_counts().reset_index()
            class_dist.columns = ['Category', 'Count']
            class_dist['Category'] = class_dist['Category'].map({0: 'Spam', 1: 'Ham'})
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(class_dist.set_index('Category'))
            with col2:
                st.write(class_dist)
            
            st.write("**Message Length Analysis**")
            raw_df['Length'] = raw_df['Message'].apply(len)
            fig, ax = plt.subplots()
            sns.boxplot(x='Category', y='Length', data=raw_df, ax=ax)
            ax.set_xticklabels(['Spam', 'Ham'])
            st.pyplot(fig)
        
        with tab3:
            st.write("**Word Frequency Visualization**")
            spam_words = ' '.join(raw_df[raw_df['Category'] == 0]['Message'])
            ham_words = ' '.join(raw_df[raw_df['Category'] == 1]['Message'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(generate_wordcloud(spam_words, "Common Spam Words"))
            with col2:
                st.pyplot(generate_wordcloud(ham_words, "Common Ham Words"))

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed with ❤️ by Mohd Shami | Spam Shield Pro v2.1 | © 2025 All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
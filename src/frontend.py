import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="SageMaker Doc Assistant", page_icon="🤖")

st.title("SageMaker AI Doc Assistant")
st.markdown("Ask questions about Amazon SageMaker. Answers are derived from the official AWS Developer Guide!")

# Initialize chat history with a welcoming greeting
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "👋 **Hello!** I am your AWS SageMaker documentation assistant.\n\nI have read the massive 9,500-page AWS Developer Guide so you don't have to! I can help you write code, deploy models, and architect your ML pipelines. \n\nHow can I help you today?"
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input from the chat box at the bottom
user_prompt = st.chat_input("Ask a question...")

# ---------------------------------------------------------
# SUGGESTED PROMPTS UI
# ---------------------------------------------------------
# Display suggested prompts ONLY if the user hasn't asked anything yet (history length is 1)
if len(st.session_state.messages) == 1:
    st.markdown("<br><br>", unsafe_allow_html=True) # Add a little spacing
    st.markdown("##### ✨ Not sure where to start? Try one of these:")
    
    # Create 3 uniform columns for the buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is SageMaker Canvas?", use_container_width=True):
            user_prompt = "What is SageMaker Canvas?"
    with col2:
        if st.button("Write code to deploy an endpoint", use_container_width=True):
            user_prompt = "Write code to deploy an endpoint"
    with col3:
        if st.button("Spot Instances vs On-Demand", use_container_width=True):
            user_prompt = "What is the difference between Managed Spot Training and On-Demand in SageMaker?"

# ---------------------------------------------------------
# CHAT LOGIC
# ---------------------------------------------------------
# React to user input (from either the chat box OR a button click)
if user_prompt:
    
    # Display user message in chat message container
    st.chat_message("user").markdown(user_prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Searching the AWS Developer Guide..."):
        try:
            # Send the question to the FastAPI backend
            response = requests.post(API_URL, json={"question": user_prompt})
            response.raise_for_status() # Raise an exception for bad status codes
            answer = response.json().get("answer", "Error: No answer returned.")
        except requests.exceptions.ConnectionError:
            answer = "Error: Could not connect to the FastAPI backend. Is main.py running?"
        except Exception as e:
            answer = f"An error occurred: {str(e)}"
            
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Rerun the app so the suggested buttons disappear now that the conversation has started
    st.rerun()

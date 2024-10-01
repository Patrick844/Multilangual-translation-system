import streamlit as st
import requests

# Initialize the session state for translation_dict if not already set
if 'translation_dict' not in st.session_state:
    st.session_state.translation_dict = []

# Title of the app
st.title("Multilingual Translator App")

# Input text for translation
text = st.text_input("Enter the text you want to translate:")

if text:
    url = "https://ideal-amoeba-specially.ngrok-free.app/inference"
    body = {"input": text}

    try:
        # Send request for translation
        response = requests.post(url, json=body, verify=False)

        # Check if the response was successful
        if response.status_code == 200:
            data = response.json()
            input_text = data.get("translation", "No translation found.")
            language = data.get("language", "Unknown")

            # Append the translation to the session state dictionary
            st.session_state.translation_dict.append({"o": text, "t": input_text, "l": language})

            # Display translation history
            for translate in st.session_state.translation_dict:
                st.write(f"**Original ({translate['l']}):** {translate['o']}")
                st.markdown(f"**Translation:** :blue[{translate['t']}]")
                st.markdown("---")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

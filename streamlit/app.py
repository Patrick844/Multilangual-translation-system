import streamlit as st
import requests
import mlflow



# def save_translation(translation,original,answer):
#     new_row = {"original":original,"translation":translation,"answer":answer}
#     st.session_state.df.append(new_row,ignore_index=True)


# Initialize the session state for translation_dict if not already set
if 'translation_dict' not in st.session_state:
    st.session_state.translation_dict = []
# if "translation_id" not in st.session_state:
#     st.session_state.translation_id=0
# if 'df' not in st.session_state:
#     st.session_state.df = pd.DataFrame(columns=['original', 'translation',"answer"])

# Title of the app
st.title("Multilingual Translator App")

# Input text for translation
text = st.text_input("Enter the text you want to translate:")

if text:
    url = "https://ideal-amoeba-specially.ngrok-free.app/inference"
    body = {"input": text}

    try:
        # Send request for translation
        response = requests.get(url, params=body, verify=False)

        # Check if the response was successful
        if response.status_code == 200:
            data = response.json()
            translated_text = data.get("translation", "No translation found.")
            language = data.get("language", "Unknown")
            errors_count = data.get("errors_count",0)
            

            # Append the translation to the session state dictionary
            st.session_state.translation_dict.append({"original": text, "translation": translated_text, "language": language,"errors_count":errors_count})
            # Display translation history
            for data in st.session_state.translation_dict:
                st.markdown("---")
                st.write(f"**Original ({data['language']}):** {data['original']}")
                if ">>ara<<" in data['translation'].split():
                   text = data['translation'].split()[1:]
                   text = " ".join(text)
                   st.markdown(f"**Translation:** :blue[{text}]")
                else:
                    st.markdown(f"**Translation:** :blue[{data['translation']}]")
                    
                rating = data['errors_count']
                rating = round(rating,4)
                st.write(f"**Rating: :red[{rating}]**")
                # accept = st.button("Accept",onclick=)
                # reject = st.button("Reject")

                st.markdown("---")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

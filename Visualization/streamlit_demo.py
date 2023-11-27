import streamlit as st
import load_model

# Streamlit app
def main():
    st.title("Named Entity Recognition with ML")

    # Text input box
    input_text = st.text_area("Enter text for analysis", "")

    # Analyze button
    if st.button("Analyse"):
        if input_text:
            # Perform analysis using the Hugging Face model
            

            # Display result in a text box
            st.subheader("Analysis Result:")
            st.text(load_model.predict_tags(input_text))
            
            # st.text(result[0]['label'])
            # st.text(f"Confidence: {result[0]['score']}")
        else:
            st.warning("Please enter some text for analysis.")

if __name__ == "__main__":
    main()

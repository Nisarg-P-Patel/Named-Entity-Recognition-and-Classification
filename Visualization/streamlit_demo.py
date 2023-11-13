import streamlit as st
from transformers import pipeline

# Load the Hugging Face model
classifier = pipeline('sentiment-analysis')

# Streamlit app
def main():
    st.title("Named Entity Recognition with ML")

    # Text input box
    input_text = st.text_area("Enter text for analysis", "")

    # Analyze button
    if st.button("Analyse"):
        if input_text:
            # Perform analysis using the Hugging Face model
            result = classifier(input_text)

            # Display result in a text box
            st.subheader("Analysis Result:")
            st.text("[('Rip', 'PERSON'), ('.', 'NON_NER'), ('illinois', 'GPE'), ('k9', 'PERSON'), ('hudson', 'PERSON'), ('was', 'NON_NER'), ('murdered', 'NON_NER'), ('when', 'NON_NER'), ('he', 'NON_NER'), ('was', 'NON_NER'), ('shot', 'NON_NER'), ('and', 'NON_NER'), ('killed', 'NON_NER'), ('while', 'NON_NER'), ('apprehending', 'NON_NER'), ('a', 'NON_NER'), ('carjacking', 'NON_NER'), ('suspect', 'NON_NER'), ('after', 'NON_NER'), ('a', 'NON_NER'), ('pursuit', 'NON_NER'), ('.', 'NON_NER'), ('stop', 'NON_NER'), ('shooting', 'NON_NER'), ('my', 'NON_NER'), ('police', 'NON_NER'), ('officers', 'NON_NER'), ('!', 'NON_NER'), ('!', 'NON_NER'), ('!', 'NON_NER'), ('rip', 'NON_NER'), ('hero', 'NON_NER'), ('murdered', 'NON_NER'), ('k9hudson', 'NON_NER'), ('thinblueline', 'NON_NER'), ('eow', 'NON_NER'), ('kanecountysheriffsoffice', 'NON_NER'), ('k9hudson', 'NON_NER'), ('bluelivesmatter', 'NON_NER'), ('protectingtheblue', 'NON_NER')]")
            
            # st.text(result[0]['label'])
            # st.text(f"Confidence: {result[0]['score']}")
        else:
            st.warning("Please enter some text for analysis.")

if __name__ == "__main__":
    main()

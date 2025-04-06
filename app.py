import streamlit as st
import pickle

# Load models and vectorizer
with open("political_lean_model.pkl", "rb") as f:
    political_model= pickle.load(f) 


with open("subreddit_model.pkl", "rb") as f:
    subreddit_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Manual mappings
leaning_map = {0: "Liberal", 1: "Conservative"}
subreddit_list = [
    "Capitalism", "Communism", "Democratic Socialism", "Liberal", "Libertarian",
    "Radical Feminism", "Social Democracy", "All the left", "Anarchial capitalism",
    "Conservatives", "Democrats", "Feminism", "Progressive", "Republicans", "Socialism"
]

# Streamlit UI
st.title("Political Sentiments Analyser")

text_input = st.text_area("Enter a Reddit Post Text", height=200)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize text
        transformed_text = vectorizer.transform([text_input])

        # Predictions
        lean_prediction = political_model.predict(transformed_text)[0]
        subreddit_prediction = subreddit_model.predict(transformed_text)[0]

        # Map to labels
        leaning_label = leaning_map.get(lean_prediction, "Unknown")
        subreddit_label = subreddit_list[subreddit_prediction] if 0 <= subreddit_prediction < len(subreddit_list) else "Unknown"

        # Display results
        st.subheader("Results:")
        st.write(f"**Political Leaning:** {leaning_label}")
        st.write(f"Based on the content, this post fits best in **{subreddit_label}** category.")

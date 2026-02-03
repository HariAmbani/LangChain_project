#import langchain_helper as lch
import streamlit as st

st.title("Pet Name Generator")

animal_type = st.sidebar.selectbox("what is your pet ?",("dog","cat","fish","cow","bird"))

if animal_type == "dog" or animal_type == "cat" or animal_type == "bird" or animal_type == "fish" or animal_type == "cow":
    pet_color = st.sidebar.text_area("what is the color of your pet ?", max_chars=20, height=30)

# if pet_color:
#     names = lch.generate_pet_name(animal_type, pet_color)
#     st.subheader("Suggested Pet Names:")
#     st.text(names['pet_names'])
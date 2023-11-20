import time
import pandas as pd
import numpy as np

import streamlit as st

"""
To run:

    streamlit run webapp.py
    
in a console
"""


#DEBUG INFO!! Nu folosim
print("Rulam!")


# Eliminam, sa vedem ce face
@st.cache_resource
def functie_heavyweight(input=0):
    if input > 0:
        time.sleep(input)


st.title('Demo Generative AI!')

mesaj = st.text(f"Un mesaj generat din cod")
obiect_text_stare = st.text("Mesaj de status")

obiect_text_stare.text("Incarcam setul de date . . .")
functie_heavyweight(10)
obiect_text_stare.text("Programul s-a incarcat!")

slider = st.slider("Set me!", 0, 10, 1)
mesaj.text(f"Slider-ul e setat la valoarea: {slider}")

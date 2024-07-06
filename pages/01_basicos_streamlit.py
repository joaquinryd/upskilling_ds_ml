import streamlit as st
from config import icono_titulo
#from config import * //mala práctica

def main(): 
    st.title(f'Hello World {icono_titulo}')
    #layout:
    # expander
    # columnas
    # tabs
    # empty/container

    espacio_slider = st.container()

    with st.expander('FAQ'):
        st.write('Answer')
        botton = st.button('Say Hello')

    col1, col2 = st.columns(2)
    with col1:
        slider = st.slider('Select a value', 0, 100, 51, key='columna')
        checkbox = st.checkbox('I agree')
    text = col2.text_input('Enter some text')
    date = col2.date_input('Pick a date')

    tab1, tab2 = st.tabs(['Tab 1', 'Tab 2'])
    with tab1:                              #izquierda, derecha, el default
        slider = st.slider('Select a value', 0, 100, 51, key='tab')
        checkbox = st.checkbox('I agreeeeeee')
    text = tab2.text_input('Enter some texto')
    initial_date = tab2.date_input('Pick a dateee')

    with espacio_slider:
        st.write(f'el valor del slider es: {slider: .2f}')
        st.write(f'la fecha es es: {initial_date}')


main()


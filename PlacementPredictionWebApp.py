import numpy as np
import pickle as pkl
import streamlit as st

loaded_model = pkl.load(open('PlacementModel1.sav', 'rb'))

def placement_model(input, name):
    inp_as_array = np.asarray(input)
    reshaped_inp = inp_as_array.reshape(1, -1)
    pred = loaded_model.predict(reshaped_inp)

    if(pred[0] == 0):
        return f"No placement for {name}"
    else:
        return f"Congrats {name} for getting placed"

def main():
    st.title("Placement Prediction")
    name = st.text_input("Enter your name")
    features = ['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','mba_p']

    user_input = []

    for feature in features:
        value = st.number_input(f"Enter {feature}", value= 0, format= '%.4f')
        user_input.append(value)

    result = ''
    if(st.button("Check")):
        result = placement_model(user_input, name)
        st.success(result)

if __name__ == '__main__':
    main()
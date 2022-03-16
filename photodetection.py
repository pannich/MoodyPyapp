from functools import lru_cache
import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np

import time
import altair as alt
import pandas as pd

@lru_cache(maxsize=128)
def load_image(img):
    im=Image.open(img)
    return im


shape=(48, 48)
reshape=(1,48,48,1)

emotion = emotions={
          0:'Angry',
          1:'Disgust',
          2:'Fear',
          3:'Happy',
          4:'Sad',
          5:'Surprise',
          6:'Neutral'}

def main():
    st.title('Face Detection App')
    st.text('Build with Streamlit and OpenCV')


    activities =["About",'Emotion_detection']
    choice = st.sidebar.selectbox('Select Activity',activities)



    #photo_upload

    if choice=='Emotion_detection':
        st.subheader('Emotion Detection')

        image_file=st.file_uploader('Upload Image',type=["jpg","png","jpeg"])

        if image_file is not None:
            our_static_image= Image.open(image_file)
            st.image(our_static_image)
            my_bar = st.progress(0)
            st.text('Calculating Emotion Step 1: resizing')
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            new_img=np.array(our_static_image.convert('RGB'))
            img=cv2.cvtColor(new_img,1)
            our_static_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #ßst.image(our_static_image)
            st.text('Calculating Emotion Step 2: passing through prediction model')
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            st.text('Calculating Emotion Step 3: calculating confidence')
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)


            from streamlit_hadjer import retrieve_model

            model = retrieve_model()

            img_resized = cv2.resize(our_static_image, shape).reshape(reshape)
            pred = model.predict(img_resized/255.)[0]
            print(pred)
            #pred=[2.18952447e-02, 9.08929738e-04, 2.18112040e-02, 5.32227278e-01,
              #4.18808281e-01, 6.75195479e-04]

            chart_data = pd.DataFrame(
            pred,
            index=["Angry","'Disgust","Fear","Happy","Sad","Surprise","Neutral"],)
            data = pd.melt(chart_data.reset_index(), id_vars=["index"])
            chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
            x=alt.X("value", type="quantitative", title=""),
            y=alt.Y("index", type="nominal", title=""),
            color=alt.Color("variable", type="nominal", title=""),
            order=alt.Order("variable", sort="descending"),
                )
                    )

            st.altair_chart(chart, use_container_width=True)


            emotion_2=emotion[np.argmax(pred)]

            st.text(f'You are looking so {emotion_2} today')



        #Emotion Prediction



    elif choice == 'About':
        st.subheader('About')


if __name__ == '__main__':
    main()

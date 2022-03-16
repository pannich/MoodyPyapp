#Import for streamlit
import streamlit as st
import av
from tensorflow.keras.models import load_model
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

import cv2
from PIL import Image,ImageEnhance
import numpy as np
import os

import time
import altair as alt
import pandas as pd


#Import for handling image
import cv2
from cvzone.FaceDetectionModule import FaceDetector


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#Create a dict for classes
emotion  ={
          0:'Angry',
          1:'Disgust',
          2:'Fear',
          3:'Happy',
          4:'Sad',
          5:'Surprise',
          6:'Neutral'}



#download model
@st.cache(allow_output_mutation=True)
def retrieve_model():

    model = load_model("model.h5")
    return model
#Main inelligence of the file, class to launch a webcam, detect faces, then detect emotion and output probability for each emotion

def app_emotion_detection():
    class EmotionPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            # Sign detector
            self.face_detector = FaceDetector(    )
            self.model = retrieve_model()
            self.queueprediction = []


        def img_convert(self,image):
            print(image.shape)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            return image


        def predict(self,image,shape,reshape):

            img_resized = cv2.resize(image, shape).reshape(reshape)
            pred = self.model.predict(img_resized/255.)[0]
            return emotion[np.argmax(pred)], np.max(pred)

        def find_faces(self, image):
            image2 = image.copy()
            image_face, faces = self.face_detector.findFaces(image)



            # loop over all faces and print them on the video + apply prediction
            for face in faces:
                if face['score'][0] < 0.9:
                    continue

                SHAPE = (48, 48)
                RESHAPE = (1,48,48,1)

                xmin = int(face['bbox'][0])
                ymin = int(face['bbox'][1])
                deltax = int(face['bbox'][2])
                deltay = int(face['bbox'][3])

                start_point = (max(0,int(xmin - 0.3*deltax)),max(0,int(ymin - 0.3*deltay)))

                end_point = (min(image2.shape[1],int(xmin + 1.3*deltax)), min(image2.shape[0],int(ymin + 1.3*deltay)))

                im2crop = image2
                im2crop = im2crop[start_point[1]:end_point[1],start_point[0]:end_point[0]]
                im2crop = self.img_convert(im2crop)
                from PIL import Image
                im = Image.fromarray(im2crop)
                im.save("your_file.jpeg")


                prediction,score = self.predict(im2crop,SHAPE,RESHAPE)
                print(prediction, score)
                self.queueprediction.append((prediction,score))

                if len(self.queueprediction)>20:
                    self.queueprediction = self.queueprediction[-20:]
                    print(self.queueprediction)



                emotions_dict =  {
                                    'Angry': 0,
                                    'Disgust':0,
                                    'Fear': 0,
                                    'Happy':0,
                                    'Sad':0,
                                    'Surprise':0,
                                    'Neutral': 0}
                emotions_responses = {
                                    'Angry': 'Wow chill out',
                                    'Disgust':'Eww',
                                    'Fear': 'TIME TO PANIC',
                                    'Happy':'Keep smiling!!',
                                    'Sad':'Aww no please do not be sad',
                                    'Surprise':'Ahhhhh',
                                    'Neutral': 'Show me your mood',
                                    'happytosad': 'Ohh no what happened '}



                for element in self.queueprediction:
                    emotions_dict[element[0]] +=1
                print(emotions_dict)

                # #draw emotion on images
                cv2.putText(image2, f'{prediction}', (start_point[0]+180, start_point[1]+300), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 0.9, (255,255,255), 2)

                #{str(score)}
                #(start_point[0]-150, start_point[1]-80)

                maxi = 0
                topemotion = 'Angry'
                for key in emotions_dict:
                    if emotions_dict[key] > maxi:
                        maxi = emotions_dict[key]
                        topemotion = key

                top_emotions_list = ['neutral','neutral']

                if maxi > 15:
                    top_emotions_list.append(topemotion)
                    top_emotions_list.pop(0)
                    if top_emotions_list[-1] == 'Neutral'and top_emotions_list[-2] == 'Happy':
                        topemotion = 'happytosad'

                test = top_emotions_list[1] == 'Neutral'and top_emotions_list[0] == 'Happy'



                cv2.putText(image2, f'{emotions_responses[topemotion]}', (150,80), cv2.FONT_HERSHEY_DUPLEX,
                                 1, (255, 0, 255), 2)










                #draw rectangle arouond face
                cv2.rectangle(image2, start_point, end_point,(255,255,255), 2)

            return faces, image2




        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            faces, annotated_image = self.find_faces(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


############################ Sidebar + launching #################################################
def main():
    our_cover_image=Image.open("new_cover_image.jpeg")
    st.image(our_cover_image)
    st.title('Welcome to MoodyPy')

    activities =["Live Emotion Detector","Upload your Emotion!", "About"]
    choice = st.sidebar.selectbox('Select Activity',activities)


    if choice=="Live Emotion Detector":
         st.markdown(f'''

        ### Deep learning for AI Facial Detector

        ### Connecting Artificial Intelligence & Human Emotion

        Artificial Intelligence technology is developing fast:
        - Whilst AI technologies stride to improve efficiency in our everyday lives, the soft side of AI is still falling behind.
        - We will be interacting with computers more than ever, and it is crucial to develop AI that communicates smoothly to us just like another human.
        - As a result we decided to develop a realtime Emotion Detector, allowing AI to emphasise more with humans.
        - This allows for endless possibilities to advance AI applications in areas such as healthcare or detecting drunk drivers.
        ### Try it out yourself: how are you feeling today? :blush:
        ''')

         app_emotion_detection()

         st.markdown(f'''

            Press Start 👈 to start the show!

            ''')




    elif choice=='Upload your Emotion!':
        st.subheader('Emotion Detection')
        image_file=st.file_uploader('Upload Image',type=["jpg","png","jpeg"])

        if image_file is not None:
            our_static_image= Image.open(image_file)
            st.image(our_static_image)
            my_bar = st.progress(0)
            st.text('Calculating Emotion Step 1: resizing')
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)
            new_img=np.array(our_static_image.convert('RGB'))
            img=cv2.cvtColor(new_img,1)
            our_static_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #ßst.image(our_static_image)
            st.text('Calculating Emotion Step 2: passing through prediction model')
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)
            st.text('Calculating Emotion Step 3: calculating confidence')
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)


            from streamlit_hadjer import retrieve_model

            model = retrieve_model()

            SHAPE = (48, 48)
            RESHAPE = (1,48,48,1)

            img_resized = cv2.resize(our_static_image, SHAPE).reshape(RESHAPE)
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
            if emotion_2 == "Happy":
                st.text(f'Great to see you are smiling today')
            if emotion_2 == "Angry":
                 st.text(f'Wow chill out please')
            if emotion_2 == "Disgust":
                 st.text(f'Eww')
            if emotion_2 == "Fear":
                 st.text(f'DO NOT PANIC')
            if emotion_2 == "Sad":
                st.text(f'Aww please do not be sad')
            if emotion_2 == "Surprise":
                st.text(f'Ahhhh')
            if emotion_2 == "Neutral":
                st.text(f'Is there anything I can do to make you smile?')




    elif choice ==  "About":
        st.subheader('Our Team')
        our_group_image=Image.open("Group_picture.png")
        st.image(our_group_image, caption="…MoodyPy is loading your \U0001F600 \U0001F60D \U0001F621 \U0001F624 \U0001F4A8")



        st.markdown(f'''

            ### About our model

            Often CNNs models are trained with the tensorflow [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16) transfer learning model. In our case, we ended up using a less complex model that is four times smaller in term of size. So that we can have it run super smooth on the website. Our accuracy of the small model is not the best across all models, but it is good enough to predict emotions.

            We used the FER 2013 (https://www.kaggle.com/msambare/fer2013) dataset, which contains 32,900 + images including 8 emotion categories – anger, contempt, disgust, fear, happiness, neutrality, sadness and surprise.

            To achieve better accuracy and an effective tradeoff with timing, we narrowed our model down to 6 emotion categories – anger, disgust, fear, happiness, sadness and surprise.

            We ended up with 64% testing accuracy.

            We hope you enjoy it!

            ''')


if __name__ == '__main__':
    main()

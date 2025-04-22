import streamlit as st
from model import classify_model, detection_model
from helper import load_image


def st_ui():
    '''
    Render the User Interface of the application endpoints
    '''
    st.title("Animal Classification")
    st.caption("Image Recognition")
    st.info("YOLO Model Implementation by Oghli")
    # hint = st.empty()
    # with hint.container():
    #     st.write("### The model detects vehicles and classify it according to:")
    #     st.write("####  • Type")
    #     st.write("####  • Color")
    st.sidebar.subheader("Upload image to classify animals")
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg"],
                                              accept_multiple_files=False, key=None,
                                              help="Image to classify animals")
    s_msg = st.empty()
    example_image = load_image('images/animal1.jpg')
    st.subheader("Input Image")
    example = st.image(example_image)
    if example:
        placeholder = st.empty()
        de_btn = placeholder.button('CLASSIFY', key='1')
    if uploaded_image:
        placeholder.empty()
        example.empty()
        # hint.empty()
        st.image(uploaded_image)
        de_btn = st.button("CLASSIFY")
        s_msg = st.sidebar.success("Image uploaded successfully")

    if de_btn:
        s_msg.empty()
        # hint.empty()
        image_de = 'images/animal1.jpg'
        if uploaded_image:
            image_de = uploaded_image
        with st.spinner('Processing Image ...'):
            detection_image, name, conf = detection_model(image_de)
            if not example:
                example.empty()
                st.image(image_de)
            st.subheader("Result Image")
            st.image(detection_image)
            st.subheader("Classification Result")
            st.success(f"#### Name: {name}")
            st.success(f"#### Confidence: {conf * 100:.2f}%")
            # byte_detect_img = pil_to_bytes(detection_image)
            # st.download_button(label="Download Result", data=byte_super_img,
            #                    file_name="detect_image.jpeg", mime="image/jpeg")


if __name__ == "__main__":
    # render the app using streamlit ui function
    st_ui()
    # image_source = "images/test.jpg"
    # detect_image = cv_vehicle_detect(image_source)
    # plt_show(detect_image)

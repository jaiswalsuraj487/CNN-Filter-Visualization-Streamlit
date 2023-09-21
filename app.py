
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Define filter kernels for Sobel edge detection
sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Set the weights of the vertical and horizontal line detection kernels
vertical_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
horizontal_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])

# Set the weights of the sharpening kernel
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])




# Define a function to apply the selected filter/kernel to the input image
def apply_filter(image, filter_type):
    filtered_image = image
    if filter_type == 'Grayscale':
        filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'Gaussian Blur':
        kernel_size = 3
        if filter_type == 'Gaussian Blur':
            # Add a selection for kernel size
            kernel_size = st.sidebar.slider('Select Kernel Size', min_value = 1, max_value=49, value=3, step=2)
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'Sobel X':
        filtered_image = cv2.filter2D(image, -1, sobel_x_kernel)
        # b,g,r = cv2.split(image)
        # sobel_yb = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_yg = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_yr = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)

        # sobel_y = cv2.merge([sobel_yb, sobel_yg, sobel_yr])

        # filtered_image = sobel_y

    elif filter_type == 'Sobel Y':
        filtered_image = cv2.filter2D(image, -1, sobel_y_kernel)

    elif filter_type == 'Sobel':
        
        # b,g,r = cv2.split(image)
        # sobel_xb = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_yb = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_xg = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_yg = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_xr = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_yr = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)

        # sobel_x = cv2.merge([sobel_xb, sobel_xg, sobel_xr])
        # sobel_y = cv2.merge([sobel_yb, sobel_yg, sobel_yr])

        sobel_x = cv2.filter2D(image, -1, sobel_x_kernel)
        sobel_y = cv2.filter2D(image, -1, sobel_y_kernel)
        output_image = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        filtered_image = output_image

    elif filter_type == 'Vertical Lines':
        # kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        filtered_image = cv2.filter2D(image, -1, vertical_kernel)
        # b,g,r = cv2.split(image)
        # b_f = cv2.filter2D(b, -1, vertical_kernel)
        # g_f = cv2.filter2D(g, -1, vertical_kernel)
        # r_f = cv2.filter2D(r, -1, vertical_kernel)
        # filtered_image = cv2.merge([b_f, g_f, r_f])

    elif filter_type == 'Horizontal Lines':
        b,g,r = cv2.split(image)
        b_f = cv2.filter2D(b, -1, horizontal_kernel)
        g_f = cv2.filter2D(g, -1, horizontal_kernel)
        r_f = cv2.filter2D(r, -1, horizontal_kernel)

        filtered_image = cv2.merge([b_f, g_f, r_f])
    elif filter_type == 'Sharpen':
        b,g,r = cv2.split(image)
        b_f = cv2.filter2D(b, -1, sharpen_kernel)
        g_f = cv2.filter2D(g, -1, sharpen_kernel)
        r_f = cv2.filter2D(r, -1, sharpen_kernel)

        filtered_image = cv2.merge([b_f, g_f, r_f])
    return filtered_image



###########################################################################################



# Prompt the user to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# print(uploaded_file)
# uploaded_file = 'cat.jpg'
if uploaded_file is not None:
    # Open the uploaded image file using PIL
    # read the uploaded file from cv2 
    image = Image.open(uploaded_file)
    image = np.array(image)
    # image = cv2.imread(uploaded_file)
    # image = cv2.imread()

    # Show the original image
    # st.image(image, caption='Original Image', width=300)

    # Get the user-selected filter/kernel
    filter_type = st.sidebar.selectbox('Select Filter/Kernel', ['Original', 'Grayscale', 'Gaussian Blur','Sobel', 'Sobel X', 'Sobel Y', 'Vertical Lines', 'Horizontal Lines', 'Sharpen'])

    # Apply the selected filter/kernel to the input image
    if filter_type != 'Original':
        filtered_image = apply_filter(image, filter_type)
    else:
        filtered_image = image

    st.image([image, filtered_image], caption=['Input Image', filter_type], width=300)
    # st.image(image, caption='Uploaded Image.')
    # st.image(filtered_image, caption='Filtered Image.')
else:
    st.write('No image file uploaded.') 



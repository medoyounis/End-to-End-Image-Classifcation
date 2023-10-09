import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import tensorflow as tf

# Assuming the model expects images of this size
IMAGE_SIZE = (256, 256)
model = None  # Initially, no model is loaded
img_array = None  # Global variable to store the current image array
photo = None  # Global variable to store the current image
output_image_label = None

app = tk.Tk()
app.title("Image Viewer with Prediction")

model_type = tk.StringVar()
model_type.set("Classification")  # default value

def draw_bounding_boxes(image, predictions, threshold=0.5):
    """
    Draw bounding boxes on the image based on YOLOv4 predictions.

    predictions: A list of detections for each scale. Each detection is of the form:
                 [center_x, center_y, width, height, objectness_score, ...class_scores]

    image: A PIL Image object where detections will be drawn.

    threshold: A confidence score threshold. Detections with a score lower than this will be ignored.

    Returns a PIL Image with the bounding boxes drawn.
    """

    draw = ImageDraw.Draw(image)
    width, height = image.size

    for prediction in predictions:
        for obj in prediction:
            # Retrieve coordinates and scores
            center_x, center_y, w, h, objectness, *class_scores = obj

            # Get the class with the highest score
            class_index = np.argmax(class_scores)
            class_score = class_scores[class_index]
            total_score = objectness * class_score

            # Filter based on threshold
            if total_score > threshold:
                # Convert center coordinates, width, and height to top-left and bottom-right coordinates
                x1 = int((center_x - w/2) * width)
                y1 = int((center_y - h/2) * height)
                x2 = int((center_x + w/2) * width)
                y2 = int((center_y + h/2) * height)

                # Draw bounding box and label
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"Class {class_index} - {total_score:.2f}", fill="red")

    return image

def default_postprocess(predictions):
    # # For example, if it's a classification task with softmax activation
    # predicted_class = np.argmax(predictions)
    # return f"Predicted Class: {predicted_class}"
    global output_image_label

    # Get the original image
    image = Image.fromarray(img_array.astype('uint8'))

    if model_type.get() == "Classification":
        predicted_class = np.argmax(predictions)
        return f"Predicted Class: {predicted_class}"

    elif model_type.get() == "Detection":
        # Draw bounding boxes on the image (this function needs to be implemented)
        image_with_boxes = draw_bounding_boxes(image, predictions)
        photo_output = ImageTk.PhotoImage(image_with_boxes)
        
        if output_image_label:
            output_image_label.config(image=photo_output)
            output_image_label.image = photo_output  # Keep a reference
        else:
            output_image_label = tk.Label(image_frame, image=photo_output)
            output_image_label.pack(padx=5, pady=5)

        return f"Detection Complete"  # You can enhance this message based on the predictions

    elif model_type.get() == "Segmentation":
            # Convert model's prediction to an image (assuming single channel output)
            # If your model has multi-channel output, adjust accordingly
            segmentation_map = (np.squeeze(predictions)).astype(np.uint8)  
            # Convert to PIL image and display
            segmentation_image = Image.fromarray(segmentation_map)
            segmentation_photo = ImageTk.PhotoImage(segmentation_image)
            
            # global output_image_label  # Ensure this variable is defined in the main part of the code
            if output_image_label:
                output_image_label.config(image=segmentation_photo)
                output_image_label.image = segmentation_photo
            else:
                output_image_label = tk.Label(image_frame, image=segmentation_photo)
                output_image_label.pack(padx=5, pady=5)

            return f"Segmentation Complete"

    else:
            return "Unknown Model Type"

postprocess = default_postprocess

def load_model():
    global model
    model_path = filedialog.askopenfilename(title="Load TensorFlow Model", filetypes=[("Model Files", "*.h5")])
    if not model_path:
        return
    model = tf.keras.models.load_model(model_path)
    status_label.config(text="Model Loaded Successfully Mohammed!")

image_label = None
prediction_label = None

def update_prediction(*args):
    global img_array, prediction_label

    if img_array is None or model is None:
        return

    if divide_var.get():
        try:
            divide_value = float(divide_entry.get())
            current_img_array = img_array / divide_value
        except ValueError:
            status_label.config(text="Invalid divide value!")
            return
    else:
        current_img_array = img_array.copy()

    predictions = model.predict(np.expand_dims(current_img_array, axis=0))
    prediction_text = postprocess(predictions)

    # Update the displayed prediction
    if prediction_label:
        prediction_label.config(text=prediction_text)
    else:
        prediction_label = tk.Label(image_frame, text=prediction_text, font=("Arial", 12, "bold"))
        prediction_label.pack(pady=20)

def open_image_and_predict():
    global img_array, photo, image_label, IMAGE_SIZE

    if not model:
        status_label.config(text="Please load a model first!")
        return

    file_path = filedialog.askopenfilename(title="Open Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if not file_path:
        return

    # Update IMAGE_SIZE from user's input
    try:
        width, height = map(int, image_size_var.get().split('x'))
        IMAGE_SIZE = (width, height)
    except ValueError:
        status_label.config(text="Invalid image size format! Please use WxH (e.g., 256x256).")
        return

    image = Image.open(file_path).resize(IMAGE_SIZE)
    photo = ImageTk.PhotoImage(image)

    if image_label:
        image_label.config(image=photo)
    else:
        image_label = tk.Label(image_frame, image=photo)
        image_label.pack(padx=5, pady=5)

    img_array = np.array(image)

    # Calculate the prediction
    update_prediction()

# Create main frame
main_frame = tk.Frame(app)
main_frame.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

# Model type dropdown menu
model_type_label = tk.Label(main_frame, text="Model Type:")
model_type_label.grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
model_type_menu = tk.OptionMenu(main_frame, model_type, "Classification", "Detection", "Segmentation")
model_type_menu.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

# Button to load model
load_model_button = tk.Button(main_frame, text="Load Model", command=load_model)
load_model_button.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)

# Image size entry box
image_size_var = tk.StringVar(value=f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
image_size_label = tk.Label(main_frame, text="Image Size (WxH):")
image_size_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
image_size_entry = tk.Entry(main_frame, textvariable=image_size_var, width=10)
image_size_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

# Label to display model status
status_label = tk.Label(main_frame, text="", font=("Arial", 12))
status_label.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5, padx=5)

def load_custom_function():
    global postprocess
    file_path = filedialog.askopenfilename(title="Load Custom Post-process Function", filetypes=[("Python Files", "*.py")])
    if not file_path:
        return
    
    local_vars = {}
    with open(file_path, 'r') as file:
        code = file.read()
        
    try:
        exec(code, {}, local_vars)
        postprocess = local_vars['custom_postprocess']
        status_label.config(text="Custom post-process function loaded successfully Mohammed!")
    except Exception as e:
        status_label.config(text=f"Error in custom function: {e}")
# Button to load custom post-process function
custom_func_button = tk.Button(main_frame, text="Load Custom Function from .py", command=load_custom_function)
custom_func_button.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)

# Checkbox and Entry for dividing by a value
divide_var = tk.BooleanVar()
divide_check = tk.Checkbutton(main_frame, text="Divide by:", variable=divide_var)
divide_check.grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
divide_entry = tk.Entry(main_frame, width=5)
divide_entry.insert(0, "255")
divide_var.trace_add("write", update_prediction)
divide_entry.bind("<KeyRelease>", update_prediction)
divide_entry.grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)

# Frame to display the image and prediction
image_frame = tk.Frame(main_frame)
image_frame.grid(row=0, column=2, rowspan=5, padx=5, pady=5, sticky=tk.E)

# Button to open the image
open_button = tk.Button(main_frame, text="Open Image and Predict", command=open_image_and_predict)
open_button.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)

app.mainloop()



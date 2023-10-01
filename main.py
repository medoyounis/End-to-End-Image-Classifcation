from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
app = FastAPI()

Model = tf.keras.models.load_model("C:/Users/hamoudi/potatodisease/plant/models/1")
classnames= ["Early Blight", "Late Blight", "Healthy"]
@app.get("/ping")
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
     image=np.array(Image.open(BytesIO(data)))
     return image

#calling predict function
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)):
        image=read_file_as_image(await file.read())
        #since the model takes batches, we need to expand dims
        image_batch=np.expand_dims(image,axis=0)
        prediction=Model.predict(image_batch)
        predicted_class=classnames[np.argmax(prediction[0])]
        confidence=np.max(prediction[0])
        return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=7000)
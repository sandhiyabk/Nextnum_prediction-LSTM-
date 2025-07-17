import numpy as np
import tensorflow as tf
import gradio as gr

# Load the trained model
model = tf.keras.models.load_model("my_model.keras")
window_size = 3  # same as used in training

# Prediction function
def predict_next_number(n1, n2, n3):
    input_seq = np.array([n1, n2, n3]).reshape(1, window_size, 1)
    prediction = model.predict(input_seq, verbose=0)
    return float(np.round(prediction[0][0], 2))

# Gradio interface
inputs = [gr.Number(label=f"Number {i+1}") for i in range(window_size)]
output = gr.Number(label="Predicted Next Number")

demo = gr.Interface(fn=predict_next_number, inputs=inputs, outputs=output, title="LSTM Sequence Predictor")
demo.launch()

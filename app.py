from fastai.vision.all import *
import gradio as gr
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


learn = load_learner('flower_model.pkl')

labels = learn.dls.vocab

title = "Tullip, Lily Flower and Sunflower Classifier"
description = "A flower classifier that trained with internet picture and using transfer learning from Resnet, made by following FastAI Deep Learning Course of 2022."


examples = ['tulip3.jpg']
interpretation = 'tullip'
enable_queue = True


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


demo = gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3), title=title,
                    description=description, examples=examples, interpretation=interpretation, enable_queue=enable_queue)

demo.launch(share=True)

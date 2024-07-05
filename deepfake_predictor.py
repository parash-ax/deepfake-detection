import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import zipfile
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

with zipfile.ZipFile("examples.zip","r") as zip_ref:
    zip_ref.extractall(".")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

EXAMPLES_FOLDER = 'examples'
examples_names = os.listdir(EXAMPLES_FOLDER)
examples = []
for example_name in examples_names:
    example_path = os.path.join(EXAMPLES_FOLDER, example_name)
    label = example_name.split('_')[0]
    example = {
        'path': example_path,
        'label': label
    }
    examples.append(example)
np.random.shuffle(examples) # shuffle

def predict(input_image:Image.Image, true_label:str):
    """Predict the label of the input_image"""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0) # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers=[model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    return confidences, true_label, face_with_mask

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Image(label="Input Image", type="pil"),  # Updated component import and type
        gr.components.Text(label="Your Text Input")  # Updated component import
    ],
    outputs=[
        gr.components.Label(label="Class"),  # Updated component import
        gr.components.Text(label="Your Text Output"),  # Updated component import
        gr.components.Image(label="Face with Explainability", type="numpy")  # Updated component import and type
    ],
    examples=[[examples[i]["path"], examples[i]["label"]] for i in range(10)],
    cache_examples=True  # Adjusted according to the new parameter for caching examples if needed
).launch()
ACTIVATE_DEEP_FAKE_DETECTION.PY
import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import zipfile
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

with zipfile.ZipFile("examples.zip","r") as zip_ref:
    zip_ref.extractall(".")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

EXAMPLES_FOLDER = 'examples'
examples_names = os.listdir(EXAMPLES_FOLDER)
examples = []
for example_name in examples_names:
    example_path = os.path.join(EXAMPLES_FOLDER, example_name)
    label = example_name.split('_')[0]
    example = {
        'path': example_path,
        'label': label
    }
    examples.append(example)
np.random.shuffle(examples) # shuffle

def predict(input_image:Image.Image, true_label:str):
    """Predict the label of the input_image"""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0) # add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # convert the face into a numpy array to be able to plot it
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    target_layers=[model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        confidences = {		
            'real': real_prediction,
            'fake': fake_prediction
        }
    return confidences, true_label, face_with_mask
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Image(label="Input Image", type="pil"),  # Updated component import and type
        gr.components.Text(label="Your Text Input")  # Updated component import
    ],
    outputs=[
        gr.components.Label(label="Class"),  # Updated component import
        gr.components.Text(label="Your Text Output"),  # Updated component import
        gr.components.Image(label="Face with Explainability", type="numpy")  # Updated component import and type
    ],
    examples=[[examples[i]["path"], examples[i]["label"]] for i in range(10)],
    cache_examples=True  # Adjusted according to the new parameter for caching examples if needed
).launch()

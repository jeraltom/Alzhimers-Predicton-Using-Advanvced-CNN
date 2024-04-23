import streamlit as st
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 30 * 30, 64)  # Adjusted size
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 4)  # Assuming 4 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 30 * 30)  # Adjusted size
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
class Net(nn.Module):
    
    
    # Defining the Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=4096, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x))) 
        x = F.relu(self.pool(self.conv2(x)))  
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))  
        
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

best_model = torch.load('CNN_98_acc.pt',map_location=torch.device('cpu'))

good_model = torch.load('CNN_simple.pt',map_location=torch.device('cpu'))

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a fixed size
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()           # Convert images to PyTorch tensors and scale to [0, 1]
])

def predict(model, image):
    model.eval()
    with torch.inference_mode():
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted


class_names= ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very MildDementia']




st.header('AlzhCare - Alzhimers Help Assistance')
st.image('https://www.drugwatch.com/wp-content/uploads/progression-alzheimers-disease.png')
st.write("Alzheimer’s disease is the most common type of dementia. It is a progressive disease beginning with mild memory loss and possibly leading to loss of the ability to carry on a conversation and respond to the environment. It can seriously affect a person’s ability to carry out daily activities.")
st.markdown('#### You are not alone we are here to assist you.')
st.markdown("Upload images of your **MRI Scan** to identify **severity.**")


with st.sidebar:
    st.title("About AlzhCare")
    st.image('https://images.newscientist.com/wp-content/uploads/2019/05/23170000/c0123811-alzheimer_s_disease-spl-2.jpg')
    st.title('Choose Model')
    selection = st.selectbox('Choose the Model: ',options=('CNN Augmented','CNN Conv'),help='Choose model for prediction')
    st.write('Valid Accuracy of different Models:')
    st.write('CNN Augmented : 97.8%')
    st.write('CNN Conv : 92.3%')
    st.download_button("Download APP",data="https://drive.google.com/file/d/1uQvQzj2yUwN8V5rRd8JqB4mY4f0aM9IY/view?usp=sharing",file_name="AlzhCareApp.apk")

if selection == 'CNN Augmented':
    st.sidebar.image('https://miro.medium.com/v2/resize:fit:1400/0*tH9evuOFqk8F41FG.png')

    st.markdown('### Upload Scans')
    file = st.file_uploader('Choose Scan Image:',type=['jpg','jpeg','png'],key="mri")

    if file:
        img = Image.open(file)
        st.markdown('##### Uploaded Scan')
        st.image(file)

        st.markdown("##### Show Result")
        if st.button("ANALYZE"):
            pred = predict(best_model, img)
            st.subheader(body = "You Have Been Diagnosed with " + class_names[int(pred)])
            if pred != 2:
                st.write("We are sorry to hear that. Check out our Mobile app for further assistance.")
            else:
                st.write("You are a very healthy person. Enjoy the Day")
else:
    st.sidebar.image('https://www.researchgate.net/publication/339008531/figure/fig1/AS:941759408402435@1601544348744/A-simple-convolutional-neural-network-CNN-and-its-main-layers.png')

    st.markdown('### Upload Scans')
    file = st.file_uploader('Choose Scan Image:',type=['jpg','jpeg','png'])

    if file:
        img = Image.open(file)
        st.markdown('##### Uploaded Scan')
        st.image(file)

        st.markdown("##### Show Result")
        if st.button("ANALYZE"):
            pred = predict(best_model, img)
            st.subheader(body = "You Have Been Diagnosed with " + class_names[int(pred)])
            if int(pred) != 2:
                st.write("We are sorry to hear that. Check out our Mobile app for further assistance.")
            else:
                st.write("You are a very healthy person. Enjoy the Day")
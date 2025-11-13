import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from database import db, UserEmotion
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotion_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["UPLOAD_FOLDER"] = "static/uploads"


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        filename = f"user_{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = Image.open(filepath)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            emotion = emotions[predicted.item()]

        new_record = UserEmotion(
            name="Guest",           # later replace with form input
            image_path=filepath,
            emotion=emotion
        )
        db.session.add(new_record)
        db.session.commit()

        return render_template("base.html", emotion=emotion, image_path=filepath)

    return render_template("base.html", emotion=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)

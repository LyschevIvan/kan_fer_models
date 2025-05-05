import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class EmotionModelBase(torch.nn.Module):
    def __init__(self, model_path, name=None):
        super(EmotionModelBase, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        print(f"Loaded model from {model_path}")
        return model
    
    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, image):
        x = self.preprocess(image)

        with torch.no_grad():
            output = self(x)
            probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
            
        result = {emotion: float(prob) for emotion, prob in zip(self.emotions, probs)}
        return result


class KANFER2013(EmotionModelBase):
    """Модель KAN-FER2013"""
    def __init__(self, model_path="src/pretrained/kan_fer2013.pt"):
        super(KANFER2013, self).__init__(model_path, name="KAN-FER2013")
        # 7 базовых эмоций для FER2013
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class KANRAFDB(EmotionModelBase):
    """Модель KAN-RAF-DB"""
    def __init__(self, model_path="src/pretrained/kan_rafdb.pt"):
        super(KANRAFDB, self).__init__(model_path, name="KAN-RAF-DB")
        # Эмоции для RAF-DB имеют другой порядок
        self.emotions = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']


class KALFER2013(EmotionModelBase):
    """Модель KAL-FER2013"""
    def __init__(self, model_path="src/pretrained/kal_fer2013.pt"):
        super(KALFER2013, self).__init__(model_path, name="KAL-FER2013")
        # 7 базовых эмоций для FER2013
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class KALRAFDB(EmotionModelBase):
    """Модель KAL-RAF-DB"""
    def __init__(self, model_path="src/pretrained/kal_rafdb.pt"):
        super(KALRAFDB, self).__init__(model_path, name="KAL-RAF-DB")
        # Эмоции для RAF-DB имеют другой порядок
        self.emotions = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
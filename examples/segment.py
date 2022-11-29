import torch
import torchvision as tv

class Segment:
    def __init__(self, device) -> None:
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        self.device = device
        self.model = tv.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        self.model.eval()
        self.model.to(self.device)
        self.preprocess = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segmentOne(self, image):
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(self.device)
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

    def segmentMany(self, images):
        input_batch = []
        for image in images:
            input_batch.append(self.preprocess(image))
        input_batch = torch.stack(input_batch).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)['out']
        output_predictions = output.argmax(1)

        o = (output_predictions > 0).type(torch.int)
        o = torch.unsqueeze(o, dim=3)

        alpha = o*255

        o = o.expand(-1,-1,-1, 3)

        images = torch.Tensor(images).to(self.device)
        mod = torch.mul(images, o)
        
        out = []
        for image, alpha_c in zip(mod, alpha):
            image = (image > 0).int() * 124
            out.append(torch.dstack([image,alpha_c]))

        out = torch.stack(out).to(self.device)

        return out
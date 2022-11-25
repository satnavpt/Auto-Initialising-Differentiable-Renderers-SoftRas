import torch
from PIL import Image
import torchvision as tv
import matplotlib.pyplot as plt

class Segment:
    def __init__(self) -> None:
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        self.model = tv.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        self.model.eval()
        self.model.cuda()
        self.preprocess = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segmentOne(self, image):
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.cuda()
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        print(output_predictions.shape)

        # # create a color pallette, selecting a color for each class
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")

        # # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((image.shape[0],image.shape[1]))
        # r.putpalette(colors)

    def segmentMany(self, images):
        input_batch = []
        for image in images:
            input_batch.append(self.preprocess(image))
        input_batch = torch.stack(input_batch).cuda()

        with torch.no_grad():
            output = self.model(input_batch)['out']
        output_predictions = output.argmax(1)

        o = (output_predictions > 0).type(torch.int)
        o = torch.unsqueeze(o, dim=3)

        alpha = o*255

        o = o.expand(-1,-1,-1, 3)

        images = torch.Tensor(images).cuda()
        mod = torch.mul(images, o)
        
        out = []
        for image, alpha_c in zip(mod, alpha):
            out.append(torch.dstack([image,alpha_c]))

        out = torch.stack(out).cuda()
        return out
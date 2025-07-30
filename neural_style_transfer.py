import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy

def load_image(path, max_size=400):
    image = Image.open(path).convert('RGB')
    size = max(image.size) if max(image.size) < max_size else max_size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = image.mul(torch.tensor([0.229, 0.224, 0.225]).view(3,1,1))
    image = image.add(torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1','5': 'conv2_1','10': 'conv3_1','19': 'conv4_1','21': 'conv4_2','28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_transfer(content, style, model, steps=2000, style_weight=1e6, content_weight=1):
    target = content.clone().requires_grad_(True)
    style_features = get_features(style, model)
    content_features = get_features(content, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    optimizer = optim.Adam([target], lr=0.003)
    for i in range(steps):
        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_loss / (target_feature.shape[1] ** 2)
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return target

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    content = load_image("content.jpg").to(device)
    style = load_image("style.jpg").to(device)
    output = style_transfer(content, style, vgg)
    result = im_convert(output)
    plt.imshow(result)
    plt.axis("off")
    plt.savefig("styled_result.jpg")
    plt.show()


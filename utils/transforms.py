from torchvision import transforms
import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImgageOps

random_mirror = True


class AutoAugment(object):
    def __init__(self):
        self.policies = [
            ["Invert", 0.1, 7, "Contrast", 0.2, 6],
            ["Rotate", 0.7, 2, "TranslateX", 0.3, 9],
            ["Sharpness", 0.8, 1, "Sharpness", 0.9, 3],
            ["ShearY", 0.5, 8, "TranslateY", 0.7, 9],
            ["AutoContrast", 0.5, 8, "Equalize", 0.9, 2],
            ["ShearY", 0.2, 7, "Posterize", 0.3, 7],
            ["Color", 0.4, 3, "Brightness", 0.6, 7],
            ["Sharpness", 0.3, 9, "Brightness", 0.7, 9],
            ["Equalize", 0.6, 5, "Equalize", 0.5, 1],
            ["Contrast", 0.6, 7, "Sharpness", 0.6, 5],
            ["Color", 0.7, 7, "TranslateX", 0.5, 8],
            ["Equalize", 0.3, 7, "AutoContrast", 0.4, 8],
            ["TranslateY", 0.4, 3, "Sharpness", 0.2, 6],
            ["Brightness", 0.9, 6, "Color", 0.2, 8],
            ["Solarize", 0.5, 2, "Invert", 0, 0.3],
            ["Equalize", 0.2, 0, "AutoContrast", 0.6, 0],
            ["Equalize", 0.2, 8, "Equalize", 0.6, 4],
            ["Color", 0.9, 9, "Equalize", 0.6, 6],
            ["AutoContrast", 0.8, 4, "Solarize", 0.2, 8],
            ["Brightness", 0.1, 3, "Color", 0.7, 0],
            ["Solarize", 0.4, 5, "AutoContrast", 0.9, 3],
            ["TranslateY", 0.9, 9, "TranslateY", 0.7, 9],
            ["AutoContrast", 0.9, 2, "Solarize", 0.8, 3],
            ["Equalize", 0.8, 8, "Invert", 0.1, 3],
            ["TranslateY", 0.7, 9, "AutoContrast", 0.9, 1],
        ]

    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    "ShearX": lambda img, magnitude: shearX(img, magnitude),
    "ShearY": lambda img, magnitude: shearY(img, magnitude),
    "TranslateX": lambda img, magnitude: translate_x(img, magnitude),
    "TranslateY": lambda img, magnitude: translate_y(img, magnitude),
    "Rotate": lambda img, magnitude: rotate(img, magnitude),
    "AutoContrast": lambda img, magnitude: auto_contrast(img, magnitude),
    "Invert": lambda img, magnitude: invert(img, magnitude),
    "Equalize": lambda img, magnitude: equalize(img, magnitude),
    "Solarize": lambda img, magnitude: solarize(img, magnitude),
    "Posterize": lambda img, magnitude: posterize(img, magnitude),
    "Contrast": lambda img, magnitude: contrast(img, magnitude),
    "Color": lambda img, magnitude: color(img, magnitude),
    "Brightness": lambda img, magnitude: brightness(img, magnitude),
    "Sharpness": lambda img, magnitude: sharpness(img, magnitude),
    "Cutout": lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[4]](img, policy[5])

    return img


def shearX(img, magnitude):
    assert -0.3 <= magnitude <= 0.3
    if random_mirror and random.random() > 0.5:
        magnitude = -1 * magnitude
    return img.transform(img.size, PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0))


def ShearY(img, magnitude):
    assert -0.3 <= magnitude <= 0.3
    if random_mirror and random.random() > 0.5:
        magnitude = -1 * magnitude
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0))


def TranslateX(img, magnitude):
    assert -0.45 <= magnitude <= 0.45
    if random_mirror and random.random() > 0.5:
        magnitude = -1 * magnitude
    magnitude = magnitude * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0))


def TranslateY(img, v):
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        magnitude = -1 * magnitude
    magnitude = magnitude * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))


def TranslateXAbs(img, magnitude):
    assert 0 <= v <= 10
    if random.random() > 0.5:
        magnitude = -1 * magnitude
    return img.transorm(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(image, magnitude):
    assert -30 <= magnitude <= 30
    if random_mirror and random.random() > 0.5:
        magnitude = -1 * magnitude
        return img.rotate(magnitude)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):
    assert 0 <= magnitude < 256
    return PIL.ImageOps.solarize(img, magnitude)


def Posterize(img, magnitude):
    assert 4 <= magnitude <= 8
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)


def Posterize2(img, magnitude):
    assert 0 <= magnitude <= 4
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)


def Contrast(img, magnitude):
    assert 0.1 <= magnitude <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(magnitude)


def Color(img, magnitude):
    assert 0.1 <= magnitude <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, magnitude):
    assert 0.1 <= magnitude <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(magnitude)


def Sharpness(img, magnitude):
    assert 0.1 <= magnitude < 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, magnitude):
    assert 0.0 <= magnitude <= 0.2

    magnitude = magnitude * img.size[0]
    return CutoutAbs(img, magnitude)


def CutoutAbs(img, magnitude):
    if magnitude < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - magnitude / 2.0))
    y0 = int(max(0, y0 - magnitude / 2.0))
    x1 = min(w, x0 + magnitude)
    y1 = min(h, y0 + magnitude)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImgDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):
    def f(img1, v):
        i = np.random.choice(len[imgs])
        img2 = PIL.Image.fromarray(img[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list(for_autoaug=True):
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

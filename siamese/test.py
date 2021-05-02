from  torchvision import datasets

a = datasets.ImageFolder('/gpfsnyu/home/xl3136/siamese/100_WebFace')
class_to_idx = a.class_to_idx
b = datasets.ImageFolder.make_dataset('/gpfsnyu/home/xl3136/siamese/100_WebFace', class_to_idx, extensions=('.jpg'))
print(b)
print('finished')

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model

# First we define transformations for the image.
# This transformaticon function does three things:
# 1.center crop the image to 224x224 in size,
# 2.transpose it to ``num_channels*height*width``,
# 3.normalize with mean and standard deviation calculated across all ImageNet images.
transform_fn = transforms.Compose([
    video.VideoCenterCrop(size=224),
    video.VideoToTensor(),
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# read image and prepare the image
im_fname = '0/0001.png'
img = image.imread(im_fname)
img_list = transform_fn([img.asnumpy()])

# load model
model_name = 'resnet50_v1b_custom'
MODEL_NAME = '0.9984-resnet50_v1b_custom-088-best.params'
MODEL_PATH = '/home/ubuntu/activity/train/2classes/'
model_path = os.path.join(MODEL_PATH, MODEL_NAME)
net = get_model(model_name, nclass=2, pretrained=False)
net.load_parameters(model_path)

# Finally, feed it to the model to get confidence score
# if >0.5 then interaction with rack detected!
pred = net(nd.array(img_list[0]).expand_dims(axis=0))
confidence = nd.softmax(pred)[0][1].asscalar()
print('Interaction probability is %s.' % confidence)

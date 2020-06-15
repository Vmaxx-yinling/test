import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model

################################################################
# Then, we download the video and extract a 32-frame clip from it.

from gluoncv.utils.filesystem import try_import_decord
decord = try_import_decord()

url = 'https://github.com/bryanyzhu/tiny-ucf101/raw/master/abseiling_k400.mp4'
video_fname = utils.download(url)
vr = decord.VideoReader(video_fname)
frame_id_list = range(0, 64, 2)
video_data = vr.get_batch(frame_id_list).asnumpy()
clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

################################################################
# Now we define transformations for the video clip.
# This transformation function does three things:
# center crop the image to 224x224 in size,
# transpose it to ``num_channels*num_frames*height*width``,
# and normalize with mean and standard deviation calculated across all ImageNet images.

transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
clip_input = transform_fn(clip_input)
clip_input = np.stack(clip_input, axis=0)
clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
print('Video data is downloaded and preprocessed.')

################################################################
# Next, we load a pre-trained I3D model.

#model_name = 'i3d_inceptionv1_kinetics400'
model_name = 'i3d_resnet50_v1_custom'
net = get_model(model_name, nclass=3, pretrained=False)
net.load_parameters('0.7203-custom-i3d_resnet50_v1_custom-029-best.params')
print('%s model is successfully loaded.' % model_name)


################################################################
# Note that if you want to use InceptionV3 series model (i.e., i3d_inceptionv3_kinetics400),
# please resize the image to have both dimensions larger than 299 (e.g., 340x450) and change input size from 224 to 299
# in the transform function. Finally, we prepare the video clip and feed it to the model.

pred = net(nd.array(clip_input))

#classes = net.classes
classes = ['No interaction', 'Interact with a left rack', 'Interact with a right rack']
topK = 3
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input video clip is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.'%
          (classes[ind[i]], nd.softmax(pred)[0][ind[i]].asscalar()))


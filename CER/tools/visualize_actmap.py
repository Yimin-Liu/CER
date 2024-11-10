"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.

Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""
import numpy as np
import os.path as osp
import argparse
import cv2
import torch
from torch.nn import functional as F
from torch import nn
from HCCL import datasets
from torch.utils.data import DataLoader
from HCCL.utils.data import transforms as T
from HCCL.utils.data.preprocessor import Preprocessor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import errno
import pickle
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
from torch.autograd import Variable
from HCCL.evaluators import Evaluator, extract_features

from HCCL import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
        model,
        test_loader,
        save_dir,
        width,
        height,
        use_gpu,
        img_mean=None,
        img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    # model.eval()

    # for target in list(test_loader.keys()):
    data_loader = test_loader  # only process query images
    # original images and activation maps are saved individually
    actmap_dir = osp.join(save_dir, 'actmap_' + 'query')
    mkdir_if_missing(actmap_dir)
    print('Visualizing activation maps for {} ...'.format('query'))
    for batch_idx, (imgs, fnames, pids, cids, _) in enumerate(data_loader):
        # imgs, paths = data['img'], data['impath']
        paths = fnames
        if use_gpu:
            imgs = imgs.cuda()

        # forward to get convolutional feature maps
        try:
            outputs = model.module.base[0:5](imgs)
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )

        if outputs.dim() != 4:
            raise ValueError(
                'The model output is supposed to have '
                'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                'Please make sure you set the model output at eval mode '
                'to be the last convolutional feature maps'.format(
                    outputs.dim()
                )
            )

        # compute activation maps
        outputs = (outputs ** 2).sum(1)
        # outputs = outputs.sum(1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.normalize(outputs, p=2, dim=1)
        # outputs = (outputs - torch.min(outputs, dim=1, keepdim=True)[0]) / (torch.max(outputs, dim=1, keepdim=True)[0] -
        #                                                                     torch.min(outputs, dim=1, keepdim=True)[0])
        outputs = outputs.view(b, h, w)

        if use_gpu:
            imgs, outputs = imgs.cpu(), outputs.cpu()

        for j in range(outputs.size(0)):
            # get image name
            path = paths[j]
            imname = osp.basename(osp.splitext(path)[0])

            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8
            )
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:,
            width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

        if (batch_idx + 1) % 10 == 0:
            print(
                '- done batch {}/{}'.format(
                    batch_idx + 1, len(data_loader)
                )
            )


def create_model(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                              num_classes=0, pooling_type=args.pooling_type)
    else:
        model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate
                              , pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
    model.cuda()
    model = nn.DataParallel(model)
    return model


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        # >>> from torchreid.utils import load_checkpoint
        # >>> fpath = 'log/my_model/model.pth.tar-10'
        # >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        # >>> from torchreid.utils import load_pretrained_weights
        # >>> weight_path = 'log/my_model/model-best.pth.tar'
        # >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # model.load_state_dict(checkpoint['state_dict'])

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        # if k.startswith('module.'):
        #     k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
                format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
            )


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model._modules.items():
            print(module_name)
            if module_name == 'fc':
                return conv_output, x
            x = module(x)  # Forward
            # print(module_name, module)
            if module_name == self.target_layer:
                print('True')
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model.fc(x)
        return conv_output


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output = self.extractor.forward_pass(input_image)
        # if target_index is None:
        #     target_index = np.argmax(model_output.data.numpy())
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_index] = 1
        # Zero grads
        # self.model.fc.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        # model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('--weights', type=str,
                        default='/home/dsp/workspace/HCCL_ReID/examples/logs/aug05/model-best.pth.tar')
    parser.add_argument('--save-dir', type=str, default='/home/dsp/workspace/HCCL_ReID/logs')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=100)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('/home/dsp/workspace/HCCL_ReID/examples', 'data'))
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pooling-type', type=str, default='avg')  # avg
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, dataset.query)

    model = create_model(args)

    if use_gpu:
        model = model.cuda()
        model.eval()
        # if args.weights and check_isfile(args.weights):

        # checkpoint = load_checkpoint(args.weights)

        load_pretrained_weights(model, args.weights)
        # model.load_state_dict(checkpoint['state_dict'])

    # evaluator = Evaluator(model)
    # evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    visactmap(
        model, test_loader, args.save_dir, args.width, args.height, use_gpu
    )

    # image = cv2.imread('/home/dsp/workspace/HCCL_ReID/examples/data/market1501/Market-1501-v15.09.15/bounding_box_train/0331_c2s1_088196_01.jpg')
    # image_prep = preprocess_image(image)
    # grad_cam = GradCam(model, target_layer='layer4')
    # cam = grad_cam.generate_cam(image_prep.cuda(), None)
    # save_class_activation_on_image(image, cam, '123')


if __name__ == '__main__':
    main()

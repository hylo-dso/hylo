import os
import torch
import torch.distributed as dist
import random
import numpy as np

from torchvision import datasets, transforms
from skimage.io import imread
from torch.utils.data import Dataset
from joblib import Parallel, delayed
# unet dataset helper import
from medpy.filter.binary import largest_connected_component
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.transform import rescale, rotate

def get_dataset(dataset, data_dir, batch_size, world_size, rank, local_rank):
        if dataset == 'fashion-mnist':
                transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.2862,), (0.3529,))])

                transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.2862,), (0.3529,))])

                os.makedirs(data_dir, exist_ok=True)

                download = True if local_rank == 0 else False
                if not download: dist.barrier()

                train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_train)
                test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform_test)

                if download: dist.barrier()
        elif dataset == 'cifar10':
                transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                
                transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                
                os.makedirs(data_dir, exist_ok=True)
                
                download = True if local_rank == 0 else False
                if not download: dist.barrier()
                
                train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
                test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform_test)
                
                if download: dist.barrier()
        elif dataset == 'cifar100':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
                
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
                
                os.makedirs(data_dir, exist_ok=True)
                
                download = True if local_rank == 0 else False
                if not download: dist.barrier()
                
                train_dataset = datasets.CIFAR100(
                        root=data_dir, train=True, download=download, transform=transform_train)
                test_dataset = datasets.CIFAR100(
                        root=data_dir, train=False, download=download, transform=transform_test)
                
                if download: dist.barrier()
        elif dataset == 'imagenet':
                train_dataset = datasets.ImageFolder(
                        data_dir + '/train',
                        transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]))
                test_dataset = datasets.ImageFolder(
                        data_dir + '/validation',
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]))
        elif dataset == 'brain-segmentation':
                return get_brain_segmentation_dataset(data_dir, batch_size, world_size, rank, local_rank)
        else:
                raise NotImplementedError
            
        return get_sampler_and_loader(train_dataset, test_dataset, batch_size, world_size, rank)


def get_sampler_and_loader(train_dataset, test_dataset, batch_size, world_size, rank):
        torch.set_num_threads(8)
        kwargs = {"num_workers": 8, "pin_memory": True}
        kwargs['pin_memory'] = False
        kwargs["prefetch_factor"] = 16
        kwargs["persistent_workers"] = True
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank)
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, sampler=test_sampler, **kwargs)
        return train_sampler, train_loader, test_sampler, test_loader

# Unet
class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=False,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]
        


        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])

        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        # self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]
        def joblib_loop():
            return Parallel(n_jobs=16)(delayed(resize_sample)(v, size=image_size) for v in self.volumes)
        # self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]
        self.volumes = joblib_loop()
        
        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform
        
        
        print("ATTENTION::DATASET, dataset length is ", len(self.patient_slice_index))

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor

def unet_transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return transforms.Compose(transform_list)


class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


def get_brain_segmentation_dataset(data_dir, batch_size, world_size, rank, local_rank ):
    train_dataset = BrainSegmentationDataset(
        images_dir=data_dir,
        subset="train",
        image_size=256,
        transform=unet_transforms(scale=0.05, angle=15, flip_prob=0.5),
    )
    val_dataset = BrainSegmentationDataset(
        images_dir=data_dir,
        subset="validation",
        image_size=256,
        random_sampling=False,
    )
    return make_sampler_and_loader(batch_size, world_size, rank, train_dataset, val_dataset, single_process_val = True)


def worker_init(worker_id):
    np.random.seed(42 + worker_id)


def make_sampler_and_loader(batch_size, world_size, rank, train_dataset, val_dataset, single_process_val = False):
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    
    if single_process_val:
        # making sampler/loader for unet
        kwargs["worker_init_fn"] = worker_init
        kwargs["drop_last"] = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=train_sampler, **kwargs)
    if single_process_val:
        kwargs["drop_last"] = False
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, **kwargs)
    
    # \TODO: check whether Unet val dataset needs sampler or not.
    else:
        kwargs["drop_last"] = False
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank)
        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, **kwargs)

    return train_sampler, train_loader, val_sampler, val_loader


def dsc(y_pred, y_true, lcc=True):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    if lcc and np.any(y_pred): 
        y_pred = largest_connected_component(y_pred)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))


def crop_sample(x):
    volume, mask = x
    volume[volume < np.max(volume) * 0.1] = 0
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask


def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask


def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume


def log_images(x, y_true, y_pred, channel=1):
    images = []
    x_np = x[:, channel].cpu().numpy()
    y_true_np = y_true[:, 0].cpu().numpy()
    y_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(x_np.shape[0]):
        image = gray2rgb(np.squeeze(x_np[i]))
        image = outline(image, y_pred_np[i], color=[255, 0, 0])
        image = outline(image, y_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images


def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image

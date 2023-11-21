import numpy as np
import torch
from kornia.augmentation import RandomErasing, RandomThinPlateSpline
from PIL import Image
import torchvision.transforms.functional as tvf

# cut the part of the circle (RandomErasing)
# def get_erased_image(image: torch.Tensor):
#     """
#     - scale: range of proportion of erased area against input image.
#     - ratio: range of aspect ratio of erased area.
#     """
#     aug = RandomErasing(scale=(0.2, 0.2), ratio=(1.0, 1.0), p=1.0, 
#                         value=0.0, same_on_batch=False, keepdim=True)
#     aug_image = aug(image)

#     n_ = 0
#     while (aug_image == image).all() or \
#     (int(torch.sum(aug_image != image)) < 1000 or int(torch.sum(aug_image != image)) > 2000):
#         aug_image = aug(image)
        
#         n_ += 1
#         if n_ > 50:
#             # import pdb; pdb.set_trace()
#             raise Exception("RandomErasing didn't cut part of the circle! \
#             Try other parameter for the RandomErasing!")

#     # aug_image_pil = tvf.to_pil_image(aug_image)
#     # aug_image_pil.save(f'img_{idx}.png')
#     # print(f"{idx} image: changed pixels = {int(torch.sum(aug_image != image))}")
#     return aug_image

# cut the part of the circle (manually)
def get_erased_image(image: torch.Tensor, mode: str):
    if mode == 'slightly':
        n_idxs_to_modify = 500
    elif mode == 'hard':
        n_idxs_to_modify = 1000

    # 3 channels in black&white image are the same, so take only the 1-st channel for simplicity
    image = image[0]
    edge_idxs = torch.argwhere(image > 0.5)

    n_ = int(len(edge_idxs) / n_idxs_to_modify)
    if n_ == 0:
        edge_idxs_to_cut = edge_idxs[:n_idxs_to_modify]
    else:
        start_i = np.random.randint(n_, size=1)[0]
        edge_idxs_to_cut = edge_idxs[start_i*n_idxs_to_modify : (start_i+1)*n_idxs_to_modify]

    linear_idxs = edge_idxs_to_cut[:, 0] * image.shape[0] + edge_idxs_to_cut[:, 1]
    aug_image = torch.reshape(image, (-1,)).detach().clone()
    aug_image[linear_idxs] = 0
    aug_image = torch.reshape(aug_image, image.shape)
    aug_image = aug_image.expand(3, aug_image.shape[0], aug_image.shape[1])
    
    # idx = np.random.randint(10, size=1)[0]
    # aug_image_pil = tvf.to_pil_image(aug_image)
    # aug_image_pil.save(f'corrupted_images_slightly/img_erased_{idx}.png')

    return aug_image


# add some noise (Gaussian - RandomPlasmaShadow / RandomGaussianNoise / RandomThinPlateSpline)
def get_noisy_image(image: torch.Tensor):
    aug = RandomThinPlateSpline(scale=0.4, align_corners=False, same_on_batch=False, p=1.0, keepdim=True)
    aug_image = aug(image)

    # idx = np.random.randint(10, size=1)[0]
    # aug_image_pil = tvf.to_pil_image(aug_image)
    # aug_image_pil.save(f'corrupted_images_hard/img_noisy_{idx}.png')
        
    return aug_image

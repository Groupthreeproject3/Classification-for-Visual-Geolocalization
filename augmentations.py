
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

#We defined this class to use it for the adjust Brightness transformation:
class BrightnessTransform:
    def __init__(self, brightness):
      """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
      self.brightness=brightness
    def __call__(self, images):
      assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
      B, C, H, W = images.shape
      brightened_images=[F.adjust_brightness(img,self.brightness).unsqueeze(0) for img in images]
      brightened_images=torch.cat(brightened_images)
      assert brightened_images.shape == torch.Size([B, C, H, W])
      return brightened_images

class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size, scale):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images

#We used this class for the RandomPerspective transformation 
class DeviceAgnosticRandomPerspective(T.RandomPerspective):
    def __init__(self, distortion_scale=0, p=0, fill=0):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(distortion_scale=distortion_scale, p=p, fill=fill)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        random_perspective = super(DeviceAgnosticRandomPerspective, self).forward
        augmented_images = [random_perspective(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

#We used this class for the Random Erasing transformation 
class DeviceAgnosticRandomErasing(T.RandomErasing):
    def __init__(self, p, scale, ratio, value, inplace):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        random_erasing = super(DeviceAgnosticRandomErasing, self).forward
        augmented_images = [random_erasing(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images

if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch = random_crop(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()


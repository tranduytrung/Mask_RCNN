import imgaug.augmenters as iaa

class AddFloat(iaa.Augmenter):
    def __init__(self, value, name=None, deterministic=False, random_state=None):
        super(AddFloat, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.value = value

    def _augment_images(self, images, random_state, parents, hooks):
        n_images = len(images)
        add_values = random_state.uniform(low=self.value[0], high=self.value[1], size=n_images)
        dest_images = []
        for idx in range(n_images):
            dest_images.append(images[idx] + add_values[idx])

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images
    
    def get_parameters(self):
        return [self.image_paths]
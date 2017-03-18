class WholeProbeToyGenerator:
    def __init__(self, data, probes, batch_size):
        self._data = data
        self._probes = probes
        self._batch_size = batch_size
        self._batch_start_index = 0
        self._size = len(probes)
        self._current_index = 0

    def __iter__(self):
        return self

    def next(self):
        res_images = []
        probe_ids = []
        image_sizes = []
        if self._current_index + self._batch_size > self._size:
            Log.info('Data ended, starting from the beginning')
            self._current_index = 0
        for probe in self._probes[self._current_index:self._current_index + self._batch_size]:
            image = probe.getImage()
            image = cv2.resize(image, dsize=(300, 300))
            image_size = image.shape[:2]
            res_images.append(image)
            image_sizes.append(image_size)

            probe_ids.append(probe.pk)
            self._current_index += self._batch_size
        # res_masks = self._data.create_binary_image(probe_ids, image_sizes)
        # # Convert to TF format
        # res_images = numpy.array(res_images)
        # res_masks = numpy.array(res_masks)
        # res_masks_shape = res_masks.shape
        # # res_masks_desired_shape = res_masks_shape + (1)
        # res_masks_desired_shape = res_masks_shape[::-1]
        # # return res_images.transpose(0, 3, 2, 1), res_masks

        labels = numpy.random.randint(0, 2, self._batch_size)
        return numpy.array(res_images), np_utils.to_categorical(labels, 2)

    def __len__(self):
        return 2
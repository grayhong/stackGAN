"""
code from
- https://github.com/Guim3/StackGAN/blob/master/data/DatasetReader.py
"""

import os  # for reading files and directories
from scipy import misc  # Read images

"""DatasetReader: reads dataset images and texts into numpy arrays.
Also, an optional normalization can be applied over all the data."""


class DatasetReader:

    def __init__(self, dataset, path=None, normalize=True):
        """ __dataset = (str) dataset type: ['cub',  'oxford-102']
            __path = (str) path to the dataset. If not specified, it will look to './datasets/{cub | oxford-102}'
            __normalization = (bool) if True, normalizes all the images of the dataset to be [-1, 1]."""

        # Check that dataset type is valid
        dataset = dataset.lower()
        assert dataset == 'cub' or dataset == 'oxford-102', \
            "Dataset not recognized: %s. Must be 'cub' or 'oxford-102'" % dataset

        self.__dataset = dataset

        # Initialize path, if not specified
        if path is None:
            self.__path =  './datasets/' + self.__dataset

        # Check if path exists
        assert os.path.exists(self.__path), "Path %s does not exist" % self.__path

        self.__normalize = normalize

    def read(self):
        """Reads the images and texts of the dataset found in self.__path.
        Output:
            · images: numpy array N x height x width x color, where N is the number of samples
            · texts: numpy array NxM, where M is the dimensionality of the texts.
            · labels: """

        print('Reading dataset...')

        if self.__dataset == 'cub' or self.__dataset == 'oxford-102':
            images, texts, labels = self.__read_cub_oxford_dataset()
        else:
            raise NameError('Dataset not implemented')

        # The normalization can be performed here as long as the image format of all datasets is the same.
        # If not, normalization will be performed inside each dataset specific methods.
        if self.__normalize:
            images = ((images / 255) * 2) - 1 # change[0, 255] to [-1, 1]

        return images, texts, labels

    def __read_cub_oxford_dataset(self):

        data_path = os.path.join(self.__path + "/images_and_texts/")
        assert data_path, "Didn't find 'images_and_texts' folder in %s" % self.__path

        # List all files
        folder_list = os.listdir(data_path)
        folder_list.sort()
        folder_iterator = filter(lambda element: os.path.isdir(data_path + element), folder_list)

        # Output variables
        images = np.empty([0,0,0,0], dtype=int) # This array shape will be set once the first image is read
        labels = []                             # This list will be converted to a 1-dimensional array
        texts = []                              # This is a list, as it will store strings of different lengths

        first_iteration = True  # We need to read an image first to get its size

        for folder_idx, folder in enumerate(folder_iterator): # Every folder contains images from the same label / class

            # List all images and text files within the folder
            file_list = os.listdir(data_path + folder)
            file_list.sort() # Order is important because image and text files need to match

            # Filter images and texts using their extension
            im_list = list(filter(lambda x: x.endswith(('.jpg')), file_list))
            txt_iterator = filter(lambda x: x.endswith(('.txt')), file_list)

            # Get image size to initialize images array
            if first_iteration == True:
                im_sz = misc.imread( os.path.join( os.path.join(data_path, folder), im_list[0] ) ).shape

            # Initialize temporal array that will be concatenated to the main 'images' array
            tmp_images = np.zeros([len(im_list), im_sz[0], im_sz[1], im_sz[2]], dtype=int)

            i = 0 # Index for 'tmp_images' array
            for im_file, txt_file in zip(im_list, txt_iterator):

                # Sanity check: make sure image and text file match
                tmp1 = im_file
                tmp2 = txt_file
                assert tmp1.rsplit( ".", 1)[0] == tmp2.rsplit( ".", 1)[0], ("Image '%s' and text file '%s' don't " + \
                         "have the same name.\n" + \
                        " It seems that some file is missing, you should check or download again the %s dataset.") \
                        % (im_file, txt_file, self.__dataset)

                element_path = os.path.join(data_path, folder)

                # Read image
                im = misc.imread(os.path.join(element_path, im_file))

                # Sanity check: make sure all images have same size
                assert im_sz == im.shape, ("Images from %s dataset have different sizes: " + \
                                           str(im_sz) +  ", " + str(im.shape) +  ". Have you preprocessed" + \
                                          " the dataset so all images have equal size?\n" + \
                                           "Conflicting image: %s") % (self.__dataset, os.path.join(element_path, im_file))

                tmp_images[i,] = im


                # Read texts
                with open(os.path.join(element_path, txt_file), 'r') as f:
                    lines = f.readlines()
                texts.append(lines)

                # Set label
                labels.append(folder_idx)

                i += 1

            if first_iteration:
                images = tmp_images
                first_iteration = False
            else:
                # Concatenate tmp_images to images (1st dimension concat)
                images = np.vstack((images, tmp_images))

            print("\t\t%d/%d" % (folder_idx, len(folder_list)), end='\r')


        print("Done!")

        labels = np.asarray(labels)

        return images, texts, labels


if __name__ == '__main__':
    rd = DatasetReader('cub')
    rd.read()

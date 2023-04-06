import cv2
import numpy
#from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST

# Global Variables
float_tolerance = 1e-7

# Test
def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    #image = image.astype('float32')
    # Blur and double size of input image - base of pyramid.
    base_image = generateBaseImage(image, sigma, assumed_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    return base_image

'''Doubles size of input image and applies gaussian blur'''
def generateBaseImage(image, sigma, assumed_blur):
    # DOubles image size
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    #
    sigma_diff = numpy.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

'''Compute number of octaves in image pyramid as function of base image shape (OpenCV default)
'''
def computeNumberOfOctaves(image_shape):
    return int(round(numpy.log(min(image_shape)) / numpy.log(2) - 1))

'''Creates a list of the amount of blur for each image in each layer.
Each layer has num_intervals + 3  images.
All images within a layer have the same width/height, but they are sucessively blurred.
+3 comes from num_intervals + 1 images to cover num_intervals steps from one blur value to its double.
    Then, another +2 for 1 blur step before the first image in the layer, and 1 blur step after
    the last image in the layer.
    Those extra are needed when theadjacent gaussian images are subtracted.'''
def generateGaussianKernels(sigma, num_intervals):
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1 / num_intervals)
    # Scale of gaussian blur necessary to go from one blur scale to the next within an octave.
    gaussian_kernels = numpy.zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = numpy.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

'''Start generating image pyramids.
Sucessively blur images with gaussian kernels.'''
def generateGaussianImages(image, num_octaves, gaussian_kernels):
    gaussian_images = []
    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)
        # Skip first value as start image already has that blur value.
        for gaussian_kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX = gaussian_kernel, sigmaY = gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        # Half third-to-last image as this has the desired blur. Use to begin next layer.
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation = cv2.INTER_NEAREST)
    return numpy.array(gaussian_images)

'''Subtract sucessive pairs of gaussian blurred images.
Uses opencv's subtract().
dog_images[2][i] = gaussian_images[2][i+1] - gaussian_images[2][i]'''
def generateDoGImages(gaussian_images):
    dog_images = []
    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(cv2.subtract(second_image, first_image))
        dog_images.append(dog_images_in_octave)
        return numpy.array(dog_images)
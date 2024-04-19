from matplotlib import pyplot as plt
import numpy as np

def get_image(image_path):
    image = plt.imread(image_path)
    return image/255.0


def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, image_path):
    plt.imsave(image_path, image)


def error(original_image: np.ndarray, clustered_image: np.ndarray) -> float:
    # Returns the Mean Squared Error between the original image and the clustered image
    sq_error = (original_image - clustered_image)**2
    return (sq_error).mean()

def plot_error(mse)-> None:
    x =['2','5','10','20','50']
    plt.xlabel("num_clusters")
    plt.ylabel("mean_squared_error")
    plt.grid(True,color='gray', linestyle='dashed')
    plt.bar(x,mse)
    plt.show()
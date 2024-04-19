from model import KMeans
from utils import get_image, show_image, save_image, error, plot_error


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape
    show_image(image)
    # reshape image
    r_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    clusters = [2,5,10,20,50]
    err=[]
    for i in clusters:
        # create model
        num_clusters = i # CHANGE THIS
        kmeans = KMeans(num_clusters)
        print("cluster k :", i)
        # fit model
        kmeans.fit(r_image)

        # replace each pixel with its closest cluster center
        imag = kmeans.replace_with_cluster_centers(r_image)

        # reshape image
        image_clustered = imag.reshape(img_shape)

        # Print the error
        MSE = error(image, image_clustered)
        err.append(MSE)
        print('MSE:', MSE)

        # show/save image
        #show_image(image)
        #show_image(image_clustered)
        save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')
    plot_error(err)


if __name__ == '__main__':
    main()

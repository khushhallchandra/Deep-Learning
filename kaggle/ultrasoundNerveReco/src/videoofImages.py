
def grays_to_RGB(img):
    # Convert a 1-channel grayscale image into 3 channel RGB image
    return np.dstack((img, img, img))


def image_plus_mask(img, mask):
    # Returns a copy of the grayscale image, converted to RGB, 
    # and with the edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # chan 0 = bright red
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color


def to_mask_path(f_image):
    # Convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(f_image)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)


def add_masks(images):
    # Return copies of the group of images with mask outlines added
    # Images are stored as dict[filepath], output is also dict[filepath]
    images_plus_masks = {} 
    for f_image in images:
        img  = images[f_image]
        mask = cv2.imread(to_mask_path(f_image))
        images_plus_masks[f_image] = image_plus_mask(img, mask)
    return images_plus_masks

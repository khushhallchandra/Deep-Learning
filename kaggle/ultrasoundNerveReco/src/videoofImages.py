
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
    
def get_patient_images(patient):
    # Return a dict of patient images, i.e. dict[filepath]
    f_path = IMAGE_DIR + '%i_*.tif' % patient 
    f_ultrasounds = [f for f in glob.glob(f_path) if 'mask' not in f]
    images = {f:get_image(f) for f in f_ultrasounds}
    return images


def image_features(img):
    return tile_features(img)   # a tile is just an image...


def tile_features(tile, tile_min_side = TILE_MIN_SIDE):
    # Recursively split a tile (image) into quadrants, down to a minimum 
    # tile size, then return flat array of the mean brightness in those tiles.
    tile_x, tile_y = tile.shape
    mid_x = tile_x / 2
    mid_y = tile_y / 2
    if (mid_x < tile_min_side) or (mid_y < tile_min_side):
        return np.array([tile.mean()]) # hit minimum tile size
    else:
        tiles = [ tile[:mid_x, :mid_y ], tile[mid_x:, :mid_y ], 
                  tile[:mid_x , mid_y:], tile[mid_x:,  mid_y:] ] 
        features = [tile_features(t) for t in tiles]
        return np.array(features).flatten()
    

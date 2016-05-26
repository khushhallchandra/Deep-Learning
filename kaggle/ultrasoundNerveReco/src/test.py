def image_plus_mask(img, mask):
    # Returns a copy of the grayscale image, converted to RGB, 
    # and with the edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # chan 0 = bright red
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color

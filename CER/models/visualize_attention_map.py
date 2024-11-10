import numpy as np
import cv2


def visualize_attention_map(attention_map):
    attention_map_color = np.zeros(shape=[attention_map.shape[0], attention_map[1], 3], dtype=np.uint8)

    red_color_map = np.zeros(shape=[attention_map.shape[0], attention_map[1]], dtype=np.uint8) + 255

    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 2] = red_color_map


    return  attention_map_color

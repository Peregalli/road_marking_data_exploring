import json

# Define labels for different datasets
dataset_labels = {
    "CeyMo": {
        "num_classes": 11,
        "labels": {
            "bus_lane": (0, 255, 255),
            "cycle_lane": (0, 128, 255),
            "diamond": (178, 102, 255),
            "junction_box": (255, 255, 51),
            "left_arrow": (255, 102, 178),
            "pedestrian_crossing": (255, 255, 0),
            "right_arrow": (255, 0, 127),
            "straight_arrow": (255, 0, 255),
            "slow": (0, 255, 0),
            "straight_left_arrow": (255, 128, 0),
            "straight_right_arrow": (255, 0, 0)
            }
    },
    "bdd100k": {
        "num_classes": 8,
        "labels": {
            "crosswalk": (219, 94, 86),
            "double other": (219, 194, 86),
            "double white": (145, 219, 86),
            "double yellow": (86, 219, 127),
            "road curb": (86, 211, 219),
            "single other": (86, 111, 219),
            "single white": (160, 86, 219),
            "single yellow": (219, 86, 178)
        }
    },
    "apolloscape_lanemark": {
        "num_classes": 38,
        "labels": {
                "void": (0,   0,   0),
                "s_w_d": (70, 130, 180),
                "s_y_d": (220,  20,  60),
                "ds_w_dn": (128,   0, 128),
                "ds_y_dn": (255, 0,   0),
                "sb_w_do": (0,   0,  60),
                "sb_y_do": (0,  60, 100),
                "b_w_g": (0,   0, 142),
                "b_y_g": (119,  11,  32),
                "db_w_g": (244,  35, 232),
                "db_y_g": (0,   0, 160),
                "db_w_s": (153, 153, 153),
                "s_w_s": (220, 220,   0),
                "ds_w_s": (250, 170,  30),
                "s_w_c": (102, 102, 156),
                "s_y_c": (128,   0,   0),
                "s_w_p": (128,  64, 128),
                "s_n_p": (238, 232, 170),
                "c_wy_z": (190, 153, 153),
                "a_w_u": (0,   0, 230),
                "a_w_t": (128, 128,   0),
                "a_w_tl": (128,  78, 160),
                "a_w_tr": (150, 100, 100),
                "a_w_tlr": (255, 165,   0),
                "a_w_l": (180, 165, 180),
                "a_w_r": (107, 142,  35),
                "a_w_lr": (201, 255, 229),
                "a_n_lu": (0,   191, 255),
                "a_w_tu": (51, 255,  51),
                "a_w_m": (250, 128, 114),
                "a_y_t": (127, 255,   0),
                "b_n_sr": (255, 128,   0),
                "d_wy_za": (0, 255, 255),
                "r_wy_np": (178, 132, 190),
                "vom_wy_n": (128, 128,  64),
                "om_n_n": (102,   0, 204),
                "noise": (0, 153, 153),
                "ignored": (255, 255, 255),
        }
    }
}


# Save dataset labels to a JSON file
def save_labels_to_json(labels, filename):
    with open(filename, "w") as file:
        json.dump(labels, file, indent=4)


# Example usage:
save_labels_to_json(dataset_labels, "dataset_labels.json")

import pdb

def write_img(group, name, data):
    first_image = data[0]
    height, width, channels = first_image.shape
    images_ds = group.create_dataset(
        name,
        shape=(len(data), height, width, channels),
        dtype=first_image.dtype,
        compression="gzip",
        compression_opts=4
    )
    for i, img in enumerate(data):
        images_ds[i] = img

def write_array(group, name, data):
    group.create_dataset(
        name,
        data=data,
        dtype=data.dtype,
        compression="gzip",
        compression_opts=4
    )
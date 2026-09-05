"""Resize reference images to the requested box and the model minimum dimensions."""

def prepare_reference(path, size, multiple):
    from PIL import Image, ImageOps
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if size < 1024:
        image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        minimum = ((64 + multiple - 1) // multiple) * multiple
        dimensions = tuple(max(minimum, edge // multiple * multiple) for edge in image.size)
    else:
        dimensions = tuple(max(64, edge) for edge in image.size)
    if dimensions != image.size:
        image = image.resize(dimensions, Image.Resampling.LANCZOS)
    return image

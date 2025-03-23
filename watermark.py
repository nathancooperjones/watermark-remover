import math
import random
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


def watermark_image(
    image: Image.Image,
    text: str = 'WATERMARK',
    font_path: Optional[str] = None,
    font_size: Union[int, Tuple[int, int]] = 36,
    color: Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]] = (200, 200, 200, 128),
    angle: Union[float, Tuple[float, float]] = 45.0,
    text_padding: Union[int, Tuple[int, int]] = 20,
    draw_grid: bool = False,
    grid_spacing: Union[int, Tuple[int, int]] = 50,
    line_thickness: Union[int, Tuple[int, int]] = 1,
    logo_path: Optional[str] = None,
    randomize_spacing: bool = True,
    randomize_angle: bool = True,
    randomize_size: bool = True,
    randomize_color: bool = True,
    multiple_colors: bool = False,
    opacity_range: Tuple[float, float] = (0.5, 0.9),
    logo_scale_range: Tuple[float, float] = (0.8, 1.2),
    logo_rotation_range: Tuple[float, float] = (-15, 15)
) -> Image.Image:
    """
    Adds a repeated diagonal watermark (text + optional recolored logo) and an optional grid to an
    image. Enhanced version with randomization and multiple colors for more challenging watermark
    removal.

    Parameters
    ----------
    image: Image.Image
        Input image (PIL Image)
    text: str
        Watermark text repeated across the image
    font_path: Optional[str]
        Path to a TrueType font. If None, uses a default PIL font with the requested size
    font_size: Union[int, Tuple[int, int]]
        Size of the watermark text font. Can be a single value or a range (min, max)
    color: Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]
        RGBA color(s) for text and grid lines. Can be a single color or a list of colors
    angle: Union[float, Tuple[float, float]]
        Angle in degrees to rotate the watermark pattern. Can be a single value or a range
    text_padding: Union[int, Tuple[int, int]]
        Extra space around each text + logo instance. Can be a single value or a range
    draw_grid: bool
        Whether to draw a (rotated) grid across the watermark pattern
    grid_spacing: Union[int, Tuple[int, int]]
        Distance between grid lines. Can be a single value or a range
    line_thickness: Union[int, Tuple[int, int]]
        Thickness of grid lines. Can be a single value or a range
    logo_path: Optional[str]
        Path to a small logo to be placed with the text
    randomize_spacing: bool
        Whether to randomize the spacing between watermark elements
    randomize_angle: bool
        Whether to randomize the rotation angle of each watermark element
    randomize_size: bool
        Whether to randomize the font size of each watermark element
    randomize_color: bool
        Whether to randomize the color of each watermark element
    multiple_colors: bool
        Whether to use multiple colors in the same watermark
    opacity_range: Tuple[float, float]
        Range for random opacity values (0.0 to 1.0)
    logo_scale_range: Tuple[float, float]
        Range for random logo scaling
    logo_rotation_range: Tuple[float, float]
        Range for random logo rotation

    Returns
    -------
    image: Image.Image
        New image with the diagonal watermark

    """
    # convert to RGBA for compositing
    base = image.convert('RGBA')
    w, h = base.size

    # calculate pattern size with extra padding to ensure coverage
    pattern_size = int(math.ceil(math.sqrt(w**2 + h**2) * 1.2))  # 20% extra padding
    pattern_layer = Image.new('RGBA', (pattern_size, pattern_size), (0, 0, 0, 0))
    draw_pattern = ImageDraw.Draw(pattern_layer)

    # load font
    if font_path:
        font = ImageFont.truetype(font=font_path, size=_get_random_value(font_size))
    else:
        font = ImageFont.load_default(size=_get_random_value(font_size))

    # load and process logo, if provided
    logo = None
    logo_width, logo_height = 0, 0
    if logo_path:
        logo = Image.open(logo_path).convert('RGBA')
        logo_width, logo_height = logo.size

    # get text bounding box
    bbox = draw_pattern.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # calculate maximum possible block size considering logo scaling and rotation
    max_logo_scale = max(logo_scale_range)

    # calculate maximum possible logo dimensions after scaling and rotation
    max_logo_dimension = max(logo_width, logo_height) * max_logo_scale
    # sqrt(2) accounts for maximum possible expansion due to rotation
    max_rotated_logo_dimension = int(max_logo_dimension * math.sqrt(2))

    # calculate base block size with maximum possible dimensions
    base_block_width = (
        max_rotated_logo_dimension + text_width + _get_random_value(value=text_padding)
    )
    base_block_height = (
        max(max_rotated_logo_dimension, text_height) + _get_random_value(value=text_padding)
    )

    # fill pattern layer with randomized spacing if requested
    y_pos = 0
    while y_pos < pattern_size:
        x_pos = 0
        while x_pos < pattern_size:
            # apply random spacing if enabled
            if randomize_spacing:
                # add random padding to both dimensions
                current_block_width = base_block_width + random.randint(-5, 5)
                current_block_height = base_block_height + random.randint(-5, 5)
            else:
                current_block_width = base_block_width
                current_block_height = base_block_height

            # create block for one repetition
            block_img = Image.new(
                mode='RGBA',
                size=(int(current_block_width), int(current_block_height)),
                color=(0, 0, 0, 0),
            )
            block_draw = ImageDraw.Draw(block_img)

            # process logo, if present
            if logo:
                # scale logo
                scale = _get_random_value(value=logo_scale_range)
                scaled_logo = logo.resize(
                    size=(int(logo_width * scale), int(logo_height * scale)),
                    resample=Image.Resampling.LANCZOS
                )

                # rotate logo
                logo_angle = _get_random_value(value=logo_rotation_range)
                rotated_logo = scaled_logo.rotate(angle=logo_angle, expand=True)

                # position logo
                logo_y = (current_block_height - rotated_logo.height) // 2

                # recolor logo
                logo_data = rotated_logo.load()
                for y in range(rotated_logo.height):
                    for x in range(rotated_logo.width):
                        r, g, b, a = logo_data[x, y]
                        if a > 0:
                            current_color = _get_random_color(
                                color=color,
                                opacity_range=opacity_range,
                                randomize_color=randomize_color or multiple_colors
                            )
                            logo_data[x, y] = (
                                current_color[0],
                                current_color[1],
                                current_color[2],
                                a,
                            )

                block_img.alpha_composite(rotated_logo, dest=(0, logo_y))

                # draw text with random properties, positioned after the rotated logo
                text_x = rotated_logo.width + 5  # small gap between logo and text
            else:
                text_x = 0

            text_y = (current_block_height - text_height) // 2

            # randomize font size if requested
            if randomize_size:
                current_font_size = _get_random_value(value=font_size)
                if font_path:
                    current_font = ImageFont.truetype(font=font_path, size=current_font_size)
                else:
                    current_font = ImageFont.load_default(size=current_font_size)
            else:
                current_font = font

            # draw text with current color
            current_color = _get_random_color(
                color=color,
                opacity_range=opacity_range,
                randomize_color=randomize_color or multiple_colors
            )
            block_draw.text(xy=(text_x, text_y), text=text, fill=current_color, font=current_font)

            # composite block onto pattern layer
            pattern_layer.alpha_composite(block_img, dest=(x_pos, y_pos))

            # move to next position with current block size
            x_pos += int(current_block_width)
        y_pos += int(current_block_height)

    # draw grid, if requested
    if draw_grid:
        current_spacing = _get_random_value(value=grid_spacing)
        current_thickness = _get_random_value(value=line_thickness)
        grid_color = _get_random_color(
            color=color,
            opacity_range=opacity_range,
            randomize_color=randomize_color or multiple_colors
        )

        for x in range(0, pattern_size, int(current_spacing)):
            draw_pattern.line(
                xy=[(x, 0), (x, pattern_size)],
                fill=grid_color,
                width=int(current_thickness),
            )
        for y in range(0, pattern_size, int(current_spacing)):
            draw_pattern.line(
                xy=[(0, y), (pattern_size, y)],
                fill=grid_color,
                width=int(current_thickness),
            )

    # rotate pattern
    final_angle = _get_random_value(value=angle) if randomize_angle else angle
    rotated_pattern = pattern_layer.rotate(angle=final_angle, expand=True)
    rp_w, rp_h = rotated_pattern.size

    # composite onto base image
    watermarked = base.copy()
    offset_x = (w - rp_w) // 2
    offset_y = (h - rp_h) // 2
    watermarked.alpha_composite(rotated_pattern, dest=(offset_x, offset_y))

    return watermarked


def _get_random_value(
    value: Union[int, float, Tuple[Union[int, float], Union[int, float]]],
) -> Union[int, float]:
    """
    Get a random value from a range.

    Parameters
    ----------
    value: int, float, or tuple of either int or float
        Value to get a random value from. Can be a single value or a range.

    Returns
    -------
    random_value: int or float
        Random value from the range.

    """
    if isinstance(value, (int, float)):
        return value

    return random.uniform(value[0], value[1])


def _get_random_color(
    color: Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]],
    opacity_range: Tuple[float, float] = (0.5, 0.9),
    randomize_color: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Get a random color.

    Parameters
    ----------
    color: tuple of ints
        Color to get a random color from. Can be a single color or a list of colors.
    opacity_range: tuple of floats
        Range for random opacity values (0.0 to 1.0)
    randomize_color: bool
        Whether to randomize the color of the watermark

    Returns
    -------
    random_color: tuple of ints
        Random color from the range.

    """
    if isinstance(color, list):
        return random.choice(color)

    if randomize_color:
        opacity = int(random.uniform(*opacity_range) * 255)

        # if we have a base color, vary it instead of completely randomizing
        if isinstance(color, tuple) and len(color) == 4:
            r, g, b, _ = color
            # vary each channel by ±30%
            r = int(max(0, min(255, r + random.uniform(-0.3, 0.3) * r)))
            g = int(max(0, min(255, g + random.uniform(-0.3, 0.3) * g)))
            b = int(max(0, min(255, b + random.uniform(-0.3, 0.3) * b)))

            return (r, g, b, opacity)

        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), opacity)

    return color

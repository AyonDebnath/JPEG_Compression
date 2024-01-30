import subprocess
from io import BytesIO
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from jpeg_decoder import *

import cv2
from tkinter import Tk, Canvas, mainloop


marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}

def calculate_ssim(image1, image2):
    # Convert the images to grayscale
    gray_image1 = image1.convert('L')
    gray_image2 = image2.convert('L')

    # Convert PIL images to numpy arrays
    array1 = np.array(gray_image1)
    array2 = np.array(gray_image2)

    # Calculate SSIM
    ssim_index, _ = ssim(array1, array2, full=True)

    return ssim_index

def convertImageWithSamplingFactor(input_image, output_image, sampling_factor):
    command = [
        "convert",
        input_image,
        "-sampling-factor",
        sampling_factor,
        output_image
    ]

    try:
        subprocess.run(command, check=True)
        print("Image converted with sampling factor 4:4:4.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def PrintMatrix(m):
    """
    A convenience function for printing matrices
    """
    for j in range(8):
        print("|", end="")
        for i in range(8):
            print("%d  |" % m[i + j * 8], end="\t")
        print()
    print()


def Clamp(col):
    """
    Makes sure col is between 0 and 255.
    """
    col = 255 if col > 255 else col
    col = 0 if col < 0 else col
    return int(col)


def ColorConversion(Y, Cr, Cb):
    """
    Converts Y, Cr and Cb to RGB color space
    """
    R = Cr * (2 - 2 * 0.299) + Y
    B = Cb * (2 - 2 * 0.114) + Y
    G = (Y - 0.114 * B - 0.299 * R) / 0.587
    return (Clamp(B + 128), Clamp(G + 128), Clamp(R + 128))


def DrawMatrix(x, y, matL, matCb, matCr, output, scaling_factor ):
    """
    Loops over a single 8x8 MCU and draws it on Tkinter canvas
    """

    for yy in range(8):
        for xx in range(8):
            x1, y1 = (x * 8 + xx) * scaling_factor, (y * 8 + yy) * scaling_factor

            # colour the entire block T\
            for i in range(scaling_factor):
                for j in range(scaling_factor):
                    output[x1+i][y1+j] = ColorConversion(
                matL[yy][xx], matCb[yy][xx], matCr[yy][xx]
            )


def DrawCompressed(x, y, comp_image, output, scaling_factor):
    comp_image = Image.open(BytesIO(comp_image))
    for yy in range(8):
        for xx in range(8):
            x1, y1 = (x * 8 + xx) * scaling_factor, (y * 8 + yy) * scaling_factor

            for i in range(scaling_factor):
                for j in range(scaling_factor):
                    output[x1+i][y1+j] = comp_image.getpixel((x, y))
    return


def RemoveFF00(data):
    """
    Removes 0x00 after 0xff in the image scan section of JPEG
    """
    datapro = []
    i = 0
    while True:
        b, bnext = unpack("BB", data[i: i + 2])
        if b == 0xFF:
            if bnext != 0:
                break
            datapro.append(data[i])
            i += 2
        else:
            datapro.append(data[i])
            i += 1
    return datapro, i


def GetArray(type, l, length):
    """
    A convenience function for unpacking an array from bitstream
    """
    s = ""
    for i in range(length):
        s = s + type
    return list(unpack(s, l[:length]))


def DecodeNumber(code, bits):
    l = 2 ** (code - 1)
    if bits >= l:
        return bits
    else:
        return bits - (2 * l - 1)

def hex_to_rgb(hex_color):
    # Convert hexadecimal color to RGB
    if(type(hex_color) == str):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:(i + 2)], 16) for i in (0, 2, 4))
    else:
        return (hex_color, hex_color, hex_color)



def create_image(hex_colors, width, height):

    # Fill the image with colors
    for x in range(width):
        for y in range(height):
            hex_color = hex_colors[x][y]
            rgb_color = hex_to_rgb(hex_color)
            hex_colors[x][y] = rgb_color

    return hex_colors

if __name__ == "__main__":

    input_image_path = "Images/lena.bmp"
    temp_converted_image_path = "Images/converted_image.jpeg"

    scaling_factor = 2

    convertImageWithSamplingFactor(input_image_path, temp_converted_image_path, "4:4:4")

    width, height = Image.open(temp_converted_image_path).size

    output = [[0 for _ in range(width*scaling_factor)] for _ in range(height*scaling_factor)]

    img = JPEG_decoder(temp_converted_image_path, output, scaling_factor)
    img.decode()

    # result_image = create_image(img.output, width*scaling_factor, height*scaling_factor)
    result_image = img.output

    # Display the image
    cv2.imshow('Result Image', np.array(result_image).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

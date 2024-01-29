import subprocess
from io import BytesIO
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from jpeg_decoder import *

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
    return (Clamp(R + 128), Clamp(G + 128), Clamp(B + 128))


def DrawMatrix(canvas, x, y, matL, matCb, matCr, scaling_factor=1):
    """
    Loops over a single 8x8 MCU and draws it on Tkinter canvas
    """
    for yy in range(8):
        for xx in range(8):
            c = "#%02x%02x%02x" % ColorConversion(
                matL[yy][xx], matCb[yy][xx], matCr[yy][xx]
            )
            x1, y1 = (x * 8 + xx) * scaling_factor, (y * 8 + yy) * scaling_factor
            x2, y2 = (x * 8 + (xx + 1)) * scaling_factor, (y * 8 + (yy + 1)) * scaling_factor
            canvas.create_rectangle(x1, y1, x2, y2, fill=c, outline=c)


def DrawCompressed(canvas, x, y, comp_image, scaling_factor=1):
    comp_image = Image.open(BytesIO(comp_image))
    for yy in range(8):
        for xx in range(8):
            c = "#%02x%02x%02x" % comp_image.getpixel((x, y))
            x1, y1 = (x * 8 + xx) * scaling_factor, (y * 8 + yy) * scaling_factor
            x2, y2 = (x * 8 + (xx + 1)) * scaling_factor, (y * 8 + (yy + 1)) * scaling_factor
            canvas.create_rectangle(x1, y1, x2, y2, fill=c, outline=c)
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

if __name__ == "__main__":

    input_image = "Images/lena.bmp"
    temp_converted_image = "Images/converted_image.jpeg"

    convertImageWithSamplingFactor(input_image, temp_converted_image, "4:4:4")

    width, height = Image.open(temp_converted_image).size

    master = Tk()
    w = Canvas(master, width=width * 2, height=height * 2)
    w.pack()
    img = JPEG_decoder(temp_converted_image, w)
    img.decode()
    mainloop()

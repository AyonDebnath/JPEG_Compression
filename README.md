# Honour's Project - Ayon 

 This project focuses on decoding baseline encoded images using Python. The code has been adapted from Yasoob Khalid's [Baseline-JPEG-Decoder](https://github.com/yasoob/Baseline-JPEG-Decoder).

## How to Run the Decoder

1. Clone this repository to your local machine.
2. Navigate to the project directory in your terminal.
3. Run the decoder by executing the `main.py` file.

```bash
python3 main.py
```

4. Enter the path of the image you want to decode when prompted.
5. Provide the x and y coordinates of the block you want to keep the same as the original image.

The decoded image will be saved in the Images directory with the name *original_file_name*_output.png.

## Note on Image Resolution
Please ensure that the images you use with the JPEG decoder have dimensions that are multiples of 8. This is necessary because the decoder breaks down the image into smaller 8 by 8 blocks during its processing.

Sample images used for testing are stored in the Images directory.

If you have any questions, please reach out.
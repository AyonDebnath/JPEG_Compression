from struct import unpack
import huffmanTable
import main
import idct as IDCT
from stream import *

class JPEG_decoder:
    """
    JPEG class for decoding a baseline encoded JPEG image
    """

    def __init__(self, image_file, output, scaling_factor, coordinate):
        self.huffman_tables = {}
        self.quant = {}
        self.quantMapping = []
        self.output = output
        self.scaling_factor = scaling_factor
        self.coordinate = coordinate
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def DefineQuantizationTables(self, data):
        while (len(data) > 0):
            (hdr,) = unpack("B", data[0:1])
            self.quant[hdr] = main.GetArray("B", data[1: 1 + 64], 64)
            data = data[65:]

    def BuildMatrix(self, st, idx, quant, olddccoeff):
        i = IDCT.IDCT()

        code = self.huffman_tables[0 + idx].GetCode(st)
        bits = st.GetBitN(code)
        dccoeff = main.DecodeNumber(code, bits) + olddccoeff

        i.base[0] = (dccoeff) * quant[0]
        l = 1
        while l < 64:
            code = self.huffman_tables[16 + idx].GetCode(st)
            if code == 0:
                break

            # The first part of the AC key_len
            # is the number of leading zeros
            if code > 15:
                l += code >> 4
                code = code & 0x0F

            bits = st.GetBitN(code)

            if l < 64:
                coeff = main.DecodeNumber(code, bits)
                i.base[l] = coeff * quant[l]
                l += 1

        i.rearrange_using_zigzag()
        i.perform_IDCT()

        return i, dccoeff

    def StartOfScan(self, data, hdrlen):
        data, lenchunk = main.RemoveFF00(data[hdrlen:])

        st = Stream(data)
        oldlumdccoeff, oldCbdccoeff, oldCrdccoeff = 0, 0, 0
        for y in range(self.height // 8):
            for x in range(self.width // 8):

                matL, oldlumdccoeff = self.BuildMatrix(
                    st, 0, self.quant[self.quantMapping[0]], oldlumdccoeff
                )
                matCr, oldCrdccoeff = self.BuildMatrix(
                    st, 1, self.quant[self.quantMapping[1]], oldCrdccoeff
                )
                matCb, oldCbdccoeff = self.BuildMatrix(
                    st, 1, self.quant[self.quantMapping[2]], oldCbdccoeff
                )
                if(x == self.coordinate.getX() and y - self.coordinate.getY()):
                    # continue
                    main.DrawCompressed(x, y, self.img_data, self.output, self.scaling_factor)
                main.DrawMatrix(x, y, matL.base, matCb.base, matCr.base, self.output, self.scaling_factor)
        return lenchunk + hdrlen

    def BaselineDCT(self, data):
        hdr, self.height, self.width, components = unpack(">BHHB", data[0:6])
        # print("size %ix%i" % (self.width,  self.height))

        for i in range(components):
            id, samp, QtbId = unpack("BBB", data[6 + i * 3: 9 + i * 3])
            self.quantMapping.append(QtbId)

    def decodeHuffman(self, data):
        while (len(data) > 0):
            offset = 0
            (header,) = unpack("B", data[offset: offset + 1])
            # print(header, header & 0x0F, (header >> 4) & 0x0F)
            offset += 1

            lengths = main.GetArray("B", data[offset: offset + 16], 16)
            offset += 16

            elements = []
            for i in lengths:
                elements += main.GetArray("B", data[offset: offset + i], i)
                offset += i

            hf = huffmanTable.HuffmanTable()
            hf.GetHuffmanBits(lengths, elements)
            self.huffman_tables[header] = hf
            data = data[offset:]

    def decode(self):
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            print(hex(marker))
            # print(marker_mapping.get(marker))
            if marker == 0xFFD8:
                lenchunk = 2
                data = data[lenchunk:]
            elif marker == 0xFFD9:
                break
            else:
                (len_chunk,) = unpack(">H", data[2:4])
                len_chunk += 2
                chunk = data[4:len_chunk]
                if marker == 0xFFC4:
                    self.decodeHuffman(chunk)
                elif marker == 0xFFDB:
                    self.DefineQuantizationTables(chunk)
                elif marker == 0xFFC0:
                    self.BaselineDCT(chunk)
                elif marker == 0xFFDA:
                    len_chunk = self.StartOfScan(data, len_chunk)
                data = data[len_chunk:]
            if len(data) == 0:
                break

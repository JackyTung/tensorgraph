//  MNIST.h
/*************************************************************************
 * MNIST Dataset Loader
 *------------------------------------------------------------------------
 * Copyright (c) 2016 Peter Baumann
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would
 *    be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not
 *    be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 *
 *************************************************************************/
//--------------------------------------------------------------------------------------
// http://yann.lecun.com/exdb/mnist/
// Yann LeCun (Courant Institute, NYU), Corinna Cortes (Google Labs, New York)
// and Christopher J.C. Burges (Microsoft Research, Redmond)
//
// The MNIST database of handwritten digits, has a training set of 60,000 examples,
// and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
// The digits have been size-normalized and centered in a fixed-size image (28x28)
//--------------------------------------------------------------------------------------

#pragma once
#ifndef MNIST_h
#define MNIST_h

#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// BIG-Endian to LITTLE-Endian byte swap
#define swap16(n) (((n&0xFF00)>>8)|((n&0x00FF)<<8))
#define swap32(n) ((swap16((n&0xFFFF0000)>>16))|((swap16(n&0x0000FFFF))<<16))

typedef unsigned char           byte;


struct MNISTchar {
    std::vector<float> pixelData;          // Store the 784 (28x28) pixel color values (0-255) of the digit-image
    std::vector<float> output;             // Store the expected output (e.g: label 5 / output 0,0,0,0,0,1,0,0,0,0)
    int label;                              // Store the handwritten digit in number form
    MNISTchar() : pixelData(std::vector<float>()), output(std::vector<float>(10)), label(0) {}
};



class MNIST {
public:
    const std::vector<MNISTchar> trainingData;  // Set of 60.000 handwritten digits to train the net
    const std::vector<MNISTchar> testData;      // Set of 10.000 different handwritten digits to test the net
    
    
    MNIST(const std::string& path)
    :   trainingData(getMNISTdata(path + "train-images-idx3-ubyte", path + "train-labels-idx1-ubyte")),
        testData(getMNISTdata(path + "t10k-images-idx3-ubyte", path + "t10k-labels-idx1-ubyte")) {
            if(!this->trainingData.size()) { std::cout <<"ERROR: parsing training data" <<std::endl; }
            if(!this->testData.size()) { std::cout <<"ERROR: parsing testing data" <<std::endl; }
    }
    
    
private:
    std::vector<MNISTchar> getMNISTdata(const std::string& imagepath, const std::string& labelpath) {
        std::vector<MNISTchar> tmpdata = std::vector<MNISTchar>();
        std::fstream file (imagepath, std::ifstream::in | std::ifstream::binary);
        int magicNum_images = 0, magicNum_labels = 0;
        int itemCount_images = 0, itemCount_labels = 0;
        // READ THE IMAGE FILE DATA
        if(file.is_open()) {
            int row_count = 0, col_count = 0;
            // FILE HEADER INFO is stored as 4 Byte Integers
            file.read((char*)&magicNum_images, 4);
            file.read((char*)&itemCount_images, 4);
            file.read((char*)&row_count, 4);
            file.read((char*)&col_count, 4);
            // Transform Byte values from big to little endian
            magicNum_images = swap32(magicNum_images);
            itemCount_images = swap32(itemCount_images);
            row_count = swap32(row_count);
            col_count= swap32(col_count);
            // Loop throug all the items and store every pixel of every row
            for (int i = 0; i < itemCount_images; i++) {
                MNISTchar tmpchar = MNISTchar();
                for(int r = 0; r < (row_count * col_count); r++) {
                    byte pixel = 0;
                    // read one byte (0-255 color value of the pixel)
                    file.read((char*)&pixel, 1);
                    tmpchar.pixelData.push_back((float)pixel / 255);
                }
                tmpdata.push_back(tmpchar);
            }
        }
        file.close();
        // READ THE LABEL FILE DATA
        file.open(labelpath, std::ifstream::in | std::ifstream::binary);
        if (file.is_open()) {
            file.read((char*)&magicNum_labels, 4);
            file.read((char*)&itemCount_labels, 4);
            magicNum_labels = swap32(magicNum_labels);
            itemCount_labels = swap32(itemCount_labels);
            if(itemCount_images == itemCount_labels) {
                // read all the labels and store them in theire MNISTchars
                for(MNISTchar& m : tmpdata) {
                    file.read((char*)&m.label, 1);
                    m.output[m.label] = 1.0f;
                }
            }
        }
        file.close();
        return tmpdata;
    }
    
    
public:
    void testPrintout(int startChar, int endChar) const {
        for(int i = startChar; i < endChar; i++) {
            std::cout <<"------------------------------" <<std::endl;
            int count = 0;
            for (const float& r : trainingData[i].pixelData) {
                if(count < 27) {
                    if(r < 0.25) std::cout <<" ";
                    else if(r < 0.5) std::cout <<"-";
                    else if(r < 0.75) std::cout <<"+";
                    else if(r <= 1.0) std::cout <<"#";
                    ++count;
                } else {
                    std::cout <<std::endl;
                    count = 0;
                }
            }
            std::cout <<" Expected Output: ";
            for(const short& x : trainingData[i].output) { std::cout <<x; }
            std::cout <<std::endl;
            std::cout <<"\t\tThis is a: " <<trainingData[i].label  <<std::endl;
            std::cout <<"------------------------------" <<std::endl;
        }
    }
    
    
};

#endif

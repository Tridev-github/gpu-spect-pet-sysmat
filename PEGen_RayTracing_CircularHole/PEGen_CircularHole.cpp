#define _CRT_SECURE_NO_WARNINGS

// PEGen_CircularHole.cpp
// Description:
//   This program calculates the photon-electric system matrix.
//
// Usage:
//   ./PEGen_CircularHole -cuda <cuda_device_id>
//
// Author: Xingchun Zheng @ tsinghua university
// Last Modified: 2024/12/22
// Version: 1.0

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <stdint.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "PESysMatGen.h"

using namespace std;

static void die(const char* msg) {
    perror(msg);
    exit(1);
}

int main(int argc, char* argv[])
{
    float* parameter_Collimator = new float[80000]();
    float* parameter_Detector   = new float[80000]();
    float* parameter_Image      = new float[100]();
    float* parameter_Physics    = new float[100]();

    FILE* fid = fopen("Params_Collimator.dat", "rb");
    if (!fid) die("open Params_Collimator.dat");
    fread(parameter_Collimator, sizeof(float), 80000, fid);
    fclose(fid);

    FILE* fid1 = fopen("Params_Detector.dat", "rb");
    if (!fid1) die("open Params_Detector.dat");
    fread(parameter_Detector, sizeof(float), 80000, fid1);
    fclose(fid1);

    FILE* fid2 = fopen("Params_Image.dat", "rb");
    if (!fid2) die("open Params_Image.dat");
    fread(parameter_Image, sizeof(float), 100, fid2);
    fclose(fid2);

    FILE* fid3 = fopen("Params_Physics.dat", "rb");
    if (!fid3) die("open Params_Physics.dat");
    fread(parameter_Physics, sizeof(float), 100, fid3);
    fclose(fid3);

    ////////////////////////////////////////////////////
    int numCollimatorLayers = (int)floor(parameter_Collimator[0] + 0.001f);
    float FOV2Collimator0 = parameter_Image[11];

    for (int id_CollimatorLayer = 0; id_CollimatorLayer < numCollimatorLayers; id_CollimatorLayer++)
    {
        cout << "############ Collimator " << id_CollimatorLayer << " ############" << endl;
        cout << "Number of collimator holes = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 0] << endl;
        cout << "Width of collimator layer(X direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 1] << "mm" << endl;
        cout << "Thickness of collimator layer(Y direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 2] << "mm" << endl;
        cout << "Height of collimator layer(Z direction) = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 3] << "mm" << endl;
        cout << "Collimator Layer to 1st Collimator Layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 4] << "mm" << endl;
        cout << "Total Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 5] << endl;
        cout << "Photon-electric Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 6] << endl;
        cout << "Compton Coeff of collimator layer = " << parameter_Collimator[(id_CollimatorLayer + 1) * 10 + 7] << endl;
    }
    cout << "FOV center to 1st Collimator = " << FOV2Collimator0 << endl;

    ////////////////////////////////////////////////////
    int cuda_id = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-cuda") == 0 && i + 1 < argc)
        {
            cuda_id = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            cout << "Usage: " << argv[0] << " [-cuda GPU_ID]" << endl;
            return 0;
        }
        else
        {
            cerr << "Unknown parameter or missing argument: " << argv[i] << endl;
            cout << "Usage: " << argv[0] << " [-cuda GPU_ID] " << endl;
            return EXIT_FAILURE;
        }
    }

    ////////////////////////////////////////////////////
    int numProjectionsingle = (int)floor(parameter_Detector[0] + 0.001f);

    int numImageVoxelX = (int)floor(parameter_Image[0] + 0.001f);
    int numImageVoxelY = (int)floor(parameter_Image[1] + 0.001f);
    int numImageVoxelZ = (int)floor(parameter_Image[2] + 0.001f);

    float widthImageVoxelX = parameter_Image[3];
    float widthImageVoxelY = parameter_Image[4];
    float widthImageVoxelZ = parameter_Image[5];

    int numRotation_ = (int)floor(parameter_Image[6] + 0.001f);
    float angelPerRotation = parameter_Image[7];

    float shiftFOVX = parameter_Image[8];
    float shiftFOVY = parameter_Image[9];
    float shiftFOVZ = parameter_Image[10];

    const int numProjectionSingle = numProjectionsingle;
    const int numRotation = numRotation_;

    const int fullX = numImageVoxelX;
    const int fullY = numImageVoxelY;
    const int fullZ = numImageVoxelZ;

    const int64_t fullImagebin = (int64_t)fullX * (int64_t)fullY * (int64_t)fullZ;
    const int64_t totalFloats  = (int64_t)numProjectionSingle * fullImagebin * (int64_t)numRotation;
    const int64_t totalBytes   = totalFloats * (int64_t)sizeof(float);

    printf("FOV dimension : %d %d %d\n", fullX, fullY, fullZ);
    printf("FOV Voxel Size(mm) : %f %f %f\n", widthImageVoxelX, widthImageVoxelY, widthImageVoxelZ);
    printf("numDet=%d  fullVox=%lld  rotations=%d  sysmatBytes=%lld (%.2f GB)\n",
           numProjectionSingle, (long long)fullImagebin, numRotation,
           (long long)totalBytes, (double)totalBytes / (1024.0*1024.0*1024.0));

    // ---------------------------
    // TRUE 3D via Z-slabs (exact)
    // ---------------------------
    const int slabZ = 4; // tune: 4 or 8. Start with 4.
    const int slabBinsMax = fullX * fullY * slabZ;

    char Fname[2048];
    sprintf(Fname, "PE_SysMat_shift_%f_%f_%f.sysmat", shiftFOVX, shiftFOVY, shiftFOVZ);

    FILE* fp = fopen(Fname, "wb+");
    if (!fp) die("open sysmat output");

    // pre-size file so random fseek writes are valid
    if (totalBytes <= 0) {
        fprintf(stderr, "ERROR: totalBytes invalid\n");
        exit(2);
    }
    fseek(fp, (long)(totalBytes - 1), SEEK_SET);
    fputc(0, fp);
    fflush(fp);

    std::vector<float> slabHost((size_t)numProjectionSingle * (size_t)slabBinsMax, 0.0f);

    // Kernel needs these two extra params
    // parameter_Image[30] = fullZ
    // parameter_Image[31] = z0
    parameter_Image[30] = (float)fullZ;

    for (int idxRotation = 0; idxRotation < numRotation; idxRotation++)
    {
        cout << "########################" << endl;
        cout << "Rotation (" << idxRotation << ") processing ..." << endl;
        cout << "########################" << endl;

        cout << "Shift FOV in X = " << shiftFOVX << "mm" << endl;
        cout << "Shift FOV in Y = " << shiftFOVY << "mm" << endl;
        cout << "Shift FOV in Z = " << shiftFOVZ << "mm" << endl;

        parameter_Image[20] = (float)idxRotation;

        float saved_fullZ_param = parameter_Image[2]; // keep original

        for (int z0 = 0; z0 < fullZ; z0 += slabZ)
        {
            int thisSlabZ = slabZ;
            if (z0 + thisSlabZ > fullZ) thisSlabZ = fullZ - z0;

            const int thisSlabBins = fullX * fullY * thisSlabZ;

            // Tell kernel local Z
            parameter_Image[2]  = (float)thisSlabZ;
            parameter_Image[31] = (float)z0;

            // compute slab into slabHost (we only use first thisSlabBins per detector)
            int q = PESysMatGen(parameter_Collimator, parameter_Detector, parameter_Image,
                                slabHost.data(), cuda_id);

            if (q != thisSlabBins) {
                fprintf(stderr, "ERROR: slab bins mismatch q=%d expected=%d\n", q, thisSlabBins);
                exit(3);
            }

            // Write slab into correct file offsets (det-major)
            for (int det = 0; det < numProjectionSingle; det++)
            {
                int64_t detBase = (int64_t)idxRotation * (int64_t)numProjectionSingle * fullImagebin
                                + (int64_t)det * fullImagebin;

                int64_t voxelOffset = (int64_t)z0 * (int64_t)fullX * (int64_t)fullY;

                int64_t fileFloatOffset = detBase + voxelOffset;
                int64_t fileByteOffset  = fileFloatOffset * (int64_t)sizeof(float);

                const float* src = slabHost.data() + (int64_t)det * (int64_t)thisSlabBins;

                fseek(fp, (long)fileByteOffset, SEEK_SET);
                fwrite(src, sizeof(float), (size_t)thisSlabBins, fp);
            }

            printf("WROTE slab z0=%d slabZ=%d bins=%d\n", z0, thisSlabZ, thisSlabBins);
            fflush(fp);
        }

        parameter_Image[2] = saved_fullZ_param;
    }

    fclose(fp);

    cout << "########################" << endl;
    cout << "Photon Electric 3D Sysmat Written (slabbed, exact)." << endl;
    cout << "########################" << endl;

    delete[] parameter_Collimator;
    delete[] parameter_Detector;
    delete[] parameter_Image;
    delete[] parameter_Physics;

    return 0;
}

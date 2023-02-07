# BIRL: Benchmark on Image Registration methods with Landmark validations

This is an adapted version of the work provided in [BIRL](https://github.com/Borda/BIRL). It incorporates the low-resolution Elastix registration stage, as outlined in the method proposed by Moles Lopez et al., [2015](https://doi.org/10.1136/amiajnl-2014-002710). Additionally, a pre-alignment step has been added, utilizing the orientation of the tissue masks for improved accuracy. The implementation of this enhanced method can be accessed [here](experiments/low-high_res_elastix.py).

## Usage

This code is designed to be used with the publicly available [dataset](https://anhir.grand-challenge.org/Data/) from the ANHIR challenge, as specified in the original pipeline. However, it can also be used with other datasets, as long as the input is formatted similarly to ANHIR's data.

For proper usage of the code, please refer to [BIRL](https://github.com/Borda/BIRL) for additional information.

For convenience, a Docker image has been created and can be found on [Docker Hub](). This allows for quick use with a few example image pairs. The command to download the image (if it has not already be done), create a container, and run it in interactive mode is provided.

```sh
docker run --rm -v /your/mounted/volume:/outputs -it aelskens/birl:low_res_prealigned bash
```
Please note, the volume argument, */your/mounted/volume*, should be changed by the user and consists of a directory which will be shared with the container in order to recover the outputs.

To use the "hello world" example, the following command should be typed in the container terminal.

```sh
./experiments/hello-world_example/hello-world_example.sh
```

The results are available in the directory that has been mounted. The output's structure is the following (with yyyymmdd corresponding to the date and x being a digit):
```
LowHighResElastix_yyyymmdd-xxxxxx
│   config.yml
│   elastix_LowREs_affine.txt
│   logging.txt
│   registration-results.csv
│   results-summary.csv
│   results-summary.txt
│
└───0
│   │   registration_visual_landmarks.jpg
│   │   image_refence-warped.jpg
│   │   result.png
│   │   ...
│   
└───1
│   │   registration_visual_landmarks.jpg
│   │   image_refence-warped.jpg
│   │   result.png
│   │   ...
│
│   ...
```

The results for each pair of images are present in each subfolders (e.g. 0, 1, 2, ...), respectively. To visualize the results, one can open the following files: 
* **fixed.png**: the fixed image
* **moving.png**: the moving image
* **registration_visual_landmarks.jpg**: the overlay of the fixed and moving images with the initial and estimated landmarks plotted
* **result.png**: the warped moving image
* **image_refence-warped.jpg**: the overlay of both warped moving and fixed images.

The quantitative results of the registration are available in **registration-results.csv** which can be found in the example root directory.

## References
Moles Lopez, X., Barbot, P., Van Eycke, Y.-R., Verset, L., Trépant, A.-L., Larbanoix, L., Salmon, I., & Decaestecker, C. (2015).
Registration of whole immunohistochemical slide images: An efficient way to characterize biomarker colocalization. 
Journal of the American Medical Informatics Association : JAMIA, 22(1), 86–99.

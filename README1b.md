
# Adaptive Histogram Equaliztion

This script performs adaptive histogram equalization for a serires of images in adaptive_hist_data.
In this script we segment the image MxN section.
We perform adaptive histogram eqaulization for each of these segmnents.


## Authors

- [@usnikchawla]


## Deployment

To deploy this project run

```bash
  python3 firstb.py M N 
```
M*N is the no of tiles the image is divied into.
Make sure M and N are less than the size of image.


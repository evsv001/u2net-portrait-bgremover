from pymatting import cutout
print('ol')
cutout(
    # input image path
    "new_image.jpg",
    # input trimap path
    "alphas.jpg",
    # output cutout path
    "lemur_cutout.png",
)
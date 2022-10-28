import os
import cv2
import math
import numpy as np


def ReadImage(ImageFolderPath):
    Images = []									
    if os.path.isdir(ImageFolderPath):
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[
                                 0]), ImageName] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x: x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1]
                             for i in range(len(ImageNames_Split))]
        for i in range(len(ImageNames_Sorted)):
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)

            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)
            Images.append(InputImage)
    else:
        print("\nEnter valid Image Folder Path.\n")

    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)

    return Images


def FindMatches(BaseImage, SecImage):
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(
        cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(
        cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])
    return GoodMatches, BaseImage_kp, SecImage_kp


def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)
    (HomographyMatrix, Status) = cv2.findHomography(
        SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)
    return HomographyMatrix, Status


def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    (Height, Width) = Sec_ImageShape
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)
    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))
    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())
    HomographyMatrix = cv2.getPerspectiveTransform(
        OldInitialPoints, NewFinalPonts)
    return [New_Height, New_Width], Correction, HomographyMatrix


def StitchImages(BaseImage, SecImage):
    SecImage_Cyl, mask_x, mask_y = ProjectOntoCylinder(SecImage)
    SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
    SecImage_Mask[mask_y, mask_x, :] = 255
    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage_Cyl)
    HomographyMatrix, Status = FindHomography(
        Matches, BaseImage_kp, SecImage_kp)
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(
        HomographyMatrix, SecImage_Cyl.shape[:2], BaseImage.shape[:2])
    SecImage_Transformed = cv2.warpPerspective(
        SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(
        SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    BaseImage_Transformed = np.zeros(
        (NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0],
                          Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage
    StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(
        BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))
    return StitchedImage


def Convert_xy(x, y):
    global center, f
    xt = (f * np.tan((x - center[0]) / f)) + center[0]
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]
    return xt, yt


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1100
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    AllCoordinates_of_ti = np.array(
        [np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]
    ii_x, ii_y = Convert_xy(ti_x, ti_y)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)
    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h-2))
    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]
    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]
    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y
    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx) * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx) * (dy)
    TransformedImage[ti_y, ti_x, :] = (weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x, :]) + \
                                      (weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :]) + \
                                      (weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x, :]) + \
                                      (weight_br[:, None] *
                                       InitialImage[ii_tl_y + 1, ii_tl_x + 1, :])
    min_x = min(ti_x)

    TransformedImage = TransformedImage[:, min_x: -min_x, :]
    return TransformedImage, ti_x-min_x, ti_y
if __name__ == "__main__":
    # Reading images.
    Images = ReadImage("InputImages/Road")

    BaseImage, _, _ = ProjectOntoCylinder(Images[0])
    for i in range(1, len(Images)):
        StitchedImage = StitchImages(BaseImage, Images[i])

        BaseImage = StitchedImage.copy()

    cv2.imwrite("Stitched_Panorama.png", BaseImage)

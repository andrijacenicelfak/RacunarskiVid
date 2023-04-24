import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def stich(img1, img2):
    detector = cv.SIFT_create()

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flan = cv.FlannBasedMatcher(index_params, search_params)
    matches = flan.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(list(good_matches)) < 10:
        print("Not enough matches...")
        return
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1,1,2)
    pts2 = cv.perspectiveTransform(np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1,1,2), M)
    pts = np.concatenate((pts1, pts2), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    image = cv.warpPerspective(img1, Ht.dot(M), (xmax - xmin, ymax - ymin))
    image[t[1]:rows2+t[1], t[0]:cols2+t[0]] = img2

    return image

if __name__ == '__main__':
    img1 = cv.imread("1.jpg")
    img2 = cv.imread("2.jpg")
    img3 = cv.imread("3.jpg")

    image = stich(img1, img2)
    image2 = stich(img3, image)
    plt.imshow(image2)
    plt.show()


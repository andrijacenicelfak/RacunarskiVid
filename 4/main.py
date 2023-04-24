# import the necessary packages
import cv2
import imutils
import numpy as np

def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image
	while True:
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                        
def contains(rect1, rect2):
      return rect1[0] >= rect2[0] and rect1[1] <= rect2[1] and rect1[2] <= rect2[2] and rect2[1] < rect1[1] + rect1[0] and rect2[2] < rect1[2] + rect1[0] and( rect1[1] + rect1[0] >= rect2[1] + rect2[0] or rect1[2] + rect1[0] >= rect2[2] + rect2[0])

if __name__ == '__main__':
    rows = open("./googlenet/synset_words.txt").read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    net = cv2.dnn.readNetFromCaffe("./googlenet/bvlc_googlenet.prototxt", "./googlenet/bvlc_googlenet.caffemodel")
    image_original = cv2.imread("./image.png")
    confidence_threshold = 0.72
    image = image_original[135:(135+720), 82:(82+1440)]
    
    rectangles = list()

    for resized in pyramid(image, scale=2, minSize=(180, 180)):
        for (x, y, img) in sliding_window(resized, 180, (180, 180)):
            if img.shape[0] != 180 or img.shape[1] != 180:
                continue

            blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (105, 117, 123))
            scale = resized.shape[0] / image.shape[0]
            net.setInput(blob)
            preds = net.forward()
            idxs = np.argsort(preds[0])[::-1][:5]
            for (i, idx) in enumerate(idxs):
                if i==0 and preds[0][idx] > confidence_threshold:
                    # print(classes[idx])
                    if not ("dog" in classes[idx] or "cat" in classes[idx]):
                          break
                    # if "dog" in classes[idx]:
                    #     cv2.imshow(str(((scale, x, y, "DOG" if "dog" in classes[idx] else "CAT"))), imutils.resize(img, width=600))
                    #     cv2.waitKey(0)
                    rectangles.append((scale, x, y, "DOG" if "dog" in classes[idx] else "CAT"))
                    break
	
    rectangles.sort(key=lambda x: x[0])
    # print(rectangles)
    done = list()
    for (scale, x, y, text) in rectangles:
        nx = int(x / scale)
        ny = int(y * scale)
        size = int(180 / scale)
        rect = (size, nx, ny)
        should_skipp = False
        for r in done:
            if contains(r, rect):
                  should_skipp = True
        if should_skipp:
            continue
        cv2.rectangle(image, (nx, ny), (nx + size, ny + size), (0, 0, 255) if text == "CAT" else (0, 255, 255), 2)
        cv2.putText(image, text, (nx + 10, ny + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if text == "CAT" else (0, 255, 255), 2)
        done.append(rect)

    # print(done)
    cv2.imwrite("output.jpg", image)
	    
    cv2.imshow("Image", image)
    cv2.waitKey(0)

